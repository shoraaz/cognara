"""
app/retrieval/reranker.py
--------------------------
Reranks hybrid search results using the Vertex AI Ranking API, the final
precision stage of Layer 2's retrieval pipeline (see ADR 0005).

WHY THIS FILE EXISTS:
  Vector search and BM25 (fused via hybrid_search.py) are both FAST,
  APPROXIMATE relevance signals — computed without ever letting a model
  actually read the query and each candidate chunk together. A reranker
  does the opposite: it takes the fused candidate list and scores each one
  with a model that reads the query and chunk TOGETHER, producing a much
  more precise relevance judgment at the cost of being too slow to run over
  the whole corpus. The standard retrieval pattern is: cheap/approximate
  methods narrow thousands of candidates to a few dozen, then an expensive/
  precise reranker picks the true best few from that narrowed set.

WHY THE VERTEX AI RANKING API, VIA LANGCHAIN'S VertexAIRank (see ADR 0005):
  Consistent with this project's GCP-first commitment (ADR 0002) and
  LangChain-as-component-layer approach (ADR 0004). VertexAIRank extends
  LangChain's BaseDocumentCompressor — so this reranker is a fully compatible
  LangChain citizen, usable inside a ContextualCompressionRetriever if needed.
  No local cross-encoder model to download, version, or run.

REQUIRES: the Discovery Engine API enabled on the GCP project (the underlying
service VertexAIRank calls — a separate API from Vertex AI's generation/
embedding endpoints).

METADATA RESTORATION — VertexAIRank silently discards custom metadata:
  VertexAIRank.compress_documents() strips all metadata except a positional
  "id" field (the 0-based index of each document in the input list, as a
  string), the relevance_score, and the field named by title_field. Our
  chunk_id, page_number, course_name, and all other fields are gone from
  the reranker's output. We track the original candidate order ourselves
  and use the positional "id" to look back up the full original Document
  and attach the reranker's relevance_score onto it.
  See BUG_FIX_LOG.md "Reranker: VertexAIRank silently discards custom metadata".

# Interview notes: local-notes/INTERVIEW_PREP.md — "app/retrieval/reranker.py"
"""

from langchain_core.documents import Document
from langchain_google_community.vertex_rank import VertexAIRank

from app.core.config import settings
from app.core.logging import get_logger
from app.retrieval.hybrid_search import HybridResult

logger = get_logger(__name__)


def _get_reranker(top_n: int) -> VertexAIRank:
    """
    Construct a VertexAIRank instance. NOT cached as a module-level singleton —
    same cautious default applied to generation.py's ChatVertexAI: VertexAIRank's
    underlying discoveryengine client may have async gRPC event-loop binding risk.
    Construction is cheap (no network round trip), so the safety default costs nothing.
    """
    return VertexAIRank(
        project_id=settings.GCP_PROJECT_ID,
        location_id="global",           # Ranking API is only available at "global" location
        ranking_config="default_ranking_config",
        title_field="topic",            # use our chunk's topic as the "title" signal for the ranker
        top_n=top_n,
    )


def rerank(query: str, candidates: list[HybridResult], top_n: int = 5) -> list[Document]:
    """
    Rerank a list of hybrid search candidates against `query` using the
    Vertex AI Ranking API, returning the top_n Documents with their
    ORIGINAL full metadata restored plus a new "relevance_score" field.

    Args:
        query: the user's question.
        candidates: HybridResult objects from hybrid_search(). Only their
            .document field is passed to the reranker — RRF scores and
            per-method ranks are not relevant to the reranker's own judgment.
        top_n: how many reranked results to return.

    Returns:
        Up to top_n Document objects, each with full original metadata
        plus "relevance_score", ordered by that score descending.
    """
    if not candidates:
        return []

    # Preserve the input order — VertexAIRank's positional "id" field in
    # the output refers to this list's indices ("0", "1", ...).
    original_documents = [c.document for c in candidates]
    reranker = _get_reranker(top_n=top_n)

    logger.info("rerank_start", candidate_count=len(original_documents))
    reranked = reranker.compress_documents(original_documents, query)
    logger.info("rerank_done", returned=len(reranked))

    # VertexAIRank's output strips all metadata except "id" (a positional
    # string: "0", "1", ...) and "relevance_score". To restore our original
    # metadata (chunk_id, page_number, course_name, etc.), we:
    #   1. Parse "id" as an integer to get the original list position.
    #   2. Look up the original Document (full metadata) at that position.
    #   3. Copy the reranker's relevance_score onto the original metadata.
    # See BUG_FIX_LOG.md "Reranker: VertexAIRank silently discards custom metadata".
    restored_results: list[Document] = []
    for reranked_doc in reranked:
        # "id" is a position-based string index ("0", "1", ...) matching the
        # input list — NOT our chunk_id.
        position = int(reranked_doc.metadata["id"])
        original_doc = original_documents[position]

        # Build restored metadata: copy original fields, then add the reranker's score.
        restored_metadata = dict(original_doc.metadata)
        restored_metadata["relevance_score"] = reranked_doc.metadata["relevance_score"]

        restored_results.append(Document(
            page_content=original_doc.page_content,
            metadata=restored_metadata,
        ))

    return restored_results
