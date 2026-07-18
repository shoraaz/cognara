"""
app/retrieval/reranker.py
----------------------------
Reranks hybrid search results using the Vertex AI Ranking API, the final
stage of Layer 2's retrieval pipeline (see ADR 0005).

WHY THIS FILE EXISTS:
  Vector search and BM25 (fused via hybrid_search.py) are both FAST,
  APPROXIMATE relevance signals — cosine similarity and term-frequency
  statistics, computed without ever letting a model actually read the
  query and each candidate chunk together. A reranker does the opposite:
  it takes the fused candidate list and scores each one with a model
  that reads the query and chunk TOGETHER, producing a much more precise
  relevance judgment at the cost of being too slow to run over the whole
  corpus. The standard retrieval pattern is: cheap/approximate methods
  narrow thousands of candidates down to a few dozen, then an expensive/
  precise reranker picks the true best few from that narrowed set.

WHY THE VERTEX AI RANKING API, VIA LANGCHAIN'S VertexAIRank (see ADR 0005):
  Consistent with this project's GCP-first commitment (ADR 0002) and
  LangChain-as-component-layer approach (ADR 0004). VertexAIRank extends
  LangChain's BaseDocumentCompressor, the standard interface for
  "narrow down a document list by relevance" — so this reranker is a
  fully compatible LangChain citizen, usable directly inside a
  ContextualCompressionRetriever if we ever want that composition.
  No local cross-encoder model to download, version, or run.

REQUIRES: the Discovery Engine API enabled on the GCP project (this is
the underlying service VertexAIRank calls — a separate API from Vertex
AI's generation/embedding endpoints, enabled specifically for this
module).

REAL BUG FOUND AND FIXED — VertexAIRank DISCARDS CUSTOM METADATA:
  The first version of this module passed our Documents straight into
  VertexAIRank.compress_documents() and returned its output directly.
  A real test caught this immediately: every returned Document's
  metadata was reduced to just {"id": "0"/"1"/..., "relevance_score":
  ..., "topic": ...} — our chunk_id, page_number, course_name, and every
  other field were silently gone. Confirmed by direct inspection of the
  real API response, not assumed: the "id" field is a synthetic,
  POSITION-BASED index (matching the input list's order — "0" is
  whatever Document was first in the list passed to
  compress_documents(), regardless of what our own chunk_id was), and
  only the field named by title_field ("topic", in our config) survives
  alongside it.

  FIX: track the original candidate list's order ourselves, and after
  reranking, use each result's positional "id" to look back up the
  ORIGINAL Document (with full metadata) from our own candidates list —
  then attach the reranker's relevance_score onto that original,
  metadata-complete Document instead of trusting VertexAIRank's
  stripped-down copy.

INTERVIEW EXPLANATION:
  "My retrieval pipeline is a funnel: hybrid search pulls a wide net of
  candidates using fast, approximate signals, then the Vertex AI Ranking
  API reranks that narrowed set with a model that actually reads the
  query and each chunk together. I found, by testing rather than
  assuming, that the LangChain wrapper around this API silently discards
  custom metadata — it only preserves a positional ID, the relevance
  score, and one designated title field. I fixed this by tracking the
  original candidate order myself and re-attaching the full original
  metadata after reranking, using the position-based ID to match results
  back to their source documents. This is a good example of why I test
  every new integration against real output instead of trusting a
  library's documented behaviour blindly."
"""

from langchain_core.documents import Document
from langchain_google_community.vertex_rank import VertexAIRank

from app.core.config import settings
from app.core.logging import get_logger
from app.retrieval.hybrid_search import HybridResult

logger = get_logger(__name__)


def _get_reranker(top_n: int) -> VertexAIRank:
    """
    Construct a VertexAIRank instance. NOT cached as a module-level
    singleton — see app/services/generation.py's documented lesson about
    async gRPC clients binding to whatever event loop is active on
    first use. VertexAIRank's underlying discoveryengine client has the
    same category of risk, so we apply the same safe default here even
    though it hasn't been proven to fail the same way (untested; cheap
    to construct, so there's no real cost to being cautious).
    """
    return VertexAIRank(
        project_id=settings.GCP_PROJECT_ID,
        location_id="global",  # the Ranking API is only available at the "global" location
        ranking_config="default_ranking_config",
        title_field="topic",   # use our chunk's topic as the "title" signal
        top_n=top_n,
    )


def rerank(query: str, candidates: list[HybridResult], top_n: int = 5) -> list[Document]:
    """
    Rerank a list of hybrid search candidates against `query` using the
    Vertex AI Ranking API, returning the top_n Documents — with their
    ORIGINAL full metadata restored (see REAL BUG FOUND AND FIXED above)
    plus a new "relevance_score" field from the reranker — in the API's
    precise relevance order.

    Args:
        query: the user's question.
        candidates: HybridResult objects from hybrid_search() — this
            function only needs their .document field; RRF scores and
            per-method ranks are not passed to the reranker (the
            reranker computes its own, independent relevance judgment).
        top_n: how many reranked results to return.

    Returns:
        Up to top_n Document objects, each with full original metadata
        plus "relevance_score", ordered by that score descending.
    """
    if not candidates:
        return []

    original_documents = [c.document for c in candidates]
    reranker = _get_reranker(top_n=top_n)

    logger.info("rerank_start", candidate_count=len(original_documents))
    reranked = reranker.compress_documents(original_documents, query)
    logger.info("rerank_done", returned=len(reranked))

    # VertexAIRank's "id" field is the POSITION of each document in the
    # list we passed in (as a string: "0", "1", ...) — use it to look
    # back up the original, metadata-complete Document. See the module
    # docstring's REAL BUG FOUND AND FIXED note for why this is necessary.
    restored_results: list[Document] = []
    for reranked_doc in reranked:
        position = int(reranked_doc.metadata["id"])
        original_doc = original_documents[position]

        restored_metadata = dict(original_doc.metadata)
        restored_metadata["relevance_score"] = reranked_doc.metadata["relevance_score"]

        restored_results.append(Document(
            page_content=original_doc.page_content,
            metadata=restored_metadata,
        ))

    return restored_results
