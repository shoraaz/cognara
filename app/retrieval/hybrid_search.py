"""
app/retrieval/hybrid_search.py
---------------------------------
Combines vector search (CognaraPGVectorStore) and keyword search
(BM25KeywordIndex) into one ranked result list using Reciprocal Rank
Fusion (RRF).

WHY THIS FILE EXISTS (see ADR 0005):
  Vector search and BM25 keyword search each miss different things.
  Vector search finds "explain overfitting" even if the chunk never uses
  the word "overfitting" — it matches MEANING. BM25 finds the exact
  chunk containing "ReLU" even if that chunk's overall meaning, embedded
  as a vector, drifts slightly from the question's phrasing — it matches
  WORDS. Neither is strictly better; they fail on different question
  shapes. Combining both, and taking the union of what each finds, is
  the entire point of hybrid retrieval.

WHY RECIPROCAL RANK FUSION, NOT A WEIGHTED SCORE AVERAGE:
  Vector search returns cosine similarity: a bounded 0..1 value. BM25
  returns an UNBOUNDED score (in our real corpus, ranging from under 1
  to over 15 — see keyword_search.py's real test output). These two
  numbers cannot be averaged or weighted together meaningfully — a BM25
  score of 8 and a cosine similarity of 0.8 are not "the same amount of
  relevance" by any principled conversion. RRF sidesteps this entirely
  by throwing away the raw scores and working ONLY with RANK POSITION
  (1st, 2nd, 3rd...) within each list — a universal, always-comparable
  signal regardless of what scoring function produced it. This is the
  standard, well-established approach (Cormack, Clarke, Büttcher 2009;
  used natively in Elasticsearch, Azure AI Search, OpenSearch, MongoDB).

THE FORMULA:
  For each document d, its RRF score is:
      score(d) = sum over every list L containing d of  1 / (k + rank_L(d))
  where rank_L(d) is d's 1-indexed position in list L, and k=60 is the
  standard smoothing constant (used by every major implementation
  surveyed — Elasticsearch calls it rank_constant). A LOW k gives a
  massive boost to whatever is ranked #1; a HIGH k rewards documents
  that appear reasonably high in MULTIPLE lists (consensus) over a
  single list's top pick. k=60 is the well-established default that
  favours consensus without being insensitive to rank — and RRF's
  performance is documented as "not critically sensitive" to k, so we
  do not tune it without a real evaluation-driven reason to.

INTERVIEW EXPLANATION:
  "I fuse my vector and BM25 result lists with Reciprocal Rank Fusion
  rather than trying to combine their raw scores, because cosine
  similarity and BM25 scores live on completely different, incomparable
  scales — there's no principled way to average a 0.8 cosine score with
  a BM25 score of 8. RRF sidesteps that by working only with each
  document's RANK POSITION in each list, which is always comparable no
  matter what scoring function produced it. A document that ranks
  reasonably well in BOTH lists usually beats a document that's #1 in
  only one — which is exactly the 'consensus' behaviour you want from
  hybrid search."
"""

from dataclasses import dataclass

from langchain_core.documents import Document

from app.core.logging import get_logger
from app.retrieval.keyword_search import BM25KeywordIndex
from app.retrieval.vector_store import CognaraPGVectorStore

logger = get_logger(__name__)

RRF_K = 60  # standard smoothing constant — see module docstring


@dataclass
class HybridResult:
    chunk_id: str
    document: Document
    rrf_score: float
    vector_rank: int | None    # 1-indexed rank in the vector list, or None if absent
    bm25_rank: int | None      # 1-indexed rank in the BM25 list, or None if absent


def _rrf_contribution(rank: int | None, k: int = RRF_K) -> float:
    """1/(k + rank) if the document appeared in this list, else 0."""
    if rank is None:
        return 0.0
    return 1.0 / (k + rank)


def hybrid_search(
    query: str,
    vector_store: CognaraPGVectorStore,
    keyword_index: BM25KeywordIndex,
    k_per_method: int = 20,
    top_k: int = 5,
    course_filter: str | None = None,
    chapter_filter: str | None = None,
) -> list[HybridResult]:
    """
    Run vector search and BM25 search independently, fuse their ranked
    lists with RRF, and return the top_k fused results.

    Args:
        query: the user's question.
        vector_store: an already-constructed CognaraPGVectorStore.
        keyword_index: an already-constructed BM25KeywordIndex.
        k_per_method: how many results to pull from EACH method before
            fusion. Deliberately larger than top_k — fusion needs a wide
            enough candidate pool from each side to find real overlap;
            asking each method for only top_k results would make RRF
            degenerate into "whichever method's top_k happens to match."
        top_k: how many fused results to return after RRF ranking.
        course_filter / chapter_filter: applied identically to BOTH
            underlying searches, so fusion never mixes filtered and
            unfiltered candidates.

    Returns:
        Up to top_k HybridResult objects, sorted by RRF score descending.
    """
    vector_results = vector_store.similarity_search_with_score(
        query, k=k_per_method, course_filter=course_filter, chapter_filter=chapter_filter,
    )
    keyword_results = keyword_index.search(
        query, k=k_per_method, course_filter=course_filter, chapter_filter=chapter_filter,
    )

    # Build rank lookups: chunk_id -> 1-indexed rank in each list.
    vector_ranks = {doc.metadata["chunk_id"]: i + 1 for i, (doc, _score) in enumerate(vector_results)}
    keyword_ranks = {r.chunk_id: i + 1 for i, r in enumerate(keyword_results)}

    # Union of every chunk_id seen in either list.
    all_chunk_ids = set(vector_ranks) | set(keyword_ranks)

    # Keep one Document per chunk_id, preferring the vector result's
    # Document (already has full metadata + text) and falling back to
    # building one from the keyword result if a chunk was found ONLY by
    # BM25.
    documents_by_id: dict[str, Document] = {
        doc.metadata["chunk_id"]: doc for doc, _score in vector_results
    }
    for r in keyword_results:
        if r.chunk_id not in documents_by_id:
            documents_by_id[r.chunk_id] = Document(page_content=r.text, metadata=r.metadata)

    fused: list[HybridResult] = []
    for chunk_id in all_chunk_ids:
        v_rank = vector_ranks.get(chunk_id)
        b_rank = keyword_ranks.get(chunk_id)
        rrf_score = _rrf_contribution(v_rank) + _rrf_contribution(b_rank)
        fused.append(HybridResult(
            chunk_id=chunk_id,
            document=documents_by_id[chunk_id],
            rrf_score=rrf_score,
            vector_rank=v_rank,
            bm25_rank=b_rank,
        ))

    fused.sort(key=lambda r: r.rrf_score, reverse=True)
    top_results = fused[:top_k]

    logger.info(
        "hybrid_search_done",
        vector_candidates=len(vector_results),
        keyword_candidates=len(keyword_results),
        fused_candidates=len(fused),
        returned=len(top_results),
    )

    return top_results
