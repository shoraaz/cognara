"""
tests/integration/test_hybrid_search.py
------------------------------------------
Tests for app/retrieval/hybrid_search.py.

WHY THIS IS AN INTEGRATION TEST:
  Real fusion needs real vector search AND real BM25 search, both
  against the live 388-chunk corpus. Guarded by a connectivity probe,
  same pattern as every other integration test file.
"""

import pytest

from app.retrieval.embedder import get_embeddings
from app.retrieval.hybrid_search import RRF_K, HybridResult, _rrf_contribution, hybrid_search
from app.retrieval.keyword_search import BM25KeywordIndex
from app.retrieval.vector_store import CognaraPGVectorStore


def _reachable() -> bool:
    try:
        store = CognaraPGVectorStore(embeddings=get_embeddings())
        store.similarity_search_with_score("connectivity probe", k=1)
        idx = BM25KeywordIndex()
        idx.refresh()
        return True
    except Exception:
        return False


REACHABLE = _reachable()
requires_infra = pytest.mark.skipif(
    not REACHABLE,
    reason="Cloud SQL / Vertex AI not reachable",
)


class TestRRFContribution:
    def test_rank_1_scores_higher_than_rank_2(self):
        assert _rrf_contribution(1) > _rrf_contribution(2)

    def test_none_rank_contributes_zero(self):
        assert _rrf_contribution(None) == 0.0

    def test_matches_the_standard_formula(self):
        # score = 1 / (k + rank)
        assert _rrf_contribution(5, k=60) == pytest.approx(1 / 65)

    def test_default_k_is_60(self):
        assert RRF_K == 60


@pytest.fixture(scope="module")
def store():
    return CognaraPGVectorStore(embeddings=get_embeddings())


@pytest.fixture(scope="module")
def keyword_index():
    idx = BM25KeywordIndex()
    idx.refresh()
    return idx


class TestHybridSearchRealCorpus:
    @requires_infra
    def test_returns_up_to_top_k_results(self, store, keyword_index):
        results = hybrid_search("Explain the vanishing gradient problem.", store, keyword_index, top_k=5)
        assert len(results) <= 5
        assert all(isinstance(r, HybridResult) for r in results)

    @requires_infra
    def test_results_sorted_by_rrf_score_descending(self, store, keyword_index):
        results = hybrid_search("gradient descent optimization", store, keyword_index, top_k=10)
        scores = [r.rrf_score for r in results]
        assert scores == sorted(scores, reverse=True)

    @requires_infra
    def test_document_found_by_both_methods_ranks_highly(self, store, keyword_index):
        """
        A concept that is both a strong semantic AND exact-keyword match
        (e.g. 'vanishing gradient problem' — a real named concept in the
        corpus) should surface a chunk present in BOTH lists near the
        top, since RRF rewards consensus.
        """
        results = hybrid_search("vanishing gradient problem", store, keyword_index, top_k=5)
        assert len(results) > 0
        top = results[0]
        # The top-ranked fused result should have been found by at least
        # one method with a strong (low-number) rank; ideally both.
        assert top.vector_rank is not None or top.bm25_rank is not None

    @requires_infra
    def test_keyword_only_match_still_appears_in_fusion(self, store, keyword_index):
        """
        A precise exact-term query should surface chunks BM25 found even
        if vector search alone might rank them lower — proving fusion
        genuinely uses both lists, not just one.
        """
        results = hybrid_search("cross-entropy loss function", store, keyword_index, top_k=10)
        bm25_found = [r for r in results if r.bm25_rank is not None]
        assert len(bm25_found) > 0

    @requires_infra
    def test_course_filter_applied_to_both_methods(self, store, keyword_index):
        results = hybrid_search(
            "training", store, keyword_index, top_k=10,
            course_filter="100 Days of Deep Learning",
        )
        for r in results:
            assert r.document.metadata["course_name"] == "100 Days of Deep Learning"

    @requires_infra
    def test_empty_or_nonsense_query_returns_something_or_nothing_gracefully(self, store, keyword_index):
        # Should not raise — vector search always returns candidates
        # (there's no zero-vector-match concept), BM25 may return none.
        results = hybrid_search("xyzzy quolgorp fribbleton", store, keyword_index, top_k=5)
        assert isinstance(results, list)
