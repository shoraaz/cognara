"""
tests/integration/test_keyword_search.py
-------------------------------------------
Tests for app/retrieval/keyword_search.py (BM25KeywordIndex).

WHY THIS IS AN INTEGRATION TEST:
  Building a real BM25 index requires reading every chunk from the live
  Cloud SQL instance — there's no meaningful offline unit-test version
  of "does this find real terms in the real corpus." Guarded by a
  connectivity probe, same pattern as every other integration test file.
"""

import pytest

from app.retrieval.keyword_search import BM25KeywordIndex, _tokenize


def _db_reachable() -> bool:
    try:
        idx = BM25KeywordIndex()
        idx.refresh()
        return True
    except Exception:
        return False


DB_REACHABLE = _db_reachable()
requires_db = pytest.mark.skipif(
    not DB_REACHABLE,
    reason="Cloud SQL not reachable — is cognara-pg started?",
)


class TestTokenize:
    def test_lowercases_and_splits_on_non_alphanumeric(self):
        assert _tokenize("ReLU Activation-Function!") == ["relu", "activation", "function"]

    def test_empty_string_returns_empty_list(self):
        assert _tokenize("") == []

    def test_numbers_are_kept_as_tokens(self):
        assert _tokenize("Layer 1 has 128 units") == ["layer", "1", "has", "128", "units"]


class TestBM25KeywordIndex:
    @requires_db
    def test_refresh_indexes_all_real_chunks(self):
        idx = BM25KeywordIndex()
        count = idx.refresh()
        assert count == 388  # the real, verified corpus size from Module 4

    @requires_db
    def test_exact_term_query_finds_relevant_chunks(self):
        """
        'cross-entropy' is a real, specific term that appears in the DL
        corpus's loss-function chapter. A working BM25 index should
        surface exactly that chapter near the top.
        """
        idx = BM25KeywordIndex()
        idx.refresh()
        results = idx.search("cross-entropy loss", k=5)
        assert len(results) > 0
        chapters = [r.metadata["chapter"] for r in results]
        assert any("Training Neural Networks" in ch or "Loss" in ch for ch in chapters) or \
               any("loss" in r.text.lower() for r in results)

    @requires_db
    def test_nonsense_query_returns_no_or_weak_matches(self):
        idx = BM25KeywordIndex()
        idx.refresh()
        results = idx.search("xyzzy quolgorp fribbleton", k=5)
        assert len(results) == 0  # no real terms match anything

    @requires_db
    def test_results_are_sorted_by_score_descending(self):
        idx = BM25KeywordIndex()
        idx.refresh()
        results = idx.search("neural network training", k=10)
        scores = [r.bm25_score for r in results]
        assert scores == sorted(scores, reverse=True)

    @requires_db
    def test_course_filter_restricts_results(self):
        idx = BM25KeywordIndex()
        idx.refresh()
        results = idx.search("learning", k=20, course_filter="100 Days of Machine Learning")
        for r in results:
            assert r.metadata["course_name"] == "100 Days of Machine Learning"

    @requires_db
    def test_lazy_refresh_on_first_search(self):
        """search() should build the index automatically if refresh()
        was never called explicitly."""
        idx = BM25KeywordIndex()
        results = idx.search("gradient descent", k=3)
        assert isinstance(results, list)

    @requires_db
    def test_zero_score_results_are_excluded(self):
        idx = BM25KeywordIndex()
        idx.refresh()
        results = idx.search("overfitting", k=50)
        assert all(r.bm25_score > 0 for r in results)
