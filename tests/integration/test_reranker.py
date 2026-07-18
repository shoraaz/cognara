"""
tests/integration/test_reranker.py
--------------------------------------
Tests for app/retrieval/reranker.py (Vertex AI Ranking API via
LangChain's VertexAIRank).

WHY THIS IS AN INTEGRATION TEST:
  Real reranking requires a real call to the Discovery Engine Ranking
  API. Guarded by a connectivity probe, same pattern as every other
  integration test file.
"""

import pytest
from langchain_core.documents import Document

from app.retrieval.hybrid_search import HybridResult
from app.retrieval.reranker import rerank


def _make_hybrid_result(text: str, topic: str, chunk_id: str) -> HybridResult:
    doc = Document(page_content=text, metadata={"chunk_id": chunk_id, "topic": topic})
    return HybridResult(chunk_id=chunk_id, document=doc, rrf_score=0.5, vector_rank=1, bm25_rank=1)


def _reranker_reachable() -> bool:
    try:
        candidates = [
            _make_hybrid_result("Paris is the capital of France.", "Geography", "c1"),
            _make_hybrid_result("The Eiffel Tower is in Paris.", "Landmarks", "c2"),
        ]
        rerank("What is the capital of France?", candidates, top_n=2)
        return True
    except Exception:
        return False


RERANKER_REACHABLE = _reranker_reachable()
requires_reranker = pytest.mark.skipif(
    not RERANKER_REACHABLE,
    reason="Vertex AI Ranking API not reachable — check discoveryengine.googleapis.com is enabled",
)


class TestRerankBasics:
    def test_empty_candidates_returns_empty_list(self):
        assert rerank("any query", [], top_n=5) == []


class TestRerankRealCall:
    @requires_reranker
    def test_reranks_by_relevance(self):
        """
        A deliberately obvious relevance ordering test: given one
        directly relevant chunk and one unrelated chunk, the reranker
        should put the relevant one first regardless of the input order.
        """
        candidates = [
            _make_hybrid_result(
                "The vanishing gradient problem occurs when gradients become "
                "extremely small during backpropagation in deep networks.",
                "Vanishing Gradients", "relevant_chunk",
            ),
            _make_hybrid_result(
                "A recipe for chocolate cake requires flour, sugar, and cocoa powder.",
                "Recipes", "irrelevant_chunk",
            ),
        ]
        results = rerank("What causes the vanishing gradient problem?", candidates, top_n=2)
        assert len(results) > 0
        assert results[0].metadata["chunk_id"] == "relevant_chunk"

    @requires_reranker
    def test_returns_at_most_top_n(self):
        candidates = [
            _make_hybrid_result(f"Chunk number {i} about neural networks.", f"Topic {i}", f"c{i}")
            for i in range(10)
        ]
        results = rerank("neural networks", candidates, top_n=3)
        assert len(results) <= 3

    @requires_reranker
    def test_reranked_documents_carry_relevance_score(self):
        candidates = [
            _make_hybrid_result("Gradient descent minimizes the loss function.", "Optimization", "c1"),
        ]
        results = rerank("How does gradient descent work?", candidates, top_n=1)
        assert len(results) == 1
        assert "relevance_score" in results[0].metadata
        assert 0.0 <= results[0].metadata["relevance_score"] <= 1.0
