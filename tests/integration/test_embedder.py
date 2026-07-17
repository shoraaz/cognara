"""
tests/integration/test_embedder.py
------------------------------------
Tests for app/retrieval/embedder.py.

WHY THIS IS AN INTEGRATION TEST:
  Like init_db.py, this module has no meaning without a real network call
  to Vertex AI's embedding endpoint — there's no pure-function logic to
  test offline. These tests call the real text-embedding-004 model via
  the Vertex AI backend and verify real, observable output shape.

REQUIREMENTS TO RUN THESE TESTS:
  - GCP_PROJECT_ID / VERTEX_AI_LOCATION / VERTEX_EMBEDDING_MODEL set in .env
  - `gcloud auth application-default login` has been run at least once
  - The aiplatform.googleapis.com API enabled on the project (it is —
    confirmed during Phase 0 provisioning)

  Guarded by a connectivity probe, same pattern as test_init_db.py — if
  the embedding call can't be made (no credentials, no network), these
  tests SKIP with a clear reason rather than failing or hanging.
"""

import pytest

from app.retrieval.embedder import get_embeddings
from app.core.config import settings


def _embeddings_reachable() -> bool:
    try:
        emb = get_embeddings()
        emb.embed_query("connectivity probe")
        return True
    except Exception:
        return False


EMBEDDINGS_REACHABLE = _embeddings_reachable()
requires_embeddings = pytest.mark.skipif(
    not EMBEDDINGS_REACHABLE,
    reason="Vertex AI embeddings not reachable — check ADC credentials and GCP_PROJECT_ID",
)


class TestGetEmbeddings:
    @requires_embeddings
    def test_returns_singleton(self):
        """Same instance every call — see module docstring for why this matters."""
        first = get_embeddings()
        second = get_embeddings()
        assert first is second

    @requires_embeddings
    def test_configured_for_vertex_ai_backend(self):
        emb = get_embeddings()
        assert emb.vertexai is True
        assert emb.project == settings.GCP_PROJECT_ID
        assert emb.model == settings.VERTEX_EMBEDDING_MODEL


class TestEmbedQuery:
    @requires_embeddings
    def test_returns_768_dim_vector(self):
        emb = get_embeddings()
        vector = emb.embed_query("What is overfitting?")
        assert len(vector) == settings.EMBEDDING_DIM  # 768
        assert all(isinstance(v, float) for v in vector)

    @requires_embeddings
    def test_similar_questions_produce_similar_vectors(self):
        """
        Sanity check on embedding QUALITY, not just shape: two questions
        about the same concept should be closer (smaller Euclidean
        distance) than two questions about unrelated concepts. This is
        the property retrieval quality depends on entirely.
        """
        emb = get_embeddings()
        v1 = emb.embed_query("What is overfitting in machine learning?")
        v2 = emb.embed_query("Explain the concept of overfitting.")
        v3 = emb.embed_query("What is the capital of France?")

        def euclidean(a, b):
            return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5

        dist_similar = euclidean(v1, v2)
        dist_different = euclidean(v1, v3)
        assert dist_similar < dist_different


class TestEmbedDocuments:
    @requires_embeddings
    def test_batch_returns_one_vector_per_text(self):
        emb = get_embeddings()
        texts = [
            "Overfitting happens when a model memorizes training data.",
            "Underfitting happens when a model is too simple.",
            "Gradient descent is an optimization algorithm.",
        ]
        vectors = emb.embed_documents(texts)
        assert len(vectors) == len(texts)
        assert all(len(v) == settings.EMBEDDING_DIM for v in vectors)
