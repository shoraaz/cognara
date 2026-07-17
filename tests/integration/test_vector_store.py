"""
tests/integration/test_vector_store.py
----------------------------------------
Tests for app/retrieval/vector_store.py (CognaraPGVectorStore).

WHY THIS IS AN INTEGRATION TEST:
  Like init_db.py and embedder.py, this module has no meaning without
  real network calls — both to Vertex AI (to embed test text) and to
  Cloud SQL (to write and search real rows). These tests exercise the
  full real path: embed -> upsert -> similarity search -> real scores.

REQUIREMENTS TO RUN THESE TESTS:
  - cognara-pg running (make db-start)
  - Vertex AI reachable (same ADC credentials as test_embedder.py)
  - The chunks table must already exist (make db-init / init_db.py)

  Guarded by a connectivity probe, same pattern as the other integration
  test files — skips cleanly with a clear reason if the dependencies
  aren't available, rather than failing or hanging.

TEST DATA HYGIENE:
  Every row this test suite writes uses a chunk_id prefixed with
  "test_vs_" and a course_name of "TEST_ONLY — vector store integration
  tests" so it is trivially distinguishable from real corpus data, and
  a fixture cleans up every row it created after each test — these tests
  do not pollute the real chunks table with leftover rows.
"""

import uuid

import pytest
from langchain_core.documents import Document

from app.retrieval.embedder import get_embeddings
from app.retrieval.vector_store import CognaraPGVectorStore

TEST_COURSE = "TEST_ONLY — vector store integration tests"


def _store_reachable() -> bool:
    try:
        store = CognaraPGVectorStore(embeddings=get_embeddings())
        store.similarity_search_with_score("connectivity probe", k=1)
        return True
    except Exception:
        return False


STORE_REACHABLE = _store_reachable()
requires_store = pytest.mark.skipif(
    not STORE_REACHABLE,
    reason="CognaraPGVectorStore not reachable — check cognara-pg is running and ADC credentials are set up",
)


def _make_test_chunk(text: str, topic: str, page_number: int) -> Document:
    return Document(
        page_content=text,
        metadata={
            "chunk_id": f"test_vs_{uuid.uuid4().hex[:12]}",
            "course_name": TEST_COURSE,
            "subject": "Test",
            "chapter": "Test Chapter",
            "topic": topic,
            "page_number": page_number,
            "page_range": None,
            "source_type": "test_fixture",
            "document_version": "test",
            "ingestion_date": "2026-07-17",
            "chunk_index_in_doc": 0,
            "char_count": len(text),
        },
    )


@pytest.fixture
def store():
    return CognaraPGVectorStore(embeddings=get_embeddings())


@pytest.fixture
def cleanup_test_rows(store):
    """Delete every TEST_COURSE row after the test, pass or fail."""
    yield
    with store.engine.connect() as conn:
        import sqlalchemy
        conn.execute(
            sqlalchemy.text("DELETE FROM chunks WHERE course_name = :c"),
            {"c": TEST_COURSE},
        )
        conn.commit()


class TestAddAndSearch:
    @requires_store
    def test_added_chunk_is_findable_by_similarity(self, store, cleanup_test_rows):
        doc = _make_test_chunk(
            "Overfitting occurs when a model learns the training data too well, "
            "including its noise, and performs poorly on new unseen data.",
            topic="Overfitting",
            page_number=42,
        )
        ids = store.add_documents([doc])
        assert len(ids) == 1
        assert ids[0] == doc.metadata["chunk_id"]

        results = store.similarity_search_with_score(
            "What is overfitting in machine learning?", k=3, course_filter=TEST_COURSE
        )
        assert len(results) > 0
        top_doc, top_score = results[0]
        assert "Overfitting" in top_doc.page_content or "overfitting" in top_doc.page_content.lower()
        assert 0.0 <= top_score <= 1.0

    @requires_store
    def test_relevant_chunk_scores_higher_than_irrelevant(self, store, cleanup_test_rows):
        relevant = _make_test_chunk(
            "Gradient descent is an optimization algorithm used to minimize "
            "the loss function by iteratively moving in the direction of "
            "steepest descent.",
            topic="Gradient Descent",
            page_number=10,
        )
        irrelevant = _make_test_chunk(
            "The recipe calls for two cups of flour, one egg, and a "
            "tablespoon of sugar, baked at 350 degrees for twenty minutes.",
            topic="Unrelated",
            page_number=11,
        )
        store.add_documents([relevant, irrelevant])

        results = store.similarity_search_with_score(
            "Explain how gradient descent optimizes a model.",
            k=2,
            course_filter=TEST_COURSE,
        )
        scores_by_topic = {doc.metadata["topic"]: score for doc, score in results}
        assert scores_by_topic["Gradient Descent"] > scores_by_topic["Unrelated"]

    @requires_store
    def test_course_filter_excludes_other_courses(self, store, cleanup_test_rows):
        doc = _make_test_chunk(
            "This chunk belongs to the test course only.",
            topic="Filter Test",
            page_number=1,
        )
        store.add_documents([doc])

        # Search with a course_filter for a DIFFERENT course — our test
        # chunk must not appear.
        results = store.similarity_search_with_score(
            "This chunk belongs to the test course only.",
            k=5,
            course_filter="100 Days of Machine Learning",
        )
        found_ids = [d.metadata["chunk_id"] for d, _ in results]
        assert doc.metadata["chunk_id"] not in found_ids

    @requires_store
    def test_upsert_updates_existing_chunk(self, store, cleanup_test_rows):
        chunk_id = f"test_vs_{uuid.uuid4().hex[:12]}"
        original = Document(
            page_content="Original text before update.",
            metadata={**_make_test_chunk("x", "Upsert Test", 1).metadata, "chunk_id": chunk_id},
        )
        store.add_documents([original])

        updated = Document(
            page_content="Updated text after upsert.",
            metadata={**_make_test_chunk("x", "Upsert Test", 1).metadata, "chunk_id": chunk_id},
        )
        store.add_documents([updated])  # same chunk_id -> ON CONFLICT DO UPDATE

        with store.engine.connect() as conn:
            import sqlalchemy
            row = conn.execute(
                sqlalchemy.text("SELECT text FROM chunks WHERE chunk_id = :id"),
                {"id": chunk_id},
            ).fetchone()
        assert row[0] == "Updated text after upsert."

    @requires_store
    def test_empty_documents_list_returns_empty_ids(self, store):
        assert store.add_documents([]) == []

    @requires_store
    def test_metadata_round_trips_correctly(self, store, cleanup_test_rows):
        doc = _make_test_chunk(
            "A chunk to verify every metadata field survives the round trip.",
            topic="Metadata Round Trip",
            page_number=99,
        )
        store.add_documents([doc])

        results = store.similarity_search_with_score(
            "metadata round trip verification", k=1, course_filter=TEST_COURSE
        )
        found_doc, _score = results[0]
        assert found_doc.metadata["topic"] == "Metadata Round Trip"
        assert found_doc.metadata["page_number"] == 99
        assert found_doc.metadata["subject"] == "Test"
        assert found_doc.metadata["source_type"] == "test_fixture"
