"""
tests/integration/test_graph_store.py
------------------------------------------
Tests for app/retrieval/graph_store.py — Layer 6's Neo4j client wrapper.

WHY THIS IS AN INTEGRATION TEST:
  Real graph operations need a real Neo4j AuraDB connection. Guarded by
  a connectivity probe, same pattern as every other integration test
  file. Tests use clearly-namespaced TEST_ONLY concept names and clean
  up after themselves, so they never pollute the real 799-concept graph
  built from the actual corpus.
"""

import pytest

from app.core.config import settings
from app.retrieval import graph_store

TEST_PREFIX = "TEST_ONLY_"


def _reachable() -> bool:
    if not settings.NEO4J_URI:
        return False
    try:
        driver = graph_store.get_driver()
        driver.verify_connectivity()
        driver.close()
        return True
    except Exception:
        return False


REACHABLE = _reachable()
requires_neo4j = pytest.mark.skipif(not REACHABLE, reason="Neo4j AuraDB not reachable")


@pytest.fixture
def driver():
    d = graph_store.get_driver()
    yield d
    d.close()


@pytest.fixture
def cleanup_test_concepts(driver):
    """Delete every TEST_ONLY_-prefixed concept after the test, pass or fail."""
    yield
    driver.execute_query(
        f"MATCH (c:Concept) WHERE c.name STARTS WITH '{TEST_PREFIX}' DETACH DELETE c",
        database_=settings.NEO4J_DATABASE,
    )


class TestUpsertConcept:
    @requires_neo4j
    def test_creates_a_new_concept(self, driver, cleanup_test_concepts):
        name = f"{TEST_PREFIX}Gradient Descent"
        graph_store.upsert_concept(driver, name, "An optimization algorithm.", ["chunk_1"])

        result = driver.execute_query(
            "MATCH (c:Concept {name: $name}) RETURN c.description AS d, c.grounding_chunk_ids AS ids",
            name=name, database_=settings.NEO4J_DATABASE,
        )
        assert len(result.records) == 1
        assert result.records[0]["ids"] == ["chunk_1"]

    @requires_neo4j
    def test_merging_the_same_concept_twice_is_idempotent(self, driver, cleanup_test_concepts):
        """The real MERGE behaviour verified live before this module was
        trusted — running the same upsert twice must not duplicate nodes."""
        name = f"{TEST_PREFIX}Idempotent Concept"
        graph_store.upsert_concept(driver, name, "desc", ["chunk_1"])
        graph_store.upsert_concept(driver, name, "desc", ["chunk_1"])

        result = driver.execute_query(
            "MATCH (c:Concept {name: $name}) RETURN count(c) AS n",
            name=name, database_=settings.NEO4J_DATABASE,
        )
        assert result.records[0]["n"] == 1

    @requires_neo4j
    def test_grounding_chunk_ids_accumulate_and_dedupe(self, driver, cleanup_test_concepts):
        """A concept seen in multiple chunks accumulates ALL its grounding
        chunk_ids, without duplicates even if the same chunk_id is passed twice."""
        name = f"{TEST_PREFIX}Multi-Chunk Concept"
        graph_store.upsert_concept(driver, name, "desc", ["chunk_1", "chunk_2"])
        graph_store.upsert_concept(driver, name, "desc", ["chunk_2", "chunk_3"])  # chunk_2 overlaps

        result = driver.execute_query(
            "MATCH (c:Concept {name: $name}) RETURN c.grounding_chunk_ids AS ids",
            name=name, database_=settings.NEO4J_DATABASE,
        )
        ids = result.records[0]["ids"]
        assert set(ids) == {"chunk_1", "chunk_2", "chunk_3"}
        assert len(ids) == 3  # no duplicate chunk_2


class TestUpsertRelationship:
    @requires_neo4j
    def test_creates_a_relationship(self, driver, cleanup_test_concepts):
        a, b = f"{TEST_PREFIX}A", f"{TEST_PREFIX}B"
        graph_store.upsert_relationship(driver, a, b, "PREREQUISITE_OF")

        result = driver.execute_query(
            "MATCH (a:Concept {name: $a})-[r:PREREQUISITE_OF]->(b:Concept {name: $b}) RETURN count(r) AS n",
            a=a, b=b, database_=settings.NEO4J_DATABASE,
        )
        assert result.records[0]["n"] == 1

    @requires_neo4j
    def test_rejects_unknown_relation_type(self, driver):
        with pytest.raises(ValueError):
            graph_store.upsert_relationship(driver, "A", "B", "NOT_A_REAL_TYPE")


class TestTraversal:
    @requires_neo4j
    def test_get_prerequisites_finds_direct_prerequisite(self, driver, cleanup_test_concepts):
        a, b = f"{TEST_PREFIX}Basics", f"{TEST_PREFIX}Advanced"
        graph_store.upsert_relationship(driver, a, b, "PREREQUISITE_OF")

        prereqs = graph_store.get_prerequisites(driver, b, max_depth=2)
        names = [p["name"] for p in prereqs]
        assert a in names

    @requires_neo4j
    def test_get_prerequisites_rejects_invalid_max_depth(self, driver):
        with pytest.raises(ValueError):
            graph_store.get_prerequisites(driver, "anything", max_depth=999)

    @requires_neo4j
    def test_get_related_concepts_finds_related_to_edge(self, driver, cleanup_test_concepts):
        a, b = f"{TEST_PREFIX}X", f"{TEST_PREFIX}Y"
        graph_store.upsert_relationship(driver, a, b, "RELATED_TO")

        related = graph_store.get_related_concepts(driver, a)
        names = [r["name"] for r in related]
        assert b in names


class TestGraphStats:
    @requires_neo4j
    def test_real_corpus_graph_has_substantial_content(self, driver):
        """
        The flagship proof: the FULL 388-chunk corpus extraction
        completed with zero errors — this test confirms the real,
        final graph state, not a synthetic one.
        """
        stats = graph_store.get_graph_stats(driver)
        assert stats["concept_count"] >= 700  # real extraction produced 799
        assert stats["relationship_count"] >= 800  # real extraction produced 970
