"""
app/retrieval/graph_store.py
--------------------------------
Layer 6: GraphRAG — Neo4j client wrapper for the concept graph.
See ADR 0010 for the full design reasoning.

WHY THIS FILE EXISTS:
  This is the only place in the codebase that runs Cypher queries — the
  graph-equivalent of vector_store.py (the only place running SQL against
  Cloud SQL). Concept extraction (graph_extraction.py) writes here; CRAG's
  search_notes tool (once extended for Layer 6) reads here for structural
  questions like "what should I learn before X."

GRAPH MODEL:
  (:Concept {name, description})
  (:Concept)-[:PREREQUISITE_OF]->(:Concept)   -- must understand A before B
  (:Concept)-[:RELATED_TO]->(:Concept)         -- A and B are connected, no ordering
  (:Concept)-[:PART_OF]->(:Concept)            -- A is a sub-topic of B
  Each Concept also carries grounding_chunk_ids: list[str] — the chunk_ids
  (from Cloud SQL's chunks table) this concept was extracted from, so any
  graph traversal result can still be cited back to real corpus pages —
  the same evidence-grounding principle used everywhere else in this
  project, applied to graph nodes instead of retrieved text chunks.

WHY MERGE, NOT CREATE, FOR EVERY WRITE:
  The extraction pipeline (graph_extraction.py) may run multiple times as
  the corpus grows or extraction logic improves. MERGE is idempotent —
  verified directly against the live AuraDB instance before building this
  module: running the same MERGE query twice creates the node/relationship
  once, and reports zero additional nodes_created/relationships_created on
  the second run. CREATE would duplicate nodes on every re-run.

TWO REAL CYPHER SYNTAX ISSUES FOUND AND FIXED BEFORE TRUSTING THIS MODULE
(caught by checking against real AuraDB behaviour/docs, not assumed):
  1. upsert_concept()'s first draft used apoc.coll.toSet() in Cypher to
     deduplicate grounding_chunk_ids on an existing concept. AuraDB Free
     only ships a SUBSET of APOC Core, and that specific function's
     presence could not be confirmed with certainty. Rather than risk a
     runtime failure on an unconfirmed procedure, deduplication is done
     in PYTHON before the query runs (see _dedupe below) — zero external
     Cypher dependency, and the Cypher itself becomes a plain SET.
  2. get_prerequisites()'s first draft tried to parameterize the
     variable-length path range as *1..$max_depth. Cypher does NOT
     support parameterizing path-length bounds the way it supports
     property value parameters — only literal integers are accepted
     there, the same restriction that already applied to relationship
     TYPES (see upsert_relationship()). Fixed by validating max_depth as
     a real bounded int and inlining it via an f-string, with the same
     "never raw user input" restriction already applied to relation_type.

INTERVIEW EXPLANATION:
  "This is the only module that speaks Cypher, the same way vector_store.py
  is the only module that speaks SQL to Cloud SQL. Every write uses MERGE,
  verified genuinely idempotent against the live database rather than
  assumed. I also hit two real Cypher restrictions while building this:
  relationship types and variable-length path bounds can't be
  parameterized the way property values can — both had to be inlined via
  f-strings, restricted to code-controlled, validated values only, never
  raw user input, to avoid Cypher injection. And rather than trust an
  unconfirmed APOC function's availability on the free tier, I moved that
  one piece of logic — deduplicating a list — into Python, where I don't
  need to guess what's installed."

# Interview notes: local-notes/INTERVIEW_PREP.md — "app/retrieval/graph_store.py"
"""

from neo4j import Driver, GraphDatabase

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

RelationType = str  # "PREREQUISITE_OF" | "RELATED_TO" | "PART_OF" | "CONTRASTS_WITH"
VALID_RELATION_TYPES = {"PREREQUISITE_OF", "RELATED_TO", "PART_OF", "CONTRASTS_WITH"}


def get_driver() -> Driver:
    """
    Construct a Neo4j Driver. Unlike ChatVertexAI (see generation.py's
    documented event-loop lesson), the Neo4j Python driver's sync API used
    here has no equivalent async gRPC binding — safe to construct once and
    reuse, but each call site constructs its own for simplicity and to
    mirror vector_store.py's per-call engine pattern rather than introduce
    an inconsistent caching strategy across two different database clients.
    """
    return GraphDatabase.driver(
        settings.NEO4J_URI, auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD),
    )


def _get_existing_chunk_ids(driver: Driver, name: str) -> list[str]:
    """Read a concept's current grounding_chunk_ids (empty list if the
    concept doesn't exist yet), so upsert_concept can dedupe in Python."""
    result = driver.execute_query(
        "MATCH (c:Concept {name: $name}) RETURN c.grounding_chunk_ids AS ids",
        name=name, database_=settings.NEO4J_DATABASE,
    )
    if not result.records or result.records[0]["ids"] is None:
        return []
    return result.records[0]["ids"]


def upsert_concept(driver: Driver, name: str, description: str, grounding_chunk_ids: list[str]) -> None:
    """
    Create or update a Concept node. MERGE on `name` — see module
    docstring for why this must be idempotent. If the concept already
    exists (found by a prior chunk's extraction), grounding_chunk_ids are
    merged and deduplicated IN PYTHON (see module docstring point 1 —
    not in Cypher via APOC, whose availability on AuraDB Free could not
    be confirmed), so a concept discussed across multiple chunks
    accumulates all of its real supporting evidence without duplicates.
    """
    existing_ids = _get_existing_chunk_ids(driver, name)
    merged_ids = list(dict.fromkeys(existing_ids + grounding_chunk_ids))  # dedupe, preserve order

    driver.execute_query(
        """
        MERGE (c:Concept {name: $name})
        ON CREATE SET c.description = $description
        SET c.grounding_chunk_ids = $chunk_ids
        """,
        name=name, description=description, chunk_ids=merged_ids,
        database_=settings.NEO4J_DATABASE,
    )


def upsert_relationship(driver: Driver, from_name: str, to_name: str, relation_type: RelationType) -> None:
    """
    Create a relationship between two Concept nodes (both MERGEd first,
    so this is safe even if a concept hasn't been separately upserted
    yet — see module docstring). relation_type is inserted via an
    f-string, NOT a parameter — Cypher does not support parameterizing
    relationship TYPES (only property values). Restricted to
    VALID_RELATION_TYPES, a fixed, code-controlled set, never raw user
    input, to avoid Cypher injection via the relationship type.
    """
    if relation_type not in VALID_RELATION_TYPES:
        raise ValueError(f"Unknown relation_type: {relation_type}")

    driver.execute_query(
        f"""
        MERGE (a:Concept {{name: $from_name}})
        MERGE (b:Concept {{name: $to_name}})
        MERGE (a)-[r:{relation_type}]->(b)
        """,
        from_name=from_name, to_name=to_name,
        database_=settings.NEO4J_DATABASE,
    )


def get_prerequisites(driver: Driver, concept_name: str, max_depth: int = 3) -> list[dict]:
    """
    Traverse PREREQUISITE_OF relationships backward from concept_name, up
    to max_depth hops — "what do I need to understand before X." This is
    the kind of query a real graph database makes natural (a variable-
    length path pattern) that would need a recursive CTE in Postgres.

    max_depth is inlined via an f-string, NOT a query parameter — Cypher
    does not support parameterizing variable-length path bounds
    (*1..N), only literal integers are accepted there (the same
    restriction already documented on relationship TYPES above).
    Validated as a real, bounded int before inlining — never raw user
    input — to avoid Cypher injection via this path.
    """
    if not isinstance(max_depth, int) or not (1 <= max_depth <= 10):
        raise ValueError(f"max_depth must be an int between 1 and 10, got: {max_depth!r}")

    result = driver.execute_query(
        f"""
        MATCH path = (prereq:Concept)-[:PREREQUISITE_OF*1..{max_depth}]->(target:Concept {{name: $name}})
        RETURN DISTINCT prereq.name AS name, prereq.description AS description,
               prereq.grounding_chunk_ids AS chunk_ids, length(path) AS depth
        ORDER BY depth
        """,
        name=concept_name, database_=settings.NEO4J_DATABASE,
    )
    return [r.data() for r in result.records]


def get_related_concepts(driver: Driver, concept_name: str) -> list[dict]:
    """One-hop RELATED_TO / PART_OF / CONTRASTS_WITH neighbors of a concept."""
    result = driver.execute_query(
        """
        MATCH (c:Concept {name: $name})-[r]-(other:Concept)
        WHERE type(r) IN ['RELATED_TO', 'PART_OF', 'CONTRASTS_WITH']
        RETURN DISTINCT other.name AS name, other.description AS description,
               other.grounding_chunk_ids AS chunk_ids, type(r) AS relation
        """,
        name=concept_name, database_=settings.NEO4J_DATABASE,
    )
    return [r.data() for r in result.records]


def get_graph_stats(driver: Driver) -> dict:
    """Real, verified counts — used to report extraction pipeline results."""
    result = driver.execute_query(
        """
        MATCH (c:Concept) WITH count(c) AS concept_count
        MATCH ()-[r]->() WITH concept_count, count(r) AS relationship_count
        RETURN concept_count, relationship_count
        """,
        database_=settings.NEO4J_DATABASE,
    )
    if not result.records:
        return {"concept_count": 0, "relationship_count": 0}
    return result.records[0].data()
