"""
app/retrieval/graph_store.py
--------------------------------
Layer 6: GraphRAG — Neo4j client wrapper for the concept graph.
See ADR 0010 for the full design reasoning.

WHY THIS FILE EXISTS:
  This is the only place in the codebase that runs Cypher queries — the
  graph-equivalent of vector_store.py (the only place running SQL against
  Cloud SQL). Concept extraction (build_concept_graph.py) writes here;
  CRAG's search_concept_graph tool (crag_agent.py) reads here for
  structural questions like "what should I learn before X."

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
  The extraction pipeline may run multiple times as the corpus grows or
  extraction logic improves. MERGE is idempotent — verified directly
  against the live AuraDB instance before building this module: running
  the same MERGE query twice creates the node/relationship once, and
  reports zero additional nodes_created/relationships_created on the
  second run. CREATE would duplicate nodes on every re-run.

TWO REAL CYPHER SYNTAX ISSUES FOUND AND FIXED BEFORE TRUSTING THIS MODULE
(caught by checking against real AuraDB behaviour/docs, not assumed):
  1. upsert_concept()'s first draft used apoc.coll.toSet() in Cypher to
     deduplicate grounding_chunk_ids on an existing concept. AuraDB Free
     only ships a SUBSET of APOC Core, and that specific function's
     presence could not be confirmed with certainty. Rather than risk a
     runtime failure on an unconfirmed procedure, deduplication is done
     in PYTHON before the query runs — zero external Cypher dependency.
  2. get_prerequisites()'s first draft tried to parameterize the
     variable-length path range as *1..$max_depth. Cypher does NOT
     support parameterizing path-length bounds the way it supports
     property value parameters — only literal integers are accepted
     there, the same restriction that already applied to relationship
     TYPES (see upsert_relationship()). Fixed by validating max_depth as
     a real bounded int and inlining it via an f-string, with the same
     "never raw user input" restriction already applied to relation_type.

REAL BUG FOUND AND FIXED — find_concept_by_name()'S FIRST VERSION
RANKED GENERIC MATCHES ABOVE SPECIFIC ONES:
  Testing "vanishing gradients" against the real 799-concept graph
  returned ['Gradients', 'Gradient', 'Vanishing Gradients'] — the
  correct, specific match ("Vanishing Gradients") ranked LAST, behind
  two generic, near-useless partial matches. Root cause: plain
  bidirectional CONTAINS matching has no sense of specificity — a short
  generic name like "Gradient" trivially satisfies "query CONTAINS
  name" for almost any gradient-related query, and Cypher's default
  result order for equally-matching rows is not guaranteed to favour
  the more specific one.
  FIX: results are now ordered by DESCENDING concept-name length before
  the LIMIT is applied — a cheap, real proxy for specificity (a longer,
  more specific canonical name is a better resolution target than a
  short, generic one that happens to also match). Verified against the
  same real query after the fix — see BUG_FIX_LOG.md for the exact
  before/after output.

LAYER 3 WIRING — find_concept_by_name():
  CRAG's search_concept_graph tool needs to resolve a user's free-text
  question (e.g. "what should I learn before understanding vanishing
  gradients") to a REAL, exact Concept node name (e.g. "Vanishing
  Gradients") before it can traverse. find_concept_by_name() does this
  with simple, bidirectional case-insensitive substring matching,
  ranked by specificity (see bug note above) — deliberately NOT a
  second embedding-similarity system, since the concept graph is small
  (799 nodes) and this is a cheap, bounded Cypher query, not a
  bottleneck worth new infrastructure for.

INTERVIEW EXPLANATION:
  "This is the only module that speaks Cypher, the same way vector_store.py
  is the only module that speaks SQL to Cloud SQL. Every write uses MERGE,
  verified genuinely idempotent against the live database rather than
  assumed. I also hit two real Cypher restrictions while building this:
  relationship types and variable-length path bounds can't be
  parameterized the way property values can. And when I wired up fuzzy
  concept-name resolution, a real test against the actual graph showed
  generic short names like 'Gradient' outranking the specific, correct
  match 'Vanishing Gradients' — simple CONTAINS matching has no built-in
  sense of specificity. I fixed it by ranking longer, more specific
  names first, a cheap proxy that fixed the real case I tested against."

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

    concept_name must be a REAL, exact node name — see
    find_concept_by_name() below for resolving free-text input first.
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
    """One-hop RELATED_TO / PART_OF / CONTRASTS_WITH neighbors of a concept.
    concept_name must be a REAL, exact node name — see find_concept_by_name()."""
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


def find_concept_by_name(driver: Driver, query_text: str, limit: int = 3) -> list[dict]:
    """
    Fuzzy-resolve free text to real Concept node(s) by name — the graph's
    real, canonical concept names (e.g. "Vanishing Gradients") rarely
    match a user's exact question phrasing (e.g. "what should I learn
    before understanding vanishing gradients"). This is the resolution
    step needed before get_prerequisites()/get_related_concepts() can
    traverse from a real starting node — see module docstring.

    Uses case-insensitive substring matching in BOTH directions (does the
    concept name appear in the query, or does the query appear in the
    concept name), ranked by DESCENDING name length — see module
    docstring's REAL BUG FOUND AND FIXED for why plain, unranked CONTAINS
    matching let short, generic names (e.g. "Gradient") outrank the
    correct, specific match (e.g. "Vanishing Gradients"). Longer name =
    treated as more specific = ranked first, a cheap, real proxy that
    fixed the observed failure case.

    Deliberately NOT a second embedding-similarity system — the concept
    graph is small (799 nodes) and this is a cheap, bounded Cypher
    query, not a bottleneck worth new infrastructure for.
    """
    result = driver.execute_query(
        """
        MATCH (c:Concept)
        WHERE toLower($query_text) CONTAINS toLower(c.name)
           OR toLower(c.name) CONTAINS toLower($query_text)
        RETURN c.name AS name, c.description AS description,
               c.grounding_chunk_ids AS chunk_ids
        ORDER BY size(c.name) DESC
        LIMIT $limit
        """,
        query_text=query_text, limit=limit, database_=settings.NEO4J_DATABASE,
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
