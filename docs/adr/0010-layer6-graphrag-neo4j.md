# ADR 0010: Layer 6 — GraphRAG via Neo4j AuraDB Free

Status: Accepted
Date: 2026-07-19

## Context

Layer 6 adds a knowledge graph of concepts and their relationships
(prerequisite-of, related-to, part-of, contrasts-with) extracted from
the corpus, used alongside vector/keyword retrieval for questions like
"what do I need to understand before X" or "what concepts relate to Y"
— structural questions plain similarity search answers poorly, since
"prerequisite" is a relationship, not a similarity.

## Decision 1: Neo4j, not Postgres tables — a deliberate break from the prior pattern

Every previous database decision in this project (ADR 0003: pgvector
over Qdrant/Vertex AI Vector Search; ADR 0005: rank_bm25 in-process over
a second search engine) followed the same reasoning: prefer the existing
Cloud SQL instance unless there's a real, demonstrated reason to add a
new system. A small concept graph for this corpus (likely dozens to a
few hundred nodes) is genuinely small enough that plain Postgres tables
(`concepts`, `concept_edges`) with recursive CTEs for traversal would
have worked, staying consistent with that pattern.

Neo4j was chosen anyway, deliberately breaking that pattern, for
reasons specific to this layer:

1. **GraphRAG is specifically named in the master prompt as its own
   layer**, distinct from "add some relational metadata" — the
   intent is to build and demonstrate real graph-database experience,
   not a SQL workaround that happens to model relationships.
2. **Cypher (Neo4j's query language) is a genuinely different, valuable
   skill** to have working code for, versus one more recursive CTE
   pattern layered onto skills already demonstrated extensively
   elsewhere in this project (SQLAlchemy, pgvector, jsonb).
3. **Graph traversal queries at even 3-4 hops are where a real graph
   database's advantage over SQL joins becomes concrete** — "what are
   all prerequisites of X, and their prerequisites" is a natural,
   fast Cypher traversal and a genuinely awkward recursive SQL query.

### Cost: AuraDB Free tier, not a self-hosted or paid instance

Neo4j AuraDB Free is a real, permanent free tier (not a time-limited
trial) — no credit card required, supporting up to 200,000 nodes and
400,000 relationships. This corpus's concept graph (extracted from ~388
chunks, likely well under 500 concept nodes) is nowhere near that
ceiling. This keeps Layer 6 consistent with the project's standing cost
discipline (stop-when-idle Cloud SQL, free-tier-first choices
throughout) despite the architectural decision to use a specialized
system.

## Decision 2: the graph is extracted from the corpus by an LLM pass, not built by hand

The corpus already has real, structured heading text from the chunker
(Module 2) — e.g. "20.1 Vanishing Gradient Problem", "20.1.7 How to
Detect Vanishing Gradient Problem". A Gemini call reads each chunk's
topic/heading and text, and extracts: the concept(s) it defines or
discusses, and its relationship to other concepts already seen
(prerequisite-of, part-of, related-to, contrasts-with) — building the
graph incrementally as chunks are processed, the same "let an LLM do
the structural extraction work, verify the result" pattern already used
throughout retrieval and generation, not a manually curated ontology.

## Consequences

- New dependency: `neo4j` Python driver.
- New GCP-adjacent-but-external service: Neo4j AuraDB Free — outside
  Cloud SQL, requiring its own connection URI and credentials in
  settings, following the same `.env` + `Settings` pattern as every
  other credential in this project.
- A new one-time extraction script builds the graph from the existing
  388 chunks — analogous to `run_ingestion.py`, but populating Neo4j
  instead of Cloud SQL.
- Retrieval gains a new path: graph traversal queries (e.g. "what
  should I learn before X") run alongside, not instead of, Layers 1-2's
  hybrid vector+keyword search — CRAG's search_notes tool gains graph
  traversal as an additional signal for structural questions, while
  Layers 1-4's existing pipeline remains unchanged for content
  questions.

## Interview summary

"I chose Neo4j for GraphRAG instead of modeling the same relationships
as plain Postgres tables, which is a real break from every earlier
database decision in this project — pgvector over Qdrant, in-process
BM25 over a second search engine — both times I stuck with the existing
database unless there was a demonstrated reason not to. Here the reason
was that GraphRAG is specifically about demonstrating real graph-
traversal patterns and Cypher, not just relationship metadata, and
AuraDB's free tier keeps that decision cost-neutral, consistent with
the cost discipline I've maintained everywhere else. The graph itself
is extracted from the corpus by an LLM reading each chunk's real
heading structure, the same 'let an LLM do the structural extraction,
verify the result' pattern I used for chunk retrieval and generation."
