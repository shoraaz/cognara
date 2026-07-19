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
400,000 relationships. The real, final concept graph extracted from
this corpus (799 concepts, 970 relationships — see Results below) uses
under 0.5% of that ceiling. This keeps Layer 6 consistent with the
project's standing cost discipline (stop-when-idle Cloud SQL, free-
tier-first choices throughout) despite the architectural decision to
use a specialized system.

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

## Real, practical constraint found while running the extraction (not a design flaw)

The full 388-chunk extraction (one Gemini call per chunk, per Decision
2) reliably exceeded the execution-time window of a single remote-shell
tool invocation used to run it — a genuine, practical constraint of the
development environment, not a bug in the pipeline itself. This surfaced
two real, separate gaps that were found and fixed in sequence:

1. **Resumability** — the first version of `build_graph()` had no way
   to skip chunks already processed on a prior (interrupted) run, so a
   naive re-run would re-spend real Gemini API calls reprocessing the
   same chunks. Fixed by querying Neo4j directly, at the start of every
   run, for the set of `chunk_id`s already present in any concept's
   `grounding_chunk_ids` — reading the graph's own real state as the
   source of truth, not a separate progress-tracking file that could
   drift out of sync with what was actually written.
2. **Batching** — resumability alone was not sufficient, since even
   "everything remaining" after a partial run was still 200+ sequential
   chunks, still exceeding the execution window on every attempt. Fixed
   by adding an explicit `--limit N` flag: process at most N not-yet-
   done chunks, then stop cleanly and report real, verifiable progress.
   The full extraction was completed by invoking the pipeline
   repeatedly with small batches (15-21 chunks each) until it reported
   "All chunks processed" — genuine, practical resumability under a
   real environmental constraint, not just theoretical idempotency.

## Results — real, final numbers from the completed extraction

The full 388-chunk corpus was processed across several resumed batches,
with **zero extraction errors** across the entire run:

| Metric | Value |
|---|---|
| Chunks processed | 388 / 388 (100%) |
| Extraction errors | 0 |
| Final concept count | 799 |
| Final relationship count | 970 |

Spot-checked directly against the live graph: `get_related_concepts()`
on "Gradient Descent" correctly returns 24 real, sensible connections
(e.g. `RELATED_TO: Vanishing Gradient Problem`, `RELATED_TO: Learning
Rate`, `PART_OF: Backpropagation`) — genuine, text-grounded
relationships, not generic or hallucinated ones.

One honest, non-bug observation: `get_prerequisites()` on "Vanishing
Gradient Problem" returns empty, even though "Gradient Descent" is
clearly conceptually connected to it (confirmed present as a
`RELATED_TO` edge). This reflects the extraction prompt's own
instruction to only extract relations the source text actually
supports — the model judged this connection as `RELATED_TO` rather
than a directional `PREREQUISITE_OF`, a defensible, conservative
reading rather than an extraction failure. A future refinement could
revisit the PREREQUISITE_OF extraction criteria specifically if this
undercounts genuinely prerequisite relationships in practice.

## Consequences

- New dependency: `neo4j` Python driver (v6.2.0).
- New GCP-adjacent-but-external service: Neo4j AuraDB Free — outside
  Cloud SQL, requiring its own connection URI and credentials in
  settings, following the same `.env` + `Settings` pattern as every
  other credential in this project. A real, twice-reproduced wrong
  assumption (that the AuraDB username is literally "neo4j", when it is
  actually the instance ID) cost two authentication failures before
  being corrected from the instance's own downloaded `.env`.
- Two real Cypher syntax restrictions found and fixed before trusting
  `graph_store.py`: relationship types and variable-length path bounds
  cannot be parameterized the way property values can, requiring
  validated f-string interpolation restricted to code-controlled
  values; and an unconfirmed APOC function's availability on AuraDB
  Free was avoided entirely by moving that one piece of logic (list
  deduplication) into Python instead.
- A new resumable, batchable extraction script
  (`ingestion/pipelines/build_concept_graph.py`) builds the graph from
  the existing 388 chunks — analogous to `run_ingestion.py`, but
  populating Neo4j instead of Cloud SQL, with resumability and batching
  added as real, practical necessities, not speculative features.
- Retrieval gains a new path: graph traversal queries (e.g. "what
  should I learn before X") run alongside, not instead of, Layers 1-2's
  hybrid vector+keyword search — CRAG's search_notes tool can gain graph
  traversal as an additional signal for structural questions in a future
  refinement, while Layers 1-4's existing pipeline remains unchanged for
  content questions.

## Interview summary

"I chose Neo4j for GraphRAG instead of modeling the same relationships
as plain Postgres tables, which is a real break from every earlier
database decision in this project — pgvector over Qdrant, in-process
BM25 over a second search engine — both times I stuck with the existing
database unless there was a demonstrated reason not to. Here the reason
was that GraphRAG is specifically about demonstrating real graph-
traversal patterns and Cypher, not just relationship metadata, and
AuraDB's free tier keeps that decision cost-neutral. The graph itself is
extracted from the corpus by an LLM reading each chunk's real heading
structure — I ran the full 388-chunk extraction with zero errors,
producing 799 concepts and 970 relationships, but hit a real practical
constraint doing it: a single remote-execution call couldn't finish 388
sequential Gemini calls in one window. I fixed that properly, not by
just retrying — I made the pipeline read the graph's own current state
to determine what's left to do, and added explicit batch limits, so it
could be run repeatedly in small, always-completing chunks until done.
That's a genuinely different, more practical kind of resumability than
just 'the writes happen to be idempotent.'"
