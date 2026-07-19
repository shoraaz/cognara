# Cognara Learn

**An evidence-verified AI learning and interview-preparation copilot, built over ML and Deep Learning study material.**

Cognara Learn is not a generic PDF chatbot. It retrieves evidence from a
curated notes corpus using hybrid search and graph traversal, has an agent
critique whether that evidence is actually strong enough to answer, checks
the generated answer for hallucination against its own evidence, cites the
exact source page, and honestly refuses to guess when it doesn't know.

## Status

**Layers 1–6 complete and verified against live infrastructure.** Layer 7
(multi-agent orchestration) in progress.

| Layer | What it does | Status |
|---|---|---|
| 1 — Retrieval | Embed a question, cosine-search 388 real corpus chunks in Cloud SQL | ✅ |
| 2 — Advanced Retrieval | Real BM25 keyword search + vector search fused via Reciprocal Rank Fusion, then precision-reranked with the Vertex AI Ranking API | ✅ |
| 3 — CRAG | An ADK agent grades retrieved evidence, retries once with a rewritten query if weak, abstains honestly if a retry doesn't help | ✅ |
| 4 — Faithfulness gate | A second, independent Gemini call checks the generated answer for unsupported claims; regenerates once, then abstains if still unfaithful | ✅ |
| 5 — Learning modes | Explain, Compare, Study-Plan (single-turn) + Quiz, Interview (stateful, persisted sessions with adaptive difficulty) | ✅ |
| 6 — GraphRAG | A 799-concept, 970-relationship knowledge graph in Neo4j, extracted from the corpus and wired into CRAG as a fourth tool for structural questions | ✅ |
| 7 — Multi-agent orchestration | An orchestrator agent delegating to specialized sub-agents (ADK `sub_agents`) | 🚧 in progress |
| 8 — Guardrails | Input/output safety checks | ⏳ planned |
| 9 — Eval / LLMOps | RAGAS-based evaluation harness | ⏳ planned |

The full, real story behind each layer — including bugs found and fixed by
actually running the code against live infrastructure, not just code review
— is in [`docs/adr/`](docs/adr/) (permanent record) and `local-notes/BUG_FIX_LOG.md`
(detailed narrative, gitignored, local-only).

## What actually works right now

A real question sent to `/api/v1/ask` goes through this full pipeline:

```
question
  → embed (Vertex AI text-embedding-004)
  → hybrid search: vector similarity + BM25 keyword search, fused with RRF
  → rerank: Vertex AI Ranking API narrows to the 5 most precise chunks
  → CRAG agent (ADK): reads the evidence, grades it, retries once if weak,
    or falls back to graph traversal (Neo4j) for structural questions
  → generate: Gemini 2.5 Flash writes an evidence-only, cited answer
  → faithfulness check: a second Gemini call verifies no hallucinated claims;
    regenerates once if needed
  → response: answer + real citations (course, chapter, page) + confidence
```

Every stage above is real, tested code — not a design sketch. See
[`docs/guides/`](docs/guides/) for detailed, step-by-step walkthroughs of a
real question traced through the whole system with real log output.

## Stack

- **API:** FastAPI (Python 3.12, `uv` for packaging)
- **Generation:** Vertex AI — `gemini-2.5-flash` (current stable; `gemini-1.5-flash`
  is fully retired as of this build, see `app/core/config.py`)
- **Embeddings:** Vertex AI `text-embedding-004`, via `langchain-google-genai`
- **Vector + relational store:** Cloud SQL for PostgreSQL + `pgvector` (see ADR 0003)
- **Keyword search:** real BM25 (`rank_bm25`, in-process) — see ADR 0005
- **Reranking:** Vertex AI Ranking API (via `langchain-google-community`)
- **Agent orchestration:** Google ADK (`google-adk`) — CRAG critic agent — see ADR 0004, 0006
- **Knowledge graph:** Neo4j AuraDB Free — concept graph for structural questions — see ADR 0010
- **Storage:** Cloud Storage (raw PDFs)
- **Observability:** structured JSON logging (`structlog`); BigQuery/Cloud Logging planned for Layer 9
- **Secrets:** `.env` (local) / Secret Manager (prod)

Everything runs against real GCP + Neo4j AuraDB APIs — no mocks, no
simulated responses. Every module in this repo has been run against live
infrastructure and its real output is what the tests assert against.

## Documentation

- **Architecture Decision Records:** [`docs/adr/`](docs/adr/) — 10 ADRs, the permanent
  record of every real design decision and why it was made
  - `0001` — focused 273-page subset vs full corpus
  - `0002` — GCP from day one
  - `0003` — pgvector on Cloud SQL, not Vertex AI Vector Search or Qdrant
  - `0004` — LangChain (components) + ADK (orchestration), not LangGraph
  - `0005` — Layer 2: real BM25 + Vertex AI reranking
  - `0006` — Layer 3: CRAG as an ADK agent (3 real framework bugs found and fixed)
  - `0007` — Layer 4: post-generation faithfulness gate
  - `0008` — Layer 5: single-turn learning modes
  - `0009` — Layer 5: stateful Quiz/Interview modes (shared session schema)
  - `0010` — Layer 6: GraphRAG via Neo4j (a deliberate break from "reuse Postgres")
- **Explainer guides:** [`docs/guides/`](docs/guides/) — detailed walkthroughs with
  real log output, for onboarding or interview prep
- **Architecture docs:** [`docs/architecture/`](docs/architecture/)
- **Document catalog schema:** [`data/catalog/`](data/catalog/)

## Deployment profiles

Two profiles are maintained side by side:

1. **GCP profile** (current, used during the build) — Vertex AI, Cloud SQL +
   pgvector, Neo4j AuraDB, Cloud Storage. Cloud Run planned for the demo deploy.
2. **BYOK profile** (planned, low-cost final deployment) — Gemini API with a
   user-supplied key, Cloud Run or Vercel, any Postgres host (the pgvector
   schema is portable — see ADR 0003) + Neo4j AuraDB Free (already
   platform-agnostic).

## Real, verified numbers

- **388 chunks** ingested from the real corpus (122 ML + 266 DL), zero NULL embeddings
- **799 concepts, 970 relationships** in the knowledge graph, extracted with zero errors
- **100+ tests**, all passing against live Cloud SQL, Vertex AI, and Neo4j — not mocked
- Every deprecation, rate limit, and framework bug encountered during the
  build is documented with the real error message and the real fix — see the ADRs

## Local setup

```bash
uv sync --no-install-project   # install deps (see Makefile for why --no-install-project)
cp .env.example .env           # fill in GCP_PROJECT_ID, CLOUD_SQL_INSTANCE, NEO4J_URI, etc.
uv run --no-sync pytest -v     # run tests (integration tests skip cleanly without live infra)

# Cloud SQL (start before any DB-touching command, stop when done — cost discipline):
make db-start
make db-init                   # creates pgvector extension, chunks + session tables
make ingest PDF_DIR=data/raw_pdfs   # populate the chunks table from raw PDFs
make build-graph LIMIT=20      # extract the concept graph into Neo4j, in batches
make db-stop

# Dev server:
make dev                       # FastAPI on :8000, hot reload
```

See `Makefile` (`make help`) for every available command, and `.env.example`
for the full list of required configuration, with notes on real gotchas
found while setting each piece up (e.g. Neo4j AuraDB's username is the
instance ID, not the literal string "neo4j").
