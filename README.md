# Cognara Learn

**An evidence-verified AI learning and interview-preparation copilot, built over ML, Deep Learning, and Generative AI study material.**

Cognara Learn is not a generic PDF chatbot. It is a production-grade learning
system that retrieves evidence from a curated notes corpus, checks whether
that evidence is strong enough to answer, cites the exact source page, and
refuses to guess when it does not know.

## Status

Phase 0 — Project Scope, Data Audit, and High-Level Architecture (complete).
Phase 1 — Reliable RAG Foundation (starting).

## Stack (Phase 0/1)

- **API:** FastAPI (Python 3.12, `uv` for packaging)
- **Models:** Vertex AI — `text-embedding-004` (embeddings), `gemini-1.5-flash` (generation)
- **Vector store:** Cloud SQL for PostgreSQL + `pgvector` (see ADR 0003)
- **Storage:** Cloud Storage (raw PDFs + processed chunk JSONL)
- **Observability:** BigQuery (query/eval logs) + Cloud Logging (structured JSON)
- **Secrets:** Secret Manager (prod), `.env` (local)

Everything runs against real GCP APIs from day one; the FastAPI app runs
locally in Phase 0/1 and moves to Cloud Run once there's a demo.

## Documentation

- Architecture Decision Records: [`docs/adr/`](docs/adr/)
  - `0001` — focused subset vs full 6,000-page corpus
  - `0002` — GCP from day one
  - `0003` — vector store: pgvector on Cloud SQL (not Vertex AI Vector Search)
- Architecture docs: [`docs/architecture/`](docs/architecture/)
- Document catalog schema: [`data/catalog/`](data/catalog/)
- Evaluation dataset plan: [`evals/datasets/`](evals/datasets/)

## Deployment profiles

Two profiles are maintained side by side:

1. **GCP profile** (used during the learning build, from day one) — Vertex AI,
   Cloud SQL + pgvector, Cloud Storage, BigQuery, Cloud Logging, Cloud Run
   (added when we deploy a demo).
2. **BYOK profile** (designed later, for low-cost final deployment) — Gemini
   API with user key, Cloud Run or Vercel, Postgres (the same pgvector schema
   runs on any Postgres, so the vector store carries over cleanly).

See [`docs/adr/0002-gcp-from-start.md`](docs/adr/0002-gcp-from-start.md) and
[`docs/adr/0003-vector-store-pgvector-cloudsql.md`](docs/adr/0003-vector-store-pgvector-cloudsql.md)
for the reasoning behind the GCP stack choices.

## Local setup

```bash
uv sync                      # install deps
cp .env.example .env         # then fill in GCP_PROJECT_ID, CLOUD_SQL_INSTANCE, etc.
uv run pytest -v             # run tests (no GCP needed for schema tests)

# Vector DB (needs the Cloud SQL Auth Proxy running in another terminal):
make db-proxy                # starts proxy on 127.0.0.1:5432
make db-init                 # creates pgvector extension, chunks table, index
```
