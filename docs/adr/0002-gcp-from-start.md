# ADR 0002: Use GCP from day one, not after a local prototype

Status: Accepted
Date: 2026-07-12

## Context

The original plan was: build and test everything locally first, then move
to GCP once the pipeline works. The user corrected this — GCP should be
used from the start, since Vertex AI / Google Cloud experience is one of
the explicit learning goals of this project, not just a deployment target.

## Problem being solved

If we build fully local first (local embeddings, local LLM calls, local
files only), we get a working RAG pipeline but zero GCP experience. Moving
to GCP later means re-doing the plumbing (auth, storage paths, API
clients) and delays the exact skill (Vertex AI / Agent Platform) that is a
stated goal of this project.

## Alternatives considered

| Option | Trade-off |
|---|---|
| Fully local (Ollama/local embeddings, local disk, no cloud) | Zero cloud cost, fast iteration, but no GCP learning until much later. Rework needed to "cloudify" later. |
| Fully GCP-managed from day one (Vertex AI Vector Search, Agent Engine, Cloud Run, Model Armor, all at once) | Matches the end goal, but violates the project's own rule: "never add a technology only because it is trending" / not yet justified by scale. Highest cost and complexity for a 250-400 page prototype. |
| **GCP for the model layer + storage, minimal/simple stack elsewhere, expand managed services as later phases justify them** | Gets real Vertex AI experience immediately, keeps cost and complexity proportional to a Phase 0/1 prototype. |

## Decision

We use GCP from the start, but only for the pieces that are justified at
Phase 0/1 scale:

- **Vertex AI (Gemini models)** — for embeddings and answer generation.
  This is the core skill we're here to learn, so it starts now, not later.
- **Cloud Storage** — for raw PDFs and processed chunks. We are ingesting
  real files from day one, so we need durable storage from day one.
- **BigQuery** — for evaluation results and request logs. Free tier is
  generous, and "observability from the beginning" is a stated project
  principle, so start the habit now with cheap, simple tables.
- **Cloud Logging** — default logging, effectively free at this volume.

Deferred until a later phase justifies them (see relevant ADRs when we get
there):

- **Vertex AI Vector Search** (managed vector DB) — not justified yet. At
  250-400 pages (a few thousand chunks), a simple vector store (e.g.
  pgvector or a local FAISS/Chroma index, loaded from Cloud Storage) is
  enough and avoids managed-vector-search cost/complexity before we even
  know our chunking strategy is good. We will revisit this in the Layer 2
  (advanced retrieval) ADR once corpus size grows.
- **Cloud Run deployment** — Phase 0/1 runs FastAPI locally against GCP
  APIs (Vertex AI, Cloud Storage, BigQuery) using Application Default
  Credentials. We deploy to Cloud Run once there is something worth
  demoing.
- **Vertex AI Agent Platform / Agent Engine, Model Armor, IAM hardening,
  CI/CD** — deferred to Layer 7/8, once there are real agents and a real
  attack surface to defend.

## Consequences

- A GCP project and billing account must exist before Phase 0 tasks are
  done (see task list).
- Every API call to Vertex AI has a real cost from day one. We keep
  evaluation batches small (dozens of questions, not thousands) and set a
  budget alert immediately, per the project's cost rules.
- The FastAPI backend takes a GCP project ID and credentials via
  environment variables from the very first line of code — this is not
  retrofitted later.
- Local-only fallback (e.g., a fully offline mode) is explicitly out of
  scope. This project's local dev machine still talks to real GCP APIs.
