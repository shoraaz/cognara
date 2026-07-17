# ADR 0003: Use Cloud SQL Postgres + pgvector as the vector store

Status: Accepted
Date: 2026-07-12 (amended 2026-07-17: Qdrant comparison added, public-IP consequence corrected)
Supersedes the "vector store" portion of ADR 0002's deferral list.

## Context

ADR 0002 committed the project to GCP from day one, but deliberately
deferred the *managed* vector store (Vertex AI Vector Search), suggesting
pgvector or a local index would be enough at Phase 0/1 scale.

During Phase 0 the requirement was tightened to: the vector store must
also be GCP-native from the start (not a local Chroma index). This ADR
records which GCP vector store we chose and why.

## Problem being solved

We need a vector store that is:
1. GCP-native and managed (IAM, backups, least-privilege service account).
2. Right-sized for a ~300-page corpus (a few thousand vectors), scaling to
   the full ~2,000-page ML+DL corpus later (low tens of thousands).
3. Cost-proportionate — no 24/7 bill for a prototype.
4. Able to hold user state (progress, quiz, interview grades) later, per
   the BYOK deployment profile, without adding a second database.

## Alternatives considered

| Option | Idle cost | Fit for corpus scale | Notes |
|---|---|---|---|
| **Vertex AI Vector Search** | High — deployed Index Endpoint is a VM billed 24/7 (~$0.50-0.75/hr) even with zero queries | Oversized; built for millions-to-billions of vectors | Rejected: wrong tool for a few thousand vectors, and the always-on endpoint cost is not justified by a prototype. Choosing it would signal poor tool-to-scale matching. |
| **Local Chroma** | ~$0 | Fine for scale | Rejected: not GCP-native. Fails the "GCP from the start" requirement for the storage layer. |
| **BigQuery `VECTOR_SEARCH`** | True $0 idle (pay per query) | Fine | Strong for batch/analytical retrieval; awkward for low-latency per-request online serving of the `/ask` endpoint. Kept as a possible *eval-time* tool, not the online store. |
| **Qdrant (managed or self-hosted)** | Managed Qdrant Cloud = another vendor bill; self-hosted = another service to run, patch, and monitor on GKE/Cloud Run | Excellent at large scale (hundreds of millions of vectors); more mature/tunable HNSW than pgvector at that scale | Rejected: not GCP-native (fails the "GCP from the start" requirement), and is a second system to operate when Postgres was already mandatory for other reasons (user/quiz/progress state). Our corpus scale (thousands to low tens of thousands of vectors) never reaches the range where Qdrant's extra headroom would be felt. Revisit only if corpus scale genuinely grows into the tens-of-millions range with strict latency SLAs — the same escape hatch already written for Vertex AI Vector Search below. |
| **Cloud SQL Postgres + pgvector** | Low; smallest tier is cheap, and the instance can be stopped between dev sessions | Right-sized (thousands to low-millions of vectors) | **Chosen.** |

## Decision

Use **Cloud SQL for PostgreSQL with the `pgvector` extension** as the
primary online vector store.

Reasons:

1. **Right-sized.** pgvector comfortably serves our corpus now and through
   the full ML+DL expansion. Correct tool for the scale.
2. **One store, many jobs.** The same managed Postgres instance will later
   hold user progress, quiz results, and interview grades (already implied
   by the BYOK profile). Vectors + relational state in one DB keeps the
   architecture simple.
3. **GCP-native and managed.** Cloud SQL provides IAM, automated backups,
   and integrates with a dedicated least-privilege service account —
   consistent with this project's standing security practice.
4. **Native metadata filtering.** `course_name` / `chapter` filters are
   plain SQL `WHERE` clauses, cleaner than a bespoke filter API.
5. **Cost control.** Smallest tier is inexpensive; the instance can be
   stopped when not developing, so we pay for use, not for uptime.

Retrieval uses cosine distance via pgvector's `<=>` operator, with an HNSW
index (`vector_cosine_ops`). The score returned to the app is
`1 - cosine_distance`, normalised to 0..1 to match `schemas.Citation`.

## Consequences

- A Cloud SQL Postgres instance and a `cognara_app` DB user (least
  privilege) must exist before Phase 1 ingestion.
- Local dev connects through the **Cloud SQL Auth Proxy** on
  `127.0.0.1:5432`. **Amendment (2026-07-17):** the instance was
  originally provisioned with `--no-assign-ip` (private IP only). In
  practice this made the instance unreachable from a local development
  machine outside the VPC — the Auth Proxy binary requires network
  adjacency to the VPC for a private-IP-only instance, which a laptop on
  a home network does not have. The instance was patched to also have a
  public IP (`--assign-ip`), with **zero authorized networks**, so no raw
  IP-based connection is possible from anywhere; the Auth Proxy's
  IAM-authenticated tunnel remains the only way in. Cloud Run (later)
  still uses `cloud-sql-python-connector` with `ip_type="PRIVATE"` for
  IAM-based, VPC-internal connections, since Cloud Run instances *are*
  attached to the VPC.
- `chromadb` is removed from dependencies; `pgvector`, `psycopg`,
  `sqlalchemy`, and `cloud-sql-python-connector` are added. Per ADR 0004,
  the ingestion/retrieval client code additionally moves to LangChain's
  `PGVector` wrapper over this same instance — the storage engine choice
  in this ADR is unaffected, only the client library used to talk to it.
- The DB password lives in `.env` locally and in **Secret Manager** in
  production — never in the repo.
- If corpus scale ever grows past what pgvector serves comfortably
  (millions of vectors with strict latency SLAs), revisit Vertex AI
  Vector Search or Qdrant in a new ADR — with the scale numbers that
  justify it.

## Interview summary

"I was asked to make the vector store fully GCP-native. The obvious
managed option, Vertex AI Vector Search, is built for millions-to-billions
of vectors and its serving endpoint bills 24/7 — the wrong tool and the
wrong cost curve for a few-thousand-vector corpus. Qdrant would give
excellent performance at much larger scale, but it's not a GCP service and
would mean operating a second system when Postgres was already mandatory
for user and progress state. I chose pgvector on Cloud SQL instead:
right-sized, fully managed, and it lets me keep vectors and user state in
one database. Retrieval is a cosine-distance ORDER BY and metadata
filtering is a SQL WHERE clause. I also learned the hard way that a
private-IP-only instance is unreachable from a laptop outside the VPC —
I added a public IP back with zero authorized networks, so the Auth
Proxy's IAM tunnel is still the only way in, which was the right balance
of security and a workable local dev loop. If scale ever demanded it, I'd
revisit Vector Search or Qdrant with real numbers — but I don't provision
infrastructure a prototype doesn't need."
