# ADR 0005: Layer 2 Retrieval — rank_bm25 for Keyword Search, Vertex AI for Reranking

Status: Accepted
Date: 2026-07-18

## Context

Layer 1 (built, proven live) uses pure vector similarity search: embed the
question, cosine-search against 388 chunk embeddings in Cloud SQL. This
works well for concept questions ("what is overfitting") but has a known
weakness for exact-term questions — acronyms, algorithm names, and
formula-adjacent terms (e.g. "ReLU", "AdaBoost", "cross-entropy") can be
semantically diluted in a 768-dimensional embedding even when they are the
literal, exact word the user is asking about. The original project master
prompt calls this out directly: ML/DL notes benefit from hybrid retrieval
because "exact terms, formulas, acronyms, algorithm names... often
benefit from keyword search, while concept questions benefit from
semantic search."

This ADR covers two Layer 2 decisions: which keyword-search
implementation to add, and which reranking approach to use once hybrid
(vector + keyword) results exist.

## Decision 1: Keyword search via rank_bm25, not Postgres full-text search

### Alternatives considered

| Option | New infra | Exact BM25 | SQL-filterable | Scales past ~50k chunks |
|---|---|---|---|---|
| **Postgres full-text search** (tsvector + ts_rank) | None — same DB | No — Postgres's own TF-based ranking, BM25-family but not the formula | Yes, native WHERE | Yes, real DB index |
| **rank_bm25 (Python library)** | New dependency, in-memory index | Yes — real BM25Okapi | No — needs separate Python-side filtering | No — full in-memory rebuild on every process start |

Postgres full-text search was the architecturally consistent choice per
ADR 0003/0004's established reasoning (prefer the existing database over
a new system unless there's a real, measured reason to add one). It was
rejected anyway, deliberately, in favor of rank_bm25.

### Reasons for choosing rank_bm25

1. **Genuine BM25, not a "BM25-family" approximation.** Postgres's
   ts_rank is a real, useful ranking function, but is not the BM25
   formula (term frequency, inverse document frequency, and length
   normalization combined the specific way Okapi BM25 defines). If this
   project claims "hybrid retrieval with BM25" — in documentation, in an
   interview — it should be true, not "BM25-adjacent."
2. **Corpus scale makes the main objection moot for now.** rank_bm25's
   real weakness (full in-memory rebuild on every process start, no
   SQL-level filtering) matters at large scale. At 388 chunks, rebuilding
   the index takes a trivial amount of time and memory. This is a
   deliberate, documented trade of long-term architectural purity for
   short-term correctness and learning value — revisit if the corpus
   grows into a range where in-memory rebuild becomes a real cost (see
   Consequences).
3. **Consistent with the project's teaching-first goal.** Implementing
   real BM25 — tokenization, the index, the scoring — is more directly
   useful preparation for explaining hybrid retrieval in an interview
   than configuring Postgres's built-in ranking function.

### Consequences

- `rank_bm25` is added as a new dependency.
- A new module owns keyword search: builds an in-memory BM25 index from
  all chunk texts (loaded from Cloud SQL once, at process startup or
  first use), tokenizes queries the same way, and returns BM25 scores.
- Because BM25 scores and cosine similarity scores are on different,
  incomparable scales, the two result sets must be merged with a
  fusion method (Reciprocal Rank Fusion — RRF) rather than combined
  directly. See the hybrid retrieval module for the real implementation.
- Course/chapter metadata filtering for the keyword path is done in
  Python (filtering the chunk list before/after BM25 scoring), not SQL —
  a direct consequence of choosing an in-memory index over Postgres FTS.
- If the corpus grows large enough that in-memory BM25 rebuild becomes a
  real startup-time or memory cost, revisit this ADR with real numbers —
  the same escape hatch pattern used in ADR 0003 for the vector store.

## Decision 2: Reranking via the Vertex AI Ranking API

### Alternatives considered

| Option | Cost/ops | Consistency |
|---|---|---|
| **Local cross-encoder** (sentence-transformers) | Free, runs on CPU, new ML dependency to manage | Not GCP-native |
| **Vertex AI Ranking API** | Managed, per-call cost, no local model to maintain | GCP-native, consistent with ADR 0002's "GCP from day one" |

### Decision

Use the Vertex AI Ranking API for the reranking step (Layer 2, stage 3:
rerank the top-N results from hybrid retrieval down to the final top-K
sent to generation).

### Reasons

1. Consistent with this project's standing GCP-first commitment (ADR
   0002) — the same reasoning already applied to embeddings and
   generation.
2. No local model weights to download, version, or run — one fewer
   moving part in an already multi-piece retrieval pipeline (vector +
   BM25 + fusion + rerank).
3. Managed infrastructure means reranking latency and availability are
   Google's operational responsibility, not this project's.

### Consequences

- A new Vertex AI API call is added to the retrieval path — adds latency
  and (small, per-call) cost on every request, which should be measured
  once implemented and weighed against the quality improvement it
  provides, per this project's standing "measure, don't assume"
  engineering principle.
- Reranking is a distinct, separate stage from BM25/vector fusion — it
  operates on the fused top-N results, not on raw chunks.

## Interview summary

"For keyword search, I chose to implement real BM25 with the rank_bm25
library rather than Postgres's built-in full-text search, even though
using Postgres would have been more consistent with my existing
GCP-and-single-database architecture. I made that trade deliberately:
Postgres's ts_rank isn't the actual BM25 formula, and I wanted a true
hybrid-retrieval implementation to reason about and explain, especially
at my current corpus scale where BM25's real weakness — rebuilding an
in-memory index — isn't yet a real cost. I combine BM25 and vector scores
with Reciprocal Rank Fusion, since the two scores aren't on comparable
scales. For reranking, I use the Vertex AI Ranking API instead of a local
cross-encoder, keeping with my GCP-first approach from earlier ADRs and
avoiding one more locally-managed model."
