# Phase 0 Architecture — Smallest Useful End-to-End System

Goal: prove one path works end to end — a PDF becomes searchable evidence,
and a question gets an answer with a correct page citation. Nothing more.

Vector store: Cloud SQL Postgres + pgvector (see ADR 0003).

## Ingestion flow (offline, run manually for now)

```
 PDF files (250-400 pages, selected subset)
        |
        v
 [Parser] -- extract text, keep page numbers (PyMuPDF)
        |
        v
 [Heading-aware chunker] -- split by chapter/topic, attach metadata
        |
        v
 [Metadata tagger] -- course_name, chapter, topic, page_number, source_type
        |
        v
 [Embedding call -> Vertex AI text-embedding-004]
        |
        v
 [Upsert -> Cloud SQL / pgvector: vector + text + metadata]
        |             \
        v              v
 Cloud Storage      chunks table
 (raw PDFs +        (embedding vector +
  processed          metadata columns;
  chunks JSONL)      HNSW cosine index)
```

## Request flow (online, FastAPI)

```
 User question
      |
      v
 [FastAPI /api/v1/ask endpoint]
      |
      v
 [Embed the question -> Vertex AI text-embedding-004]
      |
      v
 [pgvector cosine search] -- top-k chunks + metadata
      |   ORDER BY embedding <=> :q  LIMIT k
      |   (+ optional WHERE course_name / chapter filters)
      v
 [Build prompt: question + evidence chunks + citations required]
      |
      v
 [Vertex AI Gemini -> generate answer]
      |
      v
 [Format response]
      |         \
      v          v
 Answer to    Log to BigQuery
 user, with   (query, chunks used,
 citations    latency, tokens, cost)
 (doc, ch,
 page)
```

## Failure path (Phase 0 minimum version)

```
 pgvector search returns weak/empty results
      |
      v
 If top relevance_score < threshold  -->  respond:
                                "The uploaded notes do not contain enough
                                 evidence to answer this question."
                                (no CRAG retry yet -- that's Layer 3)
```

## Connectivity

```
 Local dev machine
      |
      |  Cloud SQL Auth Proxy (127.0.0.1:5432)
      v
 Cloud SQL Postgres (private IP, no public exposure)

 Cloud Run (later): cloud-sql-python-connector, IAM-based, proxy-less
```

## What Phase 0 does NOT include yet

- No hybrid (keyword + vector) search — Layer 2.
- No CRAG retry loop — Layer 3.
- No evidence-sufficiency gate — Layer 4.
- No quiz/interview modes — Layer 5.
- No graph/Neo4j — Layer 6.
- No agents/MCP — Layer 7.
- No guardrails beyond the basic threshold check — Layer 8.
- No Cloud Run deployment — FastAPI runs locally, calling real GCP APIs.

Each is added later, only when Phase 0/1 results show a real gap that
justifies it, per the project's engineering principle (problem →
alternatives → trade-offs → decision).

## GCP footprint for Phase 0

| Service | Used for | Why now |
|---|---|---|
| Vertex AI (Gemini) | Embeddings + answer generation | Core skill goal, core RAG function |
| Cloud SQL + pgvector | Vector store (+ future user state) | GCP-native, right-sized vectors (ADR 0003) |
| Cloud Storage | Raw PDFs, processed chunk JSONL | Real files from day one need durable storage |
| BigQuery | Query/eval logs | Cheap, starts the observability habit early |
| Cloud Logging | App logs | Default, effectively free at this volume |
| Secret Manager | DB password (prod) | Keeps secrets out of the repo |

Not yet: Vertex AI Vector Search, Cloud Run, Agent Engine, Model Armor —
see ADR 0002 and ADR 0003 for why each is deferred or rejected.
```
