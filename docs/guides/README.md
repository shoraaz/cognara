# Cognara Learn — Explainer Guides

These are comprehensive PDF walkthroughs generated during development,
documenting real infrastructure decisions, real bugs found and fixed,
and real, verified output from the running system. They are written for
someone with no prior GCP or LangChain knowledge, and doubling as
interview-preparation material (each includes a Q&A chapter with short
and deep answers for likely interview questions).

## How to add a guide here

These PDFs are generated in chat and shared as downloadable links. To
add one to this folder:
1. Download the PDF from the chat link
2. Save it into this folder (`docs/guides/`) using the filename below
3. Commit it — see the project root for git workflow

## Guides

| File | Covers |
|---|---|
| `01_gcp_infrastructure_guide.pdf` | Phase 0: every GCP resource provisioned — project, VPC, Cloud SQL, pgvector, Secret Manager, Cloud Storage, BigQuery, Vertex AI, Cloud Logging, Auth Proxy — explained from zero prior knowledge |
| `02_module1_pdf_parser.pdf` | Module 1: the PDF loader, PyMuPDF concepts, block-by-block code walkthrough, real test proof |
| `03_module2_chunker.pdf` | Module 2: heading-aware chunking, the `PARA_BREAK` sentinel design, two real bugs found and fixed (dropped headings, wrong-citation fallback) |
| `04_langchain_pivot.pdf` | ADR 0004 (LangChain + ADK, not LangGraph), the Module 3 DB bootstrap and a real Cloud SQL networking correction, the Modules 1-2 LangChain refactor and a genuine `TextSplitter` interface mismatch |
| `05_module4_ingestion.pdf` | The vector store design decision (custom `VectorStore` vs. `PGVector`), an embeddings deprecation migration, two real bugs (Vertex AI rate limiting, NUL bytes from a PDF), real proof: 388 chunks ingested, real semantic search working |
| `06_system_walkthrough.pdf` | How the whole system works end to end — a real question followed step by step from HTTP request to cited Gemini answer, with every real intermediate value shown |

## Source ADRs and docs referenced

These guides expand on the Architecture Decision Records in `docs/adr/`
and the architecture notes in `docs/architecture/` — the guides are the
"explain it like I'm new to this" version; the ADRs are the terse,
permanent record.
