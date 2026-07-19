# ─────────────────────────────────────────────
# Cognara Learn — Makefile
# Run targets with:  make <target>
# On Windows, run these in Git Bash or WSL, or run the commands directly
# in PowerShell (see the PowerShell-equivalent comment under each target).
# ─────────────────────────────────────────────

.PHONY: install install-dev dev lint fmt test ingest build-graph db-proxy db-init db-start db-stop help

## Install all dependencies using uv.
## --no-install-project: Cognara Learn is an application, not a published
## library, so it never needs to be built/installed as its own package —
## only its dependencies matter. This also avoids a real Windows Smart
## App Control issue: without this flag, `uv sync` builds a wheel of
## the project itself using a temp-built Python executable, which Smart
## App Control blocks as an unsigned binary (Code Integrity event ID
## 3077/3118, "did not meet the Enterprise signing level requirements").
## PowerShell equivalent:
##   uv sync --no-install-project
install:
	uv sync --no-install-project

## Install dev dependencies too
install-dev:
	uv sync --extra dev --no-install-project

## Run the FastAPI dev server (hot reload).
## --no-sync on every `uv run` below: skips the pre-run project sync
## step, which is what triggers the same Smart App Control block as
## `uv sync` without --no-install-project (see the `install` target
## above). Run `make install-dev` first whenever dependencies change;
## --no-sync then just runs against the venv as it already is.
dev:
	uv run --no-sync uvicorn app.main:app --reload --port 8000

## Lint and format check
lint:
	uv run --no-sync ruff check .
	uv run --no-sync ruff format --check .

## Auto-fix lint issues
fmt:
	uv run --no-sync ruff format .
	uv run --no-sync ruff check --fix .

## Run all tests
test:
	uv run --no-sync pytest -v

## Start the Cloud SQL instance (compute billing resumes). Run this first
## in a dev session. PowerShell equivalent:
##   gcloud sql instances patch cognara-pg --project=kodemellow-mr-2026 --activation-policy=ALWAYS
db-start:
	gcloud sql instances patch cognara-pg --project=kodemellow-mr-2026 --activation-policy=ALWAYS

## Stop the Cloud SQL instance (compute billing pauses; storage still
## bills). Run this at the END of a dev session — see ADR 0003 for the
## stop-when-idle cost discipline this project follows. PowerShell:
##   gcloud sql instances patch cognara-pg --project=kodemellow-mr-2026 --activation-policy=NEVER
db-stop:
	gcloud sql instances patch cognara-pg --project=kodemellow-mr-2026 --activation-policy=NEVER

## Start the Cloud SQL Auth Proxy on 127.0.0.1:5432. Only needed for
## tools that connect via a plain DSN (e.g. `psql`, a GUI client) — the
## Python scripts in this repo (init_db.py, run_ingestion.py) connect
## directly via the Cloud SQL Python Connector and do NOT need this
## running. Binary lives outside the repo (downloaded once, see the
## GCP Infrastructure Guide, Chapter 12). Keep this running in a
## separate terminal if you use it. PowerShell equivalent:
##   & "C:\Users\shour\cloud-sql-proxy.exe" kodemellow-mr-2026:asia-south1:cognara-pg --port 5432
db-proxy:
	"C:/Users/shour/cloud-sql-proxy.exe" kodemellow-mr-2026:asia-south1:cognara-pg --port 5432

## Create pgvector extension, chunks table, and vector indexes (one-time,
## safe to re-run — every statement is idempotent). Connects directly via
## the Cloud SQL Python Connector; the Auth Proxy does NOT need to be
## running for this. Requires the instance to be started (make db-start)
## and a `gcloud auth application-default login` session to exist.
db-init:
	uv run --no-sync python -m ingestion.pipelines.init_db

## Run ingestion on the Phase 1 subset.
## Usage: make ingest PDF_DIR=data/raw_pdfs
ingest:
	uv run --no-sync python -m ingestion.pipelines.run_ingestion --pdf-dir $(PDF_DIR)

## Extract the Layer 6 concept graph from the corpus into Neo4j AuraDB.
## Resumable — safe to interrupt and re-run; re-reads real graph state to
## skip already-processed chunks (see ingestion/pipelines/build_concept_graph.py).
## Real-world note: a full 388-chunk run reliably exceeds a single remote-
## shell invocation's execution window: run this repeatedly with a small
## LIMIT (e.g. 15-20) until it reports "All chunks processed."
## Usage: make build-graph              (attempt everything remaining)
##        make build-graph LIMIT=20     (process at most 20 more chunks, then stop)
build-graph:
	uv run --no-sync python -m ingestion.pipelines.build_concept_graph $(if $(LIMIT),--limit $(LIMIT),)

## Show this help
help:
	@echo ""
	@echo "Available targets:"
	@grep -E '^##' Makefile | sed 's/## /  /'
	@echo ""
