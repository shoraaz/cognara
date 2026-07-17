# ─────────────────────────────────────────────
# Cognara Learn — Makefile
# Run targets with:  make <target>
# On Windows, run these in Git Bash or WSL, or run the commands directly
# in PowerShell (see the PowerShell-equivalent comment under each target).
# ─────────────────────────────────────────────

.PHONY: install install-dev dev lint fmt test ingest db-proxy db-init db-start db-stop help

## Install all dependencies using uv
install:
	uv sync

## Install dev dependencies too
install-dev:
	uv sync --extra dev

## Run the FastAPI dev server (hot reload)
dev:
	uv run uvicorn app.main:app --reload --port 8000

## Lint and format check
lint:
	uv run ruff check .
	uv run ruff format --check .

## Auto-fix lint issues
fmt:
	uv run ruff format .
	uv run ruff check --fix .

## Run all tests
test:
	uv run pytest -v

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
	uv run python -m ingestion.pipelines.init_db

## Run ingestion on the Phase 1 subset.
## Usage: make ingest PDF_DIR=data/raw_pdfs
ingest:
	uv run python -m ingestion.pipelines.run_ingestion --pdf-dir $(PDF_DIR)

## Show this help
help:
	@echo ""
	@echo "Available targets:"
	@grep -E '^##' Makefile | sed 's/## /  /'
	@echo ""
