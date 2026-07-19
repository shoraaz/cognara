"""
app/core/config.py
------------------
Central configuration loaded from environment variables via pydantic-settings.

WHY THIS FILE EXISTS:
  All config in one place, typed, validated at startup. No magic strings
  scattered across the codebase. If a required variable is missing, the app
  fails loudly at boot, not silently mid-request.

DESIGN:
  - pydantic-settings reads .env automatically; every field is typed and
    validated before the first request is served.
  - extra="ignore": unknown keys in .env are silently skipped rather than
    crashing the app — see BUG_FIX_LOG.md "Config: pydantic-settings extra=forbid crash".
  - The three google-genai SDK env vars are set at module import time by
    _configure_adk_vertex_backend() so ADK's Agent uses Vertex AI, not the
    consumer Gemini API — see BUG_FIX_LOG.md "Config: ADK picks consumer Gemini API".
  - VERTEX_GENERATION_MODEL default updated from gemini-1.5-flash (retired,
    returns 404) to gemini-2.5-flash — see BUG_FIX_LOG.md "Config: Gemini 1.5 Flash retired".

# Interview notes: local-notes/INTERVIEW_PREP.md — "app/core/config.py"
"""

import os

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        # Ignore unknown .env keys instead of crashing — required fields still
        # fail loudly if absent. See BUG_FIX_LOG.md "Config: pydantic-settings extra=forbid crash".
        extra="ignore",
    )

    # ── GCP ───────────────────────────────────
    GCP_PROJECT_ID: str                         # required — app won't start without it
    GCP_REGION: str = "asia-south1"
    GCP_SERVICE_ACCOUNT_KEY_PATH: str = ""      # blank = use gcloud ADC (normal local-dev path)

    # ── Vertex AI ─────────────────────────────
    VERTEX_AI_LOCATION: str = "asia-south1"
    VERTEX_EMBEDDING_MODEL: str = "text-embedding-004"
    # gemini-2.5-flash: current stable choice — retires 2026-10-16, revisit before then.
    # Previous default gemini-1.5-flash returns 404 (fully retired).
    # See BUG_FIX_LOG.md "Config: Gemini 1.5 Flash retired".
    VERTEX_GENERATION_MODEL: str = "gemini-2.5-flash"

    # ── Cloud Storage ─────────────────────────
    GCS_BUCKET_NAME: str = "cognara-learn-dev"
    GCS_CHUNKS_PREFIX: str = "processed/chunks/"   # where JSONL chunks land in GCS

    # ── BigQuery ──────────────────────────────
    BQ_DATASET: str = "cognara_eval"
    BQ_TABLE_QUERY_LOG: str = "query_log"
    BQ_TABLE_EVAL_RESULTS: str = "eval_results"

    # ── Vector store: Cloud SQL Postgres + pgvector ───────────────────────
    # GCP-native, right-sized for our corpus, and the same DB will later hold
    # user progress / quiz / interview state. See ADR 0003.
    # Local dev connects via the Cloud SQL Python Connector (see
    # ingestion/pipelines/init_db.py) — no separate Auth Proxy process required.
    # PGVECTOR_HOST/PORT below are kept for the alternative standalone-proxy
    # connection path documented in the GCP Infrastructure Guide; they are
    # unused by the Connector path.
    PGVECTOR_HOST: str = "127.0.0.1"
    PGVECTOR_PORT: int = 5432
    PGVECTOR_DB: str = "cognara"
    PGVECTOR_USER: str = "cognara_app"
    PGVECTOR_PASSWORD: str = ""                 # local: .env  |  prod: Secret Manager
    PGVECTOR_TABLE: str = "chunks"
    CLOUD_SQL_INSTANCE: str = ""                # "project:region:instance" — used by the Connector
    EMBEDDING_DIM: int = 768                    # text-embedding-004 output dimension

    # ── Graph store: Neo4j AuraDB Free (see ADR 0010, Layer 6 GraphRAG) ───
    # A deliberate exception to "reuse Cloud SQL" (unlike pgvector/BM25) —
    # see ADR 0010 for why. AuraDB Free is genuinely free, no credit card;
    # blank defaults here mean the app still starts fine before the
    # instance/credentials exist — Layer 6 code checks for presence itself
    # rather than making these required app-wide.
    NEO4J_URI: str = ""                         # e.g. "neo4j+s://xxxxxxxx.databases.neo4j.io"
    NEO4J_USERNAME: str = "neo4j"
    NEO4J_PASSWORD: str = ""                    # local: .env  |  prod: Secret Manager

    # ── Retrieval ─────────────────────────────
    RETRIEVAL_TOP_K: int = 5
    RETRIEVAL_SCORE_THRESHOLD: float = 0.35     # used by hybrid_search/vector_store k defaults

    # ── Logging ───────────────────────────────
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"                    # "json" = Cloud Logging | "text" = local dev

    # ── App ───────────────────────────────────
    APP_ENV: str = "development"
    APP_PORT: int = 8000

    @property
    def pgvector_dsn(self) -> str:
        """
        Postgres DSN built from the parts above. Used only by the alternative
        standalone-Auth-Proxy connection path; the Cloud SQL Python Connector
        path (init_db.py) builds its own connection directly via the Connector
        library instead.
        """
        return (
            f"postgresql://{self.PGVECTOR_USER}:{self.PGVECTOR_PASSWORD}"
            f"@{self.PGVECTOR_HOST}:{self.PGVECTOR_PORT}/{self.PGVECTOR_DB}"
        )


def _configure_adk_vertex_backend(settings_obj: "Settings") -> None:
    """
    Set the three environment variables the unified google-genai SDK requires
    to route requests to Vertex AI instead of the consumer Gemini Developer API.

    ADK's Agent class (unlike GoogleGenerativeAIEmbeddings) does not accept a
    vertexai=True constructor argument — it reads these three vars exclusively.
    We derive them from our own Settings fields so .env remains the single
    source of truth. See BUG_FIX_LOG.md "Config: ADK picks consumer Gemini API".

    Uses setdefault() so an explicit OS-level override (e.g. in a deployment
    environment) is never clobbered.
    """
    os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "TRUE")
    os.environ.setdefault("GOOGLE_CLOUD_PROJECT", settings_obj.GCP_PROJECT_ID)
    os.environ.setdefault("GOOGLE_CLOUD_LOCATION", settings_obj.VERTEX_AI_LOCATION)


# Single shared instance — import this everywhere.
# _configure_adk_vertex_backend runs immediately so ADK is correctly
# initialised before any agent is instantiated.
settings = Settings()
_configure_adk_vertex_backend(settings)
