"""
app/core/config.py
------------------
Central configuration loaded from environment variables.

WHY THIS FILE EXISTS:
  All config in one place, typed, validated at startup. No magic strings
  scattered across the codebase. If a required variable is missing, the app
  fails loudly at boot, not silently mid-request.

INTERVIEW EXPLANATION:
  "We use pydantic-settings so every config value is typed and validated.
  The app fails fast at startup if GCP_PROJECT_ID is missing, rather than
  crashing on the first real request with a cryptic error."

REAL BUG FOUND AND FIXED (Module 3 / init_db.py development):
  The first time this Settings model was actually instantiated end-to-end
  (running init_db.py against the real Cloud SQL instance), it crashed
  with "Extra inputs are not permitted: gcp_service_account_key_path".
  Root cause: .env.example (and the .env generated from it) included
  GCP_SERVICE_ACCOUNT_KEY_PATH as a documented-but-optional variable, but
  this field was never actually declared on the Settings class — an
  oversight from when .env.example was written slightly ahead of the
  Python model. pydantic-settings' default behaviour is extra="forbid",
  so ANY key present in .env that isn't declared here crashes the whole
  app at import time, not just a warning.
  Fix: (1) declare the missing field explicitly, and (2) switch
  model_config to extra="ignore" as a defensive default going forward —
  a REQUIRED field like GCP_PROJECT_ID still fails loudly if missing
  (that's the behaviour we want), but an extra, not-yet-consumed key in
  .env no longer takes down the whole app. This matters in practice
  because .env.example tends to get written slightly ahead of the code
  that consumes every variable in it.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # unknown .env keys are ignored, not fatal — see docstring above
    )

    # ── GCP ───────────────────────────────────
    GCP_PROJECT_ID: str
    GCP_REGION: str = "asia-south1"
    GCP_SERVICE_ACCOUNT_KEY_PATH: str = ""  # blank = use gcloud ADC (the normal local-dev path)

    # ── Vertex AI ─────────────────────────────
    VERTEX_AI_LOCATION: str = "asia-south1"
    VERTEX_EMBEDDING_MODEL: str = "text-embedding-004"
    VERTEX_GENERATION_MODEL: str = "gemini-1.5-flash"

    # ── Cloud Storage ─────────────────────────
    GCS_BUCKET_NAME: str = "cognara-learn-dev"
    GCS_CHUNKS_PREFIX: str = "processed/chunks/"   # where JSONL chunks land in GCS

    # ── BigQuery ──────────────────────────────
    BQ_DATASET: str = "cognara_eval"
    BQ_TABLE_QUERY_LOG: str = "query_log"
    BQ_TABLE_EVAL_RESULTS: str = "eval_results"

    # ── Vector store: Cloud SQL Postgres + pgvector ───
    # GCP-native, right-sized for our corpus, and the same DB will later
    # hold user progress / quiz / interview state. See ADR 0003.
    # Local dev connects via the Cloud SQL Python Connector (see
    # ingestion/pipelines/init_db.py) — no separate Auth Proxy process
    # required. PGVECTOR_HOST/PORT below are kept for the alternative
    # standalone-proxy connection path documented in the GCP
    # Infrastructure Guide, and are unused by the Connector path.
    PGVECTOR_HOST: str = "127.0.0.1"
    PGVECTOR_PORT: int = 5432
    PGVECTOR_DB: str = "cognara"
    PGVECTOR_USER: str = "cognara_app"
    PGVECTOR_PASSWORD: str = ""            # local: .env  |  prod: Secret Manager
    PGVECTOR_TABLE: str = "chunks"
    CLOUD_SQL_INSTANCE: str = ""           # "project:region:instance" — used by the Connector
    EMBEDDING_DIM: int = 768               # text-embedding-004 output dimension

    # ── Retrieval ─────────────────────────────
    RETRIEVAL_TOP_K: int = 5
    RETRIEVAL_SCORE_THRESHOLD: float = 0.35

    # ── Logging ───────────────────────────────
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"

    # ── App ───────────────────────────────────
    APP_ENV: str = "development"
    APP_PORT: int = 8000

    @property
    def pgvector_dsn(self) -> str:
        """
        Postgres connection string built from the parts above. Used only
        by the alternative standalone-Auth-Proxy connection path; the
        Cloud SQL Python Connector path (init_db.py) builds its own
        connection directly via the Connector library instead.
        """
        return (
            f"postgresql://{self.PGVECTOR_USER}:{self.PGVECTOR_PASSWORD}"
            f"@{self.PGVECTOR_HOST}:{self.PGVECTOR_PORT}/{self.PGVECTOR_DB}"
        )


# Single shared instance — import this everywhere
settings = Settings()
