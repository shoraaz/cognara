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

REAL BUG FOUND AND FIXED (Module 5 / generation.py development):
  The default VERTEX_GENERATION_MODEL was "gemini-1.5-flash", set back
  in Phase 0. Calling it in Module 5 failed with a real 404: "Publisher
  model ... was not found". Checked against current Google documentation
  (not assumed): Gemini 1.0 and 1.5 are FULLY SHUT DOWN — every request
  to them now returns 404. Even Gemini 2.0 Flash was shut down June 1,
  2026. The default here is updated to gemini-2.5-flash, the current
  stable choice, verified with a real call in asia-south1 before
  changing this. Its own retirement date is 2026-10-16, so this default
  will need updating again before then — a cheaper newer option,
  gemini-3.1-flash-lite, exists but is still PREVIEW status (no SLA) as
  of this writing, so wasn't chosen for a project that needs a stable
  API surface right now.

REAL BUG FOUND AND FIXED (Layer 3 / ADK CRAG agent development):
  The first real run of the ADK CRAG agent (app/agents/crag_agent.py)
  crashed with "ValueError: No API key was provided" — not a Cloud SQL
  or Vertex AI project problem, but ADK silently trying to use the
  CONSUMER Gemini Developer API (needs GOOGLE_API_KEY) instead of Vertex
  AI. Root cause, found by reading the actual traceback: ADK's Agent
  class does not take a vertexai=True constructor argument the way
  GoogleGenerativeAIEmbeddings (embedder.py) does — it goes through the
  unified google-genai SDK underneath, which selects its backend PURELY
  from three environment variables: GOOGLE_GENAI_USE_VERTEXAI,
  GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION. Our .env only ever
  defined GCP_PROJECT_ID and VERTEX_AI_LOCATION — different names the
  google-genai SDK does not recognise at all.
  FIX: set the three google-genai-specific environment variables
  explicitly at process startup (see _configure_adk_vertex_backend()
  below), derived from our own settings fields, rather than requiring a
  second, duplicate set of variables in .env. This keeps .env as the
  single source of truth (GCP_PROJECT_ID, VERTEX_AI_LOCATION) while
  still satisfying google-genai's specific, non-negotiable variable
  names.
"""

import os

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
    # gemini-1.5-flash is fully shut down (404 on every request, confirmed
    # 2026-07). gemini-2.5-flash is the current stable choice — retires
    # 2026-10-16, revisit before then. See docstring above.
    VERTEX_GENERATION_MODEL: str = "gemini-2.5-flash"

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


def _configure_adk_vertex_backend(settings_obj: "Settings") -> None:
    """
    Set the three environment variables the unified google-genai SDK
    (used internally by both ADK's Agent class and, transitively, by
    google-cloud-aiplatform) requires to select the Vertex AI backend
    instead of the consumer Gemini Developer API. See this file's
    docstring — REAL BUG FOUND AND FIXED (Layer 3) — for the full story.

    Only sets a variable if it is not ALREADY set in the real OS
    environment — this respects an explicit override (e.g. in a
    deployment environment that sets these directly) rather than
    silently clobbering it.
    """
    os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "TRUE")
    os.environ.setdefault("GOOGLE_CLOUD_PROJECT", settings_obj.GCP_PROJECT_ID)
    os.environ.setdefault("GOOGLE_CLOUD_LOCATION", settings_obj.VERTEX_AI_LOCATION)


# Single shared instance — import this everywhere
settings = Settings()
_configure_adk_vertex_backend(settings)
