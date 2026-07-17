"""
app/main.py
-----------
Entry point for the Cognara Learn FastAPI application.

WHY THIS FILE EXISTS:
  FastAPI needs one place to create the app instance, attach routers,
  and run startup/shutdown lifecycle hooks. Everything else lives in
  sub-modules; this file is only wiring.

INTERVIEW EXPLANATION:
  "main.py creates the FastAPI app, registers all route groups (routers),
  and hooks into app startup — for example, to warm the vector store
  connection so the first request isn't slow."
"""

from fastapi import FastAPI
from app.api.v1 import ask
from app.core.config import settings
from app.core.logging import setup_logging

setup_logging()

app = FastAPI(
    title="Cognara Learn API",
    description="Evidence-verified AI learning and interview-preparation copilot.",
    version="0.1.0",
)

# ── Routers ───────────────────────────────────
app.include_router(ask.router, prefix="/api/v1")


@app.get("/healthz", tags=["ops"])
async def health_check() -> dict:
    """Liveness probe. Returns 200 if the process is running."""
    return {"status": "ok", "env": settings.APP_ENV}
