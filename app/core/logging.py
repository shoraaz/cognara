"""
app/core/logging.py
-------------------
Structured logging setup using structlog.

WHY THIS FILE EXISTS:
  print() is not enough in production. structlog outputs JSON lines that
  GCP Cloud Logging can parse and index automatically. Every log line
  carries the same fields (timestamp, level, module, message), so querying
  logs in BigQuery or Cloud Logging is consistent.

# Interview notes: local-notes/INTERVIEW_PREP.md — "app/core/logging.py"
"""

import logging
import structlog
from app.core.config import settings


def _build_processors() -> list[object]:
    """Build the shared structlog processor chain used by every logger."""

    processors: list[object] = [
        # Pull any context vars bound by structlog.contextvars.bind_contextvars()
        # (e.g. request_id) into every log event emitted during this request,
        # before any other processor sees the event dict.
        structlog.contextvars.merge_contextvars,
        # Adds the "level" key (e.g. "info", "warning") to every event dict.
        structlog.processors.add_log_level,
        # Adds an ISO-8601 "timestamp" key to every event dict.
        structlog.processors.TimeStamper(fmt="iso"),
    ]

    # Keep the final renderer configurable: local dev stays readable as coloured
    # text, while production emits structured JSON for Cloud Logging.
    if settings.LOG_FORMAT == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    return processors


def setup_logging() -> None:
    """Configure structlog for the app. Call once at startup in main.py."""

    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

    structlog.configure(
        processors=_build_processors(),
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        logger_factory=structlog.PrintLoggerFactory(),
        # Cache the bound logger after first use — avoids rebuilding the
        # processor chain on every log call in hot paths.
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """Return a module-level logger. Usage: logger = get_logger(__name__)"""
    return structlog.get_logger(name)
