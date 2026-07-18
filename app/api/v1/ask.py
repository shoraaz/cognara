"""
app/api/v1/ask.py
-----------------
The /ask endpoint — the only user-facing route in Phase 1.

WHY THIS FILE EXISTS:
  One route, one responsibility. The endpoint receives a question, calls
  the ask service (which handles retrieval + generation), and returns a
  typed response. The endpoint itself has no business logic.

EXECUTION FLOW:
  POST /api/v1/ask
    -> validate AskRequest (Pydantic)
    -> call ask_service.answer(request)
       (retrieval + generation live there, not here)
    -> return AskResponse

# Interview notes: local-notes/INTERVIEW_PREP.md — "app/api/v1/ask.py"
"""

import time
import uuid
from fastapi import APIRouter
from app.models.schemas import AskRequest, AskResponse
from app.services import ask_service
from app.core.logging import get_logger

router = APIRouter(tags=["ask"])
logger = get_logger(__name__)


@router.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest) -> AskResponse:
    request_id = str(uuid.uuid4())
    start = time.perf_counter()

    logger.info("ask_request_received", request_id=request_id, question=request.question)

    response = await ask_service.answer(request, request_id=request_id)

    latency_ms = (time.perf_counter() - start) * 1000
    response.latency_ms = round(latency_ms, 1)

    logger.info(
        "ask_request_done",
        request_id=request_id,
        abstained=response.abstained,
        confidence=response.confidence,
        latency_ms=response.latency_ms,
    )

    return response
