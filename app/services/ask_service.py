"""
app/services/ask_service.py
---------------------------
Orchestrates the full answer pipeline for one user question.

WHY THIS FILE EXISTS:
  The route (ask.py) handles HTTP. The retrieval module handles vector
  search. The generation module handles the LLM call. This file is the
  conductor — it calls them in order, makes the abstain decision, and
  assembles the final response. No HTTP logic here, no raw LLM calls here.

EXECUTION FLOW (Phase 1):
  1. Embed the question (via retrieval.embedder)
  2. Search the vector store (via retrieval.vector_store)
  3. Check if top score is above threshold
     - Below threshold → abstain, return "not in notes" response
  4. Build the prompt with evidence chunks
  5. Call Vertex AI Gemini (via services.generation)
  6. Parse and return AskResponse with citations

INTERVIEW EXPLANATION:
  "The service layer is where the RAG pipeline lives. Route → service →
  retrieval + generation. This separation lets us swap out the vector
  store or the LLM without touching the API layer."

NOTE: Phase 1 stub — retrieval and generation are called but not yet
implemented. They raise NotImplementedError until Phase 1 build begins.
"""

from app.models.schemas import AskRequest, AskResponse, Citation
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


async def answer(request: AskRequest, request_id: str) -> AskResponse:
    """
    Full RAG pipeline for one question.
    Returns an AskResponse (with citations) or an abstained response.
    """
    # ── Step 1: Retrieve evidence ──────────────────────────────────────────
    # TODO (Phase 1): replace stub with real retrieval
    # chunks = await retrieval.search(
    #     query=request.question,
    #     top_k=settings.RETRIEVAL_TOP_K,
    #     course_filter=request.course_filter,
    #     chapter_filter=request.chapter_filter,
    # )
    chunks: list = []  # stub

    # ── Step 2: Abstain if evidence is too weak ────────────────────────────
    if not chunks or _top_score(chunks) < settings.RETRIEVAL_SCORE_THRESHOLD:
        logger.info("abstaining_weak_evidence", request_id=request_id)
        return AskResponse(
            answer="The uploaded notes do not contain enough evidence to answer this question.",
            citations=[],
            confidence="abstained",
            abstained=True,
            abstain_reason="No sufficiently relevant chunks found in the loaded corpus.",
        )

    # ── Step 3: Generate answer with evidence ──────────────────────────────
    # TODO (Phase 1): replace stub with real generation call
    # answer_text, tokens = await generation.generate(
    #     question=request.question,
    #     chunks=chunks,
    # )
    answer_text = "[Phase 1 stub — generation not yet implemented]"
    tokens = 0

    # ── Step 4: Build citations from retrieved chunks ──────────────────────
    citations = [_chunk_to_citation(c) for c in chunks]

    return AskResponse(
        answer=answer_text,
        citations=citations,
        confidence=_confidence_label(_top_score(chunks)),
        abstained=False,
        tokens_used=tokens,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _top_score(chunks: list) -> float:
    """Return the highest relevance score from the retrieved chunks."""
    if not chunks:
        return 0.0
    return max(c.get("relevance_score", 0.0) for c in chunks)


def _confidence_label(score: float) -> str:
    if score >= 0.70:
        return "high"
    if score >= 0.50:
        return "medium"
    return "low"


def _chunk_to_citation(chunk: dict) -> Citation:
    return Citation(
        course_name=chunk.get("course_name", "unknown"),
        chapter=chunk.get("chapter", "unknown"),
        topic=chunk.get("topic"),
        page_number=chunk.get("page_number", 0),
        page_range=chunk.get("page_range"),
        relevance_score=chunk.get("relevance_score", 0.0),
    )
