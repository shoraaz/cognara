"""
app/services/ask_service.py
---------------------------
Orchestrates the full answer pipeline for one user question.

WHY THIS FILE EXISTS:
  The route (ask.py) handles HTTP. The retrieval module handles vector
  search. The generation module handles the LLM call. This file is the
  conductor — it calls them in order, makes the abstain decision, and
  assembles the final response. No HTTP logic here, no raw LLM calls here.

EXECUTION FLOW (Module 5 — real, not a stub):
  1. Embed the question and search the vector store
     (CognaraPGVectorStore.similarity_search_with_score, Module 4)
  2. Check if the top relevance score clears settings.RETRIEVAL_SCORE_THRESHOLD
     - Below threshold or zero results → abstain, return "not in notes"
  3. Build the prompt with evidence chunks and call Gemini
     (app.services.generation, Module 5)
  4. Assemble citations from the same chunks used for generation
  5. Return AskResponse with answer, citations, confidence, real token usage

INTERVIEW EXPLANATION:
  "The service layer is where the RAG pipeline lives. Route -> service ->
  retrieval + generation. This separation lets me swap the vector store
  or the LLM without touching the API layer — I proved this during
  development: CognaraPGVectorStore and the embeddings client both
  changed mid-build (a vector store schema decision, an embeddings
  deprecation migration) and ask_service.py's own code never had to
  change, because it only ever calls the module-level interfaces."
"""

from app.models.schemas import AskRequest, AskResponse, Citation
from app.core.config import settings
from app.core.logging import get_logger
from app.retrieval.embedder import get_embeddings
from app.retrieval.vector_store import CognaraPGVectorStore
from app.services import generation

logger = get_logger(__name__)

_store_instance: CognaraPGVectorStore | None = None


def _get_store() -> CognaraPGVectorStore:
    """Shared vector store instance (module-level singleton), same pattern
    as get_embeddings() — avoids reconstructing the Cloud SQL connection
    on every request."""
    global _store_instance
    if _store_instance is None:
        _store_instance = CognaraPGVectorStore(embeddings=get_embeddings())
    return _store_instance


async def answer(request: AskRequest, request_id: str) -> AskResponse:
    """
    Full RAG pipeline for one question.
    Returns an AskResponse (with citations) or an abstained response.
    """
    # ── Step 1: Retrieve evidence ──────────────────────────────────────────
    store = _get_store()
    results = store.similarity_search_with_score(
        request.question,
        k=settings.RETRIEVAL_TOP_K,
        course_filter=request.course_filter,
        chapter_filter=request.chapter_filter,
    )

    # ── Step 2: Abstain if evidence is too weak ────────────────────────────
    top_score = results[0][1] if results else 0.0
    if not results or top_score < settings.RETRIEVAL_SCORE_THRESHOLD:
        logger.info(
            "abstaining_weak_evidence",
            request_id=request_id,
            top_score=top_score,
            threshold=settings.RETRIEVAL_SCORE_THRESHOLD,
        )
        return AskResponse(
            answer="The uploaded notes do not contain enough evidence to answer this question.",
            citations=[],
            confidence="abstained",
            abstained=True,
            abstain_reason="No sufficiently relevant chunks found in the loaded corpus.",
        )

    chunks = [doc for doc, _score in results]

    # ── Step 3: Generate answer with evidence ──────────────────────────────
    answer_text, tokens = await generation.generate(request.question, chunks)

    # ── Step 4: Build citations from the SAME chunks used for generation ───
    citations = [_result_to_citation(doc, score) for doc, score in results]

    return AskResponse(
        answer=answer_text,
        citations=citations,
        confidence=_confidence_label(top_score),
        abstained=False,
        tokens_used=tokens,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _confidence_label(score: float) -> str:
    if score >= 0.70:
        return "high"
    if score >= 0.50:
        return "medium"
    return "low"


def _result_to_citation(doc, relevance_score: float) -> Citation:
    meta = doc.metadata
    return Citation(
        course_name=meta.get("course_name", "unknown"),
        chapter=meta.get("chapter", "unknown"),
        topic=meta.get("topic"),
        page_number=meta.get("page_number", 0),
        page_range=meta.get("page_range"),
        relevance_score=relevance_score,
    )
