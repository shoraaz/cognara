"""
app/services/ask_service.py
---------------------------
Orchestrates the full answer pipeline for one user question.

WHY THIS FILE EXISTS:
  The route (ask.py) handles HTTP. Retrieval, grading, and retry logic
  live in the CRAG agent (Layer 3). Generation lives in generation.py.
  Post-generation faithfulness checking lives in faithfulness.py (Layer 4).
  This file is the conductor — it calls each in order, interprets their
  decisions, and assembles the final response. No HTTP logic here, no
  raw LLM calls here.

EXECUTION FLOW:
  1. Run the CRAG agent (app.agents.crag_runner.run_crag) — this
     internally does hybrid search + rerank (Layer 2), grades the
     evidence, and retries once with a rewritten query if the first
     attempt was weak (Layer 3). Returns a grade AND the real evidence
     chunks that grade was based on.
  2. If CRAG's final decision is "abstain": return the "not in notes"
     response immediately — do not call generation at all.
  3. If CRAG's final decision is "use": build the prompt with CRAG's
     evidence chunks and call Gemini (app.services.generation).
  4. Check the generated answer's faithfulness against its own evidence
     (app.services.faithfulness — Layer 4, ADR 0007). If unfaithful,
     regenerate EXACTLY ONCE with the specific unsupported claims called
     out. If the second attempt is STILL unfaithful, fall back to an
     honest abstain rather than returning a known-unsupported answer.
  5. Assemble citations from the SAME chunks CRAG graded and generation
     used — never a separate, possibly-different re-fetch.
  6. Return AskResponse with answer, citations, confidence label,
     was_regenerated flag, and real token usage (summed across both
     generation attempts if a regeneration happened).

WHY CRAG'S GRADE REPLACES A FIXED SCORE THRESHOLD:
  The original abstention logic was a fixed numeric check:
  "abstain if top_score < settings.RETRIEVAL_SCORE_THRESHOLD". CRAG's
  critic makes a smarter decision — reading the actual evidence text, not
  just a similarity number, and retrying with a rewritten query before
  giving up. CRAG's decision now REPLACES the fixed-threshold check
  entirely. settings.RETRIEVAL_SCORE_THRESHOLD remains defined and used
  internally by hybrid_search/vector_store's k defaults, but the
  ask_service-level gate is now CRAG's decision, not a raw score check.

WHY LAYER 4's RETRY IS BOUNDED THE SAME WAY CRAG's IS (see ADR 0007):
  Exactly one regeneration attempt, then abstain — mirroring CRAG's own
  "retry once, then commit" pattern (ADR 0006). An unbounded regeneration
  loop would repeat the exact class of problem CRAG's round-1/round-2
  bugs already taught: an instruction alone ("try again until faithful")
  is not a hard guarantee, and a real system needs a hard stop.

# Interview notes: local-notes/INTERVIEW_PREP.md — "app/services/ask_service.py"
"""

from app.agents.crag_runner import run_crag
from app.models.schemas import AskRequest, AskResponse, Citation
from app.core.logging import get_logger
from app.services import faithfulness, generation

logger = get_logger(__name__)


async def answer(request: AskRequest, request_id: str) -> AskResponse:
    """
    Full RAG pipeline for one question, via the Layer 3 CRAG agent and
    the Layer 4 faithfulness gate.
    Returns an AskResponse (with citations) or an abstained response.
    """
    # ── Step 1: CRAG retrieval + grading + optional retry (Layers 2–3) ──────
    # run_crag() runs hybrid search, grades evidence quality, retries once
    # with a rewritten query if needed, and returns both the final grade AND
    # the exact evidence chunks the grade was based on (not a re-fetch).
    crag_result = await run_crag(request.question, course_filter=request.course_filter)
    grade = crag_result["grade"]
    evidence_chunks = crag_result["evidence_chunks"]

    logger.info(
        "crag_decision",
        request_id=request_id,
        decision=grade["decision"],
        relevance_score=grade["relevance_score"],
        completeness_score=grade["completeness_score"],
    )

    # ── Step 2: Abstain if CRAG's own judgment says so ───────────────────────
    # Also abstain if no chunks were returned at all (e.g. empty corpus or
    # all search results below the reranker's relevance floor).
    if grade["decision"] == "abstain" or not evidence_chunks:
        logger.info("abstaining_crag_decision", request_id=request_id, reason=grade["reason"])
        return AskResponse(
            answer="The uploaded notes do not contain enough evidence to answer this question.",
            citations=[],
            confidence="abstained",
            abstained=True,
            abstain_reason=grade["reason"],
        )

    # ── Step 3: Generate answer with CRAG's graded evidence ─────────────────
    # Convert chunk dicts to LangChain Documents so generation.generate()
    # receives its expected input type.
    langchain_docs = [_chunk_dict_to_document(c) for c in evidence_chunks]
    answer_text, tokens = await generation.generate(request.question, langchain_docs)
    total_tokens = tokens

    # ── Step 4: Layer 4 — check faithfulness, regenerate once if needed ─────
    faithfulness_result = await faithfulness.check_faithfulness(answer_text, langchain_docs)
    was_regenerated = False

    logger.info(
        "faithfulness_decision",
        request_id=request_id,
        is_faithful=faithfulness_result.is_faithful,
        unsupported_claim_count=len(faithfulness_result.unsupported_claims),
    )

    if not faithfulness_result.is_faithful:
        logger.info(
            "regenerating_unfaithful_answer",
            request_id=request_id,
            unsupported_claims=faithfulness_result.unsupported_claims,
        )
        answer_text, tokens = await generation.generate(
            request.question, langchain_docs,
            unsupported_claims=faithfulness_result.unsupported_claims,
        )
        total_tokens += tokens
        was_regenerated = True

        # Second check — did the regeneration actually fix it? If not,
        # bounded retry policy means we stop here and abstain honestly
        # rather than returning a still-unsupported answer (see ADR 0007).
        recheck = await faithfulness.check_faithfulness(answer_text, langchain_docs)
        logger.info(
            "faithfulness_recheck_decision",
            request_id=request_id,
            is_faithful=recheck.is_faithful,
            unsupported_claim_count=len(recheck.unsupported_claims),
        )
        if not recheck.is_faithful:
            logger.info("abstaining_unfaithful_after_regeneration", request_id=request_id)
            return AskResponse(
                answer="The uploaded notes do not contain enough evidence to answer this question reliably.",
                citations=[],
                confidence="abstained",
                abstained=True,
                abstain_reason=(
                    "Generated answer could not be verified as faithful to the "
                    "evidence, even after one regeneration attempt."
                ),
                was_regenerated=True,
                tokens_used=total_tokens,
            )

    # ── Step 5: Build citations from the SAME chunks CRAG graded ────────────
    # Using the identical evidence set guarantees citations match what the
    # answer was actually generated from, with no possibility of drift.
    citations = [_chunk_dict_to_citation(c) for c in evidence_chunks]

    # ── Step 6: Assemble and return the full response ────────────────────────
    return AskResponse(
        answer=answer_text,
        citations=citations,
        confidence=_confidence_label(grade["relevance_score"]),
        abstained=False,
        was_regenerated=was_regenerated,
        tokens_used=total_tokens,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _confidence_label(relevance_score: float) -> str:
    """Map CRAG's continuous relevance_score (0–1) to a human-readable label."""
    if relevance_score >= 0.70:
        return "high"
    if relevance_score >= 0.50:
        return "medium"
    return "low"


def _chunk_dict_to_document(chunk: dict):
    """
    Convert a CRAG evidence chunk dict into a langchain_core.documents.Document.
    generation.generate() and faithfulness.check_faithfulness() both expect
    Documents, so this adapter keeps their interfaces stable while CRAG's
    internal chunk-dict format can evolve independently.
    """
    from langchain_core.documents import Document
    return Document(
        page_content=chunk["text"],
        metadata={
            "course_name": chunk.get("course_name"),
            "chapter":     chunk.get("chapter"),
            "topic":       chunk.get("topic"),
            "page_number": chunk.get("page_number"),
            "page_range":  chunk.get("page_range"),
        },
    )


def _chunk_dict_to_citation(chunk: dict) -> Citation:
    """Build a Citation schema object from a CRAG evidence chunk dict."""
    return Citation(
        course_name=chunk.get("course_name", "unknown"),
        chapter=chunk.get("chapter", "unknown"),
        topic=chunk.get("topic"),
        page_number=chunk.get("page_number", 0),
        page_range=chunk.get("page_range"),
        # Use 0.0 if relevance_score is missing or None — the schema requires a float.
        relevance_score=chunk.get("relevance_score") or 0.0,
    )
