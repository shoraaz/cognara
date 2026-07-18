"""
app/services/generation.py
---------------------------
Builds the evidence-grounded prompt and calls Gemini to generate an answer.

WHY THIS FILE EXISTS:
  Retrieval finds evidence; this file turns that evidence plus the user's
  question into a real, cited answer. It owns the prompt template and the
  Gemini call — nothing else in the codebase constructs a generation
  prompt or talks to the chat model directly.

WHERE IT FITS:
  ask_service.py -> CRAG (retrieval + grading) -> generation (this file)
  -> faithfulness.py (Layer 4 check) -> AskResponse
  learning_modes.py (Layer 5) also calls generate() directly, passing a
  mode-specific system_prompt — see LAYER 5 ADDITION below.

WHY ChatVertexAI, NOT ChatGoogleGenerativeAI (a real, deliberate choice):
  Both VertexAIEmbeddings and ChatVertexAI raise LangChainDeprecationWarnings
  pointing at langchain_google_genai's replacement classes. For embeddings
  (see embedder.py) we migrated, because the same-backend replacement had no
  reported downside. For the CHAT model specifically, multiple teams have
  reported 50-90% latency increases after switching ChatVertexAI ->
  ChatGoogleGenerativeAI, attributed to the newer package using REST where
  the older one uses gRPC. Generation latency is directly user-facing, so
  this is exactly the regression worth avoiding. We keep the deprecated
  ChatVertexAI deliberately, with this reasoning documented, and will
  revisit once that regression is resolved upstream or measured acceptable
  against our own latency budget.

ASYNC gRPC CLIENT — NO MODULE-LEVEL SINGLETON:
  ChatVertexAI's async path uses grpc.aio, and a grpc.aio channel binds to
  whichever asyncio event loop is running when the channel is FIRST used.
  Caching the client as a singleton causes "RuntimeError: Event loop is closed"
  on any second event loop (e.g. each pytest-asyncio test function, a worker
  restart, or a background task on a different loop). We construct a fresh
  instance per call instead — local object setup only, no network round trip.
  See BUG_FIX_LOG.md "Generation: async gRPC client crashes on reuse across event loops".

PROMPT DESIGN — WHY CITATIONS ARE ENFORCED IN THE PROMPT:
  Cognara Learn's value proposition is trustworthy, verifiable answers. An LLM
  will often blend in outside knowledge even when given context — a well-
  documented RAG failure mode. The prompt explicitly instructs the model to
  answer ONLY from the provided evidence and cite chunk tags like [1] or [2].

LAYER 4 ADDITION — REGENERATION WITH FEEDBACK (see ADR 0007):
  generate() accepts an optional unsupported_claims parameter. When
  app.services.faithfulness.check_faithfulness() flags specific unsupported
  claims after a first generation attempt, ask_service.py calls generate()
  a SECOND time with that feedback, which is woven into the prompt as an
  explicit list of claims to avoid repeating.

LAYER 5 ADDITION — SWAPPABLE SYSTEM PROMPT FOR LEARNING MODES (see ADR 0008):
  generate() now accepts an optional system_prompt parameter, defaulting to
  the plain-Q&A SYSTEM_PROMPT below. app.services.learning_modes.py (Explain/
  Compare/Study-Plan) calls generate() directly with its own mode-specific
  system prompt, reusing EVERYTHING ELSE in this function unchanged: evidence-
  block building, citation tagging, the regeneration-feedback path, and real
  token accounting. This was a deliberate choice over duplicating generate()
  per mode — CRAG's retrieval/grading and Layer 4's faithfulness check apply
  identically regardless of which mode produced the answer, so only the
  system prompt actually needs to vary.

# Interview notes: local-notes/INTERVIEW_PREP.md — "app/services/generation.py"
"""

from langchain_core.documents import Document
from langchain_google_vertexai import ChatVertexAI

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# ── System prompt ─────────────────────────────────────────────────────────────
# Default instructions for plain Q&A: evidence-only answering, citation tags,
# and honest abstention when the evidence is insufficient. Learning modes
# (see learning_modes.py) pass their own system_prompt instead of this one.
SYSTEM_PROMPT = """You are Cognara Learn, an evidence-verified AI learning copilot.

You answer ONLY using the numbered evidence chunks provided below. Follow these rules strictly:

1. Base your answer entirely on the provided evidence. Do not use outside knowledge, even if you know the answer.
2. When you use a fact from a chunk, reference it by its tag, like [1] or [2].
3. If the evidence does not contain enough information to answer the question, say so plainly instead of guessing.
4. Write in simple, clear English. Explain technical terms the first time you use them.
5. Keep the answer focused and complete, but do not pad it with unnecessary text.
"""


def _get_llm() -> ChatVertexAI:
    """
    Construct a fresh ChatVertexAI instance per call — deliberately NOT a
    cached module-level singleton. Its async gRPC channel binds to whichever
    event loop is running at first use; reusing a cached instance across a
    different loop fails with "RuntimeError: Event loop is closed".
    See BUG_FIX_LOG.md "Generation: async gRPC client crashes on reuse across event loops".
    """
    return ChatVertexAI(
        model_name=settings.VERTEX_GENERATION_MODEL,
        project=settings.GCP_PROJECT_ID,
        location=settings.VERTEX_AI_LOCATION,
        temperature=0.2,        # low temperature: grounded, consistent answers over creative ones
        max_output_tokens=1024,
    )


def _build_evidence_block(chunks: list[Document]) -> str:
    """
    Format retrieved chunks into a numbered evidence block the model can
    reference by tag, e.g. [1], [2]. Each tag includes its citation source
    (course, chapter, topic, page) so the model can attribute facts correctly.
    """
    lines = []
    for i, doc in enumerate(chunks, start=1):
        meta = doc.metadata
        # Build a human-readable source label: "Course — Chapter — Topic"
        source = f"{meta.get('course_name', 'unknown')} — {meta.get('chapter', 'unknown')}"
        if meta.get("topic"):
            source += f" — {meta['topic']}"
        # Prefer page_range (e.g. "140-141") over a single page number when available
        page = meta.get("page_range") or meta.get("page_number")
        lines.append(f"[{i}] (Source: {source}, page {page})\n{doc.page_content}")
    return "\n\n".join(lines)


async def generate(
    question: str,
    chunks: list[Document],
    unsupported_claims: list[str] | None = None,
    system_prompt: str | None = None,
) -> tuple[str, int]:
    """
    Generate an evidence-grounded answer to `question` using `chunks` as
    the only permitted source of facts.

    Args:
        question: The user's question, verbatim.
        chunks: Retrieved evidence Documents (already filtered/thresholded by
            the CRAG agent) — this function does not re-check relevance, it
            trusts what it's given.
        unsupported_claims: Optional. If this is a REGENERATION attempt after
            Layer 4's faithfulness check flagged specific claims in a prior
            answer as unsupported (see ADR 0007), pass those claims here —
            they are woven into the prompt as an explicit list of mistakes
            to avoid repeating. None (the default) on a first attempt.
        system_prompt: Optional. Overrides the default plain-Q&A SYSTEM_PROMPT
            with a mode-specific one (see learning_modes.py, ADR 0008). None
            (the default) uses SYSTEM_PROMPT — plain /ask behaviour, unchanged.

    Returns:
        (answer_text, total_tokens_used) — total_tokens_used is the real
        figure from Gemini's response metadata, not an estimate.
    """
    # Fresh instance per call — see _get_llm() docstring for why.
    llm = _get_llm()
    evidence_block = _build_evidence_block(chunks)
    effective_system_prompt = system_prompt or SYSTEM_PROMPT

    # User turn: evidence first, then question, then citation instruction.
    # Keeping the instruction close to the question (not buried in the system
    # prompt alone) reinforces citation behaviour more reliably.
    user_prompt = (
        f"Evidence:\n{evidence_block}\n\n"
        f"Question: {question}\n\n"
        f"Answer the question using only the evidence above, citing chunk "
        f"numbers like [1] where relevant."
    )

    # Layer 4 regeneration path: append the specific unsupported claims from
    # the PREVIOUS attempt, so the model has concrete feedback rather than
    # just being asked to "try again" — see ADR 0007's retry policy.
    if unsupported_claims:
        claims_list = "\n".join(f"- {c}" for c in unsupported_claims)
        user_prompt += (
            f"\n\nIMPORTANT: a previous answer attempt included these claims "
            f"that are NOT actually supported by the evidence above:\n{claims_list}\n"
            f"Do not repeat these or any similar unsupported claims. Every "
            f"statement in your answer must be directly traceable to the "
            f"evidence provided."
        )

    logger.info(
        "generation_start",
        chunk_count=len(chunks),
        is_regeneration=bool(unsupported_claims),
        custom_system_prompt=system_prompt is not None,
    )

    response = await llm.ainvoke([
        ("system", effective_system_prompt),
        ("human", user_prompt),
    ])

    # Extract real token usage from response metadata; default to 0 if absent
    # (some API response shapes omit usage_metadata on errors or streaming).
    total_tokens = 0
    if response.usage_metadata:
        total_tokens = response.usage_metadata.get("total_tokens", 0)

    logger.info("generation_done", tokens_used=total_tokens, answer_chars=len(response.content))

    return response.content, total_tokens
