"""
app/services/faithfulness.py
-------------------------------
Layer 4: post-generation evidence-sufficiency gate. Checks whether a
GENERATED ANSWER actually stays faithful to the evidence it was
supposed to be based on — a real, different failure mode from Layer 3's
CRAG grading, which only ever checks the RETRIEVED EVIDENCE before
generation happens (see ADR 0007 for the full reasoning).

WHY THIS FILE EXISTS:
  generation.py's system prompt already instructs Gemini to answer only
  from the provided evidence — but a prompt instruction is not a
  verification step. This module is that verification step: a second,
  independent LLM call that reads the generated answer against its own
  evidence and judges whether every factual claim is actually supported.
  This is the well-documented "hallucination despite grounding" failure
  mode in RAG systems — a capable model with good evidence in front of
  it can still add an unsupported specific, blend in outside knowledge,
  or over-generalize past what the evidence actually says.

WHY A SECOND LLM CALL, NOT A CHEAPER HEURISTIC (see ADR 0007):
  A word/n-gram overlap check between the answer and evidence was
  considered and rejected — a faithful answer legitimately paraphrases
  and synthesizes across multiple chunks, so overlap would both flag
  correct paraphrasing as unsupported (false positive) and miss a
  fluent but fabricated specific number that happens to reuse the
  evidence's vocabulary (false negative). An NLI-style judgment —
  "does this evidence entail this claim" — needs real language
  understanding, not string matching.

WHY with_structured_output(), NOT ADK's output_schema:
  This module uses LangChain's ChatVertexAI.with_structured_output(),
  a different, independently-implemented mechanism from ADK's
  Agent(output_schema=...), which had a confirmed, documented bug when
  combined with tools (see crag_agent.py / ADR 0006). This module has
  NO tools — it is a single, direct structured-output call, not an
  agent with a tool-calling loop — so that specific failure mode does
  not apply here. Verified directly with a real call before building
  this module: given a claim with one fabricated detail ("it happens on
  Mars too") layered onto a true statement, with_structured_output()
  correctly flagged exactly the fabricated part, not the whole claim.

RETRY POLICY (see ADR 0007): exactly ONE regeneration attempt on
failure, mirroring CRAG's own "retry once, then commit" pattern (ADR
0006) rather than an open-ended loop. If the second attempt also fails
the faithfulness check, the caller (ask_service.py) falls back to an
honest abstain.

INTERVIEW EXPLANATION:
  "CRAG grades whether the retrieved evidence is good enough to attempt
  an answer, before generation happens. This module checks something
  CRAG structurally cannot see: whether the generated answer actually
  stays faithful to that evidence, after generation happens. I use a
  second, independent Gemini call for this rather than a cheap overlap
  heuristic, because a faithful answer legitimately paraphrases — word
  overlap would be both too strict and too loose. On failure I
  regenerate exactly once with the specific unsupported claims called
  out, then fall back to an honest abstain, mirroring the same
  bounded-retry pattern I used for CRAG's own retrieval grading."
"""

from langchain_core.documents import Document
from langchain_google_vertexai import ChatVertexAI
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class FaithfulnessCheck(BaseModel):
    """Structured judgment of whether a generated answer is faithful to its evidence."""
    is_faithful: bool = Field(..., description="True only if EVERY factual claim in the answer is directly supported by the evidence.")
    unsupported_claims: list[str] = Field(default_factory=list, description="Specific claims or details in the answer that the evidence does NOT support. Empty if is_faithful is True.")
    reason: str = Field(..., description="A short explanation of the judgment.")


FAITHFULNESS_SYSTEM_PROMPT = """You are a strict fact-checker. You will be given a set of evidence \
passages and an answer that was supposedly generated from that evidence.

Your job is to check EVERY factual claim in the answer against the evidence, and decide:
is_faithful = True only if every claim is directly supported by the evidence.
is_faithful = False if the answer contains ANY claim, number, name, or detail that the \
evidence does not actually state — even if it sounds plausible or is generally true knowledge.

List each unsupported claim specifically in unsupported_claims. Be strict: a paraphrase or \
synthesis of multiple evidence passages is fine and IS faithful; an added detail the evidence \
never mentioned is NOT faithful, regardless of whether it happens to be true.
"""


def _get_faithfulness_llm():
    """
    Construct a fresh ChatVertexAI + with_structured_output() chain per
    call — NOT a cached singleton. See generation.py's documented lesson
    about async gRPC clients binding to whichever event loop is active
    at first use; this module makes the same class of call, so the same
    cautious default applies.
    """
    llm = ChatVertexAI(
        model_name=settings.VERTEX_GENERATION_MODEL,
        project=settings.GCP_PROJECT_ID,
        location=settings.VERTEX_AI_LOCATION,
        temperature=0.0,  # deterministic fact-checking, not creative generation
    )
    return llm.with_structured_output(FaithfulnessCheck)


def _build_evidence_text(chunks: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in chunks)


async def check_faithfulness(answer_text: str, evidence_chunks: list[Document]) -> FaithfulnessCheck:
    """
    Judge whether `answer_text` is faithful to `evidence_chunks`.

    Args:
        answer_text: The generated answer to check.
        evidence_chunks: The same evidence Documents generation.generate()
            was given to produce this answer.

    Returns:
        A FaithfulnessCheck with is_faithful, unsupported_claims, reason.
    """
    structured_llm = _get_faithfulness_llm()
    evidence_text = _build_evidence_text(evidence_chunks)

    user_prompt = f"Evidence:\n{evidence_text}\n\nAnswer to check:\n{answer_text}"

    logger.info("faithfulness_check_start", answer_chars=len(answer_text), evidence_chunks=len(evidence_chunks))

    result = await structured_llm.ainvoke([
        ("system", FAITHFULNESS_SYSTEM_PROMPT),
        ("human", user_prompt),
    ])

    logger.info(
        "faithfulness_check_done",
        is_faithful=result.is_faithful,
        unsupported_claim_count=len(result.unsupported_claims),
    )

    return result
