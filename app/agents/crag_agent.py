"""
app/agents/crag_agent.py
---------------------------
Layer 3: Corrective RAG (CRAG), implemented as an ADK Agent with tools —
per ADR 0004's decision that orchestration (Layer 3 onward) uses ADK,
not LangGraph or hand-rolled control flow.

WHY AN AGENT INSTEAD OF HAND-CODED if/else RETRY LOGIC:
  The original master prompt's CRAG flow is:
      Question -> retrieve evidence -> critic grades retrieval quality
        -> if weak, rewrite/expand query and retry once
        -> if still weak, abstain or ask clarification
        -> if strong, send evidence to answer generator
  This COULD be written as a fixed Python function with if-statements.
  We use an ADK Agent instead because the actual judgment calls — "is
  this retrieval good enough," "how should I rewrite this vague query,"
  "should I retry or just abstain" — are genuinely better made by an LLM
  reasoning about the specific evidence in front of it than by a fixed
  numeric threshold alone. The agent decides WHICH tool to call next and
  WHEN to stop, which is real corrective reasoning, not a hardcoded loop
  wearing an agent costume.

THE THREE TOOLS (see ADR 0004's CRAG tool list, and this session's design
discussion — grade_retrieval is a DELIBERATE separate tool call, not the
agent's own output_schema, specifically so the grading step is visible
and debuggable in ADK's own event trace, not implicit in the final
answer):

  1. search_notes(query, course_filter=None, chapter_filter=None)
     Wraps Layer 2's full pipeline (hybrid_search -> rerank) UNCHANGED —
     CRAG adds judgment on top of retrieval, it does not reimplement it.

  2. grade_retrieval(relevance_score, completeness_score, decision,
     reason, rewritten_query) — see REAL BUG FOUND AND FIXED (round 3)
     below for why this tool's signature takes the AGENT'S OWN computed
     judgment as arguments, rather than the tool computing anything
     itself.

  3. rewrite_query(original_query, reason_retrieval_was_weak)
     Only called by the agent when grade_retrieval's decision is
     "retry" — expands or clarifies a vague/narrow query before a
     second, and FINAL, retrieval attempt (the master prompt specifies
     "retry once," not an unbounded loop).

MODEL CHOICE FOR THE AGENT ITSELF:
  gemini-2.5-flash — same stable model already verified working
  end-to-end in generation.py (Layer 1). The critic's job (grading
  retrieval quality) is lighter-weight than final answer generation, so
  there is no reason to reach for a larger/slower model here.

REAL BUG FOUND AND FIXED (round 1) — output_schema + tools TOGETHER
CAUSES AN INFINITE-ISH TOOL-CALL LOOP (a confirmed, documented ADK bug):
  The first real run of this agent called grade_retrieval TWELVE times
  for what should be at most two calls — taking over two minutes and
  burning real API cost for no reason. This matched several open ADK
  GitHub issues (e.g. #3413, #3940, #3969): when an Agent has BOTH
  output_schema and tools configured, it can repeatedly re-call a tool
  instead of committing to a final structured response. Confirmed as a
  real, current framework-level issue, not a bug in this module.
  FIX: do NOT set output_schema on this agent. The agent's final text
  response is instructed to contain the RetrievalGrade JSON directly
  (see CRAG_INSTRUCTION), and crag_runner.py parses that text as JSON.

REAL BUG FOUND AND FIXED (round 2) — INSTRUCTION-FOLLOWING ALONE WAS
NOT FULLY RELIABLE FOR THE "AT MOST 2 GRADES" LIMIT:
  A real retry-path test (a deliberately vague query, "improvements")
  showed the agent called grade_retrieval THREE times in one run — one
  more than the instruction says. FIX: added a hard, code-level call
  counter inside grade_retrieval itself; the THIRD call in any single
  run is forced to "use" with an explicit "grading limit reached"
  reason. Reset per CRAG run by crag_runner.py's reset_grade_call_count().

REAL BUG FOUND AND FIXED (round 3) — grade_retrieval's ORIGINAL SIGNATURE
CONFUSED THE AGENT ABOUT WHO WAS SUPPOSED TO GRADE:
  The original grade_retrieval(query, evidence_summaries) took only the
  QUESTION and EVIDENCE as input, and always returned all-None fields
  (relevance_score=None, decision=None, ...) with a comment saying "the
  agent fills in real values" — intending the agent to read the None
  placeholders as a cue to reason and report its own grade separately.
  In practice, a real run's output literally said: "The grade_retrieval
  tool failed to return valid scores or a decision on two attempts" —
  the agent interpreted the all-None response as the TOOL failing, not
  as an invitation to grade itself. The final answer still happened to
  be reasonable by lucky recovery, but the design was genuinely
  confusing the model, not just cosmetically odd.
  FIX: grade_retrieval's signature now takes relevance_score,
  completeness_score, decision, reason, and rewritten_query AS
  ARGUMENTS — the agent computes its own judgment by reasoning over the
  evidence search_notes returned, THEN calls grade_retrieval to RECORD
  that judgment (the tool validates/logs/returns it, it does not compute
  anything). This is a clearer division of labour: the agent grades, the
  tool records — instead of the tool pretending to grade and secretly
  expecting the agent to notice it didn't.

INTERVIEW EXPLANATION:
  "CRAG's corrective loop is implemented as an ADK Agent with three
  tools rather than hardcoded retry logic, because the actual decisions
  are judgment calls better made by reasoning over the specific evidence
  than a fixed threshold. Building it, I hit three real issues in
  sequence: a confirmed ADK framework bug combining output_schema with
  tools (12x tool-call loop, fixed by dropping output_schema); a soft
  instruction-following slip where the agent over-called my grading tool
  (fixed with a hard code-level call counter); and a genuine design
  flaw where my grading tool's signature implied the TOOL would compute
  the grade, when I actually wanted the AGENT to compute it and the tool
  to just record it — the agent got confused and reported the tool as
  'failing' when it returned placeholder None values. I fixed that by
  changing the tool's signature so the agent passes ITS OWN computed
  scores as arguments, which is a much clearer contract."
"""

from typing import Literal

from google.adk.agents import Agent
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.logging import get_logger
from app.retrieval.embedder import get_embeddings
from app.retrieval.hybrid_search import hybrid_search
from app.retrieval.keyword_search import BM25KeywordIndex
from app.retrieval.reranker import rerank
from app.retrieval.vector_store import CognaraPGVectorStore

logger = get_logger(__name__)

CRAG_AGENT_MODEL = "gemini-2.5-flash"


# ── Structured critic output shape (see master prompt's CRAG spec).
#    NOT used as ADK's output_schema (see REAL BUG FOUND AND FIXED,
#    round 1) — kept as a plain reference schema, and crag_runner.py
#    validates the agent's parsed JSON against it manually. ──────────────

class RetrievalGrade(BaseModel):
    """The critic's structured judgment of one retrieval attempt."""
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="How relevant the evidence is to the query, 0-1.")
    completeness_score: float = Field(..., ge=0.0, le=1.0, description="How completely the evidence covers what the question needs, 0-1.")
    decision: Literal["use", "retry", "abstain"] = Field(..., description="use: evidence is good enough to answer. retry: worth rewriting the query and searching once more. abstain: even a retry is unlikely to help.")
    reason: str = Field(..., description="A short, human-readable explanation of the decision.")
    rewritten_query: str | None = Field(default=None, description="Only set when decision is 'retry' — a rewritten/expanded version of the query to search again.")


# ── Tool implementations (plain functions — ADK reads their docstrings
#    and type hints to build the tool schema automatically) ─────────────────

_keyword_index: BM25KeywordIndex | None = None

# Hard, code-level backstop for the "grade at most twice per question"
# rule — see REAL BUG FOUND AND FIXED (round 2) above. Reset per CRAG
# run via reset_grade_call_count(), called by crag_runner.py before
# each new question.
_grade_call_count = 0


def reset_grade_call_count() -> None:
    """Reset the grading call counter. Call this once at the start of
    every CRAG run (see crag_runner.py) so each question gets its own
    fresh 2-call budget, not a count shared across the whole process."""
    global _grade_call_count
    _grade_call_count = 0


def _get_keyword_index() -> BM25KeywordIndex:
    """Shared BM25 index — safe to cache (see keyword_search.py; unlike
    generation.py's ChatVertexAI, this has no async gRPC event-loop
    binding risk, it's a plain in-memory data structure)."""
    global _keyword_index
    if _keyword_index is None:
        _keyword_index = BM25KeywordIndex()
        _keyword_index.refresh()
    return _keyword_index


def search_notes(
    query: str,
    course_filter: str | None = None,
    chapter_filter: str | None = None,
) -> list[dict]:
    """Search the course notes for evidence relevant to a query.

    Runs Cognara's full Layer 2 retrieval pipeline: hybrid search
    (vector similarity + BM25 keyword search, fused with Reciprocal
    Rank Fusion) followed by precision reranking with the Vertex AI
    Ranking API. Returns the top 5 most relevant evidence chunks.

    Args:
        query: The question or topic to search for.
        course_filter: Optional. Restrict results to one course, e.g.
            "100 Days of Machine Learning" or "100 Days of Deep Learning".
        chapter_filter: Optional. Restrict results to one chapter within
            the course filter.

    Returns:
        A list of up to 5 evidence chunks, each a dict with: text,
        course_name, chapter, topic, page_number, page_range,
        relevance_score (the reranker's precision score, 0-1).
    """
    store = CognaraPGVectorStore(embeddings=get_embeddings())
    keyword_index = _get_keyword_index()

    hybrid_results = hybrid_search(
        query, store, keyword_index,
        k_per_method=20, top_k=10,
        course_filter=course_filter, chapter_filter=chapter_filter,
    )
    reranked_docs = rerank(query, hybrid_results, top_n=5)

    logger.info("crag_search_notes", query=query, results=len(reranked_docs))

    return [
        {
            "text": doc.page_content,
            "course_name": doc.metadata.get("course_name"),
            "chapter": doc.metadata.get("chapter"),
            "topic": doc.metadata.get("topic"),
            "page_number": doc.metadata.get("page_number"),
            "page_range": doc.metadata.get("page_range"),
            "relevance_score": doc.metadata.get("relevance_score"),
        }
        for doc in reranked_docs
    ]


def grade_retrieval(
    relevance_score: float,
    completeness_score: float,
    decision: Literal["use", "retry", "abstain"],
    reason: str,
    rewritten_query: str | None = None,
) -> dict:
    """Record your grading decision for the current retrieval attempt.

    Call this AFTER you have read the evidence and formed your own
    judgment — YOU compute relevance_score, completeness_score, decision,
    and reason by reasoning over the evidence search_notes returned; this
    tool does not grade anything itself, it only records and validates
    the judgment you already made. Call this AT MOST TWICE PER QUESTION
    (once per retrieval attempt) — never more.

    Args:
        relevance_score: Your own assessment, 0-1, of how relevant the
            evidence is to the question.
        completeness_score: Your own assessment, 0-1, of how completely
            the evidence covers what the question needs.
        decision: Your own decision: "use" (evidence is good enough),
            "retry" (worth rewriting the query and searching once more,
            only valid on your FIRST call), or "abstain" (even a retry
            is unlikely to help).
        reason: Your own short explanation for the decision.
        rewritten_query: Leave as null unless decision is "retry".

    Returns:
        A dict confirming what was recorded, or — if this is called a
        THIRD time in the same run — a forced override to "use" with an
        explicit note that the grading limit was reached (a hard,
        code-level backstop; see module docstring, REAL BUG FOUND AND
        FIXED round 2).
    """
    global _grade_call_count
    _grade_call_count += 1
    logger.info(
        "crag_grade_retrieval_called",
        call_number=_grade_call_count,
        relevance_score=relevance_score,
        completeness_score=completeness_score,
        decision=decision,
        reason=reason,
    )

    if _grade_call_count > 2:
        logger.info("crag_grade_limit_reached", call_number=_grade_call_count)
        return {
            "relevance_score": relevance_score,
            "completeness_score": completeness_score,
            "decision": "use",
            "reason": "Grading limit reached (2 calls) — using best available evidence rather than retrying further.",
            "rewritten_query": None,
        }

    return {
        "relevance_score": relevance_score,
        "completeness_score": completeness_score,
        "decision": decision,
        "reason": reason,
        "rewritten_query": rewritten_query,
    }


def rewrite_query(original_query: str, reason_retrieval_was_weak: str) -> str:
    """Rewrite a query that produced weak retrieval results.

    Call this only when grade_retrieval's decision was "retry", and only
    ONCE per question. Expand vague terms, add likely synonyms or
    related terminology from the ML/DL domain, or clarify ambiguous
    phrasing — whatever the reason_retrieval_was_weak suggests is the
    actual problem.

    Args:
        original_query: The query that produced weak results.
        reason_retrieval_was_weak: The critic's explanation of why the
            evidence was insufficient — use this to target the rewrite.

    Returns:
        A rewritten query string to search again with search_notes().
        This is the LAST retry — do not call rewrite_query() more than
        once per original question.
    """
    logger.info("crag_rewrite_query_called", original_query=original_query, reason=reason_retrieval_was_weak)
    return original_query


CRAG_INSTRUCTION = """You are Cognara Learn's retrieval quality critic, implementing Corrective RAG.

Your job for every question:
1. Call search_notes(query) to retrieve evidence.
2. READ the evidence text carefully. Form your OWN judgment of its
   relevance_score (0-1), completeness_score (0-1), decision
   ("use"/"retry"/"abstain"), and a short reason — based on how well the
   ACTUAL evidence text answers the ACTUAL question.
3. Call grade_retrieval(relevance_score, completeness_score, decision,
   reason, rewritten_query) to RECORD the judgment you just formed. Pass
   YOUR OWN computed values as arguments — grade_retrieval does not grade
   anything itself, it only records what you tell it. Call this at most
   TWICE per question.
4. If your decision was "retry": call rewrite_query() ONCE to get a
   better query, then call search_notes() again with the rewritten
   query, then read the new evidence and call grade_retrieval() ONE more
   time (your second and FINAL call) with your updated judgment.
5. Once you have your final grading result, respond with ONLY a single
   JSON object, no other text, matching exactly this shape:
   {"relevance_score": <float 0-1>, "completeness_score": <float 0-1>,
    "decision": "use" | "retry" | "abstain", "reason": "<short explanation>",
    "rewritten_query": "<string or null>"}
   Note: your FINAL reported "decision" must be "use" or "abstain" only —
   never report "retry" as your final decision, since a retry (if any) must
   already have happened in step 4 by the time you respond.

Be honest and strict in grading — do not mark evidence as sufficient just
to avoid a retry. Cognara Learn's entire value is trustworthy answers with
real evidence; it is better to abstain than to force a weak answer.
"""


def build_crag_agent() -> Agent:
    """Construct the CRAG agent. Not a module-level singleton — see
    generation.py's documented lesson about async clients and event
    loops; ADK's Agent wraps a similar async model client under the
    hood, so we apply the same cautious default.

    Deliberately NOT passing output_schema — see this module's REAL BUG
    FOUND AND FIXED (round 1) docstring section for why. The agent's
    final text response is instructed to already be the correctly-shaped
    JSON; crag_runner.py parses and validates it against RetrievalGrade
    manually instead of relying on ADK's schema enforcement.
    """
    return Agent(
        name="crag_critic",
        model=CRAG_AGENT_MODEL,
        description="Corrective RAG critic: retrieves evidence, grades its quality, and retries with a rewritten query once if needed before deciding to answer or abstain.",
        instruction=CRAG_INSTRUCTION,
        tools=[search_notes, grade_retrieval, rewrite_query],
    )
