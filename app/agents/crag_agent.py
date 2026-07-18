"""
app/agents/crag_agent.py
------------------------
Layer 3: Corrective RAG (CRAG), implemented as an ADK Agent with three tools.
Per ADR 0004's decision that orchestration (Layer 3 onward) uses ADK.

WHY AN AGENT INSTEAD OF HAND-CODED if/else RETRY LOGIC:
  The CRAG flow is:
      question -> retrieve evidence -> critic grades retrieval quality
        -> if weak, rewrite/expand query and retry once
        -> if still weak, abstain
        -> if strong, send evidence to answer generator
  This COULD be written as a fixed Python function with if-statements. We use
  an ADK Agent because the actual judgment calls — "is this retrieval good
  enough," "how should I rewrite this vague query," "should I retry or abstain"
  — are genuinely better made by an LLM reasoning about the specific evidence
  in front of it than by a fixed numeric threshold alone.

THE THREE TOOLS:
  1. search_notes(query, course_filter, chapter_filter)
     Wraps Layer 2's full pipeline (hybrid_search -> rerank) unchanged.
     CRAG adds judgment on top of retrieval, it does not reimplement it.

  2. grade_retrieval(relevance_score, completeness_score, decision, reason,
     rewritten_query) — "agent computes, tool records" pattern. The agent
     reads the evidence, forms its own judgment, then calls this to record it.
     See BUG_FIX_LOG.md "CRAG Agent Round 3" for why the original signature
     (which had the tool grade) was wrong.

  3. rewrite_query(rewritten_query, reason) — same "agent computes, tool
     records" pattern. The agent composes the improved query text, then calls
     this to record it. See BUG_FIX_LOG.md "CRAG Agent Round 4".

MODEL CHOICE:
  gemini-2.5-flash — same stable model verified in generation.py. The critic's
  job is lighter-weight than final answer generation, so no reason to use a
  larger/slower model here.

KEY DESIGN DECISIONS (full bug stories in local-notes/BUG_FIX_LOG.md):
  - output_schema is NOT set (Round 1 bug: combined with tools it caused a
    12x tool-call loop — a confirmed ADK framework issue).
  - _grade_call_count provides a hard code-level cap at 2 grading calls per
    run (Round 2 bug: instruction alone wasn't reliable enough).

# Interview notes: local-notes/INTERVIEW_PREP.md — "app/agents/crag_agent.py"
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

# Model used for the CRAG critic — same as generation, lighter workload.
CRAG_AGENT_MODEL = "gemini-2.5-flash"


# ── Structured critic output schema ──────────────────────────────────────────
# NOT used as ADK's output_schema (see BUG_FIX_LOG.md "CRAG Agent Round 1" —
# that combination triggers an infinite tool-call loop). Kept as a reference
# schema; crag_runner.py validates the agent's parsed JSON against it manually.

class RetrievalGrade(BaseModel):
    """The critic's structured judgment of one retrieval attempt."""
    relevance_score: float = Field(
        ..., ge=0.0, le=1.0,
        description="How relevant the evidence is to the query, 0-1.",
    )
    completeness_score: float = Field(
        ..., ge=0.0, le=1.0,
        description="How completely the evidence covers what the question needs, 0-1.",
    )
    decision: Literal["use", "retry", "abstain"] = Field(
        ...,
        description="use: evidence is good enough to answer. retry: worth rewriting the query and searching once more. abstain: even a retry is unlikely to help.",
    )
    reason: str = Field(..., description="A short, human-readable explanation of the decision.")
    rewritten_query: str | None = Field(
        default=None,
        description="Only set when decision is 'retry' — a rewritten/expanded version of the query to search again.",
    )


# ── Module-level singletons ───────────────────────────────────────────────────

# BM25 index is safe to cache as a module-level singleton — it's a plain
# in-memory data structure with no async gRPC event-loop binding risk
# (contrast: ChatVertexAI in generation.py, which must NOT be cached).
_keyword_index: BM25KeywordIndex | None = None

# Hard, code-level backstop for "grade at most twice per question".
# See BUG_FIX_LOG.md "CRAG Agent Round 2" — LLM instruction-following alone
# was insufficient to reliably cap this to two calls.
_grade_call_count = 0


def reset_grade_call_count() -> None:
    """
    Reset the grading call counter to zero. Must be called at the start of
    every CRAG run (see crag_runner.py) so each question gets its own fresh
    2-call budget, not a count shared across the whole process lifetime.
    """
    global _grade_call_count
    _grade_call_count = 0


def _get_keyword_index() -> BM25KeywordIndex:
    """
    Return the shared BM25 index, building it on first call. The index is an
    in-memory data structure rebuilt from Cloud SQL on startup; caching it
    avoids an expensive full-table read on every search call.
    """
    global _keyword_index
    if _keyword_index is None:
        _keyword_index = BM25KeywordIndex()
        _keyword_index.refresh()
    return _keyword_index


# ── Tool implementations ──────────────────────────────────────────────────────
# Plain functions — ADK reads their docstrings and type hints automatically to
# build the tool schema presented to the model.

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
    # Construct a fresh vector store per call (embedder itself is a singleton;
    # the store wraps it with a SQLAlchemy engine that is also cached).
    store = CognaraPGVectorStore(embeddings=get_embeddings())
    keyword_index = _get_keyword_index()

    # k_per_method=20 gives each method a wide candidate pool before fusion;
    # too small a pool makes RRF degenerate to "whichever method's top result wins".
    hybrid_results = hybrid_search(
        query, store, keyword_index,
        k_per_method=20, top_k=10,
        course_filter=course_filter, chapter_filter=chapter_filter,
    )
    # Reranker narrows the 10 fused candidates to the 5 most precisely relevant.
    reranked_docs = rerank(query, hybrid_results, top_n=5)

    logger.info("crag_search_notes", query=query, results=len(reranked_docs))

    # Flatten LangChain Documents into plain dicts — ADK tools must return
    # JSON-serialisable types so ADK can pass results back to the model.
    return [
        {
            "text":            doc.page_content,
            "course_name":     doc.metadata.get("course_name"),
            "chapter":         doc.metadata.get("chapter"),
            "topic":           doc.metadata.get("topic"),
            "page_number":     doc.metadata.get("page_number"),
            "page_range":      doc.metadata.get("page_range"),
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
        explicit note that the grading limit was reached.
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

    # Hard cap: if somehow called a third time (instruction-following failure),
    # force "use" so the pipeline doesn't loop further. The agent's instruction
    # says "at most twice", but the code enforces it unconditionally.
    # See BUG_FIX_LOG.md "CRAG Agent Round 2".
    if _grade_call_count > 2:
        logger.info("crag_grade_limit_reached", call_number=_grade_call_count)
        return {
            "relevance_score":    relevance_score,
            "completeness_score": completeness_score,
            "decision":           "use",
            "reason":             "Grading limit reached (2 calls) — using best available evidence rather than retrying further.",
            "rewritten_query":    None,
        }

    return {
        "relevance_score":    relevance_score,
        "completeness_score": completeness_score,
        "decision":           decision,
        "reason":             reason,
        "rewritten_query":    rewritten_query,
    }


def rewrite_query(rewritten_query: str, reason: str) -> dict:
    """Record your rewritten query, to search again after weak retrieval.

    Call this only when your grade_retrieval decision was "retry", and
    only ONCE per question. YOU compose the improved query text yourself
    — expand vague terms, add likely synonyms or related ML/DL
    terminology, or clarify ambiguous phrasing, based on what was wrong
    with the original retrieval. This tool does not rewrite anything
    itself; it only records the rewrite you already composed, for
    logging and traceability.

    Args:
        rewritten_query: YOUR improved version of the original query —
            the actual new text to search with next.
        reason: A short explanation of what was wrong with the original
            query and why this rewrite should help.

    Returns:
        A dict confirming the rewrite that was recorded. Call
        search_notes() next with this exact rewritten_query text — this
        is the LAST retry, do not call rewrite_query() a second time.
    """
    # "Agent computes, tool records" pattern — the agent composed rewritten_query
    # in its reasoning step; we log it here for traceability.
    # See BUG_FIX_LOG.md "CRAG Agent Round 4" for why the original signature
    # (which returned the ORIGINAL query unchanged) was wrong.
    logger.info("crag_rewrite_query_called", rewritten_query=rewritten_query, reason=reason)
    return {"rewritten_query": rewritten_query, "reason": reason}


# ── Agent instruction ─────────────────────────────────────────────────────────
# The full prompt given to the CRAG critic. Explicit about the tool call order,
# the "at most twice" grading budget, and the required final JSON shape.

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
4. If your decision was "retry": compose your OWN improved query text
   (expand vague terms, add ML/DL synonyms, clarify ambiguity), then
   call rewrite_query(rewritten_query, reason) ONCE to record it. Then
   call search_notes() again with THAT SAME rewritten_query text. Then
   read the new evidence and call grade_retrieval() ONE more time (your
   second and FINAL call) with your updated judgment.
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
    """
    Construct the CRAG agent. NOT a module-level singleton — same reasoning as
    generation.py's ChatVertexAI: ADK's Agent wraps a similar async model
    client under the hood with an event-loop-bound gRPC channel.

    output_schema is deliberately NOT set — see BUG_FIX_LOG.md "CRAG Agent
    Round 1". The agent's final text response is instructed to already be the
    correctly-shaped JSON; crag_runner.py parses it against RetrievalGrade.
    """
    return Agent(
        name="crag_critic",
        model=CRAG_AGENT_MODEL,
        description=(
            "Corrective RAG critic: retrieves evidence, grades its quality, "
            "and retries with a rewritten query once if needed before deciding "
            "to answer or abstain."
        ),
        instruction=CRAG_INSTRUCTION,
        tools=[search_notes, grade_retrieval, rewrite_query],
    )
