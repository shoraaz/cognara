"""
app/agents/crag_agent.py
------------------------
Layer 3: Corrective RAG (CRAG) — one Agno Agent with 4 tools.

WHY AN AGENT, NOT if/else RETRY LOGIC:
The real judgment calls here — "is this evidence good enough," "how do I
fix a vague query," "retry or give up" — are things an LLM reasoning over
the actual evidence does better than a fixed score threshold. The agent
decides which tool to call and when to stop; that decision-making IS the
"corrective" part of Corrective RAG.

THE 4 TOOLS:
  1. search_notes         — hybrid search + rerank (Layer 2), for content questions
  2. search_concept_graph — Neo4j traversal (Layer 6), for structural questions
  3. grade_retrieval      — the agent grades evidence, this tool just records the score
  4. rewrite_query        — the agent rewrites a weak query, this tool just records it

Tools 3 and 4 are "agent computes, tool records" — the agent forms the
judgment in its own reasoning, then calls the tool to log it. The tools
themselves contain no intelligence; they're a place to attach a hard
safety limit (see MAX_GRADE_CALLS below) and to make the agent's
reasoning visible in RunOutput.tools for debugging.

WHY AGNO (see ADR 0011 — this replaces ADK, see ADR 0004/0006):
ADK's Agent+Runner+SessionService required ~50 lines of manual event-loop
code just to get tool results back out (see crag_runner.py's old version
in git history). Agno's agent.run() returns a RunOutput with .tools
already populated — no manual event iteration. Confirmed real behaviour
against Vertex AI before writing this: response.tools is a plain list of
ToolExecution objects with .tool_name, .tool_args, .result.

output_schema IS safe to combine with tools here (unlike ADK — see
BUG_FIX_LOG.md "CRAG Agent Round 1" for the ADK bug this used to work
around). Agno's own docs describe this combination as first-class:
tools run, THEN the final response is validated against the schema.
That means RetrievalGrade below is now the REAL output_schema, not just
a reference type crag_runner.py parses text against by hand.
"""

from typing import Literal

import sqlalchemy
from agno.agent import Agent
from agno.models.google import Gemini
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.logging import get_logger
from app.retrieval import graph_store
from app.retrieval.embedder import get_embeddings
from app.retrieval.hybrid_search import hybrid_search
from app.retrieval.keyword_search import BM25KeywordIndex
from app.retrieval.reranker import rerank
from app.retrieval.vector_store import CognaraPGVectorStore
from ingestion.pipelines.init_db import get_engine

logger = get_logger(__name__)

CRAG_AGENT_MODEL = "gemini-2.5-flash"

# Hard cap on grade_retrieval calls per run: one grade, one retry-grade,
# never a third. This is a code-level safety net, not just a prompt
# instruction — see BUG_FIX_LOG.md "CRAG Agent Round 2" for why an
# instruction alone ("call this at most twice") was not reliable enough
# with ADK, and we keep the same defensive pattern here on principle.
MAX_GRADE_CALLS = 2


class RetrievalGrade(BaseModel):
    """The critic's final verdict. This IS Agno's output_schema — the
    agent's last message is validated against this automatically."""
    relevance_score: float = Field(ge=0.0, le=1.0, description="How relevant the evidence is to the query.")
    completeness_score: float = Field(ge=0.0, le=1.0, description="How completely the evidence answers the query.")
    decision: Literal["use", "abstain"] = Field(description="Final decision — never 'retry' here, a retry must already have happened.")
    reason: str = Field(description="Short explanation of the decision.")


# ── Shared state (kept simple and explicit — this is a single-process,
#    single-worker demo app, not a distributed system) ──────────────────────
_keyword_index: BM25KeywordIndex | None = None
_grade_call_count = 0


def reset_grade_call_count() -> None:
    """Call once per question (see crag_runner.py) so each question gets
    its own fresh 2-call grading budget."""
    global _grade_call_count
    _grade_call_count = 0


def _get_keyword_index() -> BM25KeywordIndex:
    """BM25 index built once from Cloud SQL, then reused — rebuilding it
    per call would mean a full table scan on every question."""
    global _keyword_index
    if _keyword_index is None:
        _keyword_index = BM25KeywordIndex()
        _keyword_index.refresh()
    return _keyword_index


def _fetch_chunks_by_ids(chunk_ids: list[str]) -> dict[str, dict]:
    """Turn a list of chunk_ids into full chunk rows from Cloud SQL.
    Used by search_concept_graph, which only knows chunk_ids (from Neo4j)
    and needs the real text to hand back as evidence."""
    if not chunk_ids:
        return {}
    engine = get_engine(ip_type="PUBLIC")
    with engine.connect() as conn:
        rows = conn.execute(sqlalchemy.text(
            "SELECT chunk_id, text, course_name, chapter, topic, page_number, page_range "
            "FROM chunks WHERE chunk_id = ANY(:ids);"
        ), {"ids": chunk_ids}).mappings().fetchall()
    return {r["chunk_id"]: dict(r) for r in rows}


# ── Tool 1: content search (Layers 1-2) ──────────────────────────────────────

def search_notes(query: str, course_filter: str | None = None, chapter_filter: str | None = None) -> list[dict]:
    """Search course notes for evidence. Use for CONTENT questions
    ("explain X", "what is Y"). For STRUCTURAL questions about how
    concepts relate ("what should I learn before X"), use
    search_concept_graph instead.

    Args:
        query: The question or topic to search for.
        course_filter: Optional — restrict to one course.
        chapter_filter: Optional — restrict to one chapter.

    Returns:
        Up to 5 evidence chunks: text, course_name, chapter, topic,
        page_number, page_range, relevance_score.
    """
    store = CognaraPGVectorStore(embeddings=get_embeddings())
    index = _get_keyword_index()

    # Hybrid search (vector + BM25, fused with RRF) then rerank — Layer 2's
    # full pipeline, unchanged by this migration. See hybrid_search.py.
    hybrid_results = hybrid_search(query, store, index, k_per_method=20, top_k=10,
                                    course_filter=course_filter, chapter_filter=chapter_filter)
    reranked = rerank(query, hybrid_results, top_n=5)

    logger.info("crag_search_notes", query=query, results=len(reranked))
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
        for doc in reranked
    ]


# ── Tool 2: structural search (Layer 6) ──────────────────────────────────────

def search_concept_graph(
    query: str, relation: Literal["prerequisites", "related"] = "related", max_depth: int = 2,
) -> list[dict]:
    """Search the concept graph for relationships BETWEEN topics. Use for
    "what should I learn before X" (relation="prerequisites") or "what
    relates to X" (relation="related"). Returns the SAME shape as
    search_notes (real chunk text fetched from Cloud SQL for every graph
    node) so grading/generation never need to know which tool ran.

    Args:
        query: The concept to look up — natural phrasing is fine.
        relation: "prerequisites" or "related".
        max_depth: For "prerequisites" only — how many hops (1-10).

    Returns:
        Evidence chunks, same shape as search_notes. Empty list if the
        concept isn't in the graph — fall back to search_notes then.
    """
    driver = graph_store.get_driver()
    try:
        matches = graph_store.find_concept_by_name(driver, query, limit=1)
        if not matches:
            return []
        resolved = matches[0]["name"]

        if relation == "prerequisites":
            results = graph_store.get_prerequisites(driver, resolved, max_depth=max(1, min(max_depth, 10)))
        else:
            results = graph_store.get_related_concepts(driver, resolved)

        logger.info("crag_search_concept_graph", query=query, resolved=resolved, results=len(results))

        # Graph nodes only carry chunk_ids — fetch the real text once for the whole batch.
        chunk_ids = list(dict.fromkeys(cid for r in results for cid in (r.get("chunk_ids") or [])))
        chunks = _fetch_chunks_by_ids(chunk_ids)

        return [
            {
                "text": chunks[cid]["text"],
                "course_name": chunks[cid]["course_name"],
                "chapter": chunks[cid]["chapter"],
                "topic": chunks[cid]["topic"],
                "page_number": chunks[cid]["page_number"],
                "page_range": chunks[cid]["page_range"],
                "relevance_score": 1.0,  # graph-confirmed relationship, not a similarity score
            }
            for r in results for cid in (r.get("chunk_ids") or []) if cid in chunks
        ]
    finally:
        driver.close()


# ── Tool 3: grading (agent computes, tool records + enforces the cap) ───────

def grade_retrieval(relevance_score: float, completeness_score: float, decision: str, reason: str) -> dict:
    """Record YOUR grading judgment for the current evidence. You compute
    the scores by reading the evidence yourself — this tool only logs
    what you decided and enforces a hard limit of 2 calls per question.

    Args:
        relevance_score: Your assessment, 0-1.
        completeness_score: Your assessment, 0-1.
        decision: "use", "retry" (first call only), or "abstain".
        reason: Your short explanation.
    """
    global _grade_call_count
    _grade_call_count += 1
    logger.info("crag_grade_retrieval", call=_grade_call_count, decision=decision, reason=reason)

    if _grade_call_count > MAX_GRADE_CALLS:
        # Instruction-following isn't a hard guarantee (see BUG_FIX_LOG.md
        # "CRAG Agent Round 2") — this branch is the actual enforcement.
        return {"decision": "use", "reason": "Grading limit reached — using best available evidence."}
    return {"decision": decision, "reason": reason}


# ── Tool 4: query rewriting (agent computes, tool records) ─────────────────

def rewrite_query(rewritten_query: str, reason: str) -> dict:
    """Record YOUR improved query after a weak first search. You compose
    the actual new text; this tool only logs it. Call at most once, then
    search again with this exact text.

    Args:
        rewritten_query: Your improved query.
        reason: What was wrong with the original.
    """
    logger.info("crag_rewrite_query", rewritten_query=rewritten_query)
    return {"rewritten_query": rewritten_query}


CRAG_INSTRUCTION = """You are Cognara Learn's retrieval critic (Corrective RAG).

Tools: search_notes (content questions) or search_concept_graph (structural questions
about how concepts relate — "before X", "related to X"). Pick whichever fits.

1. Search for evidence with the right tool.
2. Read it. Form your own relevance_score, completeness_score, decision, reason.
3. Call grade_retrieval with YOUR judgment (max twice per question).
4. If decision was "retry": call rewrite_query once, search again with that
   text, read the new evidence, grade_retrieval one final time.
5. Your final decision must be "use" or "abstain" — never "retry".

Be strict. Abstaining honestly beats forcing a weak answer."""


def build_crag_agent() -> Agent:
    """Construct a fresh CRAG agent. Not cached — cheap to build, and
    keeps each question's run fully isolated."""
    return Agent(
        model=Gemini(
            id=CRAG_AGENT_MODEL, vertexai=True,
            project_id=settings.GCP_PROJECT_ID, location=settings.VERTEX_AI_LOCATION,
        ),
        instructions=CRAG_INSTRUCTION,
        tools=[search_notes, search_concept_graph, grade_retrieval, rewrite_query],
        output_schema=RetrievalGrade,
    )
