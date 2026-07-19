"""
app/agents/crag_runner.py
-------------------------
Executes the CRAG agent (app/agents/crag_agent.py) for one question,
wrapping ADK's Runner + SessionService machinery behind a simple async
function that matches the calling convention of app/services/generation.py.

WHY THIS FILE EXISTS:
  ADK agents are not called directly — they run inside a Runner, which
  needs a SessionService to track conversation state, even for a single,
  one-shot question. This file owns that ceremony so callers (ask_service.py)
  get a plain async function call, the same shape as generation.generate().

WHY InMemorySessionService, NOT A PERSISTENT SESSION STORE:
  Each /ask request is currently independent — there is no multi-turn
  conversation state to persist between requests yet (that is a future
  feature). InMemorySessionService creates a fresh session scoped to one
  request and discards it after — the correct, minimal choice for "one
  question in, one graded answer out."

KEY DESIGN DECISIONS (full bug stories in local-notes/BUG_FIX_LOG.md):
  - output_schema was removed from the agent to fix a 12x tool-call loop
    (ADK bug, Round 1). Agent final response is plain text; _parse_structured_output()
    parses it defensively, stripping markdown fences and falling back to a
    safe abstain result on any parse failure.
  - reset_grade_call_count() is called at the start of every run so each
    question gets a fresh 2-call grading budget (Round 2 fix).
  - Evidence chunks are extracted directly from ADK's event stream via
    event.get_function_responses(), tracking the MOST RECENT retrieval-tool
    result (Round 5 fix). On a retry path, the SECOND search's results are
    what the final grade was based on — a separate re-fetch could silently
    return different results.

REAL BUG FOUND AND FIXED (Layer 6 wiring) — EVIDENCE EXTRACTION ONLY
WATCHED search_notes, NOT search_concept_graph:
  Adding search_concept_graph as a second retrieval tool (crag_agent.py)
  surfaced a real gap immediately: a real structural question ("What
  concepts relate to vanishing gradients?") correctly made the agent
  choose search_concept_graph, correctly resolved the concept, correctly
  graded the result (relevance_score=0.9, decision=use) — but
  evidence_chunk_count came back as 0. The evidence-extraction loop below
  only ever checked `resp.name == "search_notes"`, so a real, successful
  search_concept_graph call's results were silently ignored — grading
  happened on real evidence the agent could see, but run_crag()'s caller
  (ask_service.py) would have received zero citations for a "use"
  decision, an inconsistent, broken state.
  FIX: the evidence-extraction check now accepts EITHER tool name
  (search_notes OR search_concept_graph) — both already return the
  identical dict shape (see crag_agent.py's search_concept_graph
  docstring for why that shape-matching was deliberate), so no other
  code needed to change once this check was widened.

# Interview notes: local-notes/INTERVIEW_PREP.md — "app/agents/crag_runner.py"
"""

import json
import re
import uuid

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from app.agents.crag_agent import build_crag_agent, reset_grade_call_count
from app.core.logging import get_logger

logger = get_logger(__name__)

# Identifier for this application within ADK's session management.
APP_NAME = "cognara_crag"

# Regex to strip markdown code fences the model sometimes wraps JSON in,
# e.g. ```json\n{...}\n``` or ```\n{...}\n```.
_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)

# Both retrieval tools return the identical evidence-chunk dict shape (see
# crag_agent.py's search_concept_graph docstring) — either one's result is
# valid "current evidence" for the evidence-extraction loop below.
_RETRIEVAL_TOOL_NAMES = {"search_notes", "search_concept_graph"}


async def run_crag(question: str, course_filter: str | None = None) -> dict:
    """
    Run the CRAG agent for one question: retrieve, grade, retry once if
    needed, and return the critic's final structured decision TOGETHER
    with the real evidence chunks that decision was based on.

    Args:
        question: The user's question.
        course_filter: Passed as context in the initial message text since
            ADK tool calls are agent-initiated, not directly parameterised
            by the caller. The agent's search_notes tool accepts it.

    Returns:
        A dict with:
          - grade: RetrievalGrade shape (relevance_score, completeness_score,
            decision, reason, rewritten_query)
          - evidence_chunks: list of chunk dicts from the LAST retrieval-tool
            call the agent made (search_notes OR search_concept_graph — see
            module docstring's REAL BUG FOUND AND FIXED) — the exact evidence
            the final grade was based on. See BUG_FIX_LOG.md "CRAG Runner Round 5".
    """
    # Give this question a clean 2-call grading budget — see crag_agent.py's
    # _grade_call_count and BUG_FIX_LOG.md "CRAG Agent Round 2".
    reset_grade_call_count()

    # Fresh agent instance per call — same async gRPC event-loop reasoning as
    # generation.py's _get_llm(). See BUG_FIX_LOG.md "Generation: async gRPC".
    agent = build_crag_agent()
    session_service = InMemorySessionService()

    # Each request gets its own isolated session — no state bleeds between calls.
    user_id = "cognara_user"
    session_id = str(uuid.uuid4())

    await session_service.create_session(
        app_name=APP_NAME, user_id=user_id, session_id=session_id,
    )

    runner = Runner(agent=agent, app_name=APP_NAME, session_service=session_service)

    # Build the initial message; append course_filter hint as plain text so the
    # agent can thread it into search_notes(course_filter=...) calls.
    message_text = question
    if course_filter:
        message_text += f"\n\n(Restrict search to course: {course_filter})"

    content = types.Content(role="user", parts=[types.Part(text=message_text)])

    logger.info("crag_run_start", question=question, course_filter=course_filter)

    final_response_text = None
    tool_call_count = 0
    # Tracks the most recent retrieval-tool result — updated on every
    # search_notes/search_concept_graph response so on a retry path we end
    # up with the SECOND (final) search's chunks, whichever tool made it.
    latest_evidence_chunks: list[dict] = []

    async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
        # Count every tool call for observability (logged at run end).
        function_calls = event.get_function_calls() if getattr(event, "get_function_calls", None) else None
        if function_calls:
            tool_call_count += len(function_calls)

        # Extract evidence chunks from EITHER retrieval tool's response — see
        # module docstring's REAL BUG FOUND AND FIXED for why both must be
        # checked, not just search_notes. ADK wraps non-dict tool returns
        # under a "result" key — both tools return a list, so handle both
        # shapes defensively.
        function_responses = event.get_function_responses() if getattr(event, "get_function_responses", None) else None
        if function_responses:
            for resp in function_responses:
                if resp.name in _RETRIEVAL_TOOL_NAMES and isinstance(resp.response, dict):
                    raw = resp.response.get("result", resp.response)
                    if isinstance(raw, list):
                        # Overwrite on every retrieval-tool call so we always
                        # hold the LATEST (most recently graded) result set.
                        latest_evidence_chunks = raw

        # Capture the agent's final text output (the instructed JSON blob).
        if event.is_final_response() and event.content and event.content.parts:
            final_response_text = event.content.parts[0].text

    logger.info(
        "crag_run_done",
        has_response=final_response_text is not None,
        tool_call_count=tool_call_count,
        evidence_chunk_count=len(latest_evidence_chunks),
    )

    # Guard: if the agent produced no final response at all, return a safe abstain.
    if final_response_text is None:
        return {
            "grade": {
                "relevance_score":    0.0,
                "completeness_score": 0.0,
                "decision":           "abstain",
                "reason":             "CRAG agent produced no final response.",
                "rewritten_query":    None,
            },
            "evidence_chunks": latest_evidence_chunks,
        }

    grade = _parse_structured_output(final_response_text)
    return {"grade": grade, "evidence_chunks": latest_evidence_chunks}


def _parse_structured_output(text: str) -> dict:
    """
    Parse the agent's final response text as JSON matching RetrievalGrade's
    shape. Since output_schema is deliberately NOT used (see module docstring),
    this text is instructed-but-not-enforced JSON. We:
      1. Strip markdown code fences (common LLM habit).
      2. Parse with json.loads().
      3. Validate required keys are present.
      4. Fall back to a safe abstain result on any failure, rather than
         crashing the caller with an unhandled exception.
    """
    # Strip markdown code fences like ```json ... ``` or ``` ... ```
    cleaned = _JSON_FENCE_RE.sub("", text.strip()).strip()

    try:
        parsed = json.loads(cleaned)
    except (json.JSONDecodeError, TypeError) as e:
        logger.info("crag_parse_fallback", error=str(e), raw_text=text[:200])
        return {
            "relevance_score":    0.0,
            "completeness_score": 0.0,
            "decision":           "abstain",
            "reason":             f"Could not parse CRAG agent output: {text[:200]}",
            "rewritten_query":    None,
        }

    # Shallow key-presence check — catches missing required fields without a
    # full Pydantic validation pass (which would raise, not fall back).
    required_keys = {"relevance_score", "completeness_score", "decision", "reason"}
    if not required_keys.issubset(parsed.keys()):
        logger.info("crag_parse_missing_keys", parsed_keys=list(parsed.keys()))
        return {
            "relevance_score":    0.0,
            "completeness_score": 0.0,
            "decision":           "abstain",
            "reason":             f"CRAG agent output missing required fields: {parsed}",
            "rewritten_query":    None,
        }

    # Ensure rewritten_query is always present in the returned dict (even if the
    # model omitted it for non-retry decisions) so callers don't need a .get().
    parsed.setdefault("rewritten_query", None)
    return parsed
