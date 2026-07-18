"""
app/agents/crag_runner.py
----------------------------
Executes the CRAG agent (app/agents/crag_agent.py) for one question,
wrapping ADK's Runner + SessionService machinery behind a simple async
function, matching the calling convention of app/services/generation.py.

WHY THIS FILE EXISTS:
  ADK agents are not called directly — they run inside a Runner, which
  needs a SessionService to track conversation state, even for a single,
  one-shot question. This file owns that ceremony so callers (eventually
  ask_service.py, once Layer 3 is wired in) get a plain async function
  call, the same shape as generation.generate().

WHY InMemorySessionService, NOT A PERSISTENT SESSION STORE:
  Each Cognara /ask request is currently independent — there is no
  multi-turn conversation state to persist BETWEEN requests yet (that is
  a future feature, not Layer 3's job). InMemorySessionService creates a
  fresh session scoped to one request and discards it after — the
  correct, minimal choice for "one question in, one graded answer out."
  If Cognara later adds multi-turn conversation memory, revisit this
  with VertexAiSessionService or another persistent backend.

REAL BUGS FOUND AND FIXED — SEE crag_agent.py's DOCSTRING FOR FULL DETAIL:
  Round 1: the agent no longer uses ADK's output_schema (a confirmed ADK
  bug caused a 12x tool-call loop when output_schema and tools were
  combined). This means the agent's final response is plain TEXT that is
  INSTRUCTED to be JSON, not schema-enforced JSON. This file's
  _parse_structured_output() is written defensively for that reality: it
  strips markdown code fences (```json ... ``` — a common habit even
  when an LLM is told to return "only JSON"), and falls back to a safe
  abstain result on any parse failure rather than crashing the caller.

  Round 2: reset_grade_call_count() is called at the START of every run,
  before the agent executes — this gives crag_agent.py's hard, code-level
  "grade at most twice" counter a clean slate for each new question,
  rather than counting calls across the whole process's lifetime (which
  would incorrectly start refusing legitimate grades on a SECOND
  question after the first question already used its budget).

INTERVIEW EXPLANATION:
  "ADK agents always run through a Runner with a SessionService, even
  for a single-shot question. I use InMemorySessionService because each
  Cognara question is currently independent. Since I dropped ADK's
  output_schema to work around a real framework bug, I parse the agent's
  final text response as JSON myself, defensively. I also reset my
  grading-tool call counter at the start of every run — a hard backstop
  against a soft instruction-following slip I saw once, where the agent
  called my grading tool three times instead of two — and that reset has
  to happen per-question, not just once at import time, or a SECOND
  question would incorrectly inherit the first question's used-up
  budget."
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

APP_NAME = "cognara_crag"

_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


async def run_crag(question: str, course_filter: str | None = None) -> dict:
    """
    Run the CRAG agent for one question: retrieve, grade, retry once if
    needed, and return the critic's final structured decision.

    Args:
        question: The user's question.
        course_filter: Currently informational only — passed as context
            in the initial message since ADK tool calls are agent-
            initiated, not directly parameterized by the caller. The
            agent's search_notes tool itself accepts course_filter, and
            a future refinement could thread this through more directly.

    Returns:
        A dict matching RetrievalGrade's shape: relevance_score,
        completeness_score, decision, reason, rewritten_query.
    """
    reset_grade_call_count()  # fresh 2-call budget for THIS question — see module docstring

    agent = build_crag_agent()
    session_service = InMemorySessionService()

    user_id = "cognara_user"
    session_id = str(uuid.uuid4())

    await session_service.create_session(
        app_name=APP_NAME, user_id=user_id, session_id=session_id,
    )

    runner = Runner(agent=agent, app_name=APP_NAME, session_service=session_service)

    message_text = question
    if course_filter:
        message_text += f"\n\n(Restrict search to course: {course_filter})"

    content = types.Content(role="user", parts=[types.Part(text=message_text)])

    logger.info("crag_run_start", question=question, course_filter=course_filter)

    final_response_text = None
    tool_call_count = 0
    async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
        if getattr(event, "get_function_calls", None) and event.get_function_calls():
            tool_call_count += len(event.get_function_calls())
        if event.is_final_response() and event.content and event.content.parts:
            final_response_text = event.content.parts[0].text

    logger.info("crag_run_done", has_response=final_response_text is not None, tool_call_count=tool_call_count)

    if final_response_text is None:
        return {
            "relevance_score": 0.0,
            "completeness_score": 0.0,
            "decision": "abstain",
            "reason": "CRAG agent produced no final response.",
            "rewritten_query": None,
        }

    return _parse_structured_output(final_response_text)


def _parse_structured_output(text: str) -> dict:
    """
    Parse the agent's final response text as JSON matching
    RetrievalGrade's shape. Since output_schema is deliberately NOT used
    (see module docstring), this text is instructed-but-not-enforced
    JSON — strip common LLM habits (markdown code fences) before
    parsing, and fall back to a safe abstain result on any failure
    rather than crashing the caller.
    """
    cleaned = _JSON_FENCE_RE.sub("", text.strip()).strip()

    try:
        parsed = json.loads(cleaned)
    except (json.JSONDecodeError, TypeError) as e:
        logger.info("crag_parse_fallback", error=str(e), raw_text=text[:200])
        return {
            "relevance_score": 0.0,
            "completeness_score": 0.0,
            "decision": "abstain",
            "reason": f"Could not parse CRAG agent output: {text[:200]}",
            "rewritten_query": None,
        }

    # Validate required keys are present, even if types weren't strictly
    # enforced by a schema — a shallow but useful sanity check.
    required_keys = {"relevance_score", "completeness_score", "decision", "reason"}
    if not required_keys.issubset(parsed.keys()):
        logger.info("crag_parse_missing_keys", parsed_keys=list(parsed.keys()))
        return {
            "relevance_score": 0.0,
            "completeness_score": 0.0,
            "decision": "abstain",
            "reason": f"CRAG agent output missing required fields: {parsed}",
            "rewritten_query": None,
        }

    parsed.setdefault("rewritten_query", None)
    return parsed
