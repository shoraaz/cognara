"""
app/agents/crag_runner.py
--------------------------
Runs the CRAG agent (crag_agent.py) for one question and returns its
grade plus the real evidence chunks that grade was based on.

WHY THIS FILE IS SO MUCH SHORTER THAN THE ADK VERSION:
The old ADK version (~140 lines) manually iterated
`async for event in runner.run_async(...)`, checked
event.get_function_calls()/get_function_responses(), and tracked the
latest tool result by hand — because ADK's Runner streams raw events,
not a finished result object.

Agno's agent.run() just RETURNS a finished RunOutput with everything
already collected: .content (validated against RetrievalGrade, since
output_schema is set) and .tools (a list of every tool call made, each
with .tool_name and .result).

REAL BUG FOUND AND FIXED — .result IS A STRING, NOT THE ORIGINAL LIST:
The first version checked `isinstance(call.result, list)` before trusting
a tool call's result — that check was silently always False. Agno stores
each ToolExecution.result as the Python repr STRING of whatever the tool
returned (confirmed by direct inspection: type(call.result) is str, and
the string looks like "[{'text': '...', ...}]" — a repr, not JSON, since
it uses single quotes). Our tools return real Python dicts/lists, but by
the time we read them back from RunOutput.tools, they've been through a
str()-and-back round trip that our first version never accounted for.
Fixed with ast.literal_eval() (safe for Python literals, unlike eval()) —
NOT json.loads(), which would fail on the single-quoted repr format.
"""

import ast

from app.agents.crag_agent import build_crag_agent, reset_grade_call_count
from app.core.logging import get_logger

logger = get_logger(__name__)

RETRIEVAL_TOOLS = {"search_notes", "search_concept_graph"}


async def run_crag(question: str, course_filter: str | None = None) -> dict:
    """
    Run the CRAG agent for one question.

    Returns:
        {"grade": {relevance_score, completeness_score, decision, reason},
         "evidence_chunks": [...]}  — the chunks from the LAST retrieval
        tool call, which is what the final grade was actually based on
        (on a retry path, that's the SECOND search, not the first).
    """
    reset_grade_call_count()  # fresh 2-call budget for this question

    agent = build_crag_agent()
    message = question
    if course_filter:
        message += f"\n\n(Restrict search to course: {course_filter})"

    response = await agent.arun(message)

    # Find the most recent retrieval-tool call — this is what the agent's
    # final grade was actually based on. See module docstring: .result
    # comes back as a Python-repr STRING, not the original list, so it
    # needs ast.literal_eval() to turn back into real data.
    evidence_chunks: list[dict] = []
    for call in response.tools or []:
        if call.tool_name in RETRIEVAL_TOOLS:
            try:
                parsed = ast.literal_eval(call.result) if isinstance(call.result, str) else call.result
            except (ValueError, SyntaxError):
                continue  # a malformed/empty result — skip rather than crash the whole run
            if isinstance(parsed, list):
                evidence_chunks = parsed  # overwrite each time -> ends up as the LAST call

    # response.content is a validated RetrievalGrade instance (output_schema
    # guarantees this) — or None if the model somehow failed to produce one.
    if response.content is None:
        logger.info("crag_no_content", question=question)
        grade = {"relevance_score": 0.0, "completeness_score": 0.0,
                  "decision": "abstain", "reason": "CRAG agent produced no output."}
    else:
        grade = response.content.model_dump()

    logger.info("crag_run_done", decision=grade["decision"], evidence_count=len(evidence_chunks))
    return {"grade": grade, "evidence_chunks": evidence_chunks}
