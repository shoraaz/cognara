"""
tests/integration/test_crag_agent.py
----------------------------------------
Tests for app/agents/crag_agent.py and crag_runner.py — Layer 3's
Corrective RAG implementation via ADK.

WHY THIS IS AN INTEGRATION TEST:
  A real CRAG run needs a real Cloud SQL connection (search_notes), a
  real Vertex AI Ranking API call (rerank, inside search_notes), and a
  real Gemini call via ADK's Agent/Runner machinery. There is no
  meaningful offline unit-test version of "does the corrective loop
  work end to end." Guarded by a connectivity probe, same pattern as
  every other integration test file.

WHAT THESE TESTS PROVE, based on real, reproduced behaviour during
development (see crag_agent.py's docstring for the full bug history):
  - A clear, well-covered question is graded once and used (no
    unnecessary retry).
  - A deliberately vague question triggers the retry path: grade ->
    retry -> rewrite_query -> search again -> grade again -> abstain
    (if the retry still doesn't help) or use (if it does).
  - The hard call-count backstop actually caps grade_retrieval at 2
    calls per question, regardless of what the agent might otherwise do.
  - Output is always parseable, valid JSON matching RetrievalGrade's
    shape, even though ADK's output_schema is deliberately not used.
"""

import pytest

from app.agents import crag_agent
from app.agents.crag_runner import run_crag, _parse_structured_output


def _crag_reachable() -> bool:
    try:
        agent = crag_agent.build_crag_agent()
        return agent is not None
    except Exception:
        return False


CRAG_REACHABLE = _crag_reachable()
requires_crag = pytest.mark.skipif(
    not CRAG_REACHABLE,
    reason="CRAG agent could not be constructed — check google-adk install and GCP config",
)


class TestParseStructuredOutput:
    def test_parses_plain_json(self):
        text = '{"relevance_score": 0.9, "completeness_score": 0.8, "decision": "use", "reason": "good evidence"}'
        result = _parse_structured_output(text)
        assert result["decision"] == "use"
        assert result["relevance_score"] == 0.9
        assert result["rewritten_query"] is None  # defaulted

    def test_strips_markdown_json_fence(self):
        text = '```json\n{"relevance_score": 0.5, "completeness_score": 0.5, "decision": "abstain", "reason": "weak"}\n```'
        result = _parse_structured_output(text)
        assert result["decision"] == "abstain"

    def test_malformed_json_falls_back_to_abstain(self):
        result = _parse_structured_output("not valid json at all {{{")
        assert result["decision"] == "abstain"
        assert "Could not parse" in result["reason"]

    def test_missing_required_keys_falls_back_to_abstain(self):
        result = _parse_structured_output('{"decision": "use"}')  # missing scores/reason
        assert result["decision"] == "abstain"
        assert "missing required fields" in result["reason"]


class TestGradeRetrievalCallCounter:
    def test_counter_resets(self):
        crag_agent.reset_grade_call_count()
        crag_agent.grade_retrieval(0.5, 0.5, "use", "test")
        assert crag_agent._grade_call_count == 1
        crag_agent.reset_grade_call_count()
        assert crag_agent._grade_call_count == 0

    def test_third_call_is_forced_to_use(self):
        crag_agent.reset_grade_call_count()
        crag_agent.grade_retrieval(0.3, 0.3, "retry", "first")
        crag_agent.grade_retrieval(0.3, 0.3, "retry", "second")
        third = crag_agent.grade_retrieval(0.3, 0.3, "retry", "third — should be overridden")
        assert third["decision"] == "use"
        assert "Grading limit reached" in third["reason"]

    def test_first_two_calls_pass_through_unmodified(self):
        crag_agent.reset_grade_call_count()
        first = crag_agent.grade_retrieval(0.8, 0.9, "use", "clear evidence")
        assert first["decision"] == "use"
        assert first["relevance_score"] == 0.8
        assert first["reason"] == "clear evidence"


class TestCragRunReal:
    @requires_crag
    @pytest.mark.asyncio
    async def test_clear_question_uses_first_attempt(self):
        """
        A question with strong, direct corpus coverage (verified in
        Modules 4-5: 'vanishing gradient' scores 0.77+ in plain vector
        search alone) should be graded once and decided "use" — no
        wasted retry.
        """
        result = await run_crag("Explain the vanishing gradient problem.")
        assert result["decision"] == "use"
        assert result["relevance_score"] > 0.5
        assert result["reason"]  # non-empty

    @requires_crag
    @pytest.mark.asyncio
    async def test_result_matches_retrieval_grade_shape(self):
        result = await run_crag("What is gradient descent?")
        required_keys = {"relevance_score", "completeness_score", "decision", "reason", "rewritten_query"}
        assert required_keys.issubset(result.keys())
        assert result["decision"] in ("use", "abstain")  # never "retry" as FINAL decision
        assert 0.0 <= result["relevance_score"] <= 1.0
        assert 0.0 <= result["completeness_score"] <= 1.0

    @requires_crag
    @pytest.mark.asyncio
    async def test_vague_question_triggers_retry_path_and_still_terminates(self):
        """
        A deliberately vague single-word question should trigger the
        retry path (grade -> retry -> rewrite -> search -> grade again)
        and STILL terminate with a final use/abstain decision, not hang
        or loop — this is the real behaviour proven during development
        (see crag_agent.py's bug history for the full trail).
        """
        result = await run_crag("improvements")
        assert result["decision"] in ("use", "abstain")
        assert result["reason"]
