"""
tests/integration/test_crag_agent.py
----------------------------------------
Tests for app/agents/crag_agent.py and crag_runner.py — Layer 3's
Corrective RAG, now running on Agno (see ADR 0011, replacing ADK).

WHY THIS IS AN INTEGRATION TEST:
Real CRAG needs real Cloud SQL (search_notes), real Neo4j (search_concept_graph),
real Vertex AI Ranking API (rerank), and a real Gemini call. Guarded by a
connectivity probe, same pattern as every other integration test file.

WHAT CHANGED FROM THE ADK VERSION (full story in crag_agent.py/crag_runner.py):
- No more manual JSON-text parsing tests — output_schema is now genuinely
  enforced by Agno, so response.content is always a valid RetrievalGrade
  or None, never malformed text to defensively parse.
- grade_retrieval's signature dropped rewritten_query (that's now only
  ever set via the separate rewrite_query tool).
- A new real bug (found via crag_runner.py's fix): ToolExecution.result
  comes back as a Python repr STRING, not the original list — tested
  directly below.
"""

import pytest

from app.agents import crag_agent
from app.agents.crag_runner import run_crag


def _crag_reachable() -> bool:
    try:
        return crag_agent.build_crag_agent() is not None
    except Exception:
        return False


CRAG_REACHABLE = _crag_reachable()
requires_crag = pytest.mark.skipif(not CRAG_REACHABLE, reason="CRAG agent could not be constructed — check agno install and GCP config")


class TestGradeRetrievalCallCounter:
    """The hard, code-level cap on grading calls — an instruction alone
    ("call this at most twice") isn't a guarantee, so this is enforced
    in code regardless of what the LLM does."""

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
        assert "limit reached" in third["reason"].lower()

    def test_first_two_calls_pass_through_unmodified(self):
        crag_agent.reset_grade_call_count()
        first = crag_agent.grade_retrieval(0.8, 0.9, "use", "clear evidence")
        assert first == {"decision": "use", "reason": "clear evidence"}


class TestCragRunReal:
    @requires_crag
    @pytest.mark.asyncio
    async def test_clear_question_uses_first_attempt(self):
        """A well-covered question (verified: 'vanishing gradient' scores
        0.77+ in plain vector search alone) should be graded once and used."""
        result = await run_crag("Explain the vanishing gradient problem.")
        grade = result["grade"]
        assert grade["decision"] == "use"
        assert grade["relevance_score"] > 0.5
        assert len(result["evidence_chunks"]) > 0

    @requires_crag
    @pytest.mark.asyncio
    async def test_result_matches_expected_shape(self):
        result = await run_crag("What is gradient descent?")
        grade = result["grade"]
        assert {"relevance_score", "completeness_score", "decision", "reason"}.issubset(grade.keys())
        assert grade["decision"] in ("use", "abstain")  # output_schema forbids "retry" as final
        assert 0.0 <= grade["relevance_score"] <= 1.0

    @requires_crag
    @pytest.mark.asyncio
    async def test_vague_question_terminates_cleanly(self):
        """A vague single-word question must still terminate with a real
        use/abstain decision — no hang, no loop, regardless of whether the
        agent retries first or abstains immediately (both are valid)."""
        result = await run_crag("improvements")
        assert result["grade"]["decision"] in ("use", "abstain")

    @requires_crag
    @pytest.mark.asyncio
    async def test_evidence_result_string_is_correctly_parsed(self):
        """
        Regression test for the real bug found migrating to Agno: a tool's
        .result on RunOutput.tools comes back as a Python repr STRING
        (e.g. "[{'text': '...'}]"), not the original list object. If
        run_crag() ever regresses to trusting isinstance(result, list)
        directly, this test catches it — evidence_chunks would silently
        come back empty on every "use" decision.
        """
        result = await run_crag("Explain the vanishing gradient problem.")
        if result["grade"]["decision"] == "use":
            assert len(result["evidence_chunks"]) > 0
            assert all(isinstance(c, dict) and "text" in c for c in result["evidence_chunks"])


class TestGraphToolWiring:
    """Layer 6: search_concept_graph as CRAG's second retrieval tool."""

    @requires_crag
    @pytest.mark.asyncio
    async def test_structural_question_returns_nonempty_citable_evidence(self):
        result = await run_crag("What concepts relate to vanishing gradients?")
        grade = result["grade"]
        if grade["decision"] == "use":
            assert len(result["evidence_chunks"]) > 0
            for chunk in result["evidence_chunks"]:
                assert chunk["page_number"] is not None
                assert chunk["text"]
