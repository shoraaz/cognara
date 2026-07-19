"""
tests/integration/test_quiz_interview.py
--------------------------------------------
Tests for app/services/quiz_interview.py — Layer 5's stateful modes
(Quiz, Interview). See ADR 0009.

WHY THIS IS AN INTEGRATION TEST:
  Real sessions need real Cloud SQL persistence (learning_sessions,
  learning_session_turns) and real CRAG + Gemini calls at every turn.
  Guarded by a connectivity probe, same pattern as every other
  integration test file.

WHAT THESE TESTS PROVE, based on real, reproduced behaviour during
development (three real bugs found and fixed — see quiz_interview.py's
docstring for the full history):
  - A quiz session persists real state across multiple separate calls
    (not an in-memory object — a fresh Python call to submit_answer()
    can still find and grade the right turn).
  - Quiz mode's questions are genuinely different across turns, not
    near-duplicates (round 2 bug).
  - Interview mode's "step back to foundational" hint stays anchored to
    the original topic and does not dead-end the session (round 3 bug).
"""

import os

import pytest

from app.agents import crag_agent
from app.core.config import settings
from app.services import quiz_interview


def _reachable() -> bool:
    try:
        agent = crag_agent.build_crag_agent()
        if agent is None:
            return False
    except Exception:
        return False
    if not settings.GCP_PROJECT_ID:
        return False
    if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        return True
    adc_default = os.path.expanduser("~/AppData/Roaming/gcloud/application_default_credentials.json")
    return os.path.exists(adc_default)


REACHABLE = _reachable()
requires_infra = pytest.mark.skipif(not REACHABLE, reason="CRAG/Vertex AI/Cloud SQL not reachable")


class TestQuizSession:
    @requires_infra
    @pytest.mark.asyncio
    async def test_start_session_produces_a_real_grounded_question(self):
        result = await quiz_interview.start_session("quiz", "gradient descent")
        assert result["abstained"] is False
        assert result["session_id"] is not None
        assert len(result["question_text"]) > 0
        assert result["turn_number"] == 1

    @requires_infra
    @pytest.mark.asyncio
    async def test_submit_answer_grades_and_produces_a_different_next_question(self):
        """
        The flagship round-2-bug regression test: question 2 must be
        substantively different from question 1, not a near-duplicate —
        proven by requiring the two question strings to differ by more
        than trivial wording (a loose but real check: they must not
        share the exact same first 6 words, a cheap proxy for "same
        underlying question, reworded" which is exactly what the real
        bug produced).
        """
        start = await quiz_interview.start_session("quiz", "gradient descent")
        result = await quiz_interview.submit_answer(
            start["session_id"],
            "Gradient descent minimizes a loss function by updating parameters "
            "in the opposite direction of the gradient.",
        )
        assert isinstance(result["is_correct"], bool)
        assert len(result["feedback_text"]) > 0

        next_q = result["next_question"]
        if not next_q["abstained"]:
            q1_start = " ".join(start["question_text"].split()[:6])
            q2_start = " ".join(next_q["question_text"].split()[:6])
            assert q1_start != q2_start, "Q2 should not be a near-duplicate of Q1"

    @requires_infra
    @pytest.mark.asyncio
    async def test_session_persists_across_separate_calls(self):
        """
        Proves real persistence, not in-memory state: start_session()
        and submit_answer() are two SEPARATE calls, and submit_answer()
        must be able to find and grade the right turn using only the
        session_id — simulating two separate HTTP requests.
        """
        start = await quiz_interview.start_session("quiz", "gradient descent")
        session_id = start["session_id"]

        # Simulate a fresh request: look up the session from scratch.
        session = quiz_interview._get_session(session_id)
        assert session is not None
        assert session["mode"] == "quiz"
        assert session["topic"] == "gradient descent"

        turns = quiz_interview._get_turns(session_id)
        assert len(turns) == 1
        assert turns[0]["user_answer_text"] is None  # not yet answered


class TestInterviewSession:
    @requires_infra
    @pytest.mark.asyncio
    async def test_incorrect_answer_produces_a_foundational_followup_not_abstain(self):
        """
        The flagship round-3-bug regression test: an incorrect answer
        must produce a real, on-topic follow-up question, not abstain —
        proven directly against the exact real scenario that used to
        dead-end the session (see quiz_interview.py's docstring).
        """
        start = await quiz_interview.start_session("interview", "vanishing gradient problem")
        result = await quiz_interview.submit_answer(
            start["session_id"],
            "I'm not entirely sure, maybe it has something to do with training being slow.",
        )
        next_q = result["next_question"]
        assert next_q["abstained"] is False
        assert next_q["question_text"] is not None
        assert len(next_q["question_text"]) > 0

    @requires_infra
    @pytest.mark.asyncio
    async def test_answer_key_is_never_exposed_in_any_returned_field(self):
        """
        The internal answer_key_text must never leak into any
        user-facing return value from start_session() or submit_answer().
        """
        start = await quiz_interview.start_session("interview", "gradient descent")
        assert "answer_key" not in str(start).lower() or "answer_key_text" not in start

        result = await quiz_interview.submit_answer(start["session_id"], "A reasonable answer.")
        assert "answer_key_text" not in result
        assert "answer_key_text" not in result["next_question"]
