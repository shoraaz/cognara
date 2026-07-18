"""
tests/unit/test_schemas.py
--------------------------
Tests for Pydantic request/response schema validation.

WHY THIS FILE EXISTS:
  Schema tests run in milliseconds — no network, no LLM, no vector store.
  They verify that our contracts are correct. If AskRequest rejects a
  question that's too short, or AskResponse fails when a citation is
  missing page_number, we catch it here, not in production.

# Interview notes: local-notes/INTERVIEW_PREP.md — "tests/unit/test_schemas.py"
"""

import pytest
from pydantic import ValidationError
from app.models.schemas import AskRequest, AskResponse, Citation


class TestAskRequest:
    def test_valid_question(self):
        req = AskRequest(question="What is overfitting?")
        assert req.question == "What is overfitting?"

    def test_question_too_short_raises(self):
        with pytest.raises(ValidationError):
            AskRequest(question="Hi")

    def test_optional_filters_default_to_none(self):
        req = AskRequest(question="Explain gradient descent")
        assert req.course_filter is None
        assert req.chapter_filter is None

    def test_with_course_filter(self):
        req = AskRequest(
            question="Explain gradient descent",
            course_filter="100 Days of Machine Learning",
        )
        assert req.course_filter == "100 Days of Machine Learning"


class TestCitation:
    def test_valid_citation(self):
        c = Citation(
            course_name="100 Days of Machine Learning",
            chapter="Model Evaluation",
            page_number=142,
            relevance_score=0.87,
        )
        assert c.page_number == 142

    def test_invalid_score_above_1_raises(self):
        with pytest.raises(ValidationError):
            Citation(
                course_name="100 Days of ML",
                chapter="x",
                page_number=1,
                relevance_score=1.5,  # above 1.0, invalid
            )


class TestAskResponse:
    def test_abstained_response(self):
        r = AskResponse(
            answer="Not covered in the notes.",
            citations=[],
            confidence="abstained",
            abstained=True,
            abstain_reason="No relevant chunks found.",
        )
        assert r.abstained is True
        assert r.citations == []
