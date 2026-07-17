"""
tests/integration/test_ask_service.py
----------------------------------------
Tests for app/services/ask_service.py — the full RAG pipeline, end to end.

WHY THIS IS AN INTEGRATION TEST:
  This is the top of the stack: real vector search against the real
  ingested corpus (388 chunks from Module 4), real Gemini generation.
  These tests prove the whole system works together, not just each piece
  in isolation — the same guarantee the manual verification script gave
  during Module 4, now captured as a real, repeatable test.

  Connectivity check is config-only, not a live probe with its own event
  loop — see test_generation.py's three-round bug note for exactly why:
  any asyncio.run() call made before pytest-asyncio's own session loop
  takes over silently poisons the shared ChatVertexAI client's gRPC
  channel for every real async test that runs afterward. The vector
  store's own connectivity is still checked live (it's plain sync
  SQLAlchemy — no event loop involved, no equivalent risk).
"""

import os

import pytest

from app.core.config import settings
from app.models.schemas import AskRequest
from app.services import ask_service


def _ask_service_reachable() -> bool:
    """
    Vector store: real, synchronous check (safe — no event loop).
    Generation: config-only check (safe — no event loop created here).
    See module docstring for why a live async probe is deliberately
    avoided.
    """
    try:
        store = ask_service._get_store()
        store.similarity_search_with_score("connectivity probe", k=1)
    except Exception:
        return False

    if not settings.GCP_PROJECT_ID:
        return False
    if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        return True
    adc_default = os.path.expanduser(
        "~/AppData/Roaming/gcloud/application_default_credentials.json"
    )
    return os.path.exists(adc_default)


ASK_SERVICE_REACHABLE = _ask_service_reachable()
requires_ask_service = pytest.mark.skipif(
    not ASK_SERVICE_REACHABLE,
    reason="ask_service not reachable — check cognara-pg is running and ADC credentials are set up",
)


class TestAnswerRealCorpus:
    @requires_ask_service
    @pytest.mark.asyncio
    async def test_real_question_against_real_corpus_returns_grounded_answer(self):
        """
        The flagship end-to-end test: a real question about a concept
        genuinely covered in the ingested corpus (see Module 4's
        verification: 'vanishing gradient' scored 0.77 against real
        chunks). This proves retrieval -> abstention-check -> generation
        -> citation assembly all work together against live data.
        """
        request = AskRequest(question="Explain the vanishing gradient problem.")
        response = await ask_service.answer(request, request_id="test-1")

        assert response.abstained is False
        assert len(response.answer) > 0
        assert len(response.citations) > 0
        assert response.confidence in ("high", "medium", "low")
        assert response.tokens_used > 0

        chapters = [c.chapter for c in response.citations]
        assert any("Gradient" in ch for ch in chapters)

    @requires_ask_service
    @pytest.mark.asyncio
    async def test_every_citation_has_a_real_page_number(self):
        request = AskRequest(question="What is supervised learning?")
        response = await ask_service.answer(request, request_id="test-2")

        assert not response.abstained
        for citation in response.citations:
            assert citation.page_number > 0
            assert 0.0 <= citation.relevance_score <= 1.0

    @requires_ask_service
    @pytest.mark.asyncio
    async def test_course_filter_restricts_citations_to_that_course(self):
        request = AskRequest(
            question="Explain a key concept.",
            course_filter="100 Days of Deep Learning",
        )
        response = await ask_service.answer(request, request_id="test-3")

        if not response.abstained:
            for citation in response.citations:
                assert citation.course_name == "100 Days of Deep Learning"

    @requires_ask_service
    @pytest.mark.asyncio
    async def test_nonsense_question_abstains(self):
        """
        A question with no relationship to ML/DL content should not find
        strong matches in the corpus and should abstain rather than
        force an answer from weak evidence.
        """
        request = AskRequest(question="What is the best recipe for chocolate cake?")
        response = await ask_service.answer(request, request_id="test-4")

        assert response.abstained or response.confidence == "low"

    @requires_ask_service
    def test_singleton_store_is_reused_across_calls(self):
        store1 = ask_service._get_store()
        store2 = ask_service._get_store()
        assert store1 is store2
