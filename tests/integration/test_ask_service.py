"""
tests/integration/test_ask_service.py
----------------------------------------
Tests for app/services/ask_service.py — the full RAG pipeline via the
Layer 3 CRAG agent and Layer 4 faithfulness gate, end to end.

WHY THIS IS AN INTEGRATION TEST:
  This is the top of the stack: real hybrid search + rerank (Layer 2),
  real CRAG grading and retry (Layer 3), real Gemini generation, real
  post-generation faithfulness checking (Layer 4). These tests prove the
  whole system works together, not just each piece in isolation.

REFACTOR NOTE (Layer 3 wiring):
  ask_service.py no longer manages its own CognaraPGVectorStore
  singleton — retrieval now lives entirely inside the CRAG agent
  (app.agents.crag_runner.run_crag). The connectivity probe below
  reflects that: it checks CRAG agent construction + config presence,
  not a directly-held store instance.

  Connectivity check is config-only for the LLM parts, not a live probe
  with its own event loop — see test_generation.py's bug notes for why:
  any asyncio.run() call made before pytest-asyncio's own session loop
  takes over can poison a shared async client's gRPC channel for real
  tests that run afterward.
"""

import os

import pytest

from app.agents import crag_agent
from app.core.config import settings
from app.models.schemas import AskRequest
from app.services import ask_service


def _ask_service_reachable() -> bool:
    """
    CRAG agent construction: real, synchronous check (safe — building an
    Agent object doesn't open a network connection or event loop).
    Generation/Vertex AI: config-only check (safe — no event loop
    created here). See module docstring for why a live async probe is
    deliberately avoided.
    """
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
        genuinely covered in the ingested corpus. Proves CRAG retrieval
        -> grading -> generation -> faithfulness check -> citation
        assembly all work together against live data, through the real
        Layer 3 + Layer 4 wiring.
        """
        request = AskRequest(question="Explain the vanishing gradient problem.")
        response = await ask_service.answer(request, request_id="test-1")

        assert response.abstained is False
        assert len(response.answer) > 0
        assert len(response.citations) > 0
        assert response.confidence in ("high", "medium", "low")
        assert response.tokens_used > 0
        assert isinstance(response.was_regenerated, bool)

        chapters = [c.chapter for c in response.citations]
        assert any("Gradient" in ch for ch in chapters)

    @requires_ask_service
    @pytest.mark.asyncio
    async def test_every_citation_has_a_real_page_number(self):
        request = AskRequest(question="What is supervised learning?")
        response = await ask_service.answer(request, request_id="test-2")

        if not response.abstained:
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
    async def test_nonsense_question_abstains_or_low_confidence(self):
        """
        A question with no relationship to ML/DL content should either
        be abstained by CRAG's critic directly, or — if the critic still
        decides to answer — reflect that weakness with a low confidence
        label. CRAG's own reasoning is the abstention gate now (not a
        fixed threshold), so we allow either honest outcome.
        """
        request = AskRequest(question="What is the best recipe for chocolate cake?")
        response = await ask_service.answer(request, request_id="test-4")

        assert response.abstained or response.confidence == "low"

    @requires_ask_service
    @pytest.mark.asyncio
    async def test_abstained_response_has_no_citations(self):
        """
        When CRAG abstains, the response must not carry stale or
        partial citations — an abstained answer should never look
        evidenced.
        """
        request = AskRequest(question="What is the best recipe for chocolate cake?")
        response = await ask_service.answer(request, request_id="test-5")

        if response.abstained:
            assert response.citations == []
            assert response.abstain_reason is not None

    @requires_ask_service
    @pytest.mark.asyncio
    async def test_was_regenerated_defaults_false_for_a_faithful_answer(self):
        """
        A clear, well-covered question with a genuinely faithful first
        answer should NOT trigger Layer 4's regeneration path — proven
        directly (not just schema-defaulted) against a real question
        where the first generation attempt is expected to already be
        faithful.
        """
        request = AskRequest(question="What is gradient descent?")
        response = await ask_service.answer(request, request_id="test-6")

        if not response.abstained:
            assert response.was_regenerated is False
