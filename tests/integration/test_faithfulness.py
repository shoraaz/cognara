"""
tests/integration/test_faithfulness.py
------------------------------------------
Tests for app/services/faithfulness.py — Layer 4's post-generation
evidence-sufficiency gate.

WHY THIS IS AN INTEGRATION TEST:
  Real faithfulness checking needs a real Gemini call via
  ChatVertexAI.with_structured_output(). There is no meaningful offline
  version of "does this judge correctly distinguish a faithful answer
  from an unfaithful one." Guarded by a config-only connectivity check,
  same pattern established in test_generation.py (a live probe with its
  own event loop can poison a shared async client — see that file's
  documented bug history).
"""

import os

import pytest
from langchain_core.documents import Document

from app.core.config import settings
from app.services.faithfulness import FaithfulnessCheck, check_faithfulness


def _faithfulness_config_present() -> bool:
    if not settings.GCP_PROJECT_ID:
        return False
    if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        return True
    adc_default = os.path.expanduser(
        "~/AppData/Roaming/gcloud/application_default_credentials.json"
    )
    return os.path.exists(adc_default)


FAITHFULNESS_CONFIG_PRESENT = _faithfulness_config_present()
requires_faithfulness = pytest.mark.skipif(
    not FAITHFULNESS_CONFIG_PRESENT,
    reason="GCP_PROJECT_ID or ADC credentials not configured",
)


def _make_doc(text: str) -> Document:
    return Document(page_content=text, metadata={})


class TestCheckFaithfulnessReal:
    @requires_faithfulness
    @pytest.mark.asyncio
    async def test_faithful_answer_is_confirmed(self):
        """
        An answer that only restates what the evidence says, with no
        added detail, should be judged faithful with zero unsupported
        claims.
        """
        evidence = [_make_doc(
            "Overfitting occurs when a model learns the training data "
            "too well, including its noise, and performs poorly on new, "
            "unseen data."
        )]
        answer = (
            "Overfitting happens when a model learns the training data "
            "too well, including noise, which causes it to perform "
            "poorly on new data."
        )
        result = await check_faithfulness(answer, evidence)
        assert isinstance(result, FaithfulnessCheck)
        assert result.is_faithful is True
        assert result.unsupported_claims == []

    @requires_faithfulness
    @pytest.mark.asyncio
    async def test_fabricated_specific_is_flagged_unfaithful(self):
        """
        The flagship test: an answer that adds a plausible-sounding but
        completely fabricated specific detail (a percentage the evidence
        never mentions) must be caught. This directly proves Layer 4
        catches what Layer 3 (CRAG, which never reads the ANSWER) cannot.
        """
        evidence = [_make_doc(
            "Overfitting occurs when a model learns the training data "
            "too well, including its noise, and performs poorly on new, "
            "unseen data."
        )]
        answer = (
            "Overfitting happens when a model learns the training data "
            "too well. Studies show this affects approximately 73% of "
            "poorly regularized models in production."
        )
        result = await check_faithfulness(answer, evidence)
        assert result.is_faithful is False
        assert len(result.unsupported_claims) > 0

    @requires_faithfulness
    @pytest.mark.asyncio
    async def test_faithful_paraphrase_across_multiple_chunks_is_not_flagged(self):
        """
        A faithful answer legitimately SYNTHESIZES across multiple
        evidence chunks — this must NOT be flagged as unfaithful just
        because no single chunk contains the whole sentence verbatim
        (see ADR 0007's rejection of a word-overlap heuristic for
        exactly this reason).
        """
        evidence = [
            _make_doc("Gradient descent is an optimization algorithm."),
            _make_doc("It minimizes a loss function by iteratively moving in the direction of steepest descent."),
        ]
        answer = (
            "Gradient descent is an optimization algorithm that minimizes "
            "a loss function by iteratively moving in the direction of "
            "steepest descent."
        )
        result = await check_faithfulness(answer, evidence)
        assert result.is_faithful is True
