"""
tests/integration/test_generation.py
--------------------------------------
Tests for app/services/generation.py.

WHY THIS IS AN INTEGRATION TEST:
  Like embedder.py, this module has no meaning without a real call to
  Gemini via Vertex AI. These tests call the actual model and verify
  real, observable behaviour: does it answer from evidence, does it
  cite chunk tags, does it decline when evidence doesn't cover the
  question.

REAL BUG FOUND AND FIXED (async/sync event loop mismatch, THREE rounds):
  ROUND 1: asyncio.run() wrapped around each test body failed with
  "RuntimeError: Event loop is closed" on the second/third call in one
  pytest session — grpc.aio's channel binds to whichever event loop was
  active when first used, and each asyncio.run() call tears its loop
  down on return.
  FIX ATTEMPTED: convert tests to real `async def` under pytest-asyncio,
  so all async tests share one session loop.

  ROUND 2: that alone didn't fix it, because the CONNECTIVITY PROBE
  (module-level, runs once at import) called the SYNC llm.invoke() path,
  which internally manages its own event loop under the hood — poisoning
  the shared singleton's gRPC channel before any real test ran.
  FIX ATTEMPTED: make the probe call llm.ainvoke() instead of invoke().

  ROUND 3 (the actual root cause): switching the PROBE to ainvoke() still
  failed, because the probe itself called asyncio.run(llm.ainvoke(...)) —
  and asyncio.run() ALWAYS creates a new loop and closes it on return,
  regardless of which method runs inside it. That closing is what
  poisons the shared client, not sync-vs-async. Any asyncio.run() call
  before pytest-asyncio's own managed loop takes over will break the
  singleton.
  FINAL FIX: don't perform a live network probe with its own event loop
  at all. Instead, do a cheap, purely local reachability check —
  confirm required config is present (GCP_PROJECT_ID set, ADC
  credentials file/env discoverable) — and let genuine unreachability
  surface as a real, informative test failure rather than trying to
  pre-empt it with a probe that causes the very problem it's checking
  for.

INTERVIEW EXPLANATION:
  "This took three rounds to actually fix, and each wrong fix taught me
  something specific about how grpc.aio interacts with asyncio event
  loops: the channel binds to a loop at first use, and ANY asyncio.run()
  call — not just a sync/async mismatch — tears down whatever loop it
  created when it returns. The real fix was to stop trying to
  proactively probe reachability with a separate event loop altogether,
  since pytest-asyncio already manages one shared loop for the whole
  session. I test connectivity by simply checking required config is
  present, and let an actual unreachable-service failure be a real,
  readable test failure — which is more honest than a probe that can
  itself corrupt the very client under test."
"""

import os

import pytest
from langchain_core.documents import Document

from app.core.config import settings
from app.services import generation


def _generation_config_present() -> bool:
    """
    Cheap, LOCAL-ONLY check — no network call, no event loop created.
    Confirms the config needed to attempt a Gemini call exists; does not
    guarantee the call will succeed (a real 429/403/etc. still surfaces
    as a normal, readable test failure if it happens).
    """
    if not settings.GCP_PROJECT_ID:
        return False
    # ADC discovery: either GOOGLE_APPLICATION_CREDENTIALS is set, or the
    # gcloud user ADC file exists at its default path.
    if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        return True
    adc_default = os.path.expanduser(
        "~/AppData/Roaming/gcloud/application_default_credentials.json"
    )
    return os.path.exists(adc_default)


GENERATION_CONFIG_PRESENT = _generation_config_present()
requires_generation = pytest.mark.skipif(
    not GENERATION_CONFIG_PRESENT,
    reason="GCP_PROJECT_ID or ADC credentials not configured — see .env / gcloud auth application-default login",
)


def _make_doc(text: str, course: str, chapter: str, topic: str, page: int) -> Document:
    return Document(
        page_content=text,
        metadata={
            "course_name": course, "chapter": chapter, "topic": topic,
            "page_number": page, "page_range": None,
        },
    )


class TestBuildEvidenceBlock:
    def test_formats_chunks_with_numbered_tags(self):
        chunks = [
            _make_doc("Overfitting is when a model memorizes noise.", "ML Course", "Ch1", "Overfitting", 42),
            _make_doc("Regularization helps prevent overfitting.", "ML Course", "Ch1", "Regularization", 43),
        ]
        block = generation._build_evidence_block(chunks)
        assert "[1]" in block
        assert "[2]" in block
        assert "page 42" in block
        assert "page 43" in block
        assert "Overfitting is when a model memorizes noise." in block


class TestGenerateRealCall:
    @requires_generation
    @pytest.mark.asyncio
    async def test_answers_from_provided_evidence(self):
        chunks = [
            _make_doc(
                "Overfitting occurs when a model learns the training data too "
                "well, including noise, and performs poorly on new unseen data.",
                "100 Days of Machine Learning", "Introduction", "Overfitting", 42,
            )
        ]
        answer_text, tokens = await generation.generate("What is overfitting?", chunks)
        assert len(answer_text) > 0
        assert tokens > 0
        assert "overfit" in answer_text.lower()

    @requires_generation
    @pytest.mark.asyncio
    async def test_declines_when_evidence_is_unrelated(self):
        """
        The evidence is about an unrelated topic. A well-grounded prompt
        should make Gemini say the evidence doesn't cover the question,
        not answer from outside knowledge. This is a real behavioural
        test of prompt quality, not just plumbing — it can be flaky in
        principle (LLM output isn't 100% deterministic), which is why it
        checks for a DECLINE-shaped answer rather than exact wording.
        """
        chunks = [
            _make_doc(
                "A recipe for pasta requires flour, eggs, and salt, kneaded "
                "for ten minutes and rested for thirty.",
                "Cooking Notes", "Recipes", "Pasta", 1,
            )
        ]
        answer_text, tokens = await generation.generate(
            "What is the vanishing gradient problem?", chunks
        )
        lower = answer_text.lower()
        decline_signals = ["not", "doesn't", "does not", "cannot", "no information",
                           "insufficient", "unable", "don't", "do not"]
        assert any(signal in lower for signal in decline_signals), (
            f"Expected a decline-shaped answer when evidence is unrelated, got: {answer_text}"
        )

    @requires_generation
    @pytest.mark.asyncio
    async def test_real_token_usage_is_returned(self):
        chunks = [_make_doc("Gradient descent minimizes the loss function.", "ML", "Ch2", "Gradient Descent", 10)]
        _answer, tokens = await generation.generate("What is gradient descent?", chunks)
        assert isinstance(tokens, int)
        assert tokens > 0
