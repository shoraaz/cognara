"""
app/services/learning_modes.py
---------------------------------
Layer 5: single-turn learning modes — Explain, Compare, Study-Plan.
See ADR 0008 for why these three are architecturally distinct from the
stateful modes (Quiz, Interview), which live in their own modules.

WHY THESE THREE MODES SHARE ONE FILE AND ONE PIPELINE:
  Each mode is the SAME evidence-grounded pipeline as plain /ask (CRAG
  retrieval + grading, Gemini generation, Layer 4 faithfulness checking)
  with only the system prompt changed. None of them need new retrieval
  logic, new state, or new orchestration — building them as separate
  services would duplicate Layers 1-4 for no real reason. This mirrors
  generation.py's own reasoning for adding a swappable system_prompt
  parameter rather than a separate generate_explain()/generate_compare()
  function per mode.

WHY COMPARE MODE RUNS CRAG TWICE, NOT ONCE:
  A single CRAG call retrieves evidence for ONE query. Asking it to
  retrieve balanced evidence for "gradient descent AND stochastic
  gradient descent" in one pass risks skewing toward whichever term
  dominates the embedding/BM25 signal, especially if one topic has much
  denser coverage in the corpus than the other. Running CRAG once per
  topic and merging the graded evidence guarantees BOTH topics get their
  own fair retrieval and grading pass, at the cost of double the
  retrieval latency for Compare mode specifically — a deliberate,
  documented trade (see ADR 0008).

WHY EACH MODE STILL GOES THROUGH LAYER 4'S FAITHFULNESS CHECK:
  A Compare-mode or Study-Plan-mode answer can hallucinate exactly the
  same way a plain answer can — an unsupported claim doesn't care what
  system prompt produced it. Layer 4 applies identically regardless of
  mode; this module does not reimplement or bypass it.

REAL BUG FOUND AND FIXED — COMPARE MODE TRUSTED "HAS ANY CHUNKS" INSTEAD
OF EACH SIDE'S OWN CRAG DECISION:
  A real test with one genuine topic ("gradient descent") and one
  nonsense topic ("xyzzy quolgorp fribbleton") failed unexpectedly: the
  WHOLE comparison abstained, even though the real topic had strong
  evidence (relevance 0.9, decision "use"). Root cause, found in the
  real log trace: CRAG on the nonsense side still returned 5 WEAKLY
  related chunks (the reranker's top_n=5 always returns something, even
  when nothing is truly relevant — e.g. an unrelated "Online Learning
  Resources" chunk), and the original code's abstain check
  ("both sides abstain OR both sides have zero chunks") let those weak,
  irrelevant chunks through into the merged generation prompt anyway.
  Gemini then correctly struggled to write a faithful comparison using
  noise on one side, got flagged unfaithful by Layer 4, and the
  regeneration ALSO failed — so the entire answer abstained, discarding
  the perfectly good "gradient descent" evidence along with the bad
  "xyzzy" evidence.
  FIX: only include a side's chunks in the merged generation if THAT
  SIDE's OWN CRAG grade decision was "use" — CRAG's per-side judgment
  (which already reads the actual evidence text) is a better signal than
  "did search_notes return a non-empty list," since reranking always
  returns its top_n regardless of whether any of them are genuinely
  relevant. A side whose CRAG grade was "retry" (already exhausted, see
  ADR 0006) or "abstain" is now excluded from generation entirely,
  rather than trusted alongside a genuinely strong side.

INTERVIEW EXPLANATION:
  "Explain, Compare, and Study-Plan are all the same underlying pipeline
  — CRAG retrieves and grades evidence, Gemini generates with a mode-
  specific system prompt, Layer 4 checks faithfulness — with only the
  prompt template changing per mode. Compare mode runs CRAG twice, once
  per topic, and merges the results. I found a real bug testing this: I
  was deciding whether to include a side's evidence based on 'did search
  return anything,' but retrieval always returns its top-N candidates
  even when none are truly relevant — only CRAG's own grade decision
  reliably distinguishes 'weak but present' from 'actually usable.'
  Fixing it to check each side's CRAG decision directly, not just chunk
  presence, meant a real topic paired with total nonsense correctly
  produces a one-sided (but honest) comparison instead of discarding
  good evidence because of noise on the other side."
"""

from typing import Literal

from app.agents.crag_runner import run_crag
from app.core.logging import get_logger
from app.services import faithfulness, generation

logger = get_logger(__name__)

LearningMode = Literal["explain", "compare", "study_plan"]


# ── Mode-specific system prompts ─────────────────────────────────────────────

EXPLAIN_SYSTEM_PROMPT = """You are Cognara Learn, an evidence-verified AI learning copilot, in EXPLAIN mode.

Your job is to teach the topic clearly, not just answer a narrow question. You answer ONLY using \
the numbered evidence chunks provided below.

1. Base your explanation entirely on the provided evidence. Do not use outside knowledge.
2. Structure the explanation for learning: start with a plain-English definition, then build up \
detail — mechanism, why it matters, and a concrete example if the evidence contains one.
3. When you use a fact from a chunk, reference it by its tag, like [1] or [2].
4. Explain technical terms the first time you use them, as if teaching someone new to the topic.
5. If the evidence does not contain enough information to explain the topic properly, say so \
plainly instead of guessing.
"""

COMPARE_SYSTEM_PROMPT = """You are Cognara Learn, an evidence-verified AI learning copilot, in COMPARE mode.

You are comparing TWO topics. The evidence below is grouped into labeled sections. There may be \
evidence for only ONE topic if the other topic had no usable evidence — in that case, clearly say \
you can only describe the one topic and cannot make a fair comparison. Answer ONLY using this evidence.

1. Structure your answer as a clear comparison: what each topic is, then their key similarities and \
differences, referencing evidence from BOTH sections when both are present.
2. Base every claim entirely on the provided evidence. Do not use outside knowledge.
3. When you use a fact from a chunk, reference it by its tag, like [1] or [2].
4. If the evidence for one topic is much thinner than the other, say so honestly rather than padding \
the thin side with unsupported detail.
5. If evidence for a topic is missing entirely, say so plainly instead of guessing or inventing detail.
"""

STUDY_PLAN_SYSTEM_PROMPT = """You are Cognara Learn, an evidence-verified AI learning copilot, in STUDY-PLAN mode.

Your job is to turn the evidence below into a short, ordered study plan for the requested topic. \
Answer ONLY using this evidence.

1. Break the topic into a logical sequence of sub-topics or steps, based on what the evidence actually covers.
2. For each step, briefly say what to focus on and why it matters, citing evidence chunks like [1] or [2].
3. Base every step entirely on the provided evidence — do not invent sub-topics the evidence doesn't cover.
4. Keep the plan concise: prefer 3-6 clear steps over an exhaustive list.
5. If the evidence only thinly covers the topic, produce a shorter plan and say so honestly, rather \
than padding it with unsupported steps.
"""


async def _run_crag_and_faithfulness(
    query: str,
    system_prompt: str,
    course_filter: str | None = None,
) -> dict:
    """
    Shared single-topic pipeline: CRAG retrieve+grade -> generate (with
    the given mode system_prompt) -> Layer 4 faithfulness check ->
    bounded regenerate-once-then-abstain, exactly mirroring
    ask_service.answer()'s Layer 1-4 flow. Returns a plain dict so
    Explain/Study-Plan can return it directly and Compare can merge two
    of these results before its own generation pass.
    """
    from langchain_core.documents import Document

    crag_result = await run_crag(query, course_filter=course_filter)
    grade = crag_result["grade"]
    evidence_chunks = crag_result["evidence_chunks"]

    if grade["decision"] == "abstain" or not evidence_chunks:
        return {
            "abstained": True,
            "abstain_reason": grade["reason"],
            "answer": None,
            "evidence_chunks": [],
            "grade": grade,
            "tokens": 0,
            "was_regenerated": False,
        }

    docs = [
        Document(
            page_content=c["text"],
            metadata={
                "course_name": c.get("course_name"), "chapter": c.get("chapter"),
                "topic": c.get("topic"), "page_number": c.get("page_number"),
                "page_range": c.get("page_range"),
            },
        )
        for c in evidence_chunks
    ]

    answer_text, tokens = await generation.generate(query, docs, system_prompt=system_prompt)
    total_tokens = tokens
    was_regenerated = False

    check = await faithfulness.check_faithfulness(answer_text, docs)
    if not check.is_faithful:
        answer_text, tokens = await generation.generate(
            query, docs, unsupported_claims=check.unsupported_claims, system_prompt=system_prompt,
        )
        total_tokens += tokens
        was_regenerated = True
        recheck = await faithfulness.check_faithfulness(answer_text, docs)
        if not recheck.is_faithful:
            return {
                "abstained": True,
                "abstain_reason": "Generated answer could not be verified as faithful, even after one regeneration attempt.",
                "answer": None,
                "evidence_chunks": evidence_chunks,
                "grade": grade,
                "tokens": total_tokens,
                "was_regenerated": True,
            }

    return {
        "abstained": False,
        "abstain_reason": None,
        "answer": answer_text,
        "evidence_chunks": evidence_chunks,
        "grade": grade,
        "tokens": total_tokens,
        "was_regenerated": was_regenerated,
    }


async def explain(topic: str, course_filter: str | None = None) -> dict:
    """Explain mode: teach `topic` from the corpus, evidence-grounded."""
    logger.info("learning_mode_start", mode="explain", topic=topic)
    return await _run_crag_and_faithfulness(topic, EXPLAIN_SYSTEM_PROMPT, course_filter)


async def compare(topic_a: str, topic_b: str, course_filter: str | None = None) -> dict:
    """
    Compare mode: retrieve and grade evidence for topic_a and topic_b
    SEPARATELY (see module docstring for why), then generate one
    comparison answer from the merged evidence — including a side's
    chunks ONLY if that side's own CRAG grade decided "use" (see module
    docstring, REAL BUG FOUND AND FIXED, for why "has any chunks" is not
    a reliable enough signal on its own).
    """
    logger.info("learning_mode_start", mode="compare", topic_a=topic_a, topic_b=topic_b)

    from langchain_core.documents import Document

    crag_a = await run_crag(topic_a, course_filter=course_filter)
    crag_b = await run_crag(topic_b, course_filter=course_filter)

    grade_a, grade_b = crag_a["grade"], crag_b["grade"]

    # Only trust a side's evidence if CRAG's OWN grade for that side said
    # "use" — search_notes/rerank always return their top_n candidates
    # regardless of true relevance, so "chunks is non-empty" alone is not
    # a reliable signal. See module docstring's REAL BUG FOUND AND FIXED.
    usable_chunks_a = crag_a["evidence_chunks"] if grade_a["decision"] == "use" else []
    usable_chunks_b = crag_b["evidence_chunks"] if grade_b["decision"] == "use" else []

    if not usable_chunks_a and not usable_chunks_b:
        return {
            "abstained": True,
            "abstain_reason": f"Insufficient evidence for both topics. {topic_a}: {grade_a['reason']} {topic_b}: {grade_b['reason']}",
            "answer": None,
            "evidence_chunks": [],
            "tokens": 0,
            "was_regenerated": False,
        }

    def _label_chunks(chunks: list[dict], label: str) -> list[dict]:
        return [{**c, "_compare_label": label} for c in chunks]

    merged_chunk_dicts = _label_chunks(usable_chunks_a, topic_a) + _label_chunks(usable_chunks_b, topic_b)
    docs = [
        Document(
            page_content=f"[{c['_compare_label']}] {c['text']}",
            metadata={
                "course_name": c.get("course_name"), "chapter": c.get("chapter"),
                "topic": c.get("topic"), "page_number": c.get("page_number"),
                "page_range": c.get("page_range"),
            },
        )
        for c in merged_chunk_dicts
    ]

    query = f"Compare {topic_a} and {topic_b}."
    answer_text, tokens = await generation.generate(query, docs, system_prompt=COMPARE_SYSTEM_PROMPT)
    total_tokens = tokens
    was_regenerated = False

    check = await faithfulness.check_faithfulness(answer_text, docs)
    if not check.is_faithful:
        answer_text, tokens = await generation.generate(
            query, docs, unsupported_claims=check.unsupported_claims, system_prompt=COMPARE_SYSTEM_PROMPT,
        )
        total_tokens += tokens
        was_regenerated = True
        recheck = await faithfulness.check_faithfulness(answer_text, docs)
        if not recheck.is_faithful:
            return {
                "abstained": True,
                "abstain_reason": "Generated comparison could not be verified as faithful, even after one regeneration attempt.",
                "answer": None,
                "evidence_chunks": merged_chunk_dicts,
                "tokens": total_tokens,
                "was_regenerated": True,
            }

    return {
        "abstained": False,
        "abstain_reason": None,
        "answer": answer_text,
        "evidence_chunks": merged_chunk_dicts,
        "tokens": total_tokens,
        "was_regenerated": was_regenerated,
    }


async def study_plan(topic: str, course_filter: str | None = None) -> dict:
    """Study-Plan mode: turn evidence about `topic` into an ordered study plan."""
    logger.info("learning_mode_start", mode="study_plan", topic=topic)
    return await _run_crag_and_faithfulness(topic, STUDY_PLAN_SYSTEM_PROMPT, course_filter)
