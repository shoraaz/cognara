# ADR 0007: Layer 4 — Post-Generation Evidence-Sufficiency Gate

Status: Accepted
Date: 2026-07-18

## Context

Layer 3 (CRAG, ADR 0006) grades whether RETRIEVED EVIDENCE is sufficient
to attempt an answer, before generation ever happens. This is a real,
useful gate, but it checks a different thing than what the original
master prompt's Layer 4 asks for: an "evidence-sufficiency gate" that
answers "is this GENERATED ANSWER actually faithful to its evidence."

These are genuinely different failure modes. CRAG can correctly judge
that the retrieved evidence is strong (high relevance/completeness) and
still have the generation step produce an answer that drifts from that
evidence — adds an unsupported specific number, states something the
evidence didn't actually say, or blends in outside knowledge despite the
prompt's grounding instructions (generation.py's SYSTEM_PROMPT already
tries to prevent this, but a prompt instruction is not a verification
step). CRAG never reads the generated answer at all — it only ever sees
the evidence. Layer 4 is the check that happens on the other side of
generation.

## Decision: a second LLM call, faithfulness-checking the generated answer against its own evidence, with one regenerate-then-abstain retry

**Mechanism:** after generation.generate() produces an answer, a second,
separate Gemini call (not the same call, not the same prompt) is given
ONLY the generated answer text and the evidence chunks it was supposed
to be based on, and asked to judge: does the evidence actually support
every factual claim in the answer? This is a standard "NLI-style"
faithfulness check — evidence entails claim — not a check on whether the
answer sounds good or is well-written.

**On failure:** exactly one regeneration attempt, with a stricter prompt
that explicitly lists what was flagged as unsupported, mirroring CRAG's
own "retry once, then commit" pattern (ADR 0006) rather than an
open-ended loop. If the SECOND attempt also fails the faithfulness
check, the response falls back to an honest abstain — the same
honesty-first behaviour CRAG already uses when a retry doesn't help.

## Alternatives considered and rejected

| Option | Why not |
|---|---|
| Skip Layer 4 entirely — trust generation.py's prompt instructions | A prompt instruction ("answer only from evidence") is not a verification step; this is exactly the well-documented "hallucination despite grounding" failure mode RAG systems are known to have even with a good, strict prompt |
| A cheaper, non-LLM heuristic (e.g. keyword/n-gram overlap between answer and evidence) | Too blunt — a faithful answer legitimately paraphrases and synthesizes across multiple chunks; word-overlap would flag correct paraphrasing as unsupported and miss a fluent but fabricated specific number |
| Reject and immediately abstain on first failure (no regenerate attempt) | Wastes a real chance at a good answer — many faithfulness failures are fixable by being told specifically what was wrong, without needing to give up on the question entirely |
| Unbounded regeneration loop until faithful | Directly repeats the CRAG round-1/round-2 lesson (ADR 0006) about needing a hard stop, not an instruction-only limit |

## Consequences

- One additional real Gemini call per /ask request that reaches
  generation (i.e. every non-abstained CRAG decision) — a real latency
  and cost addition, measured once implemented.
- Up to two full generation calls total on the unfaithful path (original
  + one regeneration) — still bounded, matching CRAG's own retry-once
  pattern.
- ask_service.py's flow becomes: CRAG (retrieve+grade) -> generate ->
  faithfulness check -> [regenerate once if needed] -> final response.
- A new module owns this: app/services/faithfulness.py, called from
  ask_service.py after generation.generate() and before citations are
  finalized.

## Interview summary

"CRAG (Layer 3) grades whether the RETRIEVED EVIDENCE is good enough to
attempt an answer, before generation happens. Layer 4 checks something
CRAG structurally cannot see: whether the GENERATED ANSWER actually
stays faithful to that evidence, after generation happens. I use a
second, separate Gemini call for this — an NLI-style check, evidence
entails claim — rather than a cheap heuristic like word overlap, because
a faithful answer legitimately paraphrases and synthesizes, so overlap
would produce both false positives and false negatives. On failure, I
regenerate exactly once with the specific unsupported claims called out,
then fall back to an honest abstain if the second attempt also fails —
the same bounded-retry-then-honesty pattern I already used for CRAG's
own retrieval grading."
