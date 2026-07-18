# ADR 0008: Layer 5 — Learning Modes (Explain, Compare, Quiz, Interview, Study-Plan)

Status: Accepted
Date: 2026-07-18

## Context

Layer 5 adds five distinct ways to engage with the same underlying
evidence-grounded pipeline (Layers 1-4: CRAG retrieval + grading, Gemini
generation, faithfulness checking): Explain, Compare, Quiz, Interview,
and Study-Plan.

## Decision: two genuinely different shapes, not five uniform ones

Building all five as if they were the same kind of feature would be
architecturally wrong for two of them. The real split:

**Single-turn modes (Explain, Compare, Study-Plan):** these need only a
different SYSTEM PROMPT applied to the exact same pipeline — CRAG
retrieves and grades evidence, generation.py produces a mode-specific
answer, faithfulness.py checks it, exactly as Layer 1-4 already do for
a plain question. No new state, no new retrieval logic, no new
orchestration. Each is a prompt template plus a thin wrapper function.

**Stateful modes (Quiz, Interview):** these genuinely need to track
state ACROSS multiple turns — which questions have already been asked,
what the user answered, whether an answer was correct, and what to ask
next. A single CRAG-retrieve-generate-check cycle cannot represent
"quiz me on gradient descent" as a multi-question session. These need:
(1) a session object that persists between requests, (2) generation
calls that read prior state to avoid repeating questions and to
evaluate the user's previous answer, (3) a real session store — not the
one-shot InMemorySessionService pattern CRAG uses (see ADR 0006), since
that discards state after a single request/response cycle.

## Single-turn mode implementation

`app/services/learning_modes.py` defines three prompt templates
(EXPLAIN, COMPARE, STUDY_PLAN) and a mode-aware wrapper around
generation.generate() that swaps SYSTEM_PROMPT for the selected mode's
template, but otherwise reuses generation.py's evidence-block building,
citation tagging, and token accounting completely unchanged. CRAG and
faithfulness checking apply identically regardless of mode — a
Compare-mode answer is checked for faithfulness exactly like a plain
answer.

Compare mode has one real difference: it needs evidence for BOTH things
being compared, which may span two different retrieval calls (e.g.
"compare gradient descent and stochastic gradient descent" might need
CRAG run twice, once per concept, then evidence merged) — handled by
running CRAG per detected concept and merging evidence before
generation, rather than trusting one CRAG call to retrieve balanced
evidence for two topics at once.

## Stateful mode implementation

`app/services/quiz_session.py` and `app/services/interview_session.py`
own a real session object (question history, user answers, correctness,
running topic coverage) stored in Cloud SQL (a new `quiz_sessions` /
`interview_sessions` table — reusing the existing pgvector database per
ADR 0003's stated intent that it would "later hold user progress / quiz
/ interview state"). Each turn: load session state, run CRAG for the
NEXT question's topic (informed by what hasn't been covered yet), have
Gemini generate a question grounded in that evidence, store the
question; on the user's answer, a separate grading call checks it
against the evidence and records correctness.

## Interview summary

"Layer 5 isn't five identical features — it's two different shapes.
Explain, Compare, and Study-Plan only need a different prompt template
on the exact same Layer 1-4 pipeline: CRAG retrieves, Gemini generates
with a mode-specific instruction, faithfulness checks the result. Quiz
and Interview are genuinely different — they need real state persisted
across multiple turns, which the CRAG agent's one-shot session pattern
doesn't provide. I gave those two their own session store in Cloud SQL,
reusing the same database Layer 3's ADR always intended to eventually
hold user progress state."
