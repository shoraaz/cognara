# ADR 0009: Layer 5 (Part 2) — Stateful Quiz and Interview Modes

Status: Accepted
Date: 2026-07-18

## Context

ADR 0008 split Layer 5 into single-turn modes (Explain, Compare,
Study-Plan — built, see that ADR) and stateful modes (Quiz, Interview —
this ADR). Both stateful modes need to track state ACROSS multiple
requests: which questions have already been asked, what the user
answered, whether it was correct, and what to ask next. CRAG's own
session pattern (ADR 0006) is explicitly single-shot
(InMemorySessionService, discarded after one request) and cannot
represent a multi-question quiz session.

## Decision: one shared session schema, two mode-specific generation strategies

### Schema: two new Cloud SQL tables, shared by both modes

```
learning_sessions
  session_id (PK), mode ('quiz'|'interview'), topic, course_filter,
  created_at, status ('active'|'completed')

learning_session_turns
  turn_id (PK), session_id (FK), turn_number, question_text,
  answer_key_text, user_answer_text, is_correct, feedback_text,
  evidence_chunk_ids (jsonb), created_at
```

Quiz and Interview are NOT given separate tables. Both need the
identical shape of state — an ordered sequence of (question, evidence,
answer, correctness) turns within a session. Splitting this into two
schemas would duplicate the same table twice for no structural reason;
the `mode` column on `learning_sessions` is sufficient to distinguish
them, and every turn-tracking query is identical regardless of mode.

This directly fulfils ADR 0003's original stated intent: "the same DB
will later hold user progress / quiz / interview state" — these tables
live in the existing `cognara-pg` instance, not a new database.

### What's genuinely different between the two modes

**Quiz mode** picks its next question to maximize TOPIC COVERAGE: it
reads the session's turn history, asks CRAG for evidence on the
requested topic while noting which sub-topics/chunk_ids have already
been used, and instructs Gemini to write a NEW question on a
not-yet-covered angle of the topic. Difficulty is not deliberately
varied — the goal is breadth.

**Interview mode** picks its next question to ADAPT DIFFICULTY: it
reads whether the user's previous answer was correct and how confident
the evidence-comparison judgment was, and instructs Gemini to either dig
deeper into the same sub-topic (if the answer was strong) or step back
to a more foundational related concept (if the answer was weak) —
mimicking how a real technical interviewer follows up. Topic breadth is
secondary to depth-probing on one thread at a time.

Both modes share: CRAG retrieval for the NEXT question's evidence, a
Gemini call to WRITE the question (grounded in that evidence, with an
internal answer key never shown to the user), and a separate Gemini
call to GRADE the user's submitted answer against that key +evidence.
Both reuse Layer 4's faithfulness pattern conceptually — the grading
call is itself an evidence-comparison judgment, structured the same way
faithfulness.py's check is.

## Alternatives considered and rejected

| Option | Why not |
|---|---|
| No persistence — hold session state in a browser/client-side object | The evidence and question history needs server-side grounding truth (which chunks were used, what the answer key actually was) for honest grading; trusting a client-supplied history is fragile and insecure |
| ADK's InMemorySessionService, extended to live longer | Explicitly designed for one request's lifetime (ADR 0006); stretching it to survive across separate HTTP requests fights the framework rather than using persistent storage designed for exactly this |
| Separate tables per mode | Identical schema shape; a mode column is sufficient and avoids duplicated migrations/queries |

## Interview summary

"Quiz and Interview both need real state across multiple requests — a
browser tab closing and reopening, or a different device, should still
resume the same session — so they get real Cloud SQL tables, not
in-memory state. I used ONE shared schema for both modes rather than
two separate tables, because the actual state shape — an ordered
sequence of question/evidence/answer/correctness turns — is identical;
only the logic for choosing the NEXT question differs: Quiz optimizes
for topic coverage, Interview adapts difficulty based on how the
previous answer went, like a real interviewer following a thread deeper
or backing off to a more foundational question."
