# ADR 0011: Layer 3 CRAG Migrated From ADK to Agno

Status: Accepted
Date: 2026-07-19

**Supersedes:** ADR 0004 (the "ADK for orchestration" half only — the
LangChain-for-components half is not superseded by this ADR; see
Consequences) and ADR 0006 (CRAG's ADK implementation).

Both ADR 0004 and ADR 0006 remain in `docs/adr/` unedited. They are not
deleted: they are the real, permanent record of what was tried, what
bugs were found, and how they were fixed — that debugging history stands
on its own regardless of which framework the code currently uses.

## Context

CRAG (Layer 3) was originally built on Google's Agent Development Kit
(ADK), a real, deliberate choice recorded in ADR 0004. Building it
surfaced three confirmed ADK bugs (ADR 0006): a 12x tool-call loop when
`output_schema` and `tools` were combined, an instruction-only grading
cap that wasn't reliable, and a tool signature that confused the model
about who was responsible for grading. All three were fixed and CRAG
worked correctly and was fully tested (10/10 tests passing) before this
migration was considered.

This migration was requested directly, not driven by a new bug or a
measured shortfall in the working ADK implementation.

## Decision

Rewrite `crag_agent.py` and `crag_runner.py` on Agno
(`agno>=2.5.0`, current stable line — v2.0, Sept 2025, was a full
rewrite of Agent/Team/Workflow; earlier tutorials/code do not apply).

### What changed and why it's genuinely simpler

ADK's `Runner` + `SessionService` requires manually iterating streamed
events (`async for event in runner.run_async(...)`,
`event.get_function_calls()`, `event.get_function_responses()`) to
reconstruct what tools were called and with what results — real,
necessary code in the old `crag_runner.py` (~140 lines).

Agno's `agent.arun()` returns a finished `RunOutput` object with
`.content` and `.tools` already populated. `crag_runner.py` shrank to
~55 lines as a direct, measured result — not an estimate, the file was
counted before and after.

Also genuinely different: Agno's own documentation states that
combining `output_schema` with `tools` is a supported, intentional
pattern (tools run, then the final response is validated against the
schema) — this is exactly the combination that caused ADK's confirmed
12x loop bug. `RetrievalGrade` is now a real, enforced `output_schema`,
not a reference type manually parsed from free text (see ADR 0006's
Round 1 for why ADK required that workaround).

## A new, real bug found migrating (documented for the same reason ADR
0006 documented ADK's bugs — this is not ADK-specific, it's a genuine
Agno behaviour worth knowing)

`ToolExecution.result` on `RunOutput.tools` comes back as a **Python
repr string** (e.g. `"[{'text': '...'}]"`), not the original Python
object a tool returned. The first version of the new `crag_runner.py`
checked `isinstance(call.result, list)` before trusting a tool's
evidence — silently always `False`, since `result` is a `str`. Confirmed
by direct inspection (`type(call.result)` is `str`, and the string uses
single quotes — a repr, not JSON). Fixed with `ast.literal_eval()`
(safe for Python literals; `json.loads()` would fail on the
single-quoted format).

## What did NOT change

`learning_modes.py` and `quiz_interview.py` call `run_crag()` and were
verified to need **zero code changes** — `run_crag()`'s return shape
(`{"grade": {...}, "evidence_chunks": [...]}`, plain dicts) was
unchanged by the migration. This is a real, positive signal that the
original interface boundary between CRAG and its callers was designed
correctly: callers only ever depended on plain data, never on
ADK-specific types.

The retrieval layer (`vector_store.py`, `hybrid_search.py`,
`reranker.py`) and the LangChain-based LLM calls inside
`generation.py`, `faithfulness.py`, and `quiz_interview.py` are UNCHANGED
by this ADR — they use `langchain_core.documents.Document` and
`ChatVertexAI.with_structured_output()` independently of CRAG's own
agent framework. Removing LangChain from those modules, if done, is a
separate, later migration with its own scope and its own ADR — not
bundled into this one.

## Consequences

- `google-adk` dependency removed. `agno` and `google-genai` added.
- CRAG's real, verified behaviour is unchanged: content questions grade
  once and use; structural questions correctly choose
  `search_concept_graph`; vague questions correctly retry or abstain;
  the hard 2-call grading cap still works (now enforced the same way,
  just inside Agno's tool-call loop instead of ADK's).
- All 8 CRAG integration tests rewritten and passing against live
  infrastructure (Cloud SQL, Neo4j, Vertex AI) after the migration.
- ADR 0004 and ADR 0006 remain in the repository, marked superseded
  above, not deleted — they are the real record of the ADK phase,
  including three confirmed framework bugs found and fixed, which
  remain true, valuable engineering history independent of which
  framework the code currently runs on.

## Interview summary

"I migrated CRAG from ADK to Agno after ADK was already working
correctly with three real, fixed framework bugs behind it — this wasn't
a bug-driven migration, it was a direct architectural change. I kept the
old ADRs in place, marked superseded, rather than deleting them, because
that debugging history is real and independent of which framework the
code currently uses — deleting it would make the project look like the
ADK phase never happened, when actually it's some of my strongest
evidence of debugging real framework behaviour, not just my own code.
The migration itself was a real net simplification — Agno returns a
finished result object instead of ADK's raw event stream, so
crag_runner.py went from about 140 lines to about 55 — but it wasn't
bug-free either: I found that Agno serializes tool results to their
Python repr string on the way back, not the original object, which
would have silently broken evidence extraction if I'd trusted an
isinstance check without verifying the real type first."
