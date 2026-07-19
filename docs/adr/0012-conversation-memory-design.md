# ADR 0012: Conversation Memory System — Design Now, Build With the Frontend

Status: Accepted (design only — implementation deferred, see Consequences)
Date: 2026-07-19

## Context

Every request through the system so far is stateless: `/ask`, the
learning modes, even CRAG itself, treat each question as independent.
Layer 5's Quiz/Interview sessions (ADR 0009) are the one real exception,
but they are deliberately narrow — a fixed sequence of question/answer
turns for one quiz, not general conversation memory.

A real chat frontend needs two genuinely different things, and
conflating them is a common, real mistake worth naming explicitly:

1. **Short-term (session) memory** — "what did we just discuss in this
   conversation." Needs the recent turns, close to verbatim.
2. **Long-term (user) memory** — durable facts that persist ACROSS
   separate sessions, e.g. "preparing for an ML interview," "prefers
   detailed mathematical explanations." Needs selective retrieval of
   relevant facts, not the full history replayed into every prompt.

This ADR designs both, but implementation is deferred until a real
frontend exists to actually exercise multi-turn conversation — building
this against no real caller risks the same category of problem ADR
0011 discusses avoiding: designing against assumptions instead of a
real, testable use case.

## Decision 1: use Agno's native Memory + Session features, not a hand-rolled system

Checked directly against Agno's own docs (docs.agno.com, via the Agno
Docs MCP) before deciding — Agno has first-class, built-in support for
exactly this split:

- **Session history** (`add_history_to_context=True` on an `Agent`) —
  short-term, this-conversation memory, automatically included in
  context.
- **User Memory** (`update_memory_on_run=True`, or
  `enable_agentic_memory=True` for LLM-controlled writes) — long-term,
  cross-session facts, automatically extracted and recalled. Agno's own
  docs state the distinction directly: "Memory ≠ Session History:
  Memory stores learned user facts... Session history stores
  conversation messages for continuity."
- **`PostgresDb(db_url=...)`** — real, existing Postgres persistence for
  both, via plain SQLAlchemy.

This is the same reasoning as every prior "should we build this
ourselves or does the tool already do it" question in this project
(e.g. ADR 0005 rejecting a hand-rolled second search engine): Agno
already solves this well, matches our existing Cloud SQL choice
(ADR 0003), and building a parallel system would just be reinventing a
well-tested feature.

## Decision 2: memory belongs on a NEW conversational agent, not on CRAG

CRAG (`crag_agent.py`) is deliberately a stateless, single-purpose
retrieval critic — one question in, one grade out (see ADR 0006/0011).
Giving CRAG itself memory would blur that responsibility and complicate
its own real, already-proven behaviour (the 2-call grading cap, the
retry logic) with conversational concerns it was never designed for.

The correct place for memory is a **new, higher-level conversational
agent** — the actual chat-facing agent a frontend talks to — which
calls CRAG (and the learning modes) as a tool/step, the same way
`ask_service.py` already orchestrates CRAG -> generation ->
faithfulness today. That new agent is Layer 7's Orchestrator, not a
retrofit onto Layer 3.

## Open question, deliberately left open until the build phase

`PostgresDb(db_url=...)` expects a plain connection string. Every other
Postgres connection in this project (`init_db.py`,
`quiz_interview.py`) uses the Cloud SQL Python Connector — an
IAM-authenticated tunnel, not a raw DSN with an embedded password
(see ADR 0003's networking amendment). Whether Agno's `PostgresDb`
can be constructed against a Connector-provided connection (e.g. by
passing a SQLAlchemy engine's creator function instead of a URL) or
whether it genuinely requires a plain DSN is not yet verified. This is
real, necessary investigation for the actual build, not resolved here.

## Consequences

- No code changes from this ADR alone. Deferred until a real frontend
  exists — see Context for why building this against no real caller
  would repeat a known anti-pattern.
- When built: a new `app/agents/chat_agent.py` (or similar), separate
  from `crag_agent.py`, with `db=PostgresDb(...)`,
  `add_history_to_context=True`, and `enable_agentic_memory=True`.
  Calls CRAG and the learning modes as tools/steps.
- The Cloud SQL Connector compatibility question above must be resolved
  before this can connect to the same `cognara-pg` instance everything
  else uses — a real, first task of the build phase, not assumed away.

## Interview summary

"Before building a memory system, I checked whether Agno — which I'd
already migrated CRAG onto — had a native answer, rather than assuming
I'd need to hand-roll one. It does: a real distinction between session
history (this conversation) and long-term user memory (facts across
sessions), backed by Postgres, which is consistent with my existing
Cloud SQL choice. I designed this now, while it's fresh, but deferred
actually building it until a real frontend exists to exercise multi-turn
conversation — building conversation memory against no real caller risks
designing against assumptions instead of a real use case, the same
mistake I was careful to avoid with the CRAG-to-Agno migration itself.
I also decided memory belongs on a new, higher-level conversational
agent, not retrofitted onto CRAG, which is deliberately a stateless,
single-purpose retrieval critic — mixing those responsibilities would
complicate logic that's already proven correct."
