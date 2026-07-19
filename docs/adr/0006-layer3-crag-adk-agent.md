# ADR 0006: Layer 3 CRAG — ADK Agent Design, and Three Real Bugs Found Building It

Status: **Superseded by [ADR 0011](0011-crag-migration-adk-to-agno.md)**
— CRAG was migrated from ADK to Agno on 2026-07-19, as a direct
architectural decision, not because of any unresolved issue below (all
three bugs here were found, fixed, and verified working before the
migration). This document is kept unedited as the real, permanent record
of that debugging work — it remains genuine, valuable engineering
history independent of which framework the code currently runs on.

Date: 2026-07-18

## Context

Layer 3 implements Corrective RAG (CRAG) per the original master prompt:

    Question -> retrieve evidence -> critic grades retrieval quality
      -> if weak, rewrite/expand query and retry once
      -> if still weak, abstain or ask clarification
      -> if strong, send evidence to answer generator

Per ADR 0004, this is implemented as an ADK Agent with tools, not
hand-rolled if/else retry logic or LangGraph. This ADR records the real
design decisions and three genuine bugs found while building it —
useful both as a permanent record and as concrete interview material.

## Decision: one ADK Agent, three tools, no output_schema

`app/agents/crag_agent.py` defines a single `Agent` with three plain
Python function tools:

1. `search_notes(query, course_filter, chapter_filter)` — wraps Layer
   2's full pipeline (hybrid search + Vertex AI rerank) unchanged.
2. `grade_retrieval(relevance_score, completeness_score, decision,
   reason, rewritten_query)` — records the agent's OWN computed
   judgment (see Bug 3 below for why the signature is shaped this way).
3. `rewrite_query(original_query, reason_retrieval_was_weak)` — called
   only when a retry is warranted.

`grade_retrieval` is a deliberate, separate tool call rather than the
agent's own final `output_schema` — this keeps the grading step visible
and debuggable in ADK's event trace, independent of the final answer.

## Real bug 1: output_schema + tools together caused a 12x tool-call loop

The first real run of the agent, configured with BOTH `output_schema`
and `tools`, called `grade_retrieval` **twelve times** for what should
have been at most two calls — over two minutes of wasted latency and
real API cost. Research confirmed this matches multiple open, current
ADK GitHub issues (#3413, #3940, #3969): combining `output_schema` with
`tools` on one agent is a known, unresolved framework-level bug where
the agent repeatedly re-invokes a tool instead of committing to a final
structured response.

**Fix:** `output_schema` was removed entirely. The agent's final text
response is instructed (via its system prompt) to be a single JSON
object matching the required shape; `crag_runner.py` parses that text
manually, stripping markdown code fences and validating required keys,
with a safe "abstain" fallback on any parse failure.

## Real bug 2: instruction-following alone did not reliably cap grading at 2 calls

After fixing bug 1, a real retry-path test (query: "improvements", a
deliberately vague single word) showed the agent calling
`grade_retrieval` **three** times in one run — not the 12x loop from
bug 1, but still exceeding the "at most twice" instruction. A
strongly-worded prompt instruction is a soft constraint an LLM can still
slip on.

**Fix:** added a hard, code-level call counter inside `grade_retrieval`
itself. Any call beyond the second in a single run is forced to return
`decision="use"` with an explicit "grading limit reached" reason,
regardless of what arguments the agent passes — the model has no path
to a third real grading attempt. The counter is reset per CRAG run via
`reset_grade_call_count()`, called at the start of every `run_crag()`
invocation, so a second, later question is not incorrectly starved by
the first question's already-used budget.

## Real bug 3: grade_retrieval's original signature confused the agent about who was grading

The first working version of `grade_retrieval` took only `(query,
evidence_summaries)` and always returned all-`None` fields, on the
theory that returning `None` placeholders would cue the agent to
compute and report its own grade separately. In practice, a real run's
final reasoning literally stated: *"The grade_retrieval tool failed to
return valid scores or a decision on two attempts"* — the agent
interpreted the intentional `None` placeholders as the TOOL failing,
not as an invitation to grade. The final answer was still usable by
lucky recovery, but the design was genuinely confusing the model, not
just stylistically off.

**Fix:** `grade_retrieval`'s signature now takes `relevance_score`,
`completeness_score`, `decision`, `reason`, and `rewritten_query` AS
ARGUMENTS. The agent's instruction explicitly separates the two steps:
read the evidence and form your own judgment, THEN call `grade_retrieval`
to record that judgment. The tool validates and logs what it's given; it
does not compute anything. This is a clearer division of labour and
removed the confusion entirely in the corrected verification run.

## A related, separate fix: ADK's Vertex AI backend selection

Before any of the above, the very first CRAG run failed immediately with
`ValueError: No API key was provided` — ADK's `Agent` class does not
accept a `vertexai=True` constructor argument the way
`GoogleGenerativeAIEmbeddings` does. It goes through the unified
`google-genai` SDK, which selects its backend purely from three
environment variables: `GOOGLE_GENAI_USE_VERTEXAI`, `GOOGLE_CLOUD_PROJECT`,
`GOOGLE_CLOUD_LOCATION` — names our `.env` never defined (we use
`GCP_PROJECT_ID` / `VERTEX_AI_LOCATION`). Fixed by setting these three
variables at process startup in `app/core/config.py`
(`_configure_adk_vertex_backend()`), derived from our existing settings
fields, using `os.environ.setdefault()` so an explicit external override
is still respected.

Note: test runs surfaced a further deprecation warning —
`GOOGLE_GENAI_USE_VERTEXAI is deprecated, please use
GOOGLE_GENAI_USE_ENTERPRISE instead`. The old variable still works as of
this writing (all tests pass), so no immediate action was taken, but
this is a known future revisit.

## Verified, real behaviour (not just claims)

**Clear question** ("Explain the vanishing gradient problem."): graded
once, `relevance_score=0.9`, `decision=use`, 2 total tool calls, ~10
seconds.

**Deliberately vague question** ("improvements"): grade 1
(`relevance=0.4, decision=retry`) -> `rewrite_query` (genuine semantic
broadening) -> search again -> grade 2 (`relevance=0.2, decision=abstain`)
-> correctly terminates with `abstain`, honestly reporting that even
the retry didn't resolve the vagueness. 5 total tool calls, no looping.

## Interview summary

"I built Layer 3's CRAG critic as an ADK Agent with three tools, and hit
three real, distinct bugs in sequence while making it actually work
correctly — not just compile. First, a confirmed ADK framework bug: combining
output_schema with tools caused a 12x tool-call loop, matching several
open GitHub issues; I fixed it by dropping output_schema and parsing the
agent's text response as JSON myself. Second, even without that bug, a
strongly-worded 'call this at most twice' instruction wasn't a hard
guarantee — the agent called my grading tool three times once, so I
added a real code-level call counter as a backstop. Third, and most
interesting: my grading tool's original design had the tool return
placeholder None values expecting the agent to 'notice' and grade
itself — but the agent read those Nones as the tool failing. I fixed
that by having the agent compute its own scores and pass them AS
ARGUMENTS to the tool, which just records them — a clearer contract.
Each fix taught me something different about the actual, current
limitations of agent frameworks versus their documentation."

---

**Update, 2026-07-19:** replaced by Agno — see
[ADR 0011](0011-crag-migration-adk-to-agno.md). All three bugs above
were real and are now historical; the current code no longer has this
shape.
