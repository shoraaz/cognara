# ADR 0001: Start with a focused ~250-400 page subset instead of all 6,000 pages

Status: Accepted
Date: 2026-07-12

## Context

The full CampusX notes bundle is about 6,000 pages across 9 courses. If we
ingest everything on day one, we cannot tell whether a wrong answer is
caused by bad chunking, weak retrieval, a bad prompt, or simply too much
noisy data. Debugging becomes guesswork.

## Alternatives considered

| Option | Problem |
|---|---|
| Ingest all 6,000 pages immediately | No way to isolate failure causes. Slow iteration. High embedding cost before the pipeline is even proven. |
| Ingest one full course (~1,000 pages) | Still too large to hand-check retrieval quality chunk by chunk during early debugging. |
| Ingest a focused 250-400 page subset from topics the user already knows well | User can personally verify every answer and citation, since they already know the material. Fast iteration. Cheap. |

## Decision

Phase 1 ingests a hand-picked 250-400 page subset, pulled from "100 Days of
Machine Learning," covering the full supervised-learning pipeline end to
end (not just one topic), so we can test direct questions, comparison
questions, and formula questions all in the same small corpus.

The full ML + DL corpus (~2,000 pages) is added only after this subset's
retrieval and citation accuracy is measured and acceptable. The secondary
corpus (LangChain, LangGraph, MCP, FastAPI, PyTorch, NLP, Claude Code) is
added only after the core RAG, CRAG, and evaluation pipeline all work.

## Consequences

- We can eyeball-verify almost every answer in Phase 1, because the user
  already knows this material.
- Metadata schema (course, chapter, topic, page) must be designed now, even
  though only one course is loaded, so scaling up later is a data-loading
  change, not a schema change.
- Evaluation benchmark questions for Phase 1 must come only from the loaded
  subset, not the full course index.
