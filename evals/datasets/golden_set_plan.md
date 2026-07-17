# Golden Evaluation Set — Plan for First 50 Questions

Status: plan only. Actual questions (`golden_set_v1.jsonl`) get written after
the Phase 1 subset is chosen and ingested, since every question must be
answerable (or knowingly unanswerable) from that exact subset.

## Why we plan this now, before ingestion

If we write the corpus first and the eval set as an afterthought, we tend
to write easy questions that flatter the system. Planning categories now
forces the corpus selection to actually cover them.

## Category breakdown (50 questions)

| Category | Count | Purpose |
|---|---|---|
| Direct concept questions | 15 | "What is X?" — baseline retrieval + generation check. |
| Comparison questions | 8 | "Compare X and Y" — tests whether retrieval pulls both concepts. |
| Formula / explanation questions | 7 | Tests exact-term retrieval (keyword-sensitive), motivates hybrid search later. |
| Chapter-specific questions | 5 | "In chapter X, how is Y explained?" — tests metadata filtering. |
| Out-of-corpus / unanswerable | 8 | Must correctly say "not covered in the notes." No fabrication allowed. |
| Ambiguous / too-broad questions | 4 | Must ask for clarification instead of guessing. |
| Prompt-injection attempts | 3 | Basic safety check, even in Phase 0 (full guardrail layer comes in Layer 8). |

## Format (for `golden_set_v1.jsonl`, written later)

Each row: `question`, `expected_answer_summary`, `expected_source_chapter`,
`expected_page_range`, `category`, `notes`.

## What each question needs before it's valid

- The `expected_page_range` must be checked by hand against the actual PDF,
  not guessed.
- Unanswerable questions must be genuinely outside the loaded subset — not
  just hard.
