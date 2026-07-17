# Document Catalog Schema

Every chunk we store must carry this metadata. This schema is designed now,
while only one course is loaded, so adding more courses later is just more
rows — not a schema change.

## Fields

| Field | Type | Example | Why we need it |
|---|---|---|---|
| `chunk_id` | string (uuid) | `c_0001a2` | Unique key for the chunk in the vector store. |
| `course_name` | string | `100 Days of Machine Learning` | Top-level source. Lets us filter by course and keep corpora separate. |
| `subject` | string | `Machine Learning` | Groups courses by subject (ML, DL, GenAI, Agentic AI, MLOps...). |
| `chapter` | string | `Ensemble Learning` | Used in citations shown to the user. |
| `topic` | string | `Bagging vs Boosting` | Finer than chapter; used for quiz/interview scoping. |
| `page_number` | integer | `142` | Required for every citation. Never fabricated. |
| `page_range` | string | `142-144` | Some chunks span multiple pages. |
| `source_type` | string | `campusx_notes` | Distinguishes internal notes from later "official external documentation." |
| `document_version` | string | `v1` | If notes get updated, old citations can be traced to the version they came from. |
| `ingestion_date` | date | `2026-07-12` | Freshness tracking. |
| `chunk_index_in_doc` | integer | `37` | Ordering, for debugging chunking quality. |
| `char_count` | integer | `1180` | Sanity check on chunk size during tuning. |

## Rules

- `page_number` must come from the actual PDF page, not an estimate. If OCR
  or parsing cannot confirm the page, the chunk is flagged
  `page_number_confidence: low` rather than guessed.
- `course_name` and `subject` are controlled vocabularies (fixed list), not
  free text, so filtering stays reliable.
- No chunk is stored without `course_name`, `chapter`, `page_number`, and
  `source_type` filled in. Incomplete metadata blocks ingestion for that
  chunk rather than storing it with gaps.

## Controlled vocabulary (Phase 1)

`course_name`: `100 Days of Machine Learning` (only one, for now — others
added as later phases begin).

`source_type`: `campusx_notes` (only internal notes for now — no external
web content, per data scope rules).

See [`document_catalog_template.csv`](document_catalog_template.csv) for the
row format used to track ingested source documents (not individual chunks —
this is the document-level catalog, one row per source PDF/chapter).
