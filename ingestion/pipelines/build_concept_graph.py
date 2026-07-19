"""
ingestion/pipelines/build_concept_graph.py
----------------------------------------------
Layer 6: one-time (idempotent, resumable, batchable) pipeline that
extracts a concept graph from the existing 388-chunk corpus and writes
it to Neo4j AuraDB. See ADR 0010 for the full design.

WHY THIS FILE EXISTS:
  Every earlier layer works with the CHUNKS table as-is. This pipeline
  reads the same corpus and produces a SECOND, complementary
  representation of it — a graph of concepts and their relationships —
  answering a different class of question ("what should I learn before
  X") that plain retrieval structurally cannot answer well.

EXTRACTION APPROACH — per chunk, not per corpus:
  For each of the 388 chunks (same source as Modules 1-4), a single
  Gemini call reads the chunk's real topic/heading text (e.g. "20.1.7
  How to Detect Vanishing Gradient Problem" — genuine structured
  headings already produced by the chunker, Module 2) plus its content,
  and extracts:
    - The PRIMARY concept this chunk is about (a canonical name)
    - Any RELATED concepts it mentions, with a relationship type
      (PREREQUISITE_OF / RELATED_TO / PART_OF / CONTRASTS_WITH)
  Structured output via ChatVertexAI.with_structured_output() — the
  same reliable mechanism used in faithfulness.py and quiz_interview.py
  (NOT ADK's output_schema, which had a confirmed bug — see ADR 0006).

WHY PER-CHUNK, NOT ONE GIANT CORPUS-WIDE PASS:
  Sending all 388 chunks to one Gemini call would exceed reasonable
  context and produce a single, hard-to-verify extraction. Per-chunk
  extraction means each concept's grounding_chunk_ids is exact (the
  extraction ran ON that chunk), the pipeline can be resumed/retried
  per-chunk if one call fails, and the incremental MERGE-based graph
  writes (graph_store.py) naturally accumulate a consistent graph
  across many small, independently-verifiable extraction calls.

REAL GAP FOUND AND FIXED (round 1) — THE FIRST RUN WAS NOT RESUMABLE:
  The first real full-corpus run made genuine progress (92 concepts, 80
  relationships from roughly the first third of chunks, confirmed by
  querying Neo4j directly) before the terminal session it was running in
  was lost — background execution through the remote shell tool used for
  this project is not reliable once the parent session disconnects, a
  real, practical constraint discovered by hitting it, not a code bug.
  The original build_graph() had no way to skip chunks already
  processed, meaning a naive re-run would re-spend real Gemini API calls
  on the same first third of the corpus.
  FIX: build_graph() queries Neo4j FIRST for the set of chunk_ids
  already present in any concept's grounding_chunk_ids (a real read
  against the graph itself, not a separate progress-tracking file that
  could drift out of sync with what's actually written), and skips those
  chunks entirely on this and any future run.

REAL GAP FOUND AND FIXED (round 2) — RESUMABILITY ALONE WASN'T ENOUGH,
EVERY SINGLE RUN STILL HIT THE SAME EXECUTION-TIME LIMIT:
  Even with resume logic working correctly (confirmed: 181 chunks done
  after run 1, 223 done after run 2 — genuine incremental progress each
  time), every single invocation still got cut off by the remote shell
  tool's execution-time limit before finishing the REMAINING chunks,
  because "remaining" still meant 200+ sequential Gemini calls, still
  too many for one call to finish in the available window.
  FIX: build_graph() now accepts an explicit limit parameter — process
  at most N not-yet-done chunks, then stop cleanly and report real
  progress, rather than attempting "everything left" every time and
  repeatedly hoping it finishes before the tool's time budget runs out.
  Combined with round 1's resume-by-querying-the-graph logic, this means
  the pipeline can be invoked in a series of small, always-completing
  batches (e.g. --limit 60) until the whole corpus is done — genuine,
  practical resumability under a real execution-time constraint, not
  just idempotency in theory.

RUN:
  make build-graph                          # process everything remaining
  make build-graph LIMIT=60                 # process at most 60 more chunks
  python -m ingestion.pipelines.build_concept_graph --limit 60

# Interview notes: local-notes/INTERVIEW_PREP.md — "ingestion/pipelines/build_concept_graph.py"
"""

import argparse

import sqlalchemy
from langchain_google_vertexai import ChatVertexAI
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import settings
from app.core.logging import get_logger, setup_logging
from app.retrieval import graph_store
from ingestion.pipelines.init_db import get_engine

logger = get_logger(__name__)


class ExtractedRelation(BaseModel):
    target_concept: str = Field(..., description="The name of the related concept.")
    relation_type: str = Field(..., description="One of: PREREQUISITE_OF, RELATED_TO, PART_OF, CONTRASTS_WITH. PREREQUISITE_OF means the PRIMARY concept must be understood before target_concept.")


class ConceptExtraction(BaseModel):
    primary_concept: str = Field(..., description="The single main concept this chunk is about — a short, canonical name (e.g. 'Vanishing Gradient Problem', not a full sentence).")
    description: str = Field(..., description="A one-sentence description of the primary concept, grounded in this chunk's text.")
    relations: list[ExtractedRelation] = Field(default_factory=list, description="Other concepts this chunk's primary concept relates to, if any are clearly mentioned. Empty list if none.")


EXTRACTION_PROMPT = """You are extracting a concept graph from a machine learning / deep learning \
course chapter. Given one chunk's heading and text, identify:

1. primary_concept: the single main concept this chunk is about, as a SHORT canonical name (e.g. \
"Gradient Descent", "Vanishing Gradient Problem") — not a sentence, not the full heading text.
2. description: one sentence describing it, grounded in this chunk's actual content.
3. relations: any OTHER concepts this chunk clearly relates the primary concept to, with a \
relation_type:
   - PREREQUISITE_OF: primary_concept must be understood BEFORE the target concept
   - RELATED_TO: connected, no clear ordering
   - PART_OF: primary_concept is a sub-topic of the target concept
   - CONTRASTS_WITH: commonly compared/contrasted with the target concept

Only extract relations the text ACTUALLY supports — do not invent connections. Empty relations \
list is correct and expected for many chunks.
"""


def _get_extraction_llm():
    """Fresh ChatVertexAI + with_structured_output() per call — not a
    cached singleton. See generation.py/faithfulness.py's documented
    lesson about async gRPC clients and event loops."""
    llm = ChatVertexAI(
        model_name=settings.VERTEX_GENERATION_MODEL,
        project=settings.GCP_PROJECT_ID,
        location=settings.VERTEX_AI_LOCATION,
        temperature=0.1,  # low temperature: consistent canonical naming across chunks
    )
    return llm.with_structured_output(ConceptExtraction)


def _fetch_all_chunks() -> list[dict]:
    """Read every chunk's id, topic, chapter, and text from Cloud SQL —
    the same 388-chunk corpus every other layer uses."""
    engine = get_engine(ip_type="PUBLIC")
    with engine.connect() as conn:
        rows = conn.execute(sqlalchemy.text(
            "SELECT chunk_id, topic, chapter, text FROM chunks ORDER BY chunk_id;"
        )).mappings().fetchall()
    return [dict(r) for r in rows]


def _fetch_already_processed_chunk_ids(driver) -> set[str]:
    """
    Read the set of chunk_ids already present in ANY concept's
    grounding_chunk_ids — the real, current state of the graph itself,
    not a separate progress file that could drift out of sync. See
    module docstring's REAL GAP FOUND AND FIXED (round 1) for why.
    """
    result = driver.execute_query(
        "MATCH (c:Concept) WHERE c.grounding_chunk_ids IS NOT NULL "
        "UNWIND c.grounding_chunk_ids AS chunk_id RETURN DISTINCT chunk_id",
        database_=settings.NEO4J_DATABASE,
    )
    return {r["chunk_id"] for r in result.records}


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=4, max=30), reraise=True)
def _extract_one_chunk(extractor, topic: str, chapter: str, text: str) -> ConceptExtraction:
    """Extract concepts from one chunk, with retry — see run_ingestion.py's
    documented rate-limit resilience pattern for why retries matter here too."""
    heading = topic or chapter
    prompt = f"Heading: {heading}\n\nText:\n{text}"
    return extractor.invoke([("system", EXTRACTION_PROMPT), ("human", prompt)])


def build_graph(limit: int | None = None) -> dict:
    """
    Run the extraction pipeline: read every chunk NOT ALREADY PROCESSED
    (see module docstring round 1 — real resumability, not just
    idempotent writes), extract its concept + relations, write to Neo4j.
    Safe to interrupt and re-run at any point.

    Args:
        limit: if set, process at most this many NOT-YET-DONE chunks
            this run, then stop cleanly and report real progress. See
            module docstring round 2 for why this exists — every
            attempt to process "everything remaining" in one call kept
            exceeding the remote shell tool's execution-time budget,
            even with resume logic working correctly. Running in small,
            explicit batches (e.g. limit=60) means every single
            invocation completes and reports verifiable progress.
            None (default) attempts everything remaining in one call.

    Returns a summary dict: total_chunks, chunks_skipped_already_done,
    chunks_processed_this_run, concepts_extracted, relations_extracted,
    errors, final_concept_count, final_relationship_count.
    """
    all_chunks = _fetch_all_chunks()
    logger.info("build_graph_start", total_chunk_count=len(all_chunks))

    extractor = _get_extraction_llm()
    driver = graph_store.get_driver()

    already_done = _fetch_already_processed_chunk_ids(driver)
    chunks_to_process = [c for c in all_chunks if c["chunk_id"] not in already_done]
    remaining_before_limit = len(chunks_to_process)
    if limit is not None:
        chunks_to_process = chunks_to_process[:limit]
    logger.info(
        "build_graph_resuming",
        already_done=len(already_done),
        remaining_before_limit=remaining_before_limit,
        processing_this_run=len(chunks_to_process),
    )

    concepts_extracted = 0
    relations_extracted = 0
    errors = 0

    try:
        for i, chunk in enumerate(chunks_to_process):
            try:
                extraction = _extract_one_chunk(extractor, chunk["topic"], chunk["chapter"], chunk["text"])
            except Exception as e:
                logger.info("extraction_failed", chunk_id=chunk["chunk_id"], error=str(e)[:200])
                errors += 1
                continue

            graph_store.upsert_concept(
                driver, extraction.primary_concept, extraction.description, [chunk["chunk_id"]],
            )
            concepts_extracted += 1

            for relation in extraction.relations:
                if relation.relation_type not in graph_store.VALID_RELATION_TYPES:
                    logger.info("skipping_invalid_relation_type", relation_type=relation.relation_type)
                    continue
                graph_store.upsert_relationship(
                    driver, extraction.primary_concept, relation.target_concept, relation.relation_type,
                )
                relations_extracted += 1

            if (i + 1) % 20 == 0:
                logger.info("build_graph_progress", processed=i + 1, batch_total=len(chunks_to_process))

        stats = graph_store.get_graph_stats(driver)
    finally:
        driver.close()

    summary = {
        "total_chunks": len(all_chunks),
        "chunks_skipped_already_done": len(already_done),
        "remaining_before_this_run": remaining_before_limit,
        "chunks_processed_this_run": len(chunks_to_process),
        "concepts_extracted_calls": concepts_extracted,
        "relations_extracted_calls": relations_extracted,
        "errors": errors,
        "final_concept_count": stats["concept_count"],
        "final_relationship_count": stats["relationship_count"],
    }
    logger.info("build_graph_done", **summary)
    return summary


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Cognara Learn concept graph extraction pipeline")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process at most this many not-yet-done chunks, then stop. Omit to attempt everything remaining.",
    )
    args = parser.parse_args()

    summary = build_graph(limit=args.limit)

    print("\n=== Cognara concept graph build — batch complete ===")
    print(f"Total chunks in corpus       : {summary['total_chunks']}")
    print(f"Already done (skipped)       : {summary['chunks_skipped_already_done']}")
    print(f"Remaining before this run    : {summary['remaining_before_this_run']}")
    print(f"Processed this run           : {summary['chunks_processed_this_run']}")
    print(f"Concept extraction calls     : {summary['concepts_extracted_calls']}")
    print(f"Relation extraction calls    : {summary['relations_extracted_calls']}")
    print(f"Errors                       : {summary['errors']}")
    print(f"Final concept count          : {summary['final_concept_count']}")
    print(f"Final relationship count     : {summary['final_relationship_count']}")
    still_remaining = summary["remaining_before_this_run"] - summary["chunks_processed_this_run"]
    if still_remaining > 0:
        print(f"\n{still_remaining} chunks still remaining — re-run to continue.")
    else:
        print("\nAll chunks processed.")
    print()


if __name__ == "__main__":
    main()
