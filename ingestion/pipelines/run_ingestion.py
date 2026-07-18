"""
ingestion/pipelines/run_ingestion.py
--------------------------------------
CLI entry point: ingest the Phase 1 corpus into the vector store.

WHY THIS FILE EXISTS:
  Ingestion is a one-time (or occasional) offline job, not a web request.
  Run from the command line: `make ingest PDF_DIR=data/raw_pdfs`
  This pipeline composes every module built so far:
  catalog CSV -> CognaraPDFLoader (Module 1) -> CognaraHeadingSplitter
  (Module 2) -> embeddings (Module 4a) -> CognaraPGVectorStore (Module 4b)
  writing into the chunks table (Module 3).

EXECUTION FLOW:
  1. Read data/catalog/document_catalog_v1.csv — one row per chapter,
     with course_name, subject, chapter, page_start, page_end,
     source_type, document_version, pdf_filename.
  2. For each row: CognaraPDFLoader(pdf_path, start_page, end_page).load()
     -> list[Document], one per page.
  3. Attach the row's catalog metadata onto every loaded Document
     (course_name, subject, chapter, source_type, document_version) —
     CognaraPDFLoader itself only sets page_number/source; the catalog
     metadata is the ingestion pipeline's responsibility to attach.
  4. CognaraHeadingSplitter().split_documents(docs) -> list[Document],
     one per chunk, with citation metadata attached.
  5. CognaraPGVectorStore.add_documents(chunks) — embeds each chunk's
     text and upserts into Cloud SQL. Chunks are embedded in small retried
     batches to avoid Vertex AI's rate limit (429 RESOURCE_EXHAUSTED).
  6. Log a summary: chapters, chunks, elapsed time.

RATE LIMITING:
  Vertex AI's default embedding quota is hit when many calls are made in
  rapid succession — reproduced twice during testing with a 30-45s recovery
  window. The pipeline embeds in small batches (EMBED_BATCH_SIZE=10) with
  tenacity exponential backoff (4s, 8s, 16s, 32s, 60s cap), comfortably
  longer than the observed recovery window.
  See BUG_FIX_LOG.md "Ingestion: Vertex AI embedding quota exceeded".

# Interview notes: local-notes/INTERVIEW_PREP.md — "ingestion/pipelines/run_ingestion.py"
"""

import argparse
import csv
import time
from pathlib import Path

from langchain_core.documents import Document
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.logging import get_logger, setup_logging
from app.retrieval.embedder import get_embeddings
from app.retrieval.vector_store import CognaraPGVectorStore
from ingestion.chunking.chunker import CognaraHeadingSplitter
from ingestion.parsers.pdf_parser import CognaraPDFLoader

logger = get_logger(__name__)

# Default catalog path — can be overridden via the catalog_path argument to
# ingest_catalog() for testing with alternate catalogs.
CATALOG_PATH = Path("data/catalog/document_catalog_v1.csv")

# Sized against the real 429 RESOURCE_EXHAUSTED error (confirmed twice).
# Small enough that one batch failing and retrying doesn't re-embed a large
# amount of already-successful work.
# See BUG_FIX_LOG.md "Ingestion: Vertex AI embedding quota exceeded".
EMBED_BATCH_SIZE = 10

# Import the specific error class tenacity should catch for rate-limit retries.
# Wrapped in try/except so a package-layout change doesn't crash the whole module.
try:
    from langchain_google_genai._common import GoogleGenerativeAIError as _EmbeddingError
except ImportError:  # pragma: no cover - defensive, package layout may shift
    _EmbeddingError = Exception


def _read_catalog(catalog_path: Path) -> list[dict]:
    """Read the document catalog CSV into a list of row dicts."""
    with open(catalog_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _load_chapter_documents(row: dict, pdf_dir: Path) -> list[Document]:
    """
    Load every page in one catalog row's page range and attach the row's
    shared catalog metadata onto every loaded Document.

    CognaraPDFLoader sets only page_number and source on each Document;
    the catalog metadata (course_name, subject, chapter, source_type,
    document_version) is the ingestion pipeline's responsibility to attach,
    as documented in CognaraHeadingSplitter's module docstring.
    """
    pdf_path = pdf_dir / row["pdf_filename"]
    loader = CognaraPDFLoader(
        pdf_path,
        start_page=int(row["page_start"]),
        end_page=int(row["page_end"]),
    )
    docs = loader.load()

    # Shared catalog metadata is identical for every page in this chapter.
    # We update each Document's metadata dict in place (the loader's metadata
    # is a fresh dict per Document, so this doesn't mutate any shared state).
    shared_metadata = {
        "course_name":      row["course_name"],
        "subject":          row["subject"],
        "chapter":          row["chapter"],
        "source_type":      row["source_type"],
        "document_version": row["document_version"],
    }
    for doc in docs:
        doc.metadata.update(shared_metadata)

    return docs


@retry(
    retry=retry_if_exception_type(_EmbeddingError),
    stop=stop_after_attempt(5),
    # Exponential backoff: 4s, 8s, 16s, 32s, 60s (capped).
    # Comfortably longer than the 30-45s recovery window observed for the
    # real 429 RESOURCE_EXHAUSTED quota error.
    wait=wait_exponential(multiplier=2, min=4, max=60),
    reraise=True,  # re-raise the original exception after all retries are exhausted
)
def _embed_and_store_batch(store: CognaraPGVectorStore, batch: list[Document]) -> list[str]:
    """
    Embed and upsert one small batch of chunks, retrying with exponential
    backoff specifically on the Vertex AI rate-limit error class.
    See BUG_FIX_LOG.md "Ingestion: Vertex AI embedding quota exceeded".
    """
    return store.add_documents(batch)


def _batched(items: list, size: int):
    """Yield successive slices of `items` of length `size`."""
    for i in range(0, len(items), size):
        yield items[i : i + size]


def ingest_catalog(
    catalog_path: Path = CATALOG_PATH,
    pdf_dir: Path = Path("data/raw_pdfs"),
    dry_run: bool = False,
) -> dict:
    """
    Run the full ingestion pipeline over every row in the catalog CSV.

    Args:
        catalog_path: Path to the document catalog CSV.
        pdf_dir: Directory containing the source PDFs referenced by the
            catalog's pdf_filename column.
        dry_run: If True, load and chunk every chapter (proving the catalog
            and page ranges are valid) but skip embedding and writing to
            Cloud SQL entirely. Useful for validating a catalog edit without
            spending any embedding cost.

    Returns:
        Summary dict: chapters_processed, total_chunks,
        chunks_written (0 if dry_run), elapsed_seconds.
    """
    start_time = time.monotonic()
    rows = _read_catalog(catalog_path)
    logger.info("ingestion_start", catalog=str(catalog_path), chapters=len(rows), dry_run=dry_run)

    # Only construct the vector store if we're actually writing — avoids a
    # Cloud SQL connection attempt on dry runs.
    store = None if dry_run else CognaraPGVectorStore(embeddings=get_embeddings())

    total_chunks = 0
    chunks_written = 0
    # Reuse the same splitter instance across all chapters — it's stateless.
    splitter = CognaraHeadingSplitter()

    for row in rows:
        # Load all pages for this catalog row, then split into heading-aware chunks.
        chapter_docs = _load_chapter_documents(row, pdf_dir)
        chunks = splitter.split_documents(chapter_docs)
        total_chunks += len(chunks)

        logger.info(
            "chapter_chunked",
            course_name=row["course_name"],
            chapter=row["chapter"],
            pages=f"{row['page_start']}-{row['page_end']}",
            chunk_count=len(chunks),
        )

        if dry_run:
            # Skip embedding and writing — everything above (load + chunk) is
            # still validated, which is the point of a dry run.
            continue

        # Embed and store in small batches with exponential backoff on rate-limit
        # errors. Each batch is committed independently — a batch failure only
        # requires retrying that batch, not the whole chapter.
        for batch in _batched(chunks, EMBED_BATCH_SIZE):
            ids = _embed_and_store_batch(store, batch)
            chunks_written += len(ids)

    elapsed = time.monotonic() - start_time
    summary = {
        "chapters_processed": len(rows),
        "total_chunks":       total_chunks,
        "chunks_written":     chunks_written,
        "elapsed_seconds":    round(elapsed, 1),
    }
    logger.info("ingestion_done", **summary)
    return summary


def main(pdf_dir: str, dry_run: bool = False) -> None:
    setup_logging()
    summary = ingest_catalog(pdf_dir=Path(pdf_dir), dry_run=dry_run)

    print("\n=== Cognara ingestion complete ===")
    print(f"Chapters processed : {summary['chapters_processed']}")
    print(f"Total chunks       : {summary['total_chunks']}")
    print(f"Chunks written     : {summary['chunks_written']}"
          + (" (dry run — nothing embedded or written)" if dry_run else ""))
    print(f"Elapsed            : {summary['elapsed_seconds']}s")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cognara Learn ingestion pipeline")
    parser.add_argument("--pdf-dir", required=True, help="Directory containing PDFs to ingest")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Load and chunk every chapter but skip embedding/writing — validates the catalog for free",
    )
    args = parser.parse_args()
    main(args.pdf_dir, dry_run=args.dry_run)
