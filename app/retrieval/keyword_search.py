"""
app/retrieval/keyword_search.py
--------------------------------
Real BM25 keyword search over the corpus, for exact-term queries that
pure vector similarity search can under-serve.

WHY THIS FILE EXISTS (see ADR 0005):
  Vector search finds chunks by MEANING — it's excellent for concept
  questions ("explain overfitting") but can under-rank chunks for
  exact-term queries where the literal word IS the answer: acronyms
  ("ReLU", "AdaBoost"), formula-adjacent terms ("cross-entropy loss"),
  or algorithm names. BM25 scores a document highly when it contains the
  query's exact terms, weighted by how rare those terms are across the
  whole corpus.

WHY rank_bm25, NOT Postgres full-text search (see ADR 0005):
  Postgres's ts_rank is NOT the actual BM25 formula — it's Postgres's own
  TF-based ranking, BM25-family but mathematically different. rank_bm25's
  BM25Okapi is the real, standard algorithm. The real cost of this choice:
  the index is built IN MEMORY from a full read of the chunks table and must
  be rebuilt whenever the process starts or the corpus changes. Acceptable at
  our current scale (388 chunks rebuilds near-instantly) but a real scaling
  limit to revisit if the corpus grows into the tens of thousands of chunks.

WHERE IT FITS:
  This module is called by hybrid_search.py (Layer 2's fusion module), which
  merges BM25 results with CognaraPGVectorStore's vector results using RRF.
  BM25 scores are unbounded; cosine similarity is 0..1 — they cannot be
  combined directly, which is why RRF (rank-based, not score-based) is used.

# Interview notes: local-notes/INTERVIEW_PREP.md — "app/retrieval/keyword_search.py"
"""

import re
from dataclasses import dataclass

import sqlalchemy
from rank_bm25 import BM25Okapi

from app.core.logging import get_logger
from ingestion.pipelines.init_db import get_engine

logger = get_logger(__name__)

# Simple, fast tokenizer: lowercase and split on non-alphanumeric runs.
# Deliberately basic — BM25's quality comes from term-frequency /
# document-frequency statistics, not from sophisticated tokenization.
# Acronyms like "ReLU" and "AdaBoost" survive this fine (they become
# "relu", "adaboost") because queries go through the exact same tokenizer,
# ensuring consistent matching.
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase and split text into BM25-compatible tokens."""
    return _TOKEN_RE.findall(text.lower())


@dataclass
class KeywordSearchResult:
    """One BM25 search result, parallel to vector_store's (Document, score) tuple."""
    chunk_id: str
    text: str
    metadata: dict
    bm25_score: float


class BM25KeywordIndex:
    """
    In-memory BM25 index over the chunks table. Built lazily on first search
    and cached for the process lifetime. Call refresh() to force a rebuild
    after new ingestion in a long-lived process.
    """

    def __init__(self, engine: sqlalchemy.engine.Engine | None = None):
        # Allow injecting a pre-built engine (useful in tests); otherwise
        # use the standard Cloud SQL Connector engine.
        self.engine = engine or get_engine(ip_type="PUBLIC")
        self._bm25: BM25Okapi | None = None
        self._chunk_rows: list[dict] = []

    def refresh(self) -> int:
        """
        Load every chunk from Cloud SQL and build a fresh in-memory BM25
        index. Returns the number of chunks indexed.

        Called automatically on first search(); call directly to force a
        rebuild after new ingestion in a long-lived process (e.g. a running
        API server that doesn't restart between ingestion runs).
        """
        with self.engine.connect() as conn:
            rows = conn.execute(sqlalchemy.text("""
                SELECT chunk_id, text, course_name, subject, chapter, topic,
                       page_number, page_range, source_type, document_version,
                       ingestion_date, chunk_index_in_doc, char_count
                FROM chunks;
            """)).mappings().fetchall()

        self._chunk_rows = [dict(r) for r in rows]
        # Tokenize every chunk's text upfront so BM25Okapi can build its
        # term-frequency / document-frequency statistics across the whole corpus.
        tokenized_corpus = [_tokenize(r["text"]) for r in self._chunk_rows]
        self._bm25 = BM25Okapi(tokenized_corpus)

        logger.info("bm25_index_built", chunk_count=len(self._chunk_rows))
        return len(self._chunk_rows)

    def search(
        self,
        query: str,
        k: int = 10,
        course_filter: str | None = None,
        chapter_filter: str | None = None,
    ) -> list[KeywordSearchResult]:
        """
        Return the top-k chunks by BM25 score for `query`.

        Filters (course_filter, chapter_filter) are applied in Python after
        scoring — a direct consequence of using an in-memory index instead of
        Postgres. For our corpus size this is fast; at much larger scale,
        filtering before scoring (or sharding the index) would matter more.

        Chunks with a BM25 score of exactly 0 are excluded — a score of 0
        means none of the query tokens appeared in that chunk at all, so it
        has no keyword-based evidence for relevance.
        """
        # Build index on first use — avoids an expensive full-table read
        # at import time when the module is loaded but not yet searched.
        if self._bm25 is None:
            self.refresh()

        query_tokens = _tokenize(query)
        # get_scores() returns one BM25 score per chunk, in the same order
        # as _chunk_rows (the corpus order from the SELECT above).
        scores = self._bm25.get_scores(query_tokens)

        # Zip rows with their scores so we can filter and sort together.
        scored_rows = list(zip(self._chunk_rows, scores))

        # Apply metadata filters in Python — O(n) scan over all chunks,
        # acceptable at our current corpus size.
        if course_filter:
            scored_rows = [(r, s) for r, s in scored_rows if r["course_name"] == course_filter]
        if chapter_filter:
            scored_rows = [(r, s) for r, s in scored_rows if r["chapter"] == chapter_filter]

        # Sort descending by BM25 score so the most relevant chunks come first.
        scored_rows.sort(key=lambda pair: pair[1], reverse=True)

        results = []
        for row, score in scored_rows[:k]:
            if score <= 0:
                # Score of 0 means no query token appeared in the chunk — no
                # keyword evidence at all, so exclude from results.
                continue
            # Build metadata dict from all row columns except text (which is
            # stored separately in KeywordSearchResult.text for clarity).
            metadata = {k_: row[k_] for k_ in row if k_ not in ("text",)}
            results.append(KeywordSearchResult(
                chunk_id=row["chunk_id"],
                text=row["text"],
                metadata=metadata,
                bm25_score=float(score),
            ))
        return results
