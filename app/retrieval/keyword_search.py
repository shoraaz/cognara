"""
app/retrieval/keyword_search.py
---------------------------------
Real BM25 keyword search over the corpus, for exact-term queries that
pure vector similarity search can under-serve.

WHY THIS FILE EXISTS (see ADR 0005):
  Vector search finds chunks by MEANING — it's excellent for concept
  questions ("explain overfitting") but can under-rank chunks for
  exact-term questions, where the literal word IS the answer: acronyms
  ("ReLU", "AdaBoost"), formula-adjacent terms ("cross-entropy loss"),
  or algorithm names. BM25 is the standard keyword-relevance algorithm:
  it scores a document highly when it contains the query's exact terms,
  weighted by how rare those terms are across the whole corpus (a
  document matching "ReLU" — a rare term — scores much higher than one
  matching "the" — extremely common, low signal).

WHY rank_bm25, NOT Postgres full-text search (a real, deliberate
trade — see ADR 0005 for the full alternatives comparison):
  Postgres's built-in ts_rank would need zero new infrastructure, but is
  NOT the actual BM25 formula — it's Postgres's own TF-based ranking,
  BM25-family but mathematically different. rank_bm25's BM25Okapi is the
  real, standard algorithm. The real cost of that choice: this index is
  built IN MEMORY, from a full read of the chunks table, and must be
  rebuilt whenever the process starts or the corpus changes — acceptable
  at our current scale (388 chunks rebuilds near-instantly) but a real
  scaling limit to revisit if the corpus grows into the tens of
  thousands of chunks.

WHERE IT FITS:
  This module is called by hybrid_search.py (Layer 2's fusion module),
  which merges BM25 results with CognaraPGVectorStore's vector results
  using Reciprocal Rank Fusion — the two scores are NOT on comparable
  scales (BM25 scores are unbounded; cosine similarity is 0..1), so they
  cannot be combined directly.

INTERVIEW EXPLANATION:
  "I built real BM25 keyword search using rank_bm25 rather than
  Postgres's built-in text search, because I wanted an actual
  hybrid-retrieval implementation I could reason about precisely, not an
  approximation. The trade-off is a real one: this index lives in memory
  and rebuilds from the database on first use, which doesn't scale
  cleanly to a huge corpus — but at 388 chunks that rebuild is
  effectively instant, and I've documented exactly when I'd need to
  revisit that choice."
"""

import re
from dataclasses import dataclass

import sqlalchemy
from rank_bm25 import BM25Okapi

from app.core.logging import get_logger
from ingestion.pipelines.init_db import get_engine

logger = get_logger(__name__)

# Simple, fast tokenizer: lowercase, split on non-alphanumeric runs.
# Deliberately basic — BM25's quality comes from term-frequency /
# document-frequency statistics, not from sophisticated tokenization.
# Real acronyms like "ReLU" and "AdaBoost" survive this fine (they
# become "relu", "adaboost" — matched consistently since queries go
# through the exact same tokenizer).
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class KeywordSearchResult:
    chunk_id: str
    text: str
    metadata: dict
    bm25_score: float


class BM25KeywordIndex:
    """
    In-memory BM25 index over the chunks table. Built once (lazily, on
    first search) and cached — rebuild explicitly with refresh() if the
    corpus changes after the index was built (e.g. after a new
    ingestion run in a long-lived process).
    """

    def __init__(self, engine: sqlalchemy.engine.Engine | None = None):
        self.engine = engine or get_engine(ip_type="PUBLIC")
        self._bm25: BM25Okapi | None = None
        self._chunk_rows: list[dict] = []

    def refresh(self) -> int:
        """
        Load every chunk from Cloud SQL and build a fresh in-memory BM25
        index. Returns the number of chunks indexed. Called automatically
        on first search(); call directly to force a rebuild after new
        ingestion.
        """
        with self.engine.connect() as conn:
            rows = conn.execute(sqlalchemy.text("""
                SELECT chunk_id, text, course_name, subject, chapter, topic,
                       page_number, page_range, source_type, document_version,
                       ingestion_date, chunk_index_in_doc, char_count
                FROM chunks;
            """)).mappings().fetchall()

        self._chunk_rows = [dict(r) for r in rows]
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
        Return the top-k chunks by BM25 score for `query`. Metadata
        filters are applied in Python, after scoring — a direct
        consequence of using an in-memory index instead of Postgres
        (see module docstring). For our corpus size this is fast; at
        much larger scale, filtering before scoring would matter more.
        """
        if self._bm25 is None:
            self.refresh()

        query_tokens = _tokenize(query)
        scores = self._bm25.get_scores(query_tokens)

        scored_rows = list(zip(self._chunk_rows, scores))

        if course_filter:
            scored_rows = [(r, s) for r, s in scored_rows if r["course_name"] == course_filter]
        if chapter_filter:
            scored_rows = [(r, s) for r, s in scored_rows if r["chapter"] == chapter_filter]

        scored_rows.sort(key=lambda pair: pair[1], reverse=True)

        results = []
        for row, score in scored_rows[:k]:
            if score <= 0:
                continue  # BM25 score of 0 means no query terms matched at all
            metadata = {k_: row[k_] for k_ in row if k_ not in ("text",)}
            results.append(KeywordSearchResult(
                chunk_id=row["chunk_id"],
                text=row["text"],
                metadata=metadata,
                bm25_score=float(score),
            ))
        return results
