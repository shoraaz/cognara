"""
app/retrieval/vector_store.py
------------------------------
CognaraPGVectorStore: a LangChain-compatible VectorStore backed by our
own hand-designed `chunks` table (see ingestion/pipelines/init_db.py),
instead of LangChain's auto-managed langchain_pg_embedding schema.

WHY THIS FILE EXISTS:
  The vector store is where embedded chunks live and get searched. This
  module is the only place in the codebase that runs SQL against the chunks
  table. Everything else calls it through LangChain's standard VectorStore
  interface (add_documents, similarity_search_with_score), so it's a fully
  compatible citizen of the LangChain ecosystem even though the schema
  underneath is our own.

WHY NOT langchain_postgres.PGVector (see ADR 0003, ADR 0004):
  PGVector is the obvious off-the-shelf choice — it auto-creates its own
  tables and needs zero schema code. We deliberately did NOT use it because
  its default schema stores all metadata (course_name, page_number, chapter,
  etc.) inside one JSONB column (cmetadata), not as real typed SQL columns.
  That directly undercuts ADR 0003's reason for choosing pgvector — "native
  metadata filtering: course_name / chapter filters are plain SQL WHERE
  clauses" — and makes the data harder to query from outside LangChain's
  Python client (e.g. a future Vercel/TypeScript serverless function reading
  this table directly, which is part of our planned deployment path).

  We instead extend LangChain's VectorStore base class ourselves, keeping
  our own typed `chunks` table as the source of truth. More code than
  configuring PGVector, but: (1) schema stays plain typed SQL any language
  can query directly, (2) the store is still a fully compatible LangChain
  VectorStore, (3) Module 3's schema work remains the real implementation.

LANGCHAIN INTERFACE CONTRACT:
  Two truly abstract methods: from_texts() and similarity_search(). We
  implement add_documents() directly (chunks arrive as Documents with full
  citation metadata — the natural fit) and similarity_search_with_score()
  (citation feature depends on real relevance scores, not just a list).

HOW SIMILARITY WORKS:
  pgvector's cosine distance operator is <=>. We use:
      SELECT ... ORDER BY embedding <=> :query_vec LIMIT :k
  relevance_score returned to callers is (1 - cosine_distance), so higher
  = more similar, matching the 0..1 convention in schemas.Citation.

# Interview notes: local-notes/INTERVIEW_PREP.md — "app/retrieval/vector_store.py"
"""

from typing import Any

import sqlalchemy
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from app.core.config import settings
from app.core.logging import get_logger
from ingestion.pipelines.init_db import get_engine

logger = get_logger(__name__)

# Columns on the `chunks` table that are NOT part of Document.metadata —
# they're either the primary key, the text content itself, or the embedding
# vector. All other columns map 1:1 to metadata fields.
_CORE_COLUMNS = {"chunk_id", "text", "embedding"}


class CognaraPGVectorStore(VectorStore):
    """
    LangChain VectorStore backed by the hand-designed `chunks` table in
    Cloud SQL (see ingestion/pipelines/init_db.py). See module docstring
    for why this isn't langchain_postgres.PGVector.
    """

    def __init__(self, embeddings: Embeddings, engine: sqlalchemy.engine.Engine | None = None) -> None:
        """
        Args:
            embeddings: A LangChain Embeddings instance used to embed both
                ingested chunk text and incoming search queries. The SAME model
                must be used for both — a mismatch silently produces wrong
                similarity scores because vectors from different models live in
                incomparable spaces.
            engine: Optional pre-built SQLAlchemy engine. If omitted, one is
                created via ingestion.pipelines.init_db.get_engine()
                (Cloud SQL Python Connector, ip_type="PUBLIC" — required from
                outside the VPC; see ADR 0003 amendment and init_db.py).
        """
        self.embeddings = embeddings
        self.engine = engine or get_engine(ip_type="PUBLIC")

    @property
    def embeddings(self) -> Embeddings:
        return self._embeddings

    @embeddings.setter
    def embeddings(self, value: Embeddings) -> None:
        self._embeddings = value

    # ── Writing ──────────────────────────────────────────────────────────────

    def add_documents(self, documents: list[Document], **kwargs: Any) -> list[str]:
        """
        Embed and upsert a batch of chunk Documents into the chunks table.
        Each Document's metadata must already contain every column the table
        requires (course_name, subject, chapter, page_number, source_type,
        document_version, ingestion_date — see CognaraHeadingSplitter, which
        produces exactly this shape) plus a unique chunk_id.

        The INSERT ... ON CONFLICT DO UPDATE (upsert) pattern allows re-running
        ingestion without manual cleanup — existing chunks are overwritten with
        their latest content and embeddings if the chunk_id already exists.

        Returns the list of chunk_ids written (LangChain's VectorStore
        contract: add_documents returns a list of IDs).
        """
        if not documents:
            return []

        texts = [d.page_content for d in documents]
        logger.info("embedding_batch", count=len(texts))
        vectors = self.embeddings.embed_documents(texts)

        ids: list[str] = []
        with self.engine.connect() as conn:
            for doc, vector in zip(documents, vectors):
                meta = doc.metadata
                chunk_id = meta["chunk_id"]
                ids.append(chunk_id)
                conn.execute(
                    sqlalchemy.text("""
                        INSERT INTO chunks (
                            chunk_id, text, embedding, course_name, subject,
                            chapter, topic, page_number, page_range,
                            source_type, document_version, ingestion_date,
                            chunk_index_in_doc, char_count
                        ) VALUES (
                            :chunk_id, :text, :embedding, :course_name, :subject,
                            :chapter, :topic, :page_number, :page_range,
                            :source_type, :document_version, :ingestion_date,
                            :chunk_index_in_doc, :char_count
                        )
                        ON CONFLICT (chunk_id) DO UPDATE SET
                            text = EXCLUDED.text,
                            embedding = EXCLUDED.embedding,
                            course_name = EXCLUDED.course_name,
                            subject = EXCLUDED.subject,
                            chapter = EXCLUDED.chapter,
                            topic = EXCLUDED.topic,
                            page_number = EXCLUDED.page_number,
                            page_range = EXCLUDED.page_range,
                            source_type = EXCLUDED.source_type,
                            document_version = EXCLUDED.document_version,
                            ingestion_date = EXCLUDED.ingestion_date,
                            chunk_index_in_doc = EXCLUDED.chunk_index_in_doc,
                            char_count = EXCLUDED.char_count;
                    """),
                    {
                        "chunk_id":          chunk_id,
                        "text":              doc.page_content,
                        # pgvector accepts a string literal like "[0.1,0.2,...]"
                        "embedding":         str(vector),
                        "course_name":       meta.get("course_name"),
                        "subject":           meta.get("subject"),
                        "chapter":           meta.get("chapter"),
                        "topic":             meta.get("topic"),
                        "page_number":       meta.get("page_number"),
                        "page_range":        meta.get("page_range"),
                        "source_type":       meta.get("source_type"),
                        "document_version":  meta.get("document_version"),
                        "ingestion_date":    meta.get("ingestion_date"),
                        "chunk_index_in_doc": meta.get("chunk_index_in_doc"),
                        "char_count":        meta.get("char_count"),
                    },
                )
            conn.commit()

        logger.info("upsert_done", chunks_written=len(ids))
        return ids

    # ── Reading ───────────────────────────────────────────────────────────────

    def similarity_search(self, query: str, k: int = 4, **kwargs: Any) -> list[Document]:
        """Required by VectorStore's abstract interface. Delegates to
        similarity_search_with_score() and strips the scores."""
        results = self.similarity_search_with_score(query, k=k, **kwargs)
        return [doc for doc, _score in results]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        course_filter: str | None = None,
        chapter_filter: str | None = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """
        Embed `query` and return the top-k most similar chunks with their
        relevance scores, optionally restricted by course_name/chapter.

        relevance_score = 1 - cosine_distance, in 0..1 (1 = identical,
        0 = completely unrelated) — matches app.models.schemas.Citation.
        """
        query_vector = self.embeddings.embed_query(query)

        # Build the WHERE clause dynamically from whichever filters are set.
        # Parameters are bound separately (never interpolated into the SQL
        # string directly) to prevent SQL injection.
        where_clauses = []
        params: dict[str, Any] = {"query_vector": str(query_vector), "k": k}
        if course_filter:
            where_clauses.append("course_name = :course_filter")
            params["course_filter"] = course_filter
        if chapter_filter:
            where_clauses.append("chapter = :chapter_filter")
            params["chapter_filter"] = chapter_filter
        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        # <=> is pgvector's cosine distance operator (0 = identical, 2 = opposite).
        # We ORDER BY distance ASC (most similar first) and return
        # 1 - distance as the relevance_score so higher = more similar.
        sql = sqlalchemy.text(f"""
            SELECT
                chunk_id, text, course_name, subject, chapter, topic,
                page_number, page_range, source_type, document_version,
                ingestion_date, chunk_index_in_doc, char_count,
                1 - (embedding <=> :query_vector) AS relevance_score
            FROM chunks
            {where_sql}
            ORDER BY embedding <=> :query_vector
            LIMIT :k;
        """)

        with self.engine.connect() as conn:
            rows = conn.execute(sql, params).mappings().fetchall()

        results: list[tuple[Document, float]] = []
        for row in rows:
            metadata = {
                "chunk_id":          row["chunk_id"],
                "course_name":       row["course_name"],
                "subject":           row["subject"],
                "chapter":           row["chapter"],
                "topic":             row["topic"],
                "page_number":       row["page_number"],
                "page_range":        row["page_range"],
                "source_type":       row["source_type"],
                "document_version":  row["document_version"],
                # ingestion_date comes back as a date object from Postgres;
                # convert to ISO string so it's JSON-serialisable downstream.
                "ingestion_date":    str(row["ingestion_date"]) if row["ingestion_date"] else None,
                "chunk_index_in_doc": row["chunk_index_in_doc"],
                "char_count":        row["char_count"],
            }
            doc = Document(page_content=row["text"], metadata=metadata)
            results.append((doc, float(row["relevance_score"])))

        return results

    # ── Required factory method (LangChain VectorStore contract) ─────────────

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> "CognaraPGVectorStore":
        """
        Required by VectorStore's abstract interface. Not the primary way
        chunks reach this store in Cognara's pipeline (Module 4 uses
        add_documents directly on chunks that already carry full citation
        metadata from CognaraHeadingSplitter). Provided so this class is a
        structurally complete VectorStore for any generic LangChain code that
        only knows the from_texts(texts, embedding) contract.
        """
        store = cls(embeddings=embedding)
        documents = [
            Document(page_content=text, metadata=(metadatas[i] if metadatas else {}))
            for i, text in enumerate(texts)
        ]
        store.add_documents(documents)
        return store
