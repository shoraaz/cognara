"""
tests/integration/test_init_db.py
-----------------------------------
Tests for ingestion/pipelines/init_db.py.

WHY THIS IS AN INTEGRATION TEST, NOT A UNIT TEST:
  Unlike the parser and chunker (Modules 1-2), this module has no meaning
  without a real network call to a real Cloud SQL instance — there is no
  pure-function logic to isolate and test offline. These tests connect to
  the actual cognara-pg instance via the same Cloud SQL Python Connector
  the module itself uses, and verify real, observable database state.

REQUIREMENTS TO RUN THESE TESTS:
  - cognara-pg must be running (`make db-start` / activation-policy=ALWAYS)
  - `gcloud auth application-default login` must have been run at least
    once on this machine (Application Default Credentials)
  - GCP_PROJECT_ID / CLOUD_SQL_INSTANCE / PGVECTOR_* must be set in .env

  Since none of these can be assumed true in every environment (e.g. a CI
  runner with no GCP credentials at all), every test here is guarded by
  a module-level connectivity probe, exactly like the requires_ml_pdf /
  requires_dl_pdf skip guards used in test_pdf_parser.py — if the DB
  isn't reachable, these tests SKIP with a clear reason rather than
  failing or hanging on a network timeout.
"""

import pytest
import sqlalchemy

from ingestion.pipelines.init_db import get_engine, run_ddl, verify_schema


def _db_is_reachable() -> bool:
    """Best-effort connectivity probe, used only to decide whether to skip."""
    try:
        engine = get_engine(ip_type="PUBLIC")
        with engine.connect() as conn:
            conn.execute(sqlalchemy.text("SELECT 1;"))
        engine.dispose()
        return True
    except Exception:
        return False


DB_REACHABLE = _db_is_reachable()
requires_db = pytest.mark.skipif(
    not DB_REACHABLE,
    reason="cognara-pg not reachable — is the instance started and are ADC credentials set up?",
)


class TestSchemaBootstrap:
    @requires_db
    def test_run_ddl_then_verify_reports_expected_shape(self):
        engine = get_engine(ip_type="PUBLIC")
        try:
            run_ddl(engine)
            summary = verify_schema(engine)
        finally:
            engine.dispose()

        assert summary["pgvector_installed"] is True
        assert summary["chunks_table_exists"] is True
        assert "chunks_embedding_hnsw" in summary["indexes"]
        assert "chunks_course_chapter" in summary["indexes"]
        assert "chunks_pkey" in summary["indexes"]
        assert isinstance(summary["row_count"], int)

    @requires_db
    def test_running_ddl_twice_is_idempotent(self):
        """
        The whole point of `IF NOT EXISTS` in every DDL statement: running
        the bootstrap a second time must not raise, and must report the
        exact same schema shape as the first run.
        """
        engine = get_engine(ip_type="PUBLIC")
        try:
            run_ddl(engine)
            first = verify_schema(engine)
            run_ddl(engine)  # second run — must not raise
            second = verify_schema(engine)
        finally:
            engine.dispose()

        assert first["indexes"] == second["indexes"]
        assert first["chunks_table_exists"] == second["chunks_table_exists"] is True

    @requires_db
    def test_chunks_table_has_expected_columns(self):
        engine = get_engine(ip_type="PUBLIC")
        try:
            run_ddl(engine)
            with engine.connect() as conn:
                rows = conn.execute(sqlalchemy.text(
                    "SELECT column_name, data_type, is_nullable "
                    "FROM information_schema.columns "
                    "WHERE table_name = 'chunks';"
                )).fetchall()
        finally:
            engine.dispose()

        columns = {r[0]: r for r in rows}
        expected_not_null = {
            "chunk_id", "text", "embedding", "course_name", "subject",
            "chapter", "page_number", "source_type", "document_version",
            "ingestion_date",
        }
        expected_nullable = {"topic", "page_range", "chunk_index_in_doc", "char_count"}

        for col in expected_not_null:
            assert col in columns, f"expected column {col!r} missing from chunks table"
            assert columns[col][2] == "NO", f"{col!r} should be NOT NULL"

        for col in expected_nullable:
            assert col in columns, f"expected nullable column {col!r} missing"
            assert columns[col][2] == "YES", f"{col!r} should allow NULL"

    @requires_db
    def test_embedding_column_is_vector_768(self):
        """
        Confirms the embedding column is actually pgvector's vector type
        with dimension 768 — matching text-embedding-004's real output
        size (see app/core/config.py EMBEDDING_DIM). A dimension mismatch
        here would silently break every future embedding insert.
        """
        engine = get_engine(ip_type="PUBLIC")
        try:
            run_ddl(engine)
            with engine.connect() as conn:
                row = conn.execute(sqlalchemy.text(
                    "SELECT atttypmod FROM pg_attribute "
                    "JOIN pg_class ON pg_attribute.attrelid = pg_class.oid "
                    "WHERE pg_class.relname = 'chunks' AND attname = 'embedding';"
                )).fetchone()
        finally:
            engine.dispose()

        assert row is not None
        assert row[0] == 768, f"expected embedding dimension 768, got {row[0]}"


class TestConnectionErrorHandling:
    def test_private_ip_type_fails_from_outside_vpc(self):
        """
        Documents and verifies the real networking lesson from this
        module's development (see the module docstring in init_db.py):
        ip_type='PRIVATE' cannot succeed from a machine outside the VPC,
        which this test-running machine is. This is expected, correct
        behaviour, not a bug — the test exists so a future change that
        accidentally "fixes" this by silently falling back to PUBLIC
        would be caught, instead of quietly masking the distinction.
        """
        engine = get_engine(ip_type="PRIVATE")
        try:
            with pytest.raises(Exception):
                with engine.connect():
                    pass
        finally:
            engine.dispose()
