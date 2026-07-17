"""
tests/unit/test_pdf_parser.py
------------------------------
Tests for ingestion/parsers/pdf_parser.py (CognaraPDFLoader).

WHY THIS FILE EXISTS:
  The loader is the first step of the whole ingestion pipeline. If page
  numbers are wrong here, every citation downstream is wrong. These tests
  run against the REAL Phase 1 corpus PDFs (not synthetic fixtures) because
  page-accuracy bugs often only show up on real, messy documents.

REFACTOR NOTE (LangChain pass, see ADR 0004):
  Previously this tested free functions parse_pdf()/get_page_count().
  The module was refactored into a CognaraPDFLoader class implementing
  LangChain's BaseLoader interface. Every assertion below is the same
  in spirit as before — same page counts, same verified page content,
  same error-handling behaviour — just expressed against
  loader.load() -> list[Document] instead of a plain list[dict].
"""

from pathlib import Path

import pytest

from ingestion.parsers.pdf_parser import CognaraPDFLoader, _sanitize_text

ML_PDF = Path("data/raw_pdfs/100 days ML Notes v2.pdf")
DL_PDF = Path("data/raw_pdfs/100 DAYS OF DL.pdf")

requires_ml_pdf = pytest.mark.skipif(not ML_PDF.exists(), reason="ML PDF not present locally")
requires_dl_pdf = pytest.mark.skipif(not DL_PDF.exists(), reason="DL PDF not present locally")


class TestLoaderConstruction:
    def test_missing_file_raises_immediately(self):
        # Raised in __init__, not deferred to load() — see class docstring.
        with pytest.raises(FileNotFoundError):
            CognaraPDFLoader("data/raw_pdfs/does_not_exist.pdf")


class TestGetPageCount:
    @requires_ml_pdf
    def test_ml_pdf_page_count(self):
        # Verified in Phase 0 TOC extraction: 2,434 pages
        loader = CognaraPDFLoader(ML_PDF)
        assert loader.get_page_count() == 2434

    @requires_dl_pdf
    def test_dl_pdf_page_count(self):
        # Verified in Phase 0 TOC extraction: 1,125 pages
        loader = CognaraPDFLoader(DL_PDF)
        assert loader.get_page_count() == 1125


class TestLoadReturnsDocuments:
    @requires_ml_pdf
    def test_first_ml_chapter_range(self):
        # Chapter 1: "Introduction to Machine Learning", pages 135-143
        loader = CognaraPDFLoader(ML_PDF, start_page=135, end_page=143)
        docs = loader.load()
        assert len(docs) > 0
        for d in docs:
            assert 135 <= d.metadata["page_number"] <= 143
            assert isinstance(d.page_content, str)
            assert len(d.page_content) > 0
            assert d.metadata["source"] == str(ML_PDF)

    @requires_ml_pdf
    def test_page_135_is_intro_content(self):
        # Verified in Phase 0: page 135 opens "Introduction to Machine Learning"
        loader = CognaraPDFLoader(ML_PDF, start_page=135, end_page=135)
        docs = loader.load()
        assert len(docs) == 1
        assert docs[0].metadata["page_number"] == 135
        content_lower = docs[0].page_content.lower()
        assert "machine learning" in content_lower or "ml" in content_lower

    @requires_dl_pdf
    def test_dl_part_i_starts_at_37(self):
        # Verified in Phase 0: page 37 is "Part I Introduction to Deep Learning"
        loader = CognaraPDFLoader(DL_PDF, start_page=37, end_page=37)
        docs = loader.load()
        assert len(docs) == 1
        assert "deep learning" in docs[0].page_content.lower()

    @requires_ml_pdf
    def test_invalid_range_raises_on_load(self):
        # Range validation happens inside lazy_load(), triggered by load().
        loader = CognaraPDFLoader(ML_PDF, start_page=100, end_page=50)  # end before start
        with pytest.raises(ValueError):
            loader.load()

    @requires_ml_pdf
    def test_out_of_bounds_raises_on_load(self):
        loader = CognaraPDFLoader(ML_PDF, start_page=1, end_page=999999)
        with pytest.raises(ValueError):
            loader.load()

    @requires_ml_pdf
    def test_lazy_load_yields_documents_one_at_a_time(self):
        """
        Confirms lazy_load() is a real generator (not just load() renamed),
        per LangChain's stated contract that implementations should avoid
        loading everything into memory at once.
        """
        import types

        loader = CognaraPDFLoader(ML_PDF, start_page=135, end_page=137)
        gen = loader.lazy_load()
        assert isinstance(gen, types.GeneratorType)
        first_doc = next(gen)
        assert first_doc.metadata["page_number"] == 135

    @requires_ml_pdf
    def test_full_pdf_no_range(self):
        # No start/end given -> parses ALL 2,434 pages. Slow; only run explicitly.
        pytest.skip("Full-PDF parse is slow (2,434 pages) — run manually, not in default suite")


class TestSanitizeText:
    """
    Direct unit tests for _sanitize_text(), plus a regression test against
    the exact real page that surfaced this bug during Module 4's first
    real ingestion run — see the module docstring's REAL BUG FOUND AND
    FIXED note. Page 160 of the ML PDF genuinely contains runs of NUL
    bytes in a face-clustering example figure; this test proves the fix
    against that real page, not just a synthetic string.
    """

    def test_strips_null_bytes(self):
        assert _sanitize_text("before\x00\x00\x00after") == "beforeafter"

    def test_strips_other_control_chars_but_keeps_whitespace(self):
        dirty = "line one\ttabbed\nline two\r\nline three\x01\x02\x1f"
        clean = _sanitize_text(dirty)
        assert "\x01" not in clean and "\x02" not in clean and "\x1f" not in clean
        assert "\t" in clean and "\n" in clean and "\r" in clean

    def test_normal_text_is_unchanged(self):
        normal = "Overfitting occurs when a model learns the training data too well."
        assert _sanitize_text(normal) == normal

    @requires_ml_pdf
    def test_real_page_160_contains_no_null_bytes_after_loading(self):
        """
        Direct regression test for the real bug: page 160 of the ML PDF
        (chapter 'Types of Machine Learning', section 3.8.4 'Real-World
        Example: Google Photos') is the exact page whose raw PyMuPDF
        extraction contains NUL byte runs. Confirms the loader's output
        is clean.
        """
        loader = CognaraPDFLoader(ML_PDF, start_page=160, end_page=160)
        docs = loader.load()
        assert len(docs) == 1
        assert "\x00" not in docs[0].page_content

    @requires_ml_pdf
    def test_real_page_160_has_no_control_chars_at_all(self):
        loader = CognaraPDFLoader(ML_PDF, start_page=160, end_page=160)
        docs = loader.load()
        content = docs[0].page_content
        forbidden = set(range(0x00, 0x09)) | {0x0B, 0x0C} | set(range(0x0E, 0x20)) | {0x7F}
        assert not any(ord(ch) in forbidden for ch in content)
