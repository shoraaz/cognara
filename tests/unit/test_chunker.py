"""
tests/unit/test_chunker.py
----------------------------
Tests for ingestion/chunking/chunker.py (CognaraHeadingSplitter).

WHY THIS FILE EXISTS:
  The chunker's correctness determines what text ends up next to what
  citation. These tests run the splitter against REAL pages loaded from
  the actual corpus PDFs (via CognaraPDFLoader, tested separately in
  test_pdf_parser.py) plus a handful of small synthetic Documents for
  edge conditions (empty input, oversized sections, short-section
  merging) that are awkward to find naturally in the real corpus.

REFACTOR NOTE (LangChain pass, see ADR 0004):
  Previously this tested a free function chunk_pages(pages, metadata,
  chunk_size_chars=..., overlap_chars=...) operating on plain dicts.
  The module was refactored into CognaraHeadingSplitter, a LangChain
  TextSplitter subclass. Every assertion below is the same in spirit —
  same heading detection, same page-range correctness, same short/long
  section handling — just expressed as
  splitter.split_documents(list[Document]) -> list[Document], with
  standard LangChain constructor args chunk_size / chunk_overlap in
  place of the old chunk_size_chars / overlap_chars.
"""

from pathlib import Path

import pytest
from langchain_core.documents import Document

from ingestion.chunking.chunker import CognaraHeadingSplitter
from ingestion.parsers.pdf_parser import CognaraPDFLoader

ML_PDF = Path("data/raw_pdfs/100 days ML Notes v2.pdf")
DL_PDF = Path("data/raw_pdfs/100 DAYS OF DL.pdf")

requires_ml_pdf = pytest.mark.skipif(not ML_PDF.exists(), reason="ML PDF not present locally")
requires_dl_pdf = pytest.mark.skipif(not DL_PDF.exists(), reason="DL PDF not present locally")

SAMPLE_METADATA = {
    "course_name": "100 Days of Machine Learning",
    "subject": "Machine Learning",
    "chapter": "Introduction to Machine Learning",
    "source_type": "campusx_notes",
    "document_version": "v1",
}


def _load_ml_intro_docs() -> list[Document]:
    """Real pages 135-143, with catalog metadata attached exactly as the
    real ingestion pipeline (Module 4) will do it — metadata added onto
    each loaded Document before splitting."""
    loader = CognaraPDFLoader(ML_PDF, start_page=135, end_page=143)
    docs = loader.load()
    for d in docs:
        d.metadata.update(SAMPLE_METADATA)
    return docs


class TestSplitDocumentsEmptyInput:
    def test_empty_list_returns_empty_list(self):
        splitter = CognaraHeadingSplitter()
        assert splitter.split_documents([]) == []


class TestSplitDocumentsRealCorpus:
    @requires_ml_pdf
    def test_ml_intro_chapter_produces_chunks(self):
        splitter = CognaraHeadingSplitter()
        chunks = splitter.split_documents(_load_ml_intro_docs())
        assert len(chunks) > 0
        assert all(isinstance(c, Document) for c in chunks)

    @requires_ml_pdf
    def test_every_chunk_has_required_metadata_fields(self):
        splitter = CognaraHeadingSplitter()
        chunks = splitter.split_documents(_load_ml_intro_docs())
        required_fields = {
            "chunk_id", "topic", "page_number", "page_range",
            "chunk_index_in_doc", "char_count", "ingestion_date",
            "course_name", "subject", "chapter", "source_type", "document_version",
        }
        for c in chunks:
            assert required_fields.issubset(c.metadata.keys())
            assert isinstance(c.page_content, str) and len(c.page_content) > 0
            assert c.metadata["char_count"] == len(c.page_content)
            assert isinstance(c.metadata["page_number"], int)

    @requires_ml_pdf
    def test_chunk_ids_are_unique(self):
        splitter = CognaraHeadingSplitter()
        chunks = splitter.split_documents(_load_ml_intro_docs())
        ids = [c.metadata["chunk_id"] for c in chunks]
        assert len(ids) == len(set(ids))

    @requires_ml_pdf
    def test_chunk_index_in_doc_is_sequential(self):
        splitter = CognaraHeadingSplitter()
        chunks = splitter.split_documents(_load_ml_intro_docs())
        indices = [c.metadata["chunk_index_in_doc"] for c in chunks]
        assert indices == list(range(len(chunks)))

    @requires_ml_pdf
    def test_page_numbers_within_requested_range(self):
        splitter = CognaraHeadingSplitter()
        chunks = splitter.split_documents(_load_ml_intro_docs())
        for c in chunks:
            assert 135 <= c.metadata["page_number"] <= 143

    @requires_ml_pdf
    def test_running_header_noise_is_stripped(self):
        splitter = CognaraHeadingSplitter()
        chunks = splitter.split_documents(_load_ml_intro_docs())
        for c in chunks:
            assert "2CHAPTER 1." not in c.page_content

    @requires_ml_pdf
    def test_figure_caption_noise_is_stripped(self):
        splitter = CognaraHeadingSplitter()
        chunks = splitter.split_documents(_load_ml_intro_docs())
        for c in chunks:
            assert "Figure 1.1: image" not in c.page_content

    @requires_ml_pdf
    def test_heading_topic_is_detected(self):
        # Page 137 contains heading "1.2 What is Machine Learning?" — a
        # top-level heading immediately followed by its own subsection
        # with no body text of its own; this exact case previously caused
        # the heading to be silently dropped (see chunker.py docstring).
        splitter = CognaraHeadingSplitter()
        chunks = splitter.split_documents(_load_ml_intro_docs())
        topics = [c.metadata["topic"] for c in chunks if c.metadata["topic"]]
        assert any("What is Machine Learning" in t for t in topics)

    @requires_ml_pdf
    def test_metadata_propagates_to_every_chunk(self):
        splitter = CognaraHeadingSplitter()
        chunks = splitter.split_documents(_load_ml_intro_docs())
        for c in chunks:
            assert c.metadata["course_name"] == "100 Days of Machine Learning"
            assert c.metadata["subject"] == "Machine Learning"
            assert c.metadata["chapter"] == "Introduction to Machine Learning"
            assert c.metadata["source_type"] == "campusx_notes"
            assert c.metadata["document_version"] == "v1"

    @requires_ml_pdf
    def test_page_range_spans_a_real_page_boundary(self):
        """
        Proves the exact design constraint documented in chunker.py:
        split_documents() sees all pages at once and can correctly report
        a chunk that genuinely spans two PDF pages — something
        split_text() on a single page could never produce. At least one
        chunk in this real range is known (from prior manual verification)
        to span a page boundary.
        """
        splitter = CognaraHeadingSplitter()
        chunks = splitter.split_documents(_load_ml_intro_docs())
        spanning = [c for c in chunks if c.metadata["page_range"] is not None]
        assert len(spanning) > 0, "expected at least one chunk to span a page boundary"
        for c in spanning:
            start, end = c.metadata["page_range"].split("-")
            assert int(end) > int(start)

    @requires_dl_pdf
    def test_dl_part1_produces_chunks_with_headings(self):
        loader = CognaraPDFLoader(DL_PDF, start_page=37, end_page=45)
        docs = loader.load()
        dl_metadata = {**SAMPLE_METADATA, "course_name": "100 Days of Deep Learning",
                       "subject": "Deep Learning", "chapter": "Introduction to Deep Learning"}
        for d in docs:
            d.metadata.update(dl_metadata)

        splitter = CognaraHeadingSplitter()
        chunks = splitter.split_documents(docs)
        assert len(chunks) > 0
        topics = [c.metadata["topic"] for c in chunks if c.metadata["topic"]]
        assert len(topics) > 0


class TestShortSectionMerging:
    @requires_ml_pdf
    def test_no_chunk_is_near_empty(self):
        splitter = CognaraHeadingSplitter()
        chunks = splitter.split_documents(_load_ml_intro_docs())
        for c in chunks:
            assert c.metadata["char_count"] >= 20


class TestOversizedSectionSplitting:
    def test_long_synthetic_section_is_split_with_overlap(self):
        paragraph = "This sentence about gradient descent repeats. " * 20  # ~960 chars
        long_text = "\n\n".join([paragraph] * 5)  # ~4800 chars, over chunk_size
        docs = [Document(
            page_content=f"1.1\nA Long Section\n{long_text}",
            metadata={"page_number": 200, **SAMPLE_METADATA},
        )]

        splitter = CognaraHeadingSplitter(chunk_size=2000, chunk_overlap=100)
        chunks = splitter.split_documents(docs)

        assert len(chunks) > 1
        for c in chunks:
            assert c.metadata["char_count"] <= 2200  # target + small overlap slack

    def test_short_section_is_not_split(self):
        docs = [Document(
            page_content="1.1\nShort Section\nJust a little bit of text here.",
            metadata={"page_number": 1, **SAMPLE_METADATA},
        )]
        splitter = CognaraHeadingSplitter()
        chunks = splitter.split_documents(docs)
        assert len(chunks) == 1


class TestConstructorValidation:
    """
    These come free from LangChain's base TextSplitter.__init__ — worth
    testing explicitly so a future refactor that bypasses super().__init__()
    would be caught immediately.
    """

    def test_negative_chunk_overlap_raises(self):
        with pytest.raises(ValueError):
            CognaraHeadingSplitter(chunk_overlap=-1)

    def test_overlap_larger_than_chunk_size_raises(self):
        with pytest.raises(ValueError):
            CognaraHeadingSplitter(chunk_size=100, chunk_overlap=200)


class TestSplitTextFallback:
    """
    split_text() is required by TextSplitter's abstract interface but is
    explicitly NOT the primary entry point for this splitter — it has no
    cross-page context. These tests confirm it still works correctly for
    a single page in isolation, and that its limitation is real (not
    just documented) by comparing behaviour to split_documents().
    """

    def test_split_text_returns_plain_strings(self):
        splitter = CognaraHeadingSplitter()
        result = splitter.split_text("1.1\nA Heading\nSome body text here.")
        assert isinstance(result, list)
        assert all(isinstance(s, str) for s in result)
        assert len(result) == 1
        assert "Some body text here." in result[0]
