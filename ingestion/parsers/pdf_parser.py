"""
ingestion/parsers/pdf_parser.py
--------------------------------
LangChain-compatible PDF loader: extracts text from PDF files, one
LangChain Document per page, preserving exact page numbers.

WHY THIS FILE EXISTS:
  The PDF is the raw input. Before any chunking or embedding can happen,
  we need text + page numbers out of the PDF. This is the only place that
  opens a PDF file.

LANGCHAIN INTERFACE NOTE (see ADR 0004):
  This module was originally a set of plain functions returning plain dicts.
  It now exposes CognaraPDFLoader, a LangChain BaseLoader subclass, so it
  composes with the rest of the LangChain ecosystem without glue code.

WHY NOT A STOCK LANGCHAIN LOADER (e.g. PyMuPDFLoader):
  Our corpus requires reading only a specific page RANGE out of each source
  PDF (e.g. pages 135-143 of a 2,434-page book — see
  data/catalog/document_catalog_v1.csv). No stock LangChain PDF loader
  supports page-range-restricted extraction as a constructor argument.
  Wrapping PyMuPDF ourselves, behind the same BaseLoader interface, gives
  us that page-range control while staying a drop-in LangChain citizen.

LIBRARY CHOICE — PyMuPDF (fitz):
  - Page-accurate: page 42 in the PDF = page_number 42 in the Document
  - Fast: no subprocess calls, no Java (unlike Apache Tika)
  - Handles CampusX-style slide PDFs well (text-based, not scanned)

TEXT SANITIZATION:
  PyMuPDF faithfully extracts whatever bytes are in the PDF's text layer,
  including NUL (0x00) bytes from diagram/spacing artifacts. Postgres refuses
  to store NUL bytes in a text column. We strip them (and other non-printable
  control characters) at extraction time so every downstream consumer
  receives clean text. See BUG_FIX_LOG.md "PDF Parser: NUL bytes from PDF".

# Interview notes: local-notes/INTERVIEW_PREP.md — "ingestion/parsers/pdf_parser.py"

OUTPUT: CognaraPDFLoader.lazy_load() yields one Document per non-empty page
in the requested range, with:
  - page_content: the page's extracted text, sanitized of NUL bytes and
    other non-printable control characters
  - metadata: {"page_number": int, "source": str (the PDF path)}
Empty pages are skipped (e.g. title-slide pages with only an image/logo).
"""

import re
from pathlib import Path
from typing import Iterator

import fitz  # PyMuPDF
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

# Matches NUL and other C0 control characters EXCEPT:
#   \x09 (tab), \x0A (newline), \x0D (carriage return) — normal whitespace.
# Everything else in the \x00-\x1F range (and \x7F DEL) is either extraction
# noise (like the NUL-byte runs found in real corpus page 160) or has no
# business in a text column at all.
# See BUG_FIX_LOG.md "PDF Parser: NUL bytes from PDF" for full root-cause.
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")


def _sanitize_text(text: str) -> str:
    """
    Strip NUL bytes and other non-printable control characters from extracted
    PDF text, keeping normal whitespace (space, tab, newline, carriage return).
    Called on every page immediately after PyMuPDF extraction.
    """
    return _CONTROL_CHARS_RE.sub("", text)


class CognaraPDFLoader(BaseLoader):
    """
    LangChain-compatible loader for a page-range-restricted slice of a PDF.

    Usage:
        loader = CognaraPDFLoader(
            "data/raw_pdfs/100 days ML Notes v2.pdf",
            start_page=135,
            end_page=143,
        )
        docs = loader.load()  # -> list[Document], one per non-empty page
    """

    def __init__(
        self,
        pdf_path: str | Path,
        start_page: int | None = None,
        end_page: int | None = None,
    ) -> None:
        """
        Args:
            pdf_path: Path to the PDF file.
            start_page: First page to read, 1-indexed, inclusive. None = page 1.
            end_page: Last page to read, 1-indexed, inclusive. None = last page.

        Raises:
            FileNotFoundError: if the PDF does not exist. Raised eagerly in
                __init__, not deferred to lazy_load(), so a caller building a
                list of loaders finds a bad path immediately rather than
                mid-ingestion-run.
        """
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {self.pdf_path}")
        self.start_page = start_page
        self.end_page = end_page

    def lazy_load(self) -> Iterator[Document]:
        """
        Yield one Document per non-empty page in the requested range.
        This is the method LangChain's BaseLoader.load() calls internally —
        subclasses implement lazy_load(); load() is provided for free.
        """
        doc = fitz.open(str(self.pdf_path))
        try:
            total_pages = doc.page_count
            # Apply start/end defaults only once — None means "all pages".
            first = self.start_page if self.start_page is not None else 1
            last  = self.end_page   if self.end_page   is not None else total_pages

            if first < 1 or last > total_pages or first > last:
                raise ValueError(
                    f"Invalid page range {first}-{last} for a "
                    f"{total_pages}-page PDF: {self.pdf_path.name}"
                )

            # fitz uses 0-indexed page numbers; our API is 1-indexed and inclusive.
            # range(first - 1, last) converts correctly: first=1 -> index 0.
            for page_index in range(first - 1, last):
                page = doc[page_index]
                text = _sanitize_text(page.get_text()).strip()
                # Skip entirely empty pages (title slides with only images/logos).
                if text:
                    yield Document(
                        page_content=text,
                        metadata={
                            "page_number": page_index + 1,  # back to 1-indexed
                            "source": str(self.pdf_path),
                        },
                    )
        finally:
            # Always close the fitz document, even if an exception is raised
            # mid-loop, to release the file handle immediately.
            doc.close()

    def get_page_count(self) -> int:
        """
        Return the total number of pages in the underlying PDF (not just the
        requested range). Useful for validating a catalog entry before loading
        — e.g. confirming page_end in document_catalog_v1.csv doesn't exceed
        the real PDF length.
        """
        doc = fitz.open(str(self.pdf_path))
        try:
            return doc.page_count
        finally:
            doc.close()
