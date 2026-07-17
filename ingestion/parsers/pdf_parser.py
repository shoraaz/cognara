"""
ingestion/parsers/pdf_parser.py
--------------------------------
LangChain-compatible PDF loader: extracts text from PDF files, one
LangChain Document per page, preserving exact page numbers.

WHY THIS FILE EXISTS:
  The PDF is the raw input. Before any chunking or embedding can happen,
  we need text + page numbers out of the PDF. This is the only place
  that opens a PDF file.

REFACTOR NOTE (Module 1, LangChain pass — see ADR 0004):
  This module was originally a set of plain functions (parse_pdf(),
  get_page_count()) returning plain dicts. Per ADR 0004, ingestion
  components now use LangChain's interfaces so they compose with the
  rest of the LangChain ecosystem (text splitters, retrievers, etc.)
  without any glue code. This file now exposes a CognaraPDFLoader class
  implementing LangChain's BaseLoader interface.

  WHY NOT A STOCK LANGCHAIN LOADER (e.g. PyMuPDFLoader)?
  LangChain already ships a PyMuPDFLoader, and for a generic "give me
  every page of this PDF as Documents" use case it would be the right
  choice. We do NOT use it directly because our Phase 0 corpus selection
  requires reading only a specific page RANGE out of each source PDF
  (e.g. pages 135-143 of a 2,434-page book — see
  data/catalog/document_catalog_v1.csv) and no stock LangChain PDF loader
  supports page-range-restricted extraction as a constructor argument.
  Wrapping PyMuPDF ourselves, behind the same BaseLoader interface stock
  loaders use, gives us that page-range control while staying a drop-in
  citizen of the LangChain ecosystem — any code written against
  BaseLoader works with CognaraPDFLoader with zero changes.

LANGCHAIN INTERFACE CONTRACT (from langchain_core.document_loaders.BaseLoader):
  Subclasses implement lazy_load() as a generator yielding Document
  objects. BaseLoader.load() is provided for free and just calls
  list(self.lazy_load()) — we do not override load() ourselves.

REAL BUG FOUND AND FIXED (Module 4 / first real ingestion run):
  Running the full ingestion pipeline against the real corpus failed
  partway through with a Postgres error: "invalid byte sequence for
  encoding UTF8: 0x00". Tracing it down to the exact chunk (topic
  "3.8.4 Real-World Example: Google Photos", page 160) showed the raw
  text PyMuPDF extracted from that page genuinely contains runs of NUL
  (0x00) bytes — not from the chunker's own PARA_BREAK sentinel (a
  different, unrelated internal constant), but from the PDF itself: a
  diagram/spacing artifact around bullet points like "Cluster 1" and
  "Dad" in a face-clustering example figure, most likely from how the
  source LaTeX rendered indentation or spacing before those labels.
  PyMuPDF faithfully extracts whatever bytes are present in the PDF's
  text layer, including ones that are not valid, useful text. Postgres
  correctly refuses to store NUL bytes in a text column — no database
  can store 0x00 as normal text content, that is a hard limitation most
  databases inherit from the C-string convention, not a Postgres quirk.

  FIX: sanitize extracted text at the source, in lazy_load(), before it
  ever becomes a Document — strip NUL bytes and other non-printable
  control characters (keeping normal whitespace: space, tab, newline)
  immediately after PyMuPDF returns page text. Cleaning it here, once,
  means every downstream consumer (the chunker, the embedder, the
  database) automatically receives clean text — the fix lives at the
  first point clean data can be guaranteed, not scattered as defensive
  checks in every later module.

LIBRARY CHOICE — PyMuPDF (fitz), underneath the LangChain interface:
  We use PyMuPDF because:
  - Page-accurate: page 42 in the PDF = page_number 42 in the Document
  - Fast: no subprocess calls, no Java (unlike Apache Tika)
  - Handles CampusX-style slide PDFs well (text-based, not scanned)
  We do NOT use pdfplumber (slower for our use case) or pdfminer (lower-level,
  more boilerplate for the same result).

INTERVIEW EXPLANATION:
  "PDF parsing is the first failure point in a RAG system. I wrap PyMuPDF
  in a LangChain BaseLoader subclass rather than using LangChain's stock
  PyMuPDFLoader, because I need page-range-restricted extraction that no
  stock loader supports — but by implementing the same lazy_load()
  contract, my loader is still a fully compatible citizen of the
  LangChain ecosystem. Page number accuracy matters because every
  citation we show the user must be verifiable. I also learned during
  the first real ingestion run that PDF text extraction can surface raw
  control characters like NUL bytes from diagram/spacing artifacts in
  the source document — not something you'd predict from reading the
  PDF visually, only from actually running extraction on every page and
  hitting a real database error. I sanitize text at the extraction
  source rather than downstream, so every consumer of a Document from
  this loader can trust the text is clean."

OUTPUT: CognaraPDFLoader.lazy_load() yields one Document per non-empty
page in the requested range, with:
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

# Matches NUL and other C0 control characters EXCEPT tab (\x09), newline
# (\x0A), and carriage return (\x0D) — those three are normal, meaningful
# whitespace in extracted text and must be preserved. Everything else in
# the \x00-\x1F range (and \x7F, DEL) is either extraction noise (like
# the NUL-byte runs found in real corpus page 160) or has no business in
# a text column at all.
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")


def _sanitize_text(text: str) -> str:
    """Strip NUL bytes and other non-printable control characters from
    extracted PDF text, keeping normal whitespace (space, tab, newline,
    carriage return) intact. See the module docstring's REAL BUG FOUND
    AND FIXED note for why this exists."""
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
                __init__, not deferred to lazy_load(), so a caller building
                a list of loaders finds a bad path immediately rather than
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
        This is the method LangChain's BaseLoader.load() calls under the
        hood — see the LANGCHAIN INTERFACE CONTRACT note above.
        """
        doc = fitz.open(str(self.pdf_path))
        try:
            total_pages = doc.page_count
            first = self.start_page if self.start_page is not None else 1
            last = self.end_page if self.end_page is not None else total_pages

            if first < 1 or last > total_pages or first > last:
                raise ValueError(
                    f"Invalid page range {first}-{last} for a "
                    f"{total_pages}-page PDF: {self.pdf_path.name}"
                )

            # fitz page indices are 0-indexed; our range is 1-indexed inclusive.
            for page_index in range(first - 1, last):
                page = doc[page_index]
                text = _sanitize_text(page.get_text()).strip()
                if text:
                    yield Document(
                        page_content=text,
                        metadata={
                            "page_number": page_index + 1,  # back to 1-indexed
                            "source": str(self.pdf_path),
                        },
                    )
        finally:
            doc.close()

    def get_page_count(self) -> int:
        """
        Return the total number of pages in the underlying PDF (not just
        the requested range). Useful for validating a catalog entry before
        attempting to load it — e.g. confirming page_end in
        document_catalog_v1.csv doesn't exceed the real PDF length.
        """
        doc = fitz.open(str(self.pdf_path))
        try:
            return doc.page_count
        finally:
            doc.close()
