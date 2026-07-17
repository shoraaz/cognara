"""
ingestion/chunking/chunker.py
------------------------------
LangChain-compatible text splitter: splits page-level Documents into
heading-aware chunk Documents with full citation metadata attached.

WHY THIS FILE EXISTS:
  You cannot embed an entire 1,000-page book in one vector — the LLM context
  window would overflow and retrieval would be meaningless. Chunking splits
  the text into pieces small enough to embed and retrieve individually, while
  keeping enough context to be useful.

REFACTOR NOTE (Module 2, LangChain pass — see ADR 0004):
  This module was originally a set of plain functions (chunk_pages() and
  friends) operating on plain dicts. Per ADR 0004, ingestion components
  now use LangChain's interfaces. This file now exposes a
  CognaraHeadingSplitter class extending LangChain's TextSplitter, so it
  composes directly with any LangChain loader, retriever, or vectorstore
  without glue code — e.g. loader.load_and_split(CognaraHeadingSplitter())
  works exactly like it would with a stock LangChain splitter.

  A GENUINE DESIGN CONSTRAINT, SOLVED DELIBERATELY:
  LangChain's TextSplitter contract is built around split_text(text: str),
  a method that receives ONE PLAIN STRING at a time. Its split_documents()
  helper calls split_text() once per input Document and then re-attaches
  THAT SAME Document's metadata to every resulting chunk (via
  copy.deepcopy — see langchain_text_splitters.TextSplitter.
  create_documents). That flow assumes every chunk stays within the
  boundary of a single input Document.

  Our chunker cannot fit that assumption. A real, verified example from
  our corpus: a chunk covering "1.4.3 The Paradigm Shift (Post-2010)"
  genuinely spans PDF pages 140-141 — its content starts on one page's
  Document and continues onto the next. If we only ever saw one page's
  text at a time (what split_text() would receive), we could never
  produce that correct page_range="140-141" citation; we'd either lose
  the cross-page content or invent a wrong single page number.

  RESOLUTION: rather than force our multi-page heading detection through
  a single-string interface it doesn't fit, we override split_documents()
  directly — a documented, legitimate LangChain extension point — so we
  receive the FULL LIST of per-page Documents at once, exactly like
  chunk_pages() did before the refactor. split_text() is still
  implemented (TextSplitter requires it as an abstract method), but is
  clearly marked as NOT our primary entry point — calling it directly on
  a single page loses cross-page context by construction, which is
  exactly the problem split_documents() exists to avoid.

REAL CORPUS STRUCTURE (verified against both source PDFs):
  Both "100 Days of ML" and "100 Days of DL" are LaTeX-generated documents
  with a very regular heading pattern:

      1.1
      Series Announcement
      1.1.1
      Playlist Overview

  A numbered heading marker (e.g. "1.1" or "1.1.1") sits alone on one line,
  and the heading TITLE is the line(s) immediately after it. Our heading
  detector is built around this exact pattern, confirmed by manually
  inspecting real extracted pages during Phase 1 development.

  Each page also carries a running header injected by the PDF renderer,
  e.g. "2CHAPTER 1. 100 DAYS OF MACHINE LEARNING - DAY 1: INTRODUCTION TO M"
  — a page number glued directly to an all-caps chapter title. We strip
  these; they are page furniture, not content.

  IMPORTANT SUBTLETY (found via testing, not assumption): a top-level
  heading like "1.1 Series Announcement" is very often followed IMMEDIATELY
  by its own first subsection "1.1.1 Playlist Overview" with ZERO body
  text of its own in between. A naive implementation that only keeps a
  heading if it has body text will silently DROP that heading entirely.
  We always record every detected heading as its own section (even with
  empty text), and let the short-section merge step decide how to fold
  it into neighbouring content.

  SECOND SUBTLETY (also found via testing): when a section has NO body
  text of its own, we still need a real, correct page_number for its
  citation — we cannot invent one. We remember the page the HEADING
  ITSELF was printed on (heading_page), and use that as the fallback
  instead of any hardcoded page number.

CHUNKING STRATEGY (Phase 1 — heading-aware):
  1. Concatenate all input Documents (one per page) into one continuous
     line stream, remembering which page each line came from.
  2. Strip running-header noise lines. Preserve paragraph breaks as
     explicit markers so an oversized-section split can still find real
     paragraph boundaries.
  3. Split on our detected heading pattern — each heading starts a new
     "section". Every heading always produces a section, even an empty
     one, and its page_number always traces back to a real page.
  4. Merge sections that are too short (including empty ones) forward
     into the next section, so no heading is lost and no near-empty
     chunk is produced.
  5. If a section is still too long, split further by paragraph with
     overlap so no sentence is lost at a chunk boundary.

INTERVIEW EXPLANATION:
  "I extend LangChain's TextSplitter, but override split_documents()
  instead of relying on split_text(), because split_text() only sees one
  page at a time and my chunker needs cross-page context — a real chunk
  in my corpus genuinely spans two PDF pages, and I need to report that
  as an accurate page_range citation. Overriding split_documents() is a
  documented extension point, and it means my splitter is still a fully
  compatible LangChain TextSplitter everywhere that matters — I just
  don't route through the one hook that doesn't fit my data's shape.
  Testing against the real corpus also caught two real bugs: parent
  headings with no content of their own were being silently dropped, and
  my first fix for that introduced a hardcoded fallback page number that
  would have produced wrong citations. Both were caught by tests that
  check exact page numbers and exact heading text against pages I
  manually verified."

OUTPUT: CognaraHeadingSplitter.split_documents() returns a list of
Document, one per chunk, each with metadata: chunk_id, topic (nearest
detected heading), page_number, page_range, chunk_index_in_doc,
char_count, ingestion_date, plus whatever extra metadata fields were
present on the input Documents (course_name, subject, chapter,
source_type, document_version, source — carried through from the loader).
"""

import re
import uuid
from datetime import date
from typing import Iterable

from langchain_core.documents import Document
from langchain_text_splitters import TextSplitter

# ── Heading detection ──────────────────────────────────────────────────────

# Matches a LaTeX-style section number alone on its own line, e.g.:
#   "1.1"        (section)
#   "1.1.1"      (subsection)
#   "12.3.4"     (deeper nesting, still supported)
# We require the WHOLE line to be just the number (optionally trailing
# whitespace) — this avoids false-matching "1.1" when it appears mid-sentence
# (e.g. in "version 1.1 of the library").
HEADING_NUMBER_RE = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){1,3})\s*$")

# Matches a running page-header line: a page number glued directly to an
# ALL-CAPS chapter title, e.g.:
#   "2CHAPTER 1. 100 DAYS OF MACHINE LEARNING - DAY 1: INTRODUCTION TO M"
RUNNING_HEADER_RE = re.compile(r"^\d{1,4}(CHAPTER|PART)\b.*$", re.IGNORECASE)

# A "Figure X.Y: image" caption line — not useful text content on its own.
FIGURE_CAPTION_RE = re.compile(r"^Figure\s+\d+(\.\d+)*\s*:\s*image\s*$", re.IGNORECASE)

MIN_SECTION_CHARS = 120  # sections shorter than this get merged forward

# Sentinel inserted wherever the ORIGINAL PDF text had a blank line (a
# paragraph break), so paragraph structure survives the line-flattening
# step and is available later for oversized-section splitting.
PARA_BREAK = "\x00PARA\x00"


class CognaraHeadingSplitter(TextSplitter):
    """
    Heading-aware LangChain TextSplitter for Cognara's LaTeX-numbered
    corpus. Prefer split_documents() (the primary entry point — see
    module docstring for why); split_text() is provided to satisfy
    TextSplitter's abstract interface but operates on a single page's
    text in isolation, without cross-page context.
    """

    def split_documents(self, documents: Iterable[Document]) -> list[Document]:
        """
        Primary entry point. Receives ALL per-page Documents for one
        source chapter/section at once (in page order), so heading
        detection and page-range tracking can work across page
        boundaries — see the module docstring's "genuine design
        constraint" note for why this overrides the base class instead
        of implementing split_text().

        Per-page metadata (page_number) comes from each input Document.
        Shared catalog metadata (course_name, subject, chapter,
        source_type, document_version) is expected to already be present
        on every input Document's metadata dict — CognaraPDFLoader does
        not set these; the ingestion pipeline attaches them after
        loading and before splitting (see run_ingestion.py, Module 4).
        """
        documents = list(documents)
        if not documents:
            return []

        pages = [
            {"page_number": doc.metadata["page_number"], "text": doc.page_content}
            for doc in documents
        ]
        # Shared catalog metadata is identical across all input Documents
        # for one chapter; take it from the first and strip the per-page
        # key (page_number) since that's re-derived per chunk below.
        shared_metadata = {
            k: v for k, v in documents[0].metadata.items() if k != "page_number"
        }

        chunk_dicts = self._chunk_pages(pages, shared_metadata)

        return [
            Document(
                page_content=c.pop("text"),
                metadata=c,
            )
            for c in chunk_dicts
        ]

    def split_text(self, text: str) -> list[str]:
        """
        Required by TextSplitter's abstract interface. Operates on a
        single page's text with NO cross-page context or page number —
        prefer split_documents() instead. Provided so this class is a
        structurally complete TextSplitter (e.g. for any generic code
        that only knows about the split_text(str) -> list[str] contract),
        with the limitation documented rather than silently accepted.
        """
        pages = [{"page_number": 1, "text": text}]
        chunk_dicts = self._chunk_pages(pages, metadata={})
        return [c["text"] for c in chunk_dicts]

    # ── Internal chunking pipeline (unchanged logic from the pre-refactor
    #    module — proven against the real corpus; only the outer
    #    interface changed) ───────────────────────────────────────────────

    def _chunk_pages(self, pages: list[dict], metadata: dict) -> list[dict]:
        if not pages:
            return []

        tagged_lines = self._build_tagged_lines(pages)
        sections = self._split_into_sections(tagged_lines)
        sections = self._merge_short_sections(sections)

        chunks: list[dict] = []
        for section in sections:
            if not section["text"]:
                continue
            sub_texts = self._split_long_section(section["text"])
            for sub_text in sub_texts:
                chunks.append(self._make_chunk(sub_text, section, metadata, len(chunks)))

        return chunks

    def _build_tagged_lines(self, pages: list[dict]) -> list[tuple[str, int]]:
        """Return [(line_text, page_number), ...], noise stripped, paragraph
        breaks marked with PARA_BREAK."""
        tagged: list[tuple[str, int]] = []
        for pg in pages:
            page_number = pg["page_number"]
            prev_was_blank = True
            for raw_line in pg["text"].split("\n"):
                line = raw_line.strip()

                if not line:
                    if not prev_was_blank and tagged:
                        tagged.append((PARA_BREAK, page_number))
                    prev_was_blank = True
                    continue

                if RUNNING_HEADER_RE.match(line):
                    continue
                if FIGURE_CAPTION_RE.match(line):
                    continue

                tagged.append((line, page_number))
                prev_was_blank = False
        return tagged

    def _split_into_sections(self, tagged_lines: list[tuple[str, int]]) -> list[dict]:
        """Group lines into sections at each detected heading. Every
        heading always produces a section, even an empty one."""
        sections: list[dict] = []
        current_heading: str | None = None
        current_lines: list[str] = []
        current_pages: list[int] = []
        heading_page: int | None = None
        have_content_yet = False

        def flush():
            if current_pages:
                pages_for_section = current_pages
            elif heading_page is not None:
                pages_for_section = [heading_page]
            elif tagged_lines:
                pages_for_section = [tagged_lines[0][1]]
            else:
                pages_for_section = [1]

            text = "\n".join(current_lines).strip().strip(PARA_BREAK).strip()
            sections.append({
                "heading": current_heading,
                "text": text,
                "start_page": min(pages_for_section),
                "end_page": max(pages_for_section),
            })

        i = 0
        n = len(tagged_lines)
        while i < n:
            line, page = tagged_lines[i]

            if line == PARA_BREAK:
                if current_lines:
                    current_lines.append(PARA_BREAK)
                    current_pages.append(page)
                i += 1
                continue

            if HEADING_NUMBER_RE.match(line):
                title = None
                look = i + 1
                while look < n and tagged_lines[look][0] == PARA_BREAK:
                    look += 1
                if look < n:
                    title = tagged_lines[look][0]

                if have_content_yet:
                    flush()
                have_content_yet = True

                current_heading = f"{line} {title}".strip() if title else line
                current_lines = []
                current_pages = []
                heading_page = page

                i = look + 1 if title else i + 1
                continue

            current_lines.append(line)
            current_pages.append(page)
            have_content_yet = True
            i += 1

        if have_content_yet:
            flush()

        return sections

    def _merge_short_sections(self, sections: list[dict]) -> list[dict]:
        """Fold sections shorter than MIN_SECTION_CHARS forward into the
        next section, keeping the earlier (parent) heading as the label."""
        if not sections:
            return sections

        merged: list[dict] = []
        pending: dict | None = None

        for section in sections:
            if pending is None:
                pending = dict(section)
                continue

            if len(pending["text"]) < MIN_SECTION_CHARS:
                if pending["text"] and section["text"]:
                    pending["text"] = f"{pending['text']}\n\n{section['text']}".strip()
                else:
                    pending["text"] = (pending["text"] + section["text"]).strip()
                pending["end_page"] = max(pending["end_page"], section["end_page"])
                continue

            merged.append(pending)
            pending = dict(section)

        if pending is not None:
            merged.append(pending)

        return merged

    def _split_long_section(self, text: str) -> list[str]:
        """Split an oversized section by paragraph (PARA_BREAK marker or
        fallback double-newline), with overlap between sub-chunks."""
        chunk_size_chars = self._chunk_size
        overlap_chars = self._chunk_overlap

        if len(text) <= chunk_size_chars:
            return [text.replace(PARA_BREAK, "\n\n")]

        if PARA_BREAK in text:
            paragraphs = [p.strip() for p in text.split(PARA_BREAK) if p.strip()]
        else:
            paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]

        if not paragraphs:
            paragraphs = [text.replace(PARA_BREAK, "\n\n")]

        chunks: list[str] = []
        current = ""

        for para in paragraphs:
            candidate = f"{current}\n\n{para}".strip() if current else para
            if len(candidate) <= chunk_size_chars or not current:
                current = candidate
            else:
                chunks.append(current)
                tail = current[-overlap_chars:] if overlap_chars > 0 else ""
                current = f"{tail}\n\n{para}".strip() if tail else para

        if current:
            chunks.append(current)

        return chunks

    def _make_chunk(self, text: str, section: dict, metadata: dict, index_in_doc: int) -> dict:
        start_page = section["start_page"]
        end_page = section["end_page"]
        page_range = None if start_page == end_page else f"{start_page}-{end_page}"

        return {
            "chunk_id": uuid.uuid4().hex[:12],
            "text": text,
            "topic": section["heading"],
            "page_number": start_page,
            "page_range": page_range,
            "chunk_index_in_doc": index_in_doc,
            "char_count": len(text),
            "ingestion_date": date.today().isoformat(),
            **metadata,
        }
