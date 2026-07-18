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

WHY WE OVERRIDE split_documents() INSTEAD OF split_text():
  LangChain's TextSplitter contract is built around split_text(text: str) —
  one string at a time. Its split_documents() helper calls split_text() once
  per input Document and re-attaches THAT SAME Document's metadata to every
  resulting chunk, assuming each chunk stays within a single Document's boundary.

  Our chunker cannot fit that assumption. A verified example from our corpus:
  a chunk covering "1.4.3 The Paradigm Shift (Post-2010)" genuinely spans PDF
  pages 140-141. If we only ever saw one page's text at a time, we could never
  produce the correct page_range="140-141" citation.

  RESOLUTION: we override split_documents() directly — a documented, legitimate
  LangChain extension point — so we receive the FULL LIST of per-page Documents
  at once. split_text() is still implemented (TextSplitter requires it), but is
  clearly marked as NOT the primary entry point.

REAL CORPUS STRUCTURE (verified against both source PDFs):
  Both "100 Days of ML" and "100 Days of DL" are LaTeX-generated documents with
  a very regular heading pattern: a numbered marker (e.g. "1.1") alone on one
  line, followed immediately by the heading title on the next line. Our heading
  detector is built around this exact pattern.

  Each page also carries a running header injected by the PDF renderer, e.g.:
    "2CHAPTER 1. 100 DAYS OF MACHINE LEARNING - DAY 1: INTRODUCTION TO M"
  We strip these — they are page furniture, not content.

  IMPORTANT SUBTLETY: a top-level heading like "1.1" is often followed
  IMMEDIATELY by its first subsection "1.1.1" with ZERO body text in between.
  A naive implementation that only keeps headings with body text will silently
  DROP these parent headings. We always record every detected heading as its
  own section (even empty ones) and let the short-section merge step decide.

  SECOND SUBTLETY: when a section has NO body text, we still need a real page
  number for its citation. We remember the page the HEADING ITSELF was printed
  on (heading_page) and use that as the fallback.

CHUNKING STRATEGY (Phase 1 — heading-aware):
  1. Concatenate all input Documents into one continuous line stream, remembering
     which page each line came from.
  2. Strip running-header noise lines. Preserve paragraph breaks as explicit
     markers so oversized-section splitting can still find real boundaries.
  3. Split on detected heading pattern — each heading starts a new section.
     Every heading always produces a section, even an empty one.
  4. Merge sections that are too short forward into the next section, so no
     heading is lost and no near-empty chunk is produced.
  5. If a section is still too long, split further by paragraph with overlap.

# Interview notes: local-notes/INTERVIEW_PREP.md — "ingestion/chunking/chunker.py"

OUTPUT: CognaraHeadingSplitter.split_documents() returns a list of Document,
one per chunk, each with metadata: chunk_id, topic (nearest detected heading),
page_number, page_range, chunk_index_in_doc, char_count, ingestion_date, plus
whatever extra metadata fields were present on the input Documents (course_name,
subject, chapter, source_type, document_version, source).
"""

import re
import uuid
from datetime import date
from typing import Iterable

from langchain_core.documents import Document
from langchain_text_splitters import TextSplitter

# ── Heading detection ─────────────────────────────────────────────────────────

# Matches a LaTeX-style section number alone on its own line, e.g.:
#   "1.1"        (section)
#   "1.1.1"      (subsection)
#   "12.3.4"     (deeper nesting, still supported)
# We require the WHOLE line to be just the number (^...$) to avoid false-
# matching "1.1" when it appears mid-sentence (e.g. "version 1.1 of the library").
HEADING_NUMBER_RE = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){1,3})\s*$")

# Matches a running page-header line: a page number glued directly to an
# ALL-CAPS chapter title, e.g.:
#   "2CHAPTER 1. 100 DAYS OF MACHINE LEARNING - DAY 1: INTRODUCTION TO M"
# These are injected by the PDF renderer and are page furniture, not content.
RUNNING_HEADER_RE = re.compile(r"^\d{1,4}(CHAPTER|PART)\b.*$", re.IGNORECASE)

# A "Figure X.Y: image" caption line — not useful text content on its own
# (the image itself doesn't appear in the text layer; only the label does).
FIGURE_CAPTION_RE = re.compile(r"^Figure\s+\d+(\.\d+)*\s*:\s*image\s*$", re.IGNORECASE)

# Sections shorter than this (in characters) are merged forward into the next
# section to avoid near-empty chunks that would embed poorly.
MIN_SECTION_CHARS = 120

# Sentinel string inserted wherever the original PDF text had a blank line
# (a paragraph break). Surviving the line-flattening step means oversized-
# section splitting can still break at real paragraph boundaries rather than
# arbitrary character positions.
PARA_BREAK = "\x00PARA\x00"


class CognaraHeadingSplitter(TextSplitter):
    """
    Heading-aware LangChain TextSplitter for Cognara's LaTeX-numbered corpus.
    Prefer split_documents() (the primary entry point — see module docstring
    for why); split_text() is provided to satisfy TextSplitter's abstract
    interface but operates on a single page's text in isolation.
    """

    def split_documents(self, documents: Iterable[Document]) -> list[Document]:
        """
        Primary entry point. Receives ALL per-page Documents for one source
        chapter/section at once (in page order), so heading detection and
        page-range tracking can work across page boundaries.

        Per-page metadata (page_number) comes from each input Document.
        Shared catalog metadata (course_name, subject, chapter, source_type,
        document_version) must already be present on every input Document's
        metadata dict — CognaraPDFLoader does not set these; the ingestion
        pipeline attaches them after loading and before splitting (see
        run_ingestion.py, Module 4).
        """
        documents = list(documents)
        if not documents:
            return []

        # Extract per-page data: only page_number varies between Documents;
        # all other fields are the shared catalog metadata.
        pages = [
            {"page_number": doc.metadata["page_number"], "text": doc.page_content}
            for doc in documents
        ]
        # Shared metadata is identical across all input Documents for one chapter;
        # take it from the first Document and exclude page_number (re-derived per chunk).
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
        Required by TextSplitter's abstract interface. Operates on a single
        page's text with NO cross-page context or page number — prefer
        split_documents() instead. Provided so this class is a structurally
        complete TextSplitter for any generic code that only knows the
        split_text(str) -> list[str] contract, with the limitation documented
        rather than silently accepted.
        """
        # Wrap the single text as a fake page 1 so _chunk_pages() can run.
        pages = [{"page_number": 1, "text": text}]
        chunk_dicts = self._chunk_pages(pages, metadata={})
        return [c["text"] for c in chunk_dicts]

    # ── Internal chunking pipeline ────────────────────────────────────────────
    # The logic below is the pre-refactor chunk_pages() function, proven against
    # the real corpus. Only the outer interface changed (functions -> class methods).

    def _chunk_pages(self, pages: list[dict], metadata: dict) -> list[dict]:
        """
        Full chunking pipeline: tag lines with page numbers, detect headings,
        merge short sections, and split any oversized sections by paragraph.
        Returns a list of chunk dicts (not yet wrapped in Document objects).
        """
        if not pages:
            return []

        tagged_lines = self._build_tagged_lines(pages)
        sections = self._split_into_sections(tagged_lines)
        sections = self._merge_short_sections(sections)

        chunks: list[dict] = []
        for section in sections:
            if not section["text"]:
                # Section ended up empty after merging — skip it entirely.
                continue
            # A single section may produce multiple sub-chunks if it's oversized.
            sub_texts = self._split_long_section(section["text"])
            for sub_text in sub_texts:
                chunks.append(self._make_chunk(sub_text, section, metadata, len(chunks)))

        return chunks

    def _build_tagged_lines(self, pages: list[dict]) -> list[tuple[str, int]]:
        """
        Flatten all pages into a single list of (line_text, page_number) tuples,
        stripping noise (running headers, figure captions) and inserting
        PARA_BREAK sentinels wherever the original PDF text had a blank line.

        The result preserves paragraph structure across page boundaries,
        which is essential for accurate page_range citation and meaningful
        paragraph-level splitting of oversized sections.
        """
        tagged: list[tuple[str, int]] = []
        for pg in pages:
            page_number = pg["page_number"]
            prev_was_blank = True   # treat the start of each page as preceded by a blank
            for raw_line in pg["text"].split("\n"):
                line = raw_line.strip()

                if not line:
                    # A blank line marks a paragraph boundary — record PARA_BREAK
                    # once per consecutive blank-line run (prev_was_blank guard).
                    if not prev_was_blank and tagged:
                        tagged.append((PARA_BREAK, page_number))
                    prev_was_blank = True
                    continue

                # Skip noise lines: running page headers and figure captions.
                if RUNNING_HEADER_RE.match(line):
                    continue
                if FIGURE_CAPTION_RE.match(line):
                    continue

                tagged.append((line, page_number))
                prev_was_blank = False
        return tagged

    def _split_into_sections(self, tagged_lines: list[tuple[str, int]]) -> list[dict]:
        """
        Group tagged lines into sections, starting a new section at each
        detected heading. Every heading always produces a section, even an
        empty one, so no heading is silently dropped.

        Each section dict has: heading (str|None), text (str), start_page (int),
        end_page (int).
        """
        sections: list[dict] = []
        current_heading: str | None = None
        current_lines: list[str] = []
        current_pages: list[int] = []
        heading_page: int | None = None
        have_content_yet = False   # don't flush before any content has been seen

        def flush():
            """Flush the current section into `sections`."""
            # Determine the page range for this section. If no content lines
            # were collected (empty section), fall back to the heading's own
            # page so the citation has a real, correct page number.
            if current_pages:
                pages_for_section = current_pages
            elif heading_page is not None:
                pages_for_section = [heading_page]
            elif tagged_lines:
                pages_for_section = [tagged_lines[0][1]]
            else:
                pages_for_section = [1]

            # Strip leading/trailing PARA_BREAK sentinels and whitespace
            # from the assembled text before storing.
            text = "\n".join(current_lines).strip().strip(PARA_BREAK).strip()
            sections.append({
                "heading":    current_heading,
                "text":       text,
                "start_page": min(pages_for_section),
                "end_page":   max(pages_for_section),
            })

        i = 0
        n = len(tagged_lines)
        while i < n:
            line, page = tagged_lines[i]

            if line == PARA_BREAK:
                # Paragraph break: add the sentinel to the current section's
                # line buffer (only if we already have some content, to avoid
                # a PARA_BREAK at the very start of a section).
                if current_lines:
                    current_lines.append(PARA_BREAK)
                    current_pages.append(page)
                i += 1
                continue

            if HEADING_NUMBER_RE.match(line):
                # Found a heading number (e.g. "1.1"). The heading TITLE is the
                # next non-PARA_BREAK line. Skip any PARA_BREAK between them.
                title = None
                look = i + 1
                while look < n and tagged_lines[look][0] == PARA_BREAK:
                    look += 1
                if look < n:
                    title = tagged_lines[look][0]

                # Flush the previous section before starting the new one.
                if have_content_yet:
                    flush()
                have_content_yet = True

                # Build the full heading label: "1.1 Series Announcement"
                current_heading = f"{line} {title}".strip() if title else line
                current_lines = []
                current_pages = []
                # Remember this heading's page as the fallback citation page
                # for sections with no body text of their own.
                heading_page = page

                # Advance past the title line (or just past the number line
                # if no title was found).
                i = look + 1 if title else i + 1
                continue

            # Normal content line — add to the current section.
            current_lines.append(line)
            current_pages.append(page)
            have_content_yet = True
            i += 1

        # Flush the final section after the loop.
        if have_content_yet:
            flush()

        return sections

    def _merge_short_sections(self, sections: list[dict]) -> list[dict]:
        """
        Fold sections shorter than MIN_SECTION_CHARS forward into the next
        section, keeping the earlier (parent) heading as the merged chunk's label.

        This handles the common case where a parent heading like "1.1" has
        no body text of its own — it would produce a near-empty chunk if left
        alone, so we merge its content with the next section.
        """
        if not sections:
            return sections

        merged: list[dict] = []
        # `pending` holds the current section waiting to be either emitted or
        # merged forward into the next one.
        pending: dict | None = None

        for section in sections:
            if pending is None:
                # First section — just start accumulating.
                pending = dict(section)
                continue

            if len(pending["text"]) < MIN_SECTION_CHARS:
                # pending is too short — absorb the current section into it.
                # Keep pending's heading (parent label) as the merged chunk's topic.
                if pending["text"] and section["text"]:
                    # Both have content: join with a paragraph break.
                    pending["text"] = f"{pending['text']}\n\n{section['text']}".strip()
                else:
                    # One or both are empty: simple concatenation.
                    pending["text"] = (pending["text"] + section["text"]).strip()
                # Extend the page range to cover the absorbed section.
                pending["end_page"] = max(pending["end_page"], section["end_page"])
                # Do NOT update heading — we keep the parent heading as the label.
                continue

            # pending is long enough — emit it and start a fresh pending.
            merged.append(pending)
            pending = dict(section)

        # Don't forget the last pending section.
        if pending is not None:
            merged.append(pending)

        return merged

    def _split_long_section(self, text: str) -> list[str]:
        """
        Split an oversized section into sub-chunks by paragraph, with overlap
        between adjacent sub-chunks so no sentence is lost at a chunk boundary.

        Splits preferentially at PARA_BREAK sentinels (real paragraph boundaries
        from the original PDF); falls back to double-newlines if no sentinels
        are present. Returns the text of each sub-chunk as a clean string
        (PARA_BREAK sentinels replaced by double-newlines in the output).
        """
        chunk_size_chars = self._chunk_size
        overlap_chars = self._chunk_overlap

        # Short enough to keep as-is — most sections take this path.
        if len(text) <= chunk_size_chars:
            return [text.replace(PARA_BREAK, "\n\n")]

        # Split into paragraphs, preferring PARA_BREAK sentinels (which mark
        # real blank lines from the PDF) over arbitrary double-newlines.
        if PARA_BREAK in text:
            paragraphs = [p.strip() for p in text.split(PARA_BREAK) if p.strip()]
        else:
            paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]

        if not paragraphs:
            # Edge case: text has no discernible paragraph structure — return as-is.
            paragraphs = [text.replace(PARA_BREAK, "\n\n")]

        chunks: list[str] = []
        current = ""

        for para in paragraphs:
            # Try adding this paragraph to the current accumulator.
            candidate = f"{current}\n\n{para}".strip() if current else para
            if len(candidate) <= chunk_size_chars or not current:
                # Fits within the size limit, or this is the very first paragraph
                # (always include it even if it alone exceeds the limit).
                current = candidate
            else:
                # Adding this paragraph would overflow. Emit the current accumulator
                # as a chunk, then start a new one with an overlap tail from the
                # previous chunk to preserve context across the boundary.
                chunks.append(current)
                tail = current[-overlap_chars:] if overlap_chars > 0 else ""
                current = f"{tail}\n\n{para}".strip() if tail else para

        if current:
            chunks.append(current)

        return chunks

    def _make_chunk(self, text: str, section: dict, metadata: dict, index_in_doc: int) -> dict:
        """
        Build the final chunk dict from a section's text and the shared
        catalog metadata. This dict becomes Document.metadata when wrapped
        by split_documents().
        """
        start_page = section["start_page"]
        end_page   = section["end_page"]
        # page_range is only set when a chunk genuinely spans multiple pages;
        # single-page chunks use only page_number.
        page_range = None if start_page == end_page else f"{start_page}-{end_page}"

        return {
            # Short random ID generated at chunk creation time — see init_db.py's
            # "WHY chunk_id IS text" note for why we don't use a DB serial.
            "chunk_id":           uuid.uuid4().hex[:12],
            "text":               text,
            "topic":              section["heading"],   # nearest detected heading
            "page_number":        start_page,
            "page_range":         page_range,
            "chunk_index_in_doc": index_in_doc,
            "char_count":         len(text),
            "ingestion_date":     date.today().isoformat(),
            # Spread the shared catalog metadata (course_name, subject, chapter,
            # source_type, document_version, source) into the chunk dict.
            **metadata,
        }
