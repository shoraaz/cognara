"""
scripts/extract_toc.py
----------------------
One-off helper: print the table of contents (bookmarks) and page count
of a PDF, so we can choose the Phase 1 corpus subset.

Usage:
    py scripts/extract_toc.py "path/to/file.pdf"
"""

import sys
import fitz  # PyMuPDF


def main(pdf_path: str) -> None:
    doc = fitz.open(pdf_path)
    print(f"FILE: {pdf_path}")
    print(f"TOTAL_PAGES: {doc.page_count}")
    toc = doc.get_toc()  # [[level, title, page], ...]
    if not toc:
        print("NO_EMBEDDED_TOC")
    else:
        print(f"TOC_ENTRIES: {len(toc)}")
        print("---")
        for level, title, page in toc:
            indent = "  " * (level - 1)
            print(f"{page:>5} | {indent}{title}")
    doc.close()


if __name__ == "__main__":
    main(sys.argv[1])
