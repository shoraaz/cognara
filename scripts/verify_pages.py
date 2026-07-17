"""
scripts/verify_pages.py
-----------------------
Print the first ~200 chars of specific pages from a PDF, to verify that
chapter-boundary page numbers in the catalog are correct before ingestion.

Usage:
    py scripts/verify_pages.py "path/to/file.pdf" 96 135 162 191 230 251
"""

import sys
import fitz


def main(pdf_path: str, pages: list[int]) -> None:
    doc = fitz.open(pdf_path)
    for p in pages:
        # PDF pages are 0-indexed internally; TOC pages are 1-indexed
        page = doc[p - 1]
        text = page.get_text().strip().replace("\n", " ")
        snippet = text[:180]
        print(f"--- page {p} ---")
        print(snippet)
        print()
    doc.close()


if __name__ == "__main__":
    main(sys.argv[1], [int(x) for x in sys.argv[2:]])
