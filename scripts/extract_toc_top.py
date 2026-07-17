"""
scripts/extract_toc_top.py
--------------------------
Print only top-level (level 1) TOC entries with their start pages.
Used to pick clean chapter boundaries for the Phase 1 subset.

Usage:
    py scripts/extract_toc_top.py "path/to/file.pdf"
"""

import sys
import fitz


def main(pdf_path: str) -> None:
    doc = fitz.open(pdf_path)
    total = doc.page_count
    toc = doc.get_toc()
    tops = [(title.strip(), page) for level, title, page in toc if level == 1]
    print(f"TOTAL_PAGES: {total}")
    print(f"TOP_LEVEL_ENTRIES: {len(tops)}")
    print("---")
    for i, (title, page) in enumerate(tops):
        # end page = next top's start - 1 (or total for the last one)
        end = (tops[i + 1][1] - 1) if i + 1 < len(tops) else total
        span = end - page + 1
        print(f"{i+1:>3}. p{page:>4}-{end:<4} ({span:>3}p) | {title}")
    doc.close()


if __name__ == "__main__":
    main(sys.argv[1])
