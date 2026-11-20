"""Step 2: text cleaning. Conservative by design: never invent or drop real content."""
from __future__ import annotations

import re
import unicodedata
from collections import Counter

from rag.ingestion.loader import Page

_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_PAGE_NUM = re.compile(r"^\s*(page\s+)?\d{1,4}(\s*(of|/)\s*\d{1,4})?\s*$", re.I)
_BULLET = re.compile(r"^\s*([-*•–]|\d+[.)]|[a-z][.)])\s+")


def normalize_for_hash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _base_clean(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)           # ligatures (ﬁ), fullwidth chars, nbsp
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CTRL.sub("", text)
    text = text.replace("­", "")                    # soft hyphen
    return text


def _is_heading(line: str, nxt: str) -> bool:
    s = line.strip()
    return bool(s) and len(s) <= 60 and not re.search(r"[.,;:!?]$", s) and bool(nxt.strip()) and nxt.strip()[0].isupper()


def _unwrap_pdf_lines(text: str) -> str:
    """PDF text has a newline at every visual line wrap. Re-join wrapped lines inside a paragraph,
    but keep bullets, headings and blank-line paragraph breaks."""
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    out: list[str] = []
    lines = text.split("\n")
    for i, line in enumerate(lines):
        line = line.rstrip()
        if not out or not line.strip():
            out.append(line)
            continue
        prev = out[-1]
        nxt_is_new_block = bool(_BULLET.match(line))
        prev_ends = bool(re.search(r"[.:;!?]$", prev.strip())) or not prev.strip()
        if nxt_is_new_block or prev_ends or _is_heading(prev, line):
            out.append(line)
        else:
            out[-1] = prev + " " + line.strip()
    return "\n".join(out)


def _strip_repeated_lines(pages: list[Page]) -> list[Page]:
    """Drop headers/footers: lines that repeat on most pages, and bare page numbers."""
    if len(pages) < 2:
        return [Page(p.number, "\n".join(l for l in p.text.split("\n") if not _PAGE_NUM.match(l))) for p in pages]
    counts: Counter[str] = Counter()
    for p in pages:
        counts.update({l.strip() for l in p.text.split("\n") if l.strip()})
    repeated = {l for l, c in counts.items() if c >= max(2, 0.6 * len(pages)) and len(l) < 120}
    cleaned = []
    for p in pages:
        keep = [l for l in p.text.split("\n") if l.strip() not in repeated and not _PAGE_NUM.match(l)]
        cleaned.append(Page(p.number, "\n".join(keep)))
    return cleaned


def preprocess(pages: list[Page], *, is_pdf: bool) -> list[Page]:
    pages = [Page(p.number, _base_clean(p.text)) for p in pages]
    if is_pdf:
        pages = _strip_repeated_lines(pages)
        pages = [Page(p.number, _unwrap_pdf_lines(p.text)) for p in pages]
    result = []
    for p in pages:
        text = re.sub(r"[ \t]+", " ", p.text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        if text:
            result.append(Page(p.number, text))
    return result
