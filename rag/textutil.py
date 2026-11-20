"""Shared tokenisation helpers (used by BM25, the offline embedder and the grounding checks)."""
from __future__ import annotations

import re

STOPWORDS = frozenset(
    """a an and are as at be been but by can could did do does for from had has have how i if in into is it its
    me my of on or our so than that the their them then there these they this those to was we were what when
    where which who whom why will with would you your about also any each per not no""".split()
)
_TOKEN = re.compile(r"[a-z0-9]+(?:[.'][a-z0-9]+)*")
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")


def stem(tok: str) -> str:
    """Very light suffix stripping so 'policies'/'policy', 'reset'/'resetting' line up."""
    if len(tok) <= 4 or any(c.isdigit() for c in tok):
        return tok
    if tok.endswith("ies"):
        return tok[:-3] + "y"
    for suf in ("ing", "ed", "es", "s"):
        if tok.endswith(suf) and len(tok) - len(suf) >= 3:
            return tok[: -len(suf)]
    return tok


def tokenize(text: str, *, keep_stopwords: bool = False) -> list[str]:
    toks = _TOKEN.findall(text.lower())
    return [stem(t) for t in toks if keep_stopwords or t not in STOPWORDS]


def numbers(text: str) -> set[str]:
    """Numeric tokens, normalised ('1,000' -> '1000'). Used for duplicate-vs-conflict decisions
    and for checking that numbers in an answer actually occur in the sources."""
    return {m.group(0).replace(",", "") for m in _NUMBER.finditer(text)}
