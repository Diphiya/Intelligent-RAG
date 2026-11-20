"""Step 3: recursive, structure-aware chunking with overlap.

Split on the largest natural boundary that keeps pieces <= chunk_size (paragraph > line > sentence >
clause > word), then greedily pack the pieces back together. Chunks never cross page boundaries so
every citation has an exact page number.
"""
from __future__ import annotations

import hashlib

from rag.ingestion.loader import Page
from rag.ingestion.preprocess import normalize_for_hash
from rag.schemas import Chunk

_SEPS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "]


def _atoms(text: str, size: int, level: int = 0) -> list[str]:
    """Break text into pieces <= size that concatenate back to the original text."""
    if len(text) <= size:
        return [text]
    if level >= len(_SEPS):                       # no separator left: hard cut (e.g. a very long token)
        return [text[i : i + size] for i in range(0, len(text), size)]
    sep = _SEPS[level]
    parts = text.split(sep)
    if len(parts) == 1:
        return _atoms(text, size, level + 1)
    atoms: list[str] = []
    for i, part in enumerate(parts):
        piece = part + (sep if i < len(parts) - 1 else "")
        atoms.extend(_atoms(piece, size, level + 1) if len(piece) > size else [piece])
    return atoms


def _overlap_tail(chunk: str, overlap: int) -> str:
    if overlap <= 0 or len(chunk) <= overlap:
        return ""
    tail = chunk[-overlap:]
    cut = tail.find(" ")                          # start the overlap on a word boundary
    return tail[cut + 1 :] if 0 <= cut < len(tail) - 1 else tail


def split_text(text: str, size: int, overlap: int) -> list[str]:
    if overlap >= size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    chunks: list[str] = []
    current = ""
    for atom in _atoms(text, size):
        if len(current) + len(atom) <= size:
            current += atom
            continue
        if current.strip():
            chunks.append(current.strip())
        tail = _overlap_tail(current, overlap)
        current = (tail + atom) if len(tail) + len(atom) <= size else atom
    if current.strip():
        chunks.append(current.strip())
    return chunks


def chunk_document(doc_id: str, source: str, pages: list[Page], size: int, overlap: int, min_chars: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    idx = 0
    for page in pages:
        pieces = split_text(page.text, size, overlap)
        for piece in pieces:
            if len(piece) < min_chars and len(pieces) > 1:
                continue                          # drop tiny trailing fragments (stray headers)
            chunks.append(
                Chunk(
                    id=f"{doc_id}-p{page.number or 0}-c{idx}",
                    doc_id=doc_id,
                    source=source,
                    page=page.number,
                    index=idx,
                    text=piece,
                    norm_hash=hashlib.sha256(normalize_for_hash(piece).encode()).hexdigest()[:16],
                )
            )
            idx += 1
    return chunks
