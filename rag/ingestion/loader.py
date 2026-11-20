"""Step 1-2: validate an upload and extract raw text per page."""
from __future__ import annotations

import io
from dataclasses import dataclass

from pypdf import PdfReader
from pypdf.errors import PyPdfError

from rag.exceptions import CorruptFile, EmptyDocument, FileTooLarge, UnsupportedFileType

ALLOWED = {".pdf", ".txt"}


@dataclass
class Page:
    number: int | None   # 1-based for PDF, None for TXT
    text: str


def extension(filename: str) -> str:
    return "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""


def extract_pages(filename: str, data: bytes, max_bytes: int) -> list[Page]:
    ext = extension(filename)
    if ext not in ALLOWED:
        raise UnsupportedFileType(f"'{filename}': only .pdf and .txt are supported (got '{ext or 'no extension'}').")
    if len(data) > max_bytes:
        raise FileTooLarge(f"'{filename}' is {len(data) / 1e6:.1f} MB; the limit is {max_bytes / 1e6:.0f} MB.")
    if not data.strip():
        raise EmptyDocument(f"'{filename}' is empty.")
    pages = _read_pdf(filename, data) if ext == ".pdf" else _read_txt(filename, data)
    if not any(p.text.strip() for p in pages):
        raise EmptyDocument(
            f"'{filename}' contains no extractable text. If it is a scanned/image-only PDF, "
            "it needs OCR, which this system does not support."
        )
    return pages


def _read_txt(filename: str, data: bytes) -> list[Page]:
    if b"\x00" in data[:4096]:
        raise CorruptFile(f"'{filename}' looks like a binary file, not text.")
    for enc in ("utf-8-sig", "cp1252"):
        try:
            return [Page(None, data.decode(enc))]
        except UnicodeDecodeError:
            continue
    raise CorruptFile(f"'{filename}' could not be decoded as text.")


def _read_pdf(filename: str, data: bytes) -> list[Page]:
    if not data.lstrip().startswith(b"%PDF"):
        raise CorruptFile(f"'{filename}' is not a valid PDF (missing %PDF header).")
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted and not reader.decrypt(""):
            raise CorruptFile(f"'{filename}' is password-protected.")
        pages = []
        for i, page in enumerate(reader.pages, start=1):
            try:
                pages.append(Page(i, page.extract_text() or ""))
            except Exception:  # one bad page must not lose the whole document
                pages.append(Page(i, ""))
        return pages
    except CorruptFile:
        raise
    except (PyPdfError, ValueError, KeyError, OSError, RecursionError) as exc:
        raise CorruptFile(f"'{filename}' could not be parsed as a PDF: {exc}") from exc
