import io
from dataclasses import replace
from pathlib import Path

import pytest
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from rag.config import Settings
from rag.pipeline import RAGPipeline

SAMPLES = Path(__file__).resolve().parent.parent / "data" / "sample_docs"


class ScriptedLLM:
    """Deterministic stand-in for Claude: pops scripted structured answers / text, records every call."""

    def __init__(self, structured=None, text=None):
        self.structured, self.text = list(structured or []), list(text or [])
        self.calls: list[dict] = []

    def generate_structured(self, **kw):
        self.calls.append({"kind": "structured", **kw})
        item = self.structured.pop(0)
        if isinstance(item, Exception):
            raise item
        return item(kw) if callable(item) else item

    def complete_text(self, **kw):
        self.calls.append({"kind": "text", **kw})
        item = self.text.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def answer(**kw):
    base = dict(answerable=True, answer="", citations=[], confidence="high", conflict_detected=False,
                conflict_note="", unanswerable_reason="")
    base.update(kw)
    return base


def make_pdf(pages: list[str]) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    for text in pages:
        y = 780
        for line in text.split("\n"):
            c.drawString(50, y, line)
            y -= 14
        c.showPage()
    c.save()
    return buf.getvalue()


@pytest.fixture
def settings():
    return replace(Settings(), embedding_backend="hashing", min_relevance=0.08, persist=False)


@pytest.fixture
def make_pipeline(settings):
    def _make(llm=None, load_samples=True, **overrides):
        p = RAGPipeline.build(replace(settings, **overrides), llm=llm)
        if load_samples:
            for f in sorted(SAMPLES.iterdir()):
                p.ingest(f.name, f.read_bytes())
        return p
    return _make
