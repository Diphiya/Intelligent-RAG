import pytest

from rag.chunking import chunk_document, split_text
from rag.ingestion.loader import Page
from rag.ingestion.preprocess import preprocess


def test_chunks_respect_size_and_cover_all_text():
    text = "\n\n".join(f"Paragraph {i}. " + "word " * 60 for i in range(12))
    chunks = split_text(text, 400, 60)
    assert all(len(c) <= 400 for c in chunks)
    assert len(chunks) > 3
    for i in range(12):
        assert any(f"Paragraph {i}." in c for c in chunks)


def test_overlap_repeats_context_between_neighbours():
    text = " ".join(f"sentence{i} is here." for i in range(80))
    a, b = split_text(text, 300, 80)[:2]
    assert set(a.split()[-3:]) & set(b.split()[:12])


def test_very_long_token_is_hard_split_not_dropped():
    chunks = split_text("x" * 2500, 1000, 100)
    assert sum(len(c) for c in chunks) >= 2500 and all(len(c) <= 1000 for c in chunks)


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        split_text("abc", 100, 100)


def test_chunks_never_cross_pages_and_keep_page_numbers():
    pages = [Page(1, "Alpha " * 200), Page(2, "Beta " * 200)]
    chunks = chunk_document("d", "f.pdf", pages, 300, 50, 20)
    assert {c.page for c in chunks} == {1, 2}
    assert all(("Alpha" in c.text) != ("Beta" in c.text) for c in chunks)


def test_preprocess_rejoins_wrapped_lines_and_hyphenation():
    raw = "The quick brown fox jumped over the\nlazy dog and the exam-\nple continues here.\n\n- bullet one\n- bullet two"
    out = preprocess([Page(1, raw)], is_pdf=True)[0].text
    assert "over the lazy dog" in out and "example continues" in out
    assert "- bullet one\n- bullet two" in out


def test_preprocess_removes_repeating_footer_and_page_numbers():
    pages = [Page(i, f"Company Confidential\nReal content number {i} is here.\n{i}") for i in (1, 2, 3)]
    out = preprocess(pages, is_pdf=True)
    assert all("Confidential" not in p.text and p.text.strip().endswith("here.") for p in out)


def test_preprocess_keeps_txt_line_structure():
    out = preprocess([Page(None, "Heading\r\nline one\r\nline two")], is_pdf=False)[0].text
    assert out == "Heading\nline one\nline two"
