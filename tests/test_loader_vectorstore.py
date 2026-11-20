import numpy as np
import pytest

from rag.exceptions import CorruptFile, EmptyDocument, FileTooLarge, IndexMismatch, UnsupportedFileType, DocumentNotFound
from rag.ingestion.loader import extract_pages
from rag.schemas import Chunk, DocumentRecord
from rag.vectorstore import VectorStore
from tests.conftest import make_pdf

MB = 1_000_000


def test_reads_txt_and_pdf_with_page_numbers():
    assert extract_pages("a.txt", b"hello world", MB)[0].text == "hello world"
    pages = extract_pages("a.pdf", make_pdf(["first page text", "second page text"]), MB)
    assert [p.number for p in pages] == [1, 2] and "second" in pages[1].text


def test_cp1252_text_is_decoded():
    assert "café" in extract_pages("a.txt", "café".encode("cp1252"), MB)[0].text


@pytest.mark.parametrize("name,data,exc", [
    ("a.docx", b"x", UnsupportedFileType),
    ("noext", b"x", UnsupportedFileType),
    ("a.txt", b"", EmptyDocument),
    ("a.txt", b"   \n  ", EmptyDocument),
    ("a.pdf", b"this is not a pdf", CorruptFile),
    ("a.pdf", b"%PDF-1.4 garbage garbage", CorruptFile),
    ("a.txt", b"abc\x00\x01\x02binary", CorruptFile),
])
def test_invalid_inputs_raise_typed_errors(name, data, exc):
    with pytest.raises(exc):
        extract_pages(name, data, MB)


def test_oversized_file_rejected():
    with pytest.raises(FileTooLarge):
        extract_pages("a.txt", b"x" * 2000, 1000)


def test_pdf_without_text_layer_is_reported_as_needing_ocr():
    from reportlab.pdfgen import canvas
    import io
    buf = io.BytesIO(); c = canvas.Canvas(buf); c.rect(10, 10, 100, 100, fill=1); c.save()
    with pytest.raises(EmptyDocument, match="OCR"):
        extract_pages("scan.pdf", buf.getvalue(), MB)


def _store(dim=4):
    s = VectorStore(dim, "test")
    chunks = [Chunk(id=f"c{i}", doc_id="d1" if i < 2 else "d2", source="f", index=i, text=f"t{i}", norm_hash=str(i)) for i in range(3)]
    vecs = np.eye(4, dtype="float32")[:3]
    for did, sl in (("d1", slice(0, 2)), ("d2", slice(2, 3))):
        s.add_document(DocumentRecord(doc_id=did, filename=did, content_hash=did, pages=1, chunks=len(chunks[sl]), chars=1, ingested_at="x"),
                       chunks[sl], vecs[sl])
    return s


def test_search_returns_nearest_first():
    ids, scores = _store().search(np.array([0, 1, 0, 0], dtype="float32"), 2)
    assert ids[0] == 1 and scores[0] == pytest.approx(1.0)


def test_remove_document_rebuilds_index():
    s = _store()
    s.remove_document("d1")
    assert len(s) == 1 and s.search(np.array([0, 0, 1, 0], dtype="float32"), 3)[0] == [0]
    with pytest.raises(DocumentNotFound):
        s.remove_document("d1")


def test_persistence_roundtrip_and_embedder_mismatch(tmp_path):
    s = _store(); s.save(tmp_path)
    loaded = VectorStore.load(tmp_path, 4, "test")
    assert len(loaded) == 3 and set(loaded.documents) == {"d1", "d2"}
    with pytest.raises(IndexMismatch):
        VectorStore.load(tmp_path, 4, "other-embedder")
