import pytest
from fastapi.testclient import TestClient

from rag.api import create_app
from tests.conftest import ScriptedLLM, answer, make_pdf


@pytest.fixture
def client_factory(make_pipeline):
    def _make(llm=None, load_samples=False, **kw):
        return TestClient(create_app(make_pipeline(llm, load_samples=load_samples, **kw)))
    return _make


def test_upload_multiple_files_with_mixed_results(client_factory):
    c = client_factory()
    files = [("files", ("a.txt", b"Alpha beta gamma. The warranty lasts two years.", "text/plain")),
             ("files", ("b.pdf", make_pdf(["PDF page about batteries and runtime."]), "application/pdf")),
             ("files", ("c.docx", b"zzz", "application/octet-stream")),
             ("files", ("d.txt", b"Alpha beta gamma. The warranty lasts two years.", "text/plain"))]
    r = c.post("/documents", files=files)
    assert r.status_code == 200
    by = {x["filename"]: x for x in r.json()}
    assert by["a.txt"]["status"] == "ingested" and by["b.pdf"]["status"] == "ingested" and by["b.pdf"]["pages"] == 1
    assert by["c.docx"]["status"] == "failed" and by["c.docx"]["error_code"] == "unsupported_file_type"
    assert by["d.txt"]["status"] == "duplicate" and by["d.txt"]["duplicate_of"] == "a.txt"
    assert len(c.get("/documents").json()) == 2


@pytest.mark.parametrize("name,data,status,code", [
    ("x.docx", b"1", 415, "unsupported_file_type"),
    ("x.pdf", b"not a pdf", 422, "corrupt_file"),
    ("x.txt", b"", 422, "no_extractable_text"),
])
def test_single_bad_upload_returns_matching_http_status(client_factory, name, data, status, code):
    r = client_factory().post("/documents", files=[("files", (name, data, "application/octet-stream"))])
    assert r.status_code == status and r.json()[0]["error_code"] == code


def test_oversized_upload_returns_413(client_factory):
    c = client_factory(max_upload_mb=0)
    r = c.post("/documents", files=[("files", ("big.txt", b"x" * 10, "text/plain"))])
    assert r.status_code == 413 and r.json()[0]["error_code"] == "file_too_large"


def test_ask_returns_structured_json(client_factory):
    c = client_factory(ScriptedLLM([answer(answer="Warranty is 2 years [S1].", citations=["S1"])]), load_samples=True)
    r = c.post("/ask", json={"question": "How long is the Orbit Home Hub warranty?"})
    body = r.json()
    assert r.status_code == 200 and body["answerable"] and body["citations"][0]["source"] == "orbit_home_hub_faq.txt"
    assert {"session_id", "rewritten_query", "confidence", "retrieval", "warnings", "latency_ms"} <= body.keys()


def test_ask_validation_and_error_mapping(client_factory):
    c = client_factory(ScriptedLLM(), load_samples=True)
    assert c.post("/ask", json={"question": ""}).status_code == 422
    assert c.post("/ask", json={}).status_code == 422
    c2 = client_factory(None, load_samples=True)
    r = c2.post("/ask", json={"question": "What is the minimum password length?"})
    assert r.status_code == 503 and r.json()["error"]["code"] == "llm_unavailable"


def test_delete_document_and_404(client_factory):
    c = client_factory(load_samples=True)
    doc = c.get("/documents").json()[0]
    assert c.delete(f"/documents/{doc['doc_id']}").status_code == 200
    r = c.delete(f"/documents/{doc['doc_id']}")
    assert r.status_code == 404 and r.json()["error"]["code"] == "document_not_found"


def test_health_and_ui(client_factory):
    c = client_factory(load_samples=True)
    h = c.get("/health").json()
    assert h["status"] == "ok" and h["documents"] == 6 and h["llm_configured"] is False
    assert c.get("/").status_code == 200
