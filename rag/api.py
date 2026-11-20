"""FastAPI layer. Thin: validation + error mapping only; all logic lives in RAGPipeline."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from rag.exceptions import RAGError
from rag.pipeline import RAGPipeline
from rag.schemas import AskRequest, AskResponse, DocumentRecord, IngestResult

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
STATIC = Path(__file__).resolve().parent.parent / "static"


def create_app(pipeline: RAGPipeline | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.pipeline = pipeline or RAGPipeline.build()
        yield

    app = FastAPI(title="Document Q&A (RAG) API", version="1.0.0", lifespan=lifespan,
                  description="Upload PDF/TXT documents and ask grounded, cited questions. Powered by Claude.")
    if pipeline is not None:
        app.state.pipeline = pipeline

    @app.exception_handler(RAGError)
    async def rag_error_handler(_: Request, exc: RAGError):
        return JSONResponse(status_code=exc.status_code, content={"error": {"code": exc.code, "message": exc.message}})

    @app.exception_handler(Exception)
    async def unhandled(_: Request, exc: Exception):
        logging.getLogger("rag.api").exception("Unhandled error")
        return JSONResponse(status_code=500, content={"error": {"code": "internal_error", "message": "Unexpected server error."}})

    def pl(request: Request) -> RAGPipeline:
        return request.app.state.pipeline

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/health")
    def health(request: Request):
        p = pl(request)
        return {"status": "ok", "documents": len(p.store.documents), "chunks": len(p.store),
                "embedder": p.embedder.name, "llm_configured": p.llm is not None, "llm_model": p.s.llm_model}

    @app.post("/documents", response_model=list[IngestResult])
    async def upload(request: Request, files: list[UploadFile] = File(...)):
        """Upload one or more PDF/TXT files. Always returns a per-file result list.
        Status is 200 if at least one file was ingested/duplicate, else the first error's status code."""
        p = pl(request)
        limit = p.s.max_upload_mb * 1_000_000
        results: list[IngestResult] = []
        for f in files:
            data = await f.read(limit + 1)             # never buffer more than limit+1 bytes
            results.append(p.ingest(f.filename or "unnamed", data))
        ok = any(r.status != "failed" for r in results)
        status = 200 if ok else results[0].http_status
        return JSONResponse(status_code=status, content=[r.model_dump() for r in results])

    @app.get("/documents", response_model=list[DocumentRecord])
    def list_documents(request: Request):
        return pl(request).list_documents()

    @app.delete("/documents/{doc_id}", response_model=DocumentRecord)
    def delete_document(request: Request, doc_id: str):
        return pl(request).delete_document(doc_id)

    @app.post("/ask", response_model=AskResponse)
    def ask(request: Request, body: AskRequest):
        return pl(request).ask(body.question, body.session_id)

    @app.delete("/sessions/{session_id}")
    def clear_session(request: Request, session_id: str):
        return {"cleared": pl(request).memory.clear(session_id)}

    return app


app = create_app()
