"""Orchestrates the whole RAG flow:  ingest -> chunk -> embed -> index   |   ask -> rewrite -> retrieve -> gate -> generate -> verify."""
from __future__ import annotations

import hashlib
import logging
import threading
import time
import uuid
from datetime import datetime, timezone

from rag.chunking import chunk_document
from rag.config import Settings
from rag.embeddings import Embedder, get_embedder
from rag.exceptions import InvalidQuestion, LLMBadResponse, LLMUnavailable, RAGError
from rag.generation import Generator
from rag.ingestion.loader import extension, extract_pages
from rag.ingestion.preprocess import normalize_for_hash, preprocess
from rag.llm import AnthropicLLM, LLMClient
from rag.memory import ConversationMemory, Turn, rewrite_question
from rag.retrieval import Retriever
from rag.schemas import AskResponse, Citation, DocumentRecord, IngestResult, RetrievalInfo
from rag.vectorstore import VectorStore

log = logging.getLogger("rag.pipeline")

NOT_FOUND = "I couldn't find information about that in the uploaded documents."
NO_DOCS = "No documents have been uploaded yet. Upload a PDF or TXT file first."


class RAGPipeline:
    def __init__(self, settings: Settings, embedder: Embedder, store: VectorStore, llm: LLMClient | None):
        self.s, self.embedder, self.store, self.llm = settings, embedder, store, llm
        self.retriever = Retriever(store, embedder, settings)
        self.generator = Generator(llm, settings.llm_model) if llm else None
        self.memory = ConversationMemory(settings.max_history_turns)
        self._lock = threading.RLock()

    @classmethod
    def build(cls, settings: Settings | None = None, llm: LLMClient | None = None, embedder: Embedder | None = None) -> "RAGPipeline":
        settings = settings or Settings.from_env()
        embedder = embedder or get_embedder(settings)
        store = VectorStore.load(settings.data_dir, embedder.dim, embedder.name) if settings.persist else VectorStore(embedder.dim, embedder.name)
        if llm is None and settings.anthropic_api_key:
            llm = AnthropicLLM(settings)
        if llm is None:
            log.warning("No ANTHROPIC_API_KEY: ingestion/retrieval work, /ask will return 503.")
        return cls(settings, embedder, store, llm)

    # ------------------------------------------------------------------ ingestion
    def ingest(self, filename: str, data: bytes) -> IngestResult:
        filename = (filename or "unnamed").replace("\\", "/").rsplit("/", 1)[-1]
        try:
            pages = extract_pages(filename, data, self.s.max_upload_mb * 1_000_000)
            pages = preprocess(pages, is_pdf=extension(filename) == ".pdf")
            if not pages:
                from rag.exceptions import EmptyDocument
                raise EmptyDocument(f"'{filename}' has no usable text after cleaning.")
            # Hash the CLEANED text so re-saved copies (CRLF, trailing spaces, renamed) are caught as duplicates.
            content_hash = hashlib.sha256(normalize_for_hash(" ".join(p.text for p in pages)).encode()).hexdigest()
            with self._lock:
                existing = self.store.find_by_content_hash(content_hash)
                if existing:
                    return IngestResult(filename=filename, status="duplicate", doc_id=existing.doc_id,
                                        duplicate_of=existing.filename,
                                        message=f"Identical content already indexed as '{existing.filename}'; skipped.")
                display = self._unique_name(filename)
                doc_id = content_hash[:12]
                chunks = chunk_document(doc_id, display, pages, self.s.chunk_size, self.s.chunk_overlap, self.s.min_chunk_chars)
                vectors = self.embedder.embed_documents([c.embedding_text for c in chunks])
                record = DocumentRecord(doc_id=doc_id, filename=display, content_hash=content_hash, pages=len(pages),
                                        chunks=len(chunks), chars=sum(len(p.text) for p in pages),
                                        ingested_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
                self.store.add_document(record, chunks, vectors)
                self._persist()
            log.info("ingested %s: %d pages, %d chunks", display, len(pages), len(chunks))
            return IngestResult(filename=display, status="ingested", doc_id=doc_id, pages=len(pages), chunks=len(chunks))
        except RAGError as exc:
            log.warning("ingest failed for %s: %s (%s)", filename, exc.code, exc.message)
            return IngestResult(filename=filename, status="failed", error_code=exc.code, message=exc.message, http_status=exc.status_code)

    def _unique_name(self, filename: str) -> str:
        if not self.store.filename_exists(filename):
            return filename
        stem, dot, ext = filename.rpartition(".")
        n = 2
        while self.store.filename_exists(f"{stem} (v{n}).{ext}"):
            n += 1
        return f"{stem} (v{n}).{ext}"

    def delete_document(self, doc_id: str) -> DocumentRecord:
        with self._lock:
            rec = self.store.remove_document(doc_id)
            self._persist()
            return rec

    def list_documents(self) -> list[DocumentRecord]:
        return list(self.store.documents.values())

    def _persist(self) -> None:
        if self.s.persist:
            self.store.save(self.s.data_dir)

    # ------------------------------------------------------------------ question answering
    def ask(self, question: str, session_id: str | None = None) -> AskResponse:
        t0 = time.perf_counter()
        question = (question or "").strip()
        if not question:
            raise InvalidQuestion("Question must not be empty.")
        if len(question) > self.s.max_question_chars:
            raise InvalidQuestion(f"Question is longer than {self.s.max_question_chars} characters.")
        session_id = session_id or uuid.uuid4().hex[:12]
        history = self.memory.history(session_id)

        def respond(**kw) -> AskResponse:
            resp = AskResponse(session_id=session_id, question=question, latency_ms=int((time.perf_counter() - t0) * 1000), **kw)
            self.memory.add(session_id, Turn(question, resp.rewritten_query, resp.answer, resp.answerable))
            return resp

        if not len(self.store):
            return respond(rewritten_query=question, answerable=False, answer=NO_DOCS, confidence="none",
                           unanswerable_reason="no_documents",
                           retrieval=RetrievalInfo(mode=self.s.retrieval_mode, top_score=0, threshold=self.s.min_relevance, passed_gate=False, chunks_used=0))
        if self.llm is None or self.generator is None:
            raise LLMUnavailable("ANTHROPIC_API_KEY is not configured, so answers cannot be generated.")

        standalone = rewrite_question(self.llm, self.s.rewrite_model, question, history)
        result = self.retriever.retrieve(standalone)
        info = RetrievalInfo(
            mode=result.mode, top_score=result.top_score, threshold=result.threshold, passed_gate=result.passed,
            chunks_used=len(result.chunks),
            nearest=[{"source": n.chunk.source, "page": n.chunk.page, "score": n.dense_score} for n in result.nearest],
        )
        # Gate: if nothing clears the relevance threshold we do not even call the LLM. Cheaper, faster, and it
        # removes the opportunity to hallucinate from irrelevant context.
        if not result.passed or not result.chunks:
            return respond(rewritten_query=standalone, answerable=False, answer=NOT_FOUND, confidence="none",
                           unanswerable_reason="no_relevant_passages", retrieval=info)

        answer, labelled, warnings = self.generator.generate(question, standalone, history, result)
        citations = [
            Citation(label=lab, chunk_id=labelled[lab].chunk.id, source=labelled[lab].chunk.source, page=labelled[lab].chunk.page,
                     score=labelled[lab].dense_score, snippet=labelled[lab].chunk.text[:300], also_in=labelled[lab].also_in)
            for lab in answer.citations
        ]
        return respond(
            rewritten_query=standalone, answerable=answer.answerable, answer=answer.answer,
            confidence=answer.confidence if answer.answerable else "none", citations=citations,
            conflict_detected=answer.conflict_detected, conflict_note=answer.conflict_note,
            unanswerable_reason=answer.unanswerable_reason, warnings=warnings, retrieval=info,
        )
