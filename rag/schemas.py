"""Pydantic models: internal data objects and the public API contract."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# ---------- internal ----------
class Chunk(BaseModel):
    id: str
    doc_id: str
    source: str                      # display filename
    page: int | None = None          # 1-based page for PDFs, None for TXT
    index: int                       # chunk number within the document
    text: str
    norm_hash: str                   # hash of whitespace/case-normalised text (exact-duplicate detection)

    @property
    def title(self) -> str:
        stem = self.source.rsplit(".", 1)[0]
        return stem.replace("_", " ").replace("-", " ")

    @property
    def embedding_text(self) -> str:
        # Contextual header: the document title disambiguates chunks such as 2023 vs 2024 policies.
        return f"{self.title}\n{self.text}"


class DocumentRecord(BaseModel):
    doc_id: str
    filename: str
    content_hash: str
    pages: int
    chunks: int
    chars: int
    ingested_at: str


class RetrievedChunk(BaseModel):
    chunk: Chunk
    dense_score: float               # cosine similarity (absolute scale -> used for the relevance gate)
    bm25_score: float = 0.0
    rank_score: float = 0.0          # score used for ordering (RRF for hybrid)
    also_in: list[str] = Field(default_factory=list)   # other sources holding a duplicate of this chunk


class ConflictHint(BaseModel):
    chunk_a: str
    chunk_b: str
    similarity: float
    numbers_a: list[str]
    numbers_b: list[str]


# ---------- LLM structured output ----------
class LLMAnswer(BaseModel):
    answerable: bool
    answer: str = ""
    citations: list[str] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"] = "low"
    conflict_detected: bool = False
    conflict_note: str = ""
    unanswerable_reason: str = ""


# ---------- API contract ----------
class IngestResult(BaseModel):
    filename: str
    status: Literal["ingested", "duplicate", "failed"]
    doc_id: str | None = None
    pages: int = 0
    chunks: int = 0
    duplicate_of: str | None = None
    error_code: str | None = None
    message: str | None = None
    http_status: int = 200


class Citation(BaseModel):
    label: str                       # S1, S2 ... as used inline in the answer
    chunk_id: str
    source: str
    page: int | None
    score: float
    snippet: str
    also_in: list[str] = Field(default_factory=list)


class RetrievalInfo(BaseModel):
    mode: str
    top_score: float
    threshold: float
    passed_gate: bool
    chunks_used: int
    nearest: list[dict] = Field(default_factory=list)   # closest chunks, shown when the gate rejects


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    session_id: str | None = None


class AskResponse(BaseModel):
    session_id: str
    question: str
    rewritten_query: str
    answerable: bool
    answer: str
    confidence: Literal["high", "medium", "low", "none"]
    citations: list[Citation] = Field(default_factory=list)
    conflict_detected: bool = False
    conflict_note: str = ""
    unanswerable_reason: str = ""
    warnings: list[str] = Field(default_factory=list)
    retrieval: RetrievalInfo
    latency_ms: int = 0


class ErrorBody(BaseModel):
    code: str
    message: str
