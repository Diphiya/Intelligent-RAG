"""Step 4: FAISS vector index + chunk/document registry, persisted to disk.

IndexFlatIP over L2-normalised vectors == exact cosine search. Exact search is the right choice at
this scale (thousands of chunks); an ANN index (HNSW/IVF) only pays off at 100k+ chunks.
The raw vectors are kept next to the index so documents can be deleted by rebuilding the index.
"""
from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from rag.exceptions import DocumentNotFound, IndexMismatch
from rag.schemas import Chunk, DocumentRecord


class VectorStore:
    def __init__(self, dim: int, embedder_name: str):
        self.dim = dim
        self.embedder_name = embedder_name
        self.chunks: list[Chunk] = []
        self.documents: dict[str, DocumentRecord] = {}
        self.vectors = np.zeros((0, dim), dtype="float32")
        self.version = 0                       # bumped on every change (invalidates the BM25 cache)
        self._index = faiss.IndexFlatIP(dim)

    def __len__(self) -> int:
        return len(self.chunks)

    # ---- mutation ----
    def add_document(self, record: DocumentRecord, chunks: list[Chunk], vectors: np.ndarray) -> None:
        assert len(chunks) == len(vectors)
        self.documents[record.doc_id] = record
        self.chunks.extend(chunks)
        self.vectors = np.vstack([self.vectors, vectors.astype("float32")])
        self._index.add(vectors.astype("float32"))
        self.version += 1

    def remove_document(self, doc_id: str) -> DocumentRecord:
        if doc_id not in self.documents:
            raise DocumentNotFound(f"No document with id '{doc_id}'.")
        record = self.documents.pop(doc_id)
        keep = [i for i, c in enumerate(self.chunks) if c.doc_id != doc_id]
        self.chunks = [self.chunks[i] for i in keep]
        self.vectors = self.vectors[keep] if keep else np.zeros((0, self.dim), dtype="float32")
        self._rebuild()
        self.version += 1
        return record

    def _rebuild(self) -> None:
        self._index = faiss.IndexFlatIP(self.dim)
        if len(self.vectors):
            self._index.add(self.vectors)

    # ---- query ----
    def search(self, qvec: np.ndarray, k: int) -> tuple[list[int], list[float]]:
        if not len(self.chunks):
            return [], []
        scores, ids = self._index.search(qvec.reshape(1, -1).astype("float32"), min(k, len(self.chunks)))
        pairs = [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]
        return [i for i, _ in pairs], [s for _, s in pairs]

    def find_by_content_hash(self, content_hash: str) -> DocumentRecord | None:
        return next((d for d in self.documents.values() if d.content_hash == content_hash), None)

    def filename_exists(self, filename: str) -> bool:
        return any(d.filename == filename for d in self.documents.values())

    # ---- persistence ----
    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "vectors.npy", self.vectors)
        payload = {
            "embedder": self.embedder_name,
            "dim": self.dim,
            "chunks": [c.model_dump() for c in self.chunks],
            "documents": [d.model_dump() for d in self.documents.values()],
        }
        tmp = directory / "store.json.tmp"
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(directory / "store.json")  # atomic: a crash never leaves a half-written store

    @classmethod
    def load(cls, directory: Path, dim: int, embedder_name: str) -> "VectorStore":
        meta = directory / "store.json"
        if not meta.exists():
            return cls(dim, embedder_name)
        payload = json.loads(meta.read_text(encoding="utf-8"))
        if payload["embedder"] != embedder_name or payload["dim"] != dim:
            raise IndexMismatch(
                f"Stored index was built with '{payload['embedder']}' but the configured embedder is "
                f"'{embedder_name}'. Delete '{directory}' (or re-upload documents) to rebuild."
            )
        store = cls(dim, embedder_name)
        store.chunks = [Chunk(**c) for c in payload["chunks"]]
        store.documents = {d["doc_id"]: DocumentRecord(**d) for d in payload["documents"]}
        store.vectors = np.load(directory / "vectors.npy").astype("float32")
        store._rebuild()
        return store
