"""Step 4: embedding backends behind one interface.

Anthropic does not offer an embedding model, so semantic search uses a local open-source model.
 - SentenceTransformerEmbedder: real semantic embeddings (default, BAAI/bge-small-en-v1.5).
 - HashingEmbedder: dependency-free lexical fallback for offline tests/CI. It is NOT semantic.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np

from rag.config import Settings
from rag.exceptions import EmbeddingError
from rag.textutil import tokenize

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder(Protocol):
    name: str
    dim: int

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


def _l2(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (x / norms).astype("float32")


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbeddingError("sentence-transformers is not installed (pip install sentence-transformers).") from exc
        try:
            self._model = SentenceTransformer(model_name)
        except Exception as exc:  # network / bad model name
            raise EmbeddingError(f"Could not load embedding model '{model_name}': {exc}") from exc
        self.name = f"st:{model_name}"
        dim_fn = getattr(self._model, "get_embedding_dimension", None) or self._model.get_sentence_embedding_dimension
        self.dim = int(dim_fn())
        lowered = model_name.lower()
        self._query_prefix = BGE_QUERY_PREFIX if "bge" in lowered and "-en" in lowered else ""

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        try:
            vecs = self._model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
        except Exception as exc:
            raise EmbeddingError(f"Embedding failed: {exc}") from exc
        return np.asarray(vecs, dtype="float32")

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([self._query_prefix + text])[0]


class HashingEmbedder:
    """Stateless hashed unigram+bigram bag-of-words with log-TF and L2 norm."""

    def __init__(self, n_features: int = 2**14):
        from sklearn.feature_extraction.text import HashingVectorizer

        self.name = f"hashing:{n_features}"
        self.dim = n_features
        self._vec = HashingVectorizer(
            analyzer=self._analyze, n_features=n_features, alternate_sign=False, norm=None
        )

    @staticmethod
    def _analyze(text: str) -> list[str]:
        toks = tokenize(text)
        return toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        m = self._vec.transform(texts).toarray().astype("float32")
        return _l2(np.log1p(m))

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]


def get_embedder(settings: Settings) -> Embedder:
    if settings.embedding_backend == "hashing":
        return HashingEmbedder()
    if settings.embedding_backend == "sentence-transformers":
        return SentenceTransformerEmbedder(settings.embedding_model)
    raise EmbeddingError(f"Unknown embedding backend '{settings.embedding_backend}'.")
