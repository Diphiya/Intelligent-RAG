"""Central configuration. Every tunable lives here and can be overridden by env var."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Cosine-similarity gate per embedder. These are starting points: score scales differ
# between embedders, so ALWAYS re-calibrate (python -m evaluation.run_eval --calibrate).
DEFAULT_MIN_RELEVANCE = {"sentence-transformers": 0.60, "hashing": 0.08}


def load_dotenv(path: Path = Path(".env")) -> None:
    """Minimal .env reader (KEY=VALUE lines). Real environment variables always win."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _f(name: str, default: float) -> float:
    return float(os.getenv(name, default))


def _i(name: str, default: int) -> int:
    return int(os.getenv(name, default))


@dataclass(frozen=True)
class Settings:
    # --- LLM (Anthropic) ---
    anthropic_api_key: str | None = None
    llm_model: str = "claude-sonnet-5-5"
    rewrite_model: str = "claude-haiku-5-5"
    llm_timeout_s: float = 60.0
    llm_max_retries: int = 3          # SDK retries 429/5xx/connection errors with backoff
    llm_max_tokens: int = 2048
    llm_temperature: float = 0.0      # dropped automatically if the model rejects it
    # --- Embeddings ---
    embedding_backend: str = "sentence-transformers"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    # --- Chunking (characters; ~4 chars per token). 400 chars (~100 tokens) ~ one short policy/FAQ paragraph. Chosen from the
    #     real-embedding sweep (evaluation.run_eval --chunk-sweep, see docs/decisions.md D1): Hit@1 1.00 at 250/400 chars vs 0.80 at 1000+. ---
    chunk_size: int = 400
    chunk_overlap: int = 66
    min_chunk_chars: int = 40
    # --- Retrieval ---
    retrieval_mode: str = "hybrid"    # dense | bm25 | hybrid
    top_k: int = 5
    candidate_k: int = 20
    min_relevance: float = 0.50
    rrf_k: int = 60
    dedup_similarity: float = 0.95
    conflict_similarity: float = 0.75
    # --- Limits ---
    max_upload_mb: int = 20
    max_question_chars: int = 2000
    max_history_turns: int = 6
    # --- Storage ---
    data_dir: Path = Path("storage")
    persist: bool = True

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        backend = os.getenv("RAG_EMBEDDING_BACKEND", "sentence-transformers")
        return cls(
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY") or None,
            llm_model=os.getenv("RAG_LLM_MODEL", cls.llm_model),
            rewrite_model=os.getenv("RAG_REWRITE_MODEL", cls.rewrite_model),
            llm_timeout_s=_f("RAG_LLM_TIMEOUT_S", cls.llm_timeout_s),
            embedding_backend=backend,
            embedding_model=os.getenv("RAG_EMBEDDING_MODEL", cls.embedding_model),
            chunk_size=_i("RAG_CHUNK_SIZE", cls.chunk_size),
            chunk_overlap=_i("RAG_CHUNK_OVERLAP", cls.chunk_overlap),
            retrieval_mode=os.getenv("RAG_RETRIEVAL_MODE", cls.retrieval_mode),
            top_k=_i("RAG_TOP_K", cls.top_k),
            min_relevance=_f("RAG_MIN_RELEVANCE", DEFAULT_MIN_RELEVANCE.get(backend, 0.5)),
            max_upload_mb=_i("RAG_MAX_UPLOAD_MB", cls.max_upload_mb),
            data_dir=Path(os.getenv("RAG_DATA_DIR", "storage")),
        )
