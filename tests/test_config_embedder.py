import sys
import types

import numpy as np

from rag.config import Settings, load_dotenv


def test_dotenv_is_loaded_but_real_env_wins(tmp_path, monkeypatch):
    f = tmp_path / ".env"
    f.write_text('# c\nANTHROPIC_API_KEY="sk-from-file"\nRAG_TOP_K=7\nBAD LINE\n')
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("RAG_TOP_K", "3")
    load_dotenv(f)
    s = Settings.from_env()
    assert s.anthropic_api_key == "sk-from-file" and s.top_k == 3


def test_min_relevance_defaults_follow_the_embedder(monkeypatch):
    monkeypatch.setenv("RAG_EMBEDDING_BACKEND", "hashing")
    monkeypatch.delenv("RAG_MIN_RELEVANCE", raising=False)
    assert Settings.from_env().min_relevance < 0.2


def test_sentence_transformer_wrapper_with_fake_model(monkeypatch):
    seen = {}

    class FakeST:
        def __init__(self, name): seen["name"] = name
        def get_sentence_embedding_dimension(self): return 3
        def encode(self, texts, **kw):
            seen["texts"] = texts
            return np.ones((len(texts), 3), dtype="float32")

    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=FakeST))
    from rag.embeddings import BGE_QUERY_PREFIX, SentenceTransformerEmbedder
    e = SentenceTransformerEmbedder("BAAI/bge-small-en-v1.5")
    assert e.dim == 3 and e.embed_documents(["a", "b"]).shape == (2, 3)
    e.embed_query("hello")
    assert seen["texts"] == [BGE_QUERY_PREFIX + "hello"]          # queries get the bge instruction, documents don't
    e2 = SentenceTransformerEmbedder("sentence-transformers/all-MiniLM-L6-v2")
    e2.embed_query("hello")
    assert seen["texts"] == ["hello"]


def test_model_load_failure_is_a_typed_error(monkeypatch):
    import pytest
    from rag.exceptions import EmbeddingError

    def boom(name): raise OSError("offline")
    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=boom))
    from rag.embeddings import SentenceTransformerEmbedder
    with pytest.raises(EmbeddingError, match="Could not load"):
        SentenceTransformerEmbedder("x")
