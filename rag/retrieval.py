"""Step 5: hybrid retrieval with an explicit relevance gate, de-duplication and conflict hints.

Pipeline:  dense (FAISS) top-N  +  BM25 top-N  ->  RRF fusion  ->  relevance gate (cosine)
           -> near-duplicate collapse (numbers-aware) -> top-k  -> conflict hints
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict

import numpy as np

from rag.config import Settings
from rag.embeddings import Embedder
from rag.schemas import ConflictHint, RetrievedChunk
from rag.textutil import numbers, tokenize
from rag.vectorstore import VectorStore


class BM25:
    """Okapi BM25 over an inverted index."""

    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.n = len(docs)
        self.k1, self.b = k1, b
        self.doc_len = np.array([len(d) for d in docs], dtype="float32")
        self.avgdl = float(self.doc_len.mean()) if self.n else 0.0
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, toks in enumerate(docs):
            for tok, tf in Counter(toks).items():
                self.postings[tok].append((i, tf))

    def scores(self, query: list[str]) -> np.ndarray:
        out = np.zeros(self.n, dtype="float32")
        for tok in set(query):
            plist = self.postings.get(tok)
            if not plist:
                continue
            idf = math.log(1 + (self.n - len(plist) + 0.5) / (len(plist) + 0.5))
            for i, tf in plist:
                denom = tf + self.k1 * (1 - self.b + self.b * self.doc_len[i] / (self.avgdl or 1))
                out[i] += idf * tf * (self.k1 + 1) / denom
        return out


class RetrievalResult:
    def __init__(self, query, mode, chunks, passed, top_score, threshold, nearest, conflicts, considered):
        self.query, self.mode, self.chunks = query, mode, chunks
        self.passed, self.top_score, self.threshold = passed, top_score, threshold
        self.nearest, self.conflicts, self.considered = nearest, conflicts, considered


class Retriever:
    def __init__(self, store: VectorStore, embedder: Embedder, settings: Settings):
        self.store, self.embedder, self.s = store, embedder, settings
        self._bm25: BM25 | None = None
        self._bm25_version = -1

    def _get_bm25(self) -> BM25:
        if self._bm25 is None or self._bm25_version != self.store.version:
            self._bm25 = BM25([tokenize(c.embedding_text) for c in self.store.chunks])
            self._bm25_version = self.store.version
        return self._bm25

    def retrieve(
        self,
        query: str,
        *,
        mode: str | None = None,
        top_k: int | None = None,
        min_relevance: float | None = None,
        apply_gate: bool = True,
    ) -> RetrievalResult:
        mode = mode or self.s.retrieval_mode
        top_k = top_k or self.s.top_k
        threshold = self.s.min_relevance if min_relevance is None else min_relevance
        if not len(self.store):
            return RetrievalResult(query, mode, [], False, 0.0, threshold, [], [], 0)

        q = self.embedder.embed_query(query)
        cand_k = min(self.s.candidate_k, len(self.store))
        dense_ids, dense_scores = self.store.search(q, cand_k)
        bm = self._get_bm25().scores(tokenize(query))
        bm_ids = [int(i) for i in np.argsort(-bm)[:cand_k] if bm[i] > 0]

        if mode == "dense":
            ranked = {i: 1.0 / (1 + r) for r, i in enumerate(dense_ids)}
        elif mode == "bm25":
            ranked = {i: 1.0 / (1 + r) for r, i in enumerate(bm_ids)}
        else:  # hybrid: reciprocal rank fusion (robust: no score-scale tuning between the two systems)
            ranked = defaultdict(float)
            for lst in (dense_ids, bm_ids):
                for r, i in enumerate(lst):
                    ranked[i] += 1.0 / (self.s.rrf_k + r + 1)
        order = sorted(ranked, key=lambda i: -ranked[i])

        cos = self.store.vectors[order] @ q if order else np.array([])
        cos_by_id = dict(zip(order, cos.tolist()))
        top_score = dense_scores[0] if dense_scores else 0.0
        passed = top_score >= threshold

        pool = [i for i in order if (cos_by_id[i] >= threshold or not apply_gate)]
        kept = self._collapse_duplicates(pool, top_k)
        results = [
            RetrievedChunk(
                chunk=self.store.chunks[i],
                dense_score=round(cos_by_id[i], 4),
                bm25_score=round(float(bm[i]), 4),
                rank_score=round(ranked[i], 5),
                also_in=sorted(also),
            )
            for i, also in kept
        ]
        nearest = [
            RetrievedChunk(chunk=self.store.chunks[i], dense_score=round(s, 4)) for i, s in zip(dense_ids[:3], dense_scores[:3])
        ]
        return RetrievalResult(
            query, mode, results, passed, round(top_score, 4), threshold, nearest,
            self._conflict_hints(results), len(order),
        )

    def _collapse_duplicates(self, pool: list[int], top_k: int) -> list[tuple[int, set[str]]]:
        """Merge exact and near duplicates, but NEVER merge chunks whose numbers differ:
        two policy versions that differ only in '20 days' vs '25 days' are a conflict, not a duplicate."""
        kept: list[tuple[int, set[str]]] = []
        for i in pool:
            ci = self.store.chunks[i]
            dup_of = None
            for j, also in kept:
                cj = self.store.chunks[j]
                same_text = ci.norm_hash == cj.norm_hash
                near = float(self.store.vectors[i] @ self.store.vectors[j]) >= self.s.dedup_similarity
                if same_text or (near and numbers(ci.text) == numbers(cj.text)):
                    dup_of = also
                    if ci.source != cj.source:
                        also.add(ci.source)
                    break
            if dup_of is None:
                kept.append((i, set()))
                if len(kept) >= top_k:
                    break
        return kept

    def _conflict_hints(self, results: list[RetrievedChunk]) -> list[ConflictHint]:
        """Heuristic hint for the LLM: near-identical topic, different documents, different numbers."""
        hints = []
        idx = {c.id: n for n, c in enumerate(self.store.chunks)}
        for a in range(len(results)):
            for b in range(a + 1, len(results)):
                ca, cb = results[a].chunk, results[b].chunk
                if ca.doc_id == cb.doc_id:
                    continue
                sim = float(self.store.vectors[idx[ca.id]] @ self.store.vectors[idx[cb.id]])
                na, nb = numbers(ca.text), numbers(cb.text)
                if sim >= self.s.conflict_similarity and na and nb and na != nb:
                    hints.append(ConflictHint(chunk_a=ca.id, chunk_b=cb.id, similarity=round(sim, 3),
                                              numbers_a=sorted(na), numbers_b=sorted(nb)))
        return hints
