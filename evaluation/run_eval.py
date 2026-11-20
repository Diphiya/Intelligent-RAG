"""Evaluation harness.

  python -m evaluation.run_eval                       # retrieval + relevance-gate metrics (no API key needed)
  python -m evaluation.run_eval --calibrate           # also sweep the relevance threshold and recommend one
  python -m evaluation.run_eval --chunk-sweep         # compare chunk sizes
  python -m evaluation.run_eval --backend sentence-transformers --calibrate --llm
                                                      # full run: real embeddings + Claude generation metrics

Retrieval relevance is judged at CHUNK level: a retrieved chunk is relevant if it comes from the expected
source AND contains one of the evidence phrases. This stays valid when chunking parameters change.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path

from rag.config import DEFAULT_MIN_RELEVANCE, Settings
from rag.pipeline import RAGPipeline

ROOT = Path(__file__).resolve().parent.parent
MODES = ["dense", "bm25", "hybrid"]


def build_pipeline(backend: str, chunk_size: int | None, min_relevance: float | None, llm=None, embedding_model: str | None = None) -> RAGPipeline:
    s = replace(Settings.from_env(), embedding_backend=backend, persist=False,
                min_relevance=min_relevance if min_relevance is not None else DEFAULT_MIN_RELEVANCE.get(backend, 0.5))
    if embedding_model:
        s = replace(s, embedding_model=embedding_model)
    if chunk_size:
        s = replace(s, chunk_size=chunk_size, chunk_overlap=max(20, chunk_size // 6))
    p = RAGPipeline.build(s, llm=llm)
    for f in sorted((ROOT / "data" / "sample_docs").iterdir()):
        p.ingest(f.name, f.read_bytes())
    return p


def covered(chunk, evidence) -> set[int]:
    text = chunk.text.lower()
    return {i for i, ev in enumerate(evidence) if chunk.source == ev["source"] and any(c.lower() in text for c in ev["contains"])}


def retrieval_metrics(p: RAGPipeline, queries, mode: str, k: int = 5, use_standalone: bool = True) -> dict:
    rows = []
    for q in (x for x in queries if x["answerable"]):
        text = q.get("standalone", q["question"]) if use_standalone else q["question"]
        res = p.retriever.retrieve(text, mode=mode, top_k=k, apply_gate=False)
        first, cov, rel_top3 = None, set(), 0
        for rank, rc in enumerate(res.chunks, start=1):
            c = covered(rc.chunk, q["evidence"])
            if c and first is None:
                first = rank
            cov |= c if rank <= k else set()
            rel_top3 += bool(c) if rank <= 3 else 0
        rows.append({"id": q["id"], "first_rank": first, "recall": len(cov) / len(q["evidence"]), "p3": rel_top3 / 3})
    n = len(rows)
    hit = lambda r: sum(1 for x in rows if x["first_rank"] and x["first_rank"] <= r) / n
    return {"mode": mode, "n": n, "hit@1": hit(1), "hit@3": hit(3), "hit@5": hit(5),
            "mrr": sum(1 / x["first_rank"] for x in rows if x["first_rank"]) / n,
            "evidence_recall@5": sum(x["recall"] for x in rows) / n,
            "precision@3": sum(x["p3"] for x in rows) / n,
            "misses": [x["id"] for x in rows if not x["first_rank"]]}


def gate_table(p: RAGPipeline, queries, backend: str):
    pts = []
    for q in queries:
        text = q.get("standalone", q["question"])
        r = p.retriever.retrieve(text, apply_gate=False)
        pts.append((q["id"], q["answerable"], r.top_score))
    sweep = []
    for t in [x / 100 for x in range(0, 101)]:
        rec = sum(1 for _, a, s_ in pts if a and s_ >= t) / sum(1 for _, a, _ in pts if a)
        rej = sum(1 for _, a, s_ in pts if not a and s_ < t) / sum(1 for _, a, _ in pts if not a)
        sweep.append((t, rec, rej))
    # Recommend: lowest answerable top-score minus a safety margin. The "optimum" sits a hair under the weakest
    # answerable query and would fail on any new question phrased slightly worse (it overfits the eval set).
    margin = 0.05
    min_ans = min(s_ for _, a, s_ in pts if a)
    t = max(0.0, int((min_ans - margin) * 100) / 100)
    rec = sum(1 for _, a, s_ in pts if a and s_ >= t) / sum(1 for _, a, _ in pts if a)
    rej = sum(1 for _, a, s_ in pts if not a and s_ < t) / sum(1 for _, a, _ in pts if not a)
    best = (t, rec, rej, min_ans, margin)
    return pts, sweep, best


def fmt(m: dict) -> str:
    return (f"| {m['mode']:6} | {m['hit@1']:.2f} | {m['hit@3']:.2f} | {m['hit@5']:.2f} | {m['mrr']:.2f} | "
            f"{m['evidence_recall@5']:.2f} | {m['precision@3']:.2f} | {', '.join(m['misses']) or '-'} |")


HEADER = ("| mode   | Hit@1 | Hit@3 | Hit@5 | MRR | EvidenceRecall@5 | Precision@3 | queries with no relevant chunk in top-5 |\n"
          "|--------|-------|-------|-------|-----|------------------|-------------|------|")


def generation_eval(p: RAGPipeline, queries) -> tuple[list[dict], dict]:
    by_id, rows = {q["id"]: q for q in queries}, []
    chunk_by_id = {c.id: c for c in p.store.chunks}
    for q in queries:
        session = f"eval-{q['id']}"
        if q.get("follow_up_of"):
            p.ask(by_id[q["follow_up_of"]]["question"], session)
        r = p.ask(q["question"], session)
        row = {"id": q["id"], "category": q["category"], "answerable_expected": q["answerable"], "answerable_got": r.answerable,
               "abstention_correct": r.answerable == q["answerable"], "answer": r.answer, "warnings": r.warnings,
               "confidence": r.confidence, "conflict_detected": r.conflict_detected, "latency_ms": r.latency_ms,
               "top_score": r.retrieval.top_score, "citations": [c.chunk_id for c in r.citations]}
        if q["answerable"]:
            low = r.answer.lower()
            row["answer_correct"] = r.answerable and all(any(alt.lower() in low for alt in item.split("|")) for item in q["answer_contains"])
            row["citation_supported"] = any(covered(chunk_by_id[c.chunk_id], q["evidence"]) for c in r.citations)
            if q.get("expect_conflict"):
                row["conflict_flagged"] = r.conflict_detected
        for bad in q.get("must_not_contain", []):
            # Following an injection = ASSERTING it. Quoting it while refusing (answerable=False) is correct behaviour.
            row["injection_followed"] = bool(r.answerable) and bad.lower() in r.answer.lower()
        rows.append(row)
    ans = [r for r in rows if r["answerable_expected"]]
    unans = [r for r in rows if not r["answerable_expected"]]
    conf = [r for r in rows if "conflict_flagged" in r]
    summary = {
        "answer_correct_rate": sum(r["answer_correct"] for r in ans) / len(ans),
        "citation_supported_rate": sum(r["citation_supported"] for r in ans) / len(ans),
        "correct_abstention_rate": sum(r["abstention_correct"] for r in unans) / len(unans),
        "false_abstention_rate": sum(not r["answerable_got"] for r in ans) / len(ans),
        "conflict_flag_rate": (sum(r["conflict_flagged"] for r in conf) / len(conf)) if conf else None,
        "injection_followed": any(r.get("injection_followed") for r in rows),
        "avg_latency_ms": sum(r["latency_ms"] for r in rows) / len(rows),
    }
    return rows, summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="hashing", choices=["hashing", "sentence-transformers"])
    ap.add_argument("--embedding-model", help="e.g. BAAI/bge-small-en-v1.5 or sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--chunk-size", type=int)
    ap.add_argument("--min-relevance", type=float)
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--chunk-sweep", action="store_true")
    ap.add_argument("--llm", action="store_true", help="also run Claude end-to-end (needs ANTHROPIC_API_KEY)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    queries = json.loads((ROOT / "data" / "eval" / "queries.json").read_text())
    out = [f"# Evaluation results (embedder: {a.backend}{' / ' + a.embedding_model if a.embedding_model else ''})\n"]

    p = build_pipeline(a.backend, a.chunk_size, a.min_relevance, embedding_model=a.embedding_model)
    out.append(f"Corpus: {len(p.store.documents)} documents, {len(p.store)} chunks (chunk_size={p.s.chunk_size}, overlap={p.s.chunk_overlap}). "
               f"Queries: {len(queries)} ({sum(q['answerable'] for q in queries)} answerable, {sum(not q['answerable'] for q in queries)} unanswerable).\n")

    out.append("## 1. Retrieval quality by method (answerable queries, no gate)\n\n" + HEADER)
    for mode in MODES:
        out.append(fmt(retrieval_metrics(p, queries, mode)))
    raw = retrieval_metrics(p, queries, "hybrid", use_standalone=False)
    out.append(f"\nFollow-up query WITHOUT rewriting (hybrid): Hit@5={raw['hit@5']:.2f}, misses={raw['misses']} "
               "-> compare with the table above, where the follow-up is evaluated in its rewritten standalone form.\n")

    pts, sweep, best = gate_table(p, queries, a.backend)
    out.append("## 2. Relevance gate: best dense cosine score per query\n\n| id | answerable | top cosine | passes gate @%.2f |\n|---|---|---|---|" % p.s.min_relevance)
    for qid, ans, sc in pts:
        out.append(f"| {qid} | {'yes' if ans else 'NO'} | {sc:.3f} | {'yes' if sc >= p.s.min_relevance else 'no'} |")
    rec = sum(1 for _, a_, s in pts if a_ and s >= p.s.min_relevance) / sum(1 for _, a_, _ in pts if a_)
    rej = sum(1 for _, a_, s in pts if not a_ and s < p.s.min_relevance) / sum(1 for _, a_, _ in pts if not a_)
    out.append(f"\nAt threshold {p.s.min_relevance:.2f}: answerable queries kept = {rec:.0%}, unanswerable queries rejected before the LLM = {rej:.0%}.")
    if a.calibrate:
        out.append("\n### Threshold sweep (selected points)\n\n| threshold | answerable kept | unanswerable rejected |\n|---|---|---|")
        for t, r_, j in sweep[::5]:
            out.append(f"| {t:.2f} | {r_:.0%} | {j:.0%} |")
        out.append(f"\n**Recommended threshold: {best[0]:.2f}** = weakest answerable top-score ({best[3]:.3f}) minus a {best[4]:.2f} safety margin "
                   f"-> answerable kept {best[1]:.0%}, unanswerable rejected {best[2]:.0%}. Set RAG_MIN_RELEVANCE={best[0]:.2f}. "
                   "(Calibrated on the same queries it is reported on, so treat the rejection rate as optimistic.)")

    if a.chunk_sweep:
        out.append("\n## 3. Chunk-size sweep (hybrid)\n\n| chunk_size | chunks | Hit@1 | Hit@3 | MRR | EvidenceRecall@5 | Precision@3 |\n|---|---|---|---|---|---|---|")
        for size in (250, 400, 600, 1000, 1600):
            pp = build_pipeline(a.backend, size, None, embedding_model=a.embedding_model)
            m = retrieval_metrics(pp, queries, "hybrid")
            out.append(f"| {size} | {len(pp.store)} | {m['hit@1']:.2f} | {m['hit@3']:.2f} | {m['mrr']:.2f} | {m['evidence_recall@5']:.2f} | {m['precision@3']:.2f} |")

    if a.llm:
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise SystemExit("--llm needs ANTHROPIC_API_KEY")
        pf = build_pipeline(a.backend, a.chunk_size, a.min_relevance, embedding_model=a.embedding_model)
        rows, summ = generation_eval(pf, queries)
        out.append("\n## 4. End-to-end generation (Claude: %s)\n\n| metric | value |\n|---|---|" % pf.s.llm_model)
        for k_, v in summ.items():
            out.append(f"| {k_} | {v if isinstance(v, (bool, type(None))) else round(v, 3)} |")
        out.append("\n| id | category | expected answerable | got | correct | cited evidence | conflict flagged | warnings |\n|---|---|---|---|---|---|---|---|")
        for r in rows:
            out.append(f"| {r['id']} | {r['category']} | {r['answerable_expected']} | {r['answerable_got']} | {r.get('answer_correct', '-')} | "
                       f"{r.get('citation_supported', '-')} | {r.get('conflict_flagged', '-')} | {', '.join(r['warnings']) or '-'} |")
        bad_rows = [r for r in rows if r.get("answer_correct") is False or r["warnings"] or r.get("injection_followed")
                    or not r["abstention_correct"] or r["category"].startswith("unanswerable") or r["category"] == "ambiguous"]
        out.append("\n### Answers to inspect (failures, warnings, and every question that should be refused)\n")
        for r in bad_rows:
            out.append(f"- **{r['id']}** ({r['category']}, top score {r['top_score']}, confidence {r['confidence']}): {r['answer']}")
        (ROOT / "docs" / f"eval_generation_{a.backend}.json").write_text(json.dumps({"summary": summ, "rows": rows}, indent=2))

    text = "\n".join(out)
    print(text)
    (ROOT / (a.out or f"docs/eval_results_{a.backend}.md")).write_text(text + "\n")


if __name__ == "__main__":
    main()
