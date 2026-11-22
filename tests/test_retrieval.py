from rag.retrieval import BM25
from rag.textutil import tokenize


def test_bm25_ranks_exact_term_match_first():
    docs = [tokenize(t) for t in ["cats and dogs", "quarterly revenue report", "revenue revenue growth"]]
    scores = BM25(docs).scores(tokenize("revenue"))
    assert scores.argmax() == 2 and scores[0] == 0


def test_relevant_query_passes_gate_and_finds_right_source(make_pipeline):
    p = make_pipeline()
    res = p.retriever.retrieve("What is the revenue in Q3?")
    assert res.passed and res.chunks[0].chunk.source == "quarterly_report_q3.pdf"
    # Known limitation (docs/evaluation_report.md, failure case 2): a lexical retriever can rank the
    # "Regional Breakdown" chunk first because it repeats "revenue"; the evidence chunk must still be in the top 3.
    assert any("4.2 million" in rc.chunk.text for rc in res.chunks[:3])


def test_unrelated_query_is_rejected_by_gate(make_pipeline):
    res = make_pipeline().retriever.retrieve("Who won the 2018 FIFA World Cup?")
    assert not res.passed and res.chunks == []


def test_exact_duplicate_document_is_skipped_at_ingestion(make_pipeline):
    p = make_pipeline()
    assert "hr_leave_policy_2024_copy.txt" not in {d.filename for d in p.list_documents()}
    again = p.ingest("renamed.txt", (p.store.chunks[0].text + "\n").encode())
    assert again.status in ("ingested", "duplicate")


def test_duplicate_detected_even_with_different_whitespace_and_name(make_pipeline):
    p = make_pipeline(load_samples=False)
    assert p.ingest("a.txt", b"Alpha beta gamma delta.\nSecond line here.").status == "ingested"
    r = p.ingest("b.txt", b"Alpha   beta gamma delta.\r\nSecond line   here.  ")
    assert r.status == "duplicate" and r.duplicate_of == "a.txt"


def test_conflicting_versions_are_both_retrieved_not_collapsed(make_pipeline):
    res = make_pipeline().retriever.retrieve("How many days of paid annual leave do full-time employees get?")
    sources = {rc.chunk.source for rc in res.chunks}
    assert {"hr_leave_policy_2023.txt", "hr_leave_policy_2024.txt"} <= sources
    assert res.conflicts, "different numbers across versions must raise a conflict hint"


def test_near_duplicate_chunks_with_same_numbers_are_merged(make_pipeline):
    p = make_pipeline(load_samples=False)
    base = " ".join(f"Rule {i}: employees must report incidents within 24 hours of discovery to the security team." for i in range(3))   # < chunk_size, so each file is a single chunk
    p.ingest("a.txt", (base + " Badges are blue.").encode())
    p.ingest("b.txt", (base + " Badges are red.").encode())
    res = p.retriever.retrieve("report incidents within 24 hours", top_k=5)
    assert len(res.chunks) == 1 and res.chunks[0].also_in


def test_modes_all_work(make_pipeline):
    p = make_pipeline()
    for mode in ("dense", "bm25", "hybrid"):
        assert p.retriever.retrieve("factory reset hub", mode=mode, apply_gate=False).chunks


def test_empty_store_returns_nothing(make_pipeline):
    res = make_pipeline(load_samples=False).retriever.retrieve("anything")
    assert not res.passed and not res.chunks


def test_version_siblings_bring_in_the_other_policy_version(make_pipeline):
    r = make_pipeline().retriever
    q = "How many days of paid annual leave do full-time employees get?"
    plain = r.retrieve(q, top_k=1, apply_gate=False)
    wide = r.retrieve(q, top_k=1, apply_gate=False, include_versions=True)
    assert len({rc.chunk.source for rc in plain.chunks}) == 1
    assert {"hr_leave_policy_2023.txt", "hr_leave_policy_2024.txt"} <= {rc.chunk.source for rc in wide.chunks}
    assert len(wide.chunks) <= 1 + 2 and wide.conflicts          # capped, and the conflict hint is produced
