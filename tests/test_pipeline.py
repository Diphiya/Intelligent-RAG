import pytest

from rag.exceptions import InvalidQuestion, LLMBadResponse, LLMUnavailable
from tests.conftest import ScriptedLLM, answer


def label_containing(prompt: str, needle: str) -> str:
    """What a competent model does: cite the passage that actually contains the fact."""
    import re
    return next(m.group(1) for m in re.finditer(r'<passage label="(S\d+)"[^>]*>\n(.*?)\n</passage>', prompt, re.S) if needle in m.group(2))


def test_answerable_flow_returns_structured_cited_response(make_pipeline):
    def scripted(kw):
        lab = label_containing(kw["user"], "4.2 million")
        return answer(answer=f"Revenue was 4.2 million USD [{lab}].", citations=[lab])
    llm = ScriptedLLM([scripted])
    r = make_pipeline(llm).ask("What was the revenue in Q3?")
    assert r.answerable and r.citations and r.confidence == "high" and r.retrieval.passed_gate
    assert r.citations[0].source == "quarterly_report_q3.pdf" and r.citations[0].page in (1, 2)
    assert "4.2 million" in r.answer and r.warnings == []


def test_prompt_contains_numbered_passages_and_rules(make_pipeline):
    llm = ScriptedLLM([answer(answer="x [S1]", citations=["S1"])])
    make_pipeline(llm).ask("What is the minimum password length?")
    call = llm.calls[0]
    assert '<passage label="S1"' in call["user"] and "ONLY from the numbered passages" in call["system"]
    assert call["schema"]["required"]


def test_irrelevant_question_never_calls_the_llm(make_pipeline):
    llm = ScriptedLLM()
    r = make_pipeline(llm).ask("Who won the 2018 FIFA World Cup?")
    assert not r.answerable and r.unanswerable_reason == "no_relevant_passages" and llm.calls == []
    assert r.retrieval.passed_gate is False and r.confidence == "none"


def test_llm_abstention_on_near_miss_is_respected(make_pipeline):
    llm = ScriptedLLM([answer(answerable=False, answer="The documents do not mention parental leave.",
                              unanswerable_reason="Only annual, sick, unpaid and bereavement leave are covered.", confidence="low")])
    r = make_pipeline(llm).ask("What is the parental leave policy?")
    assert not r.answerable and r.citations == [] and "parental" in r.answer


def test_hallucinated_uncited_answer_is_blocked(make_pipeline):
    uncited = answer(answer="Parental leave is 16 weeks.", citations=[])
    llm = ScriptedLLM([uncited, uncited, uncited])                       # still uncited after two repair attempts
    r = make_pipeline(llm).ask("What is the parental leave policy?")
    assert not r.answerable and "answer_without_valid_citation" in r.warnings and "16" not in r.answer
    assert len(llm.calls) == 3 and "cited no valid passage label" in llm.calls[1]["user"]


def test_uncited_answer_is_repaired_by_one_retry(make_pipeline):
    def cited(kw):                                                       # cites whichever label holds the fact
        lab = label_containing(kw["user"], "14 characters")
        return answer(answer=f"Passwords need 14 characters [{lab}].", citations=[lab])
    llm = ScriptedLLM([answer(answer="Passwords need 14 characters.", citations=[]), cited])
    r = make_pipeline(llm).ask("What is the minimum password length?")
    assert r.answerable and r.citations and "uncited_answer_retry" in r.warnings
    assert "answer_without_valid_citation" not in r.warnings


def test_conflict_hint_is_sent_to_the_llm_and_flag_is_returned(make_pipeline):
    llm = ScriptedLLM([answer(answer="2024: 25 days [S1]; 2023: 20 days [S2].", citations=["S1", "S2"], conflict_detected=True,
                              conflict_note="2023 and 2024 policies differ; 2024 supersedes.", confidence="medium")])
    r = make_pipeline(llm).ask("How many days of paid annual leave do full-time employees get?")
    assert "<possible_conflicts>" in llm.calls[0]["user"]
    assert r.conflict_detected and "supersedes" in r.conflict_note


def test_followup_is_rewritten_with_history_and_retrieval_uses_rewrite(make_pipeline):
    llm = ScriptedLLM(
        structured=[answer(answer="14 characters [S1].", citations=["S1"]), answer(answer="Every 180 days [S1].", citations=["S1"])],
        text=["How often must passwords be rotated?"])
    p = make_pipeline(llm)
    p.ask("What is the minimum password length?", "s1")
    r = p.ask("And how often does it have to be changed?", "s1")
    assert r.rewritten_query == "How often must passwords be rotated?"
    assert "180 days" in llm.calls[-1]["user"]                       # retrieval used the rewritten query
    assert "<conversation>" in llm.calls[-1]["user"] and "minimum password length" in llm.calls[-1]["user"]
    assert r.answerable


def test_rewrite_failure_falls_back_to_heuristic(make_pipeline):
    llm = ScriptedLLM(structured=[answer(answer="a [S1]", citations=["S1"]), answer(answer="b [S1]", citations=["S1"])],
                      text=[LLMUnavailable("down")])
    p = make_pipeline(llm)
    p.ask("What is the minimum password length?", "s")
    r = p.ask("And how often is it changed?", "s")
    assert "minimum password length" in r.rewritten_query.lower()


def test_sessions_are_isolated(make_pipeline):
    llm = ScriptedLLM([answer(answer="a [S1]", citations=["S1"]), answer(answer="b [S1]", citations=["S1"])])
    p = make_pipeline(llm)
    p.ask("What is the minimum password length?", "A")
    r = p.ask("How long is the Orbit Home Hub warranty?", "B")
    assert r.rewritten_query == "How long is the Orbit Home Hub warranty?" and not [c for c in llm.calls if c["kind"] == "text"]


def test_invalid_structured_output_is_retried_once_then_succeeds(make_pipeline):
    llm = ScriptedLLM([{"answerable": "maybe"}, answer(answer="14 [S1]", citations=["S1"])])
    r = make_pipeline(llm).ask("What is the minimum password length?")
    assert r.answerable and "invalid_model_output_retry" in r.warnings and "invalid" in llm.calls[1]["user"]


def test_invalid_structured_output_repeatedly_raises_bad_response(make_pipeline):
    llm = ScriptedLLM([{"nope": 1}, {"nope": 2}, {"nope": 3}])
    with pytest.raises(LLMBadResponse):
        make_pipeline(llm).ask("What is the minimum password length?")


def test_llm_outage_surfaces_as_503_error_type(make_pipeline):
    with pytest.raises(LLMUnavailable):
        make_pipeline(ScriptedLLM([LLMUnavailable("timeout")])).ask("What is the minimum password length?")


def test_missing_api_key_is_a_clear_error(make_pipeline):
    with pytest.raises(LLMUnavailable, match="ANTHROPIC_API_KEY"):
        make_pipeline(None).ask("What is the minimum password length?")


def test_no_documents_uploaded(make_pipeline):
    r = make_pipeline(ScriptedLLM(), load_samples=False).ask("anything?")
    assert not r.answerable and r.unanswerable_reason == "no_documents"


@pytest.mark.parametrize("q", ["", "   ", "x" * 3000])
def test_invalid_questions_rejected(make_pipeline, q):
    with pytest.raises(InvalidQuestion):
        make_pipeline(ScriptedLLM()).ask(q)


def test_same_filename_different_content_is_kept_as_new_version(make_pipeline):
    p = make_pipeline(load_samples=False)
    assert p.ingest("policy.txt", b"Version one text about leave.").filename == "policy.txt"
    assert p.ingest("policy.txt", b"Version two text about leave, changed.").filename == "policy (v2).txt"


def test_delete_document_removes_it_from_retrieval(make_pipeline):
    p = make_pipeline(ScriptedLLM())
    doc = next(d for d in p.list_documents() if d.filename == "quarterly_report_q3.pdf")
    p.delete_document(doc.doc_id)
    assert all(c.source != "quarterly_report_q3.pdf" for c in p.store.chunks)
    assert not p.retriever.retrieve("What is the revenue in Q3?").chunks or all(
        rc.chunk.source != "quarterly_report_q3.pdf" for rc in p.retriever.retrieve("revenue Q3").chunks)


def test_incomplete_tool_call_is_rejected_and_retried(make_pipeline):
    def cited(kw):
        lab = label_containing(kw["user"], "14 characters")
        return answer(answer=f"Passwords need 14 characters [{lab}].", citations=[lab])
    llm = ScriptedLLM([{"answerable": True}, cited])                      # first call: only one field, rest missing
    r = make_pipeline(llm).ask("What is the minimum password length?")
    assert r.answerable and "invalid_model_output_retry" in r.warnings and "invalid" in llm.calls[1]["user"]
