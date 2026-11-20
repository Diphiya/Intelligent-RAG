"""Step 7/10: deterministic checks applied AFTER the LLM answers. The model is never trusted blindly."""
from __future__ import annotations

import re

from rag.schemas import LLMAnswer, RetrievedChunk
from rag.textutil import numbers

_MARKER = re.compile(r"\[\s*(S\d+)\s*\]", re.I)


def normalize_label(raw: str) -> str:
    m = re.search(r"S\s*(\d+)", raw, re.I)
    return f"S{m.group(1)}" if m else raw.strip().upper()


def validate_answer(ans: LLMAnswer, labelled: dict[str, RetrievedChunk], question: str) -> tuple[LLMAnswer, list[str]]:
    """Returns a corrected answer plus warnings.

    Rules:
      1. Only citations that refer to a passage we actually supplied survive.
      2. An 'answerable' answer with zero valid citations is downgraded to unanswerable
         (an uncited answer cannot be verified, so it is treated as unsupported).
      3. Numbers in the answer that appear in none of the cited passages are flagged and confidence is capped.
    """
    warnings: list[str] = []
    ans = ans.model_copy(deep=True)

    if not ans.answerable:
        ans.citations = []
        ans.confidence = "low"
        if not ans.answer.strip():
            ans.answer = "I could not find this information in the uploaded documents."
        return ans, warnings

    claimed = [normalize_label(c) for c in ans.citations] + [normalize_label(m) for m in _MARKER.findall(ans.answer)]
    valid, seen = [], set()
    for lab in claimed:
        if lab in labelled:
            if lab not in seen:
                seen.add(lab)
                valid.append(lab)
        else:
            warnings.append(f"invalid_citation:{lab}")
    ans.citations = valid

    if not valid:
        warnings.append("answer_without_valid_citation")
        ans.answerable = False
        ans.unanswerable_reason = "The generated answer could not be tied to any retrieved passage, so it was withheld."
        ans.answer = "I could not verify an answer against the uploaded documents, so I am not going to guess."
        ans.confidence = "low"
        return ans, warnings

    # The model is shown each passage's source name (e.g. hr_leave_policy_2024.txt), so numbers taken from it are grounded too.
    evidence = " ".join(f"{labelled[l].chunk.text} {labelled[l].chunk.source}" for l in valid)
    answer_text = _MARKER.sub("", ans.answer)
    unsupported = sorted(numbers(answer_text) - numbers(evidence) - numbers(question))
    if unsupported:
        warnings.extend(f"unsupported_number:{n}" for n in unsupported)
        ans.confidence = "low"
    return ans, warnings
