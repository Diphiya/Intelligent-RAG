"""Step 8: conversational context. Per-session sliding window + follow-up query rewriting."""
from __future__ import annotations

import re
import threading
from collections import OrderedDict, deque
from dataclasses import dataclass

from rag.exceptions import LLMBadResponse, LLMUnavailable
from rag.llm import LLMClient


@dataclass
class Turn:
    question: str
    standalone: str
    answer: str
    answerable: bool


class ConversationMemory:
    def __init__(self, max_turns: int = 6, max_sessions: int = 500):
        self.max_turns, self.max_sessions = max_turns, max_sessions
        self._s: OrderedDict[str, deque[Turn]] = OrderedDict()
        self._lock = threading.Lock()

    def history(self, session_id: str) -> list[Turn]:
        with self._lock:
            return list(self._s.get(session_id, []))

    def add(self, session_id: str, turn: Turn) -> None:
        with self._lock:
            self._s.setdefault(session_id, deque(maxlen=self.max_turns)).append(turn)
            self._s.move_to_end(session_id)
            while len(self._s) > self.max_sessions:
                self._s.popitem(last=False)

    def clear(self, session_id: str) -> bool:
        with self._lock:
            return self._s.pop(session_id, None) is not None


REWRITE_SYSTEM = (
    "You rewrite a follow-up question into a single self-contained search query, using the conversation. "
    "Resolve pronouns and omitted subjects (it, that, they, 'and for 2024?'). Keep every specific term, "
    "name and number. If the question is already self-contained, return it unchanged. "
    "Output ONLY the rewritten question, nothing else. Never answer the question."
)
_FOLLOWUP_HINT = re.compile(r"^(and|what about|how about|also|then|so|but)\b|\b(it|that|this|they|them|those|these|its|their)\b", re.I)


def format_history(history: list[Turn]) -> str:
    return "\n".join(f"User: {t.question}\nAssistant: {t.answer[:300]}" for t in history)


def rewrite_question(llm: LLMClient | None, model: str | None, question: str, history: list[Turn]) -> str:
    """Return a standalone query. Falls back to a heuristic if the LLM is unavailable."""
    if not history:
        return question
    if llm is not None:
        try:
            out = llm.complete_text(
                system=REWRITE_SYSTEM, model=model, max_tokens=120,
                user=f"Conversation:\n{format_history(history)}\n\nFollow-up question: {question}\n\nStandalone question:",
            )
            out = out.strip().strip('"')
            if 3 <= len(out) <= 500:
                return out
        except (LLMUnavailable, LLMBadResponse):
            pass
    # Fallback: prepend the previous standalone question when the new one looks like a follow-up.
    if len(question.split()) <= 8 or _FOLLOWUP_HINT.search(question):
        return f"{history[-1].standalone} {question}"
    return question
