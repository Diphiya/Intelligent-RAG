"""Anthropic Claude client wrapper: structured output, retries, timeouts, graceful degradation.

Notes on the Anthropic API (verified against current docs):
 * Newer models (Sonnet 5.5 / Opus 5.5 / Fable 5.x) reject FORCED tool use (tool_choice any/tool -> 400).
   So structured output uses tool_choice=auto + a strict tool + an explicit instruction, and the
   wrapper re-prompts once if the model answers in plain text.
 * Optional parameters (temperature, strict) are dropped automatically if a model rejects them.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Protocol

import anthropic

from rag.config import Settings
from rag.exceptions import LLMBadResponse, LLMUnavailable

log = logging.getLogger(__name__)
_UNEXPECTED_KW = re.compile(r"unexpected keyword argument '(\w+)'")


class LLMClient(Protocol):
    def generate_structured(self, *, system: str, user: str, tool_name: str, tool_description: str,
                            schema: dict, max_tokens: int | None = None, model: str | None = None) -> dict: ...

    def complete_text(self, *, system: str, user: str, max_tokens: int = 200, model: str | None = None) -> str: ...


class AnthropicLLM:
    def __init__(self, settings: Settings, client: Any | None = None):
        self.s = settings
        if client is None:
            if not settings.anthropic_api_key:
                raise LLMUnavailable("ANTHROPIC_API_KEY is not set.")
            client = anthropic.Anthropic(
                api_key=settings.anthropic_api_key,
                timeout=settings.llm_timeout_s,
                max_retries=settings.llm_max_retries,   # exponential backoff on 429 / 5xx / network errors
            )
        self.client = client
        self._unsupported: set[str] = set()              # optional params this model rejected

    # ---- low level ----
    def _create(self, **kwargs: Any) -> Any:
        for _ in range(4):
            call = dict(kwargs)
            if "temperature" not in self._unsupported:
                call["temperature"] = self.s.llm_temperature
            for name in self._unsupported:             # parameters this SDK/model has already refused
                call.pop(name, None)
            if "strict" in self._unsupported and "tools" in call:
                call["tools"] = [{k: v for k, v in t.items() if k != "strict"} for t in call["tools"]]
            try:
                return self.client.messages.create(**call)
            except TypeError as exc:
                # Newer SDK versions reject some keyword arguments locally (before any HTTP request).
                m = _UNEXPECTED_KW.search(str(exc))
                if not m or m.group(1) in self._unsupported:
                    raise
                log.warning("SDK does not accept '%s'; retrying without it.", m.group(1))
                self._unsupported.add(m.group(1))
            except anthropic.BadRequestError as exc:
                msg = str(exc).lower()
                dropped = next((p for p in ("temperature", "strict") if p in msg and p not in self._unsupported), None)
                if dropped is None:
                    raise LLMBadResponse(f"Anthropic rejected the request: {exc}") from exc
                log.warning("Model rejected '%s'; retrying without it.", dropped)
                self._unsupported.add(dropped)
            except (anthropic.APITimeoutError, anthropic.APIConnectionError) as exc:
                raise LLMUnavailable(f"Could not reach the Anthropic API: {exc}") from exc
            except anthropic.AuthenticationError as exc:
                raise LLMUnavailable("Anthropic API key was rejected (check ANTHROPIC_API_KEY).") from exc
            except anthropic.RateLimitError as exc:
                raise LLMUnavailable("Anthropic rate limit hit after retries; try again shortly.") from exc
            except anthropic.APIStatusError as exc:
                raise LLMUnavailable(f"Anthropic API error {exc.status_code}: {exc.message}") from exc
        raise LLMBadResponse("Anthropic rejected the request after dropping optional parameters.")

    # ---- public ----
    def generate_structured(self, *, system, user, tool_name, tool_description, schema, max_tokens=None, model=None) -> dict:
        tool = {"name": tool_name, "description": tool_description, "input_schema": schema, "strict": True}
        base = dict(model=model or self.s.llm_model, max_tokens=max_tokens or self.s.llm_max_tokens,
                    system=system, tools=[tool])
        content = user
        for attempt in range(3):
            resp = self._create(messages=[{"role": "user", "content": content}], **base)
            stop = getattr(resp, "stop_reason", None)
            if stop == "max_tokens":
                # The output was cut off, so a tool call may be missing fields. Retry once with a larger budget.
                log.warning("Model output truncated at max_tokens=%s (attempt %d); retrying with a larger budget.",
                            base["max_tokens"], attempt + 1)
                if attempt < 2:
                    base["max_tokens"] = base["max_tokens"] * 2
                    continue
            elif stop not in (None, "tool_use", "end_turn"):
                log.warning("Unexpected stop_reason=%r", stop)
            for block in resp.content:
                if getattr(block, "type", "") == "tool_use" and block.name == tool_name:
                    return dict(block.input)
            text = "".join(getattr(b, "text", "") for b in resp.content)
            parsed = _json_from_text(text)
            if parsed is not None:
                return parsed
            content = user + f"\n\nYou MUST respond by calling the `{tool_name}` tool. Do not reply with plain text."
        raise LLMBadResponse("Model did not return the required structured output.")

    def complete_text(self, *, system, user, max_tokens=200, model=None) -> str:
        resp = self._create(model=model or self.s.llm_model, max_tokens=max_tokens, system=system,
                            messages=[{"role": "user", "content": user}])
        return "".join(getattr(b, "text", "") for b in resp.content).strip()


def _json_from_text(text: str) -> dict | None:
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None
