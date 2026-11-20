from types import SimpleNamespace as NS

import anthropic
import httpx
import pytest

from rag.config import Settings
from rag.exceptions import LLMBadResponse, LLMUnavailable
from rag.llm import AnthropicLLM


def tool_resp(inp):
    return NS(content=[NS(type="tool_use", name="submit_answer", input=inp)])


def text_resp(t):
    return NS(content=[NS(type="text", text=t)])


class FakeMessages:
    def __init__(self, script):
        self.script, self.calls = list(script), []

    def create(self, **kw):
        self.calls.append(kw)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def llm(script):
    m = FakeMessages(script)
    return AnthropicLLM(Settings(), client=NS(messages=m)), m


def bad_request(msg):
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return anthropic.BadRequestError(msg, response=httpx.Response(400, request=req), body=None)


def call(l):
    return l.generate_structured(system="s", user="u", tool_name="submit_answer", tool_description="d", schema={"type": "object"})


def test_tool_use_block_is_returned_and_forced_tool_choice_is_not_sent():
    l, m = llm([tool_resp({"answerable": True})])
    assert call(l) == {"answerable": True}
    assert "tool_choice" not in m.calls[0]          # newer Claude models reject forced tool use
    assert m.calls[0]["tools"][0]["strict"] is True and m.calls[0]["temperature"] == 0.0


def test_plain_text_reply_triggers_one_reprompt():
    l, m = llm([text_resp("sure!"), tool_resp({"ok": 1})])
    assert call(l) == {"ok": 1} and "MUST respond by calling" in m.calls[1]["messages"][0]["content"]


def test_json_in_text_is_accepted_as_last_resort():
    l, _ = llm([text_resp('Here: {"answerable": false}')])
    assert call(l) == {"answerable": False}


def test_never_structured_raises_bad_response():
    l, _ = llm([text_resp("no"), text_resp("still no"), text_resp("nope")])
    with pytest.raises(LLMBadResponse):
        call(l)


def test_unsupported_temperature_is_dropped_and_remembered():
    l, m = llm([bad_request("temperature is not supported for this model"), tool_resp({"a": 1}), tool_resp({"a": 2})])
    assert call(l) == {"a": 1} and "temperature" not in m.calls[1]
    call(l)
    assert "temperature" not in m.calls[2]


def test_unsupported_strict_is_dropped():
    l, m = llm([bad_request("tools.0.strict: extra inputs are not permitted"), tool_resp({"a": 1})])
    call(l)
    assert "strict" not in m.calls[1]["tools"][0]


def test_unrelated_bad_request_is_not_retried_forever():
    l, m = llm([bad_request("prompt is too long")])
    with pytest.raises(LLMBadResponse):
        call(l)
    assert len(m.calls) == 1


def test_connection_error_maps_to_unavailable():
    l, _ = llm([anthropic.APIConnectionError(request=httpx.Request("POST", "https://x"))])
    with pytest.raises(LLMUnavailable):
        call(l)


def test_missing_key_without_client_fails_fast():
    with pytest.raises(LLMUnavailable):
        AnthropicLLM(Settings())


class StrictSignatureMessages(FakeMessages):
    """Mimics a newer SDK whose Messages.create() has no `temperature` parameter at all."""

    def create(self, **kw):
        if "temperature" in kw:
            raise TypeError("Messages.create() got an unexpected keyword argument 'temperature'")
        return super().create(**kw)


def test_sdk_that_rejects_temperature_locally_is_handled():
    m = StrictSignatureMessages([tool_resp({"a": 1}), tool_resp({"a": 2})])
    l = AnthropicLLM(Settings(), client=NS(messages=m))
    assert call(l) == {"a": 1} and "temperature" not in m.calls[0]
    assert call(l) == {"a": 2} and "temperature" not in m.calls[1]


def test_unrelated_type_error_is_not_swallowed():
    class Boom(FakeMessages):
        def create(self, **kw):
            raise TypeError("something else entirely")
    l = AnthropicLLM(Settings(), client=NS(messages=Boom([])))
    with pytest.raises(TypeError):
        call(l)


def test_truncated_output_is_retried_with_a_larger_token_budget():
    cut = NS(content=[NS(type="tool_use", name="submit_answer", input={"answerable": True})], stop_reason="max_tokens")
    full = NS(content=[NS(type="tool_use", name="submit_answer", input={"answerable": True, "answer": "ok"})], stop_reason="tool_use")
    l, m = llm([cut, full])
    assert call(l)["answer"] == "ok"
    assert m.calls[1]["max_tokens"] == 2 * m.calls[0]["max_tokens"]
