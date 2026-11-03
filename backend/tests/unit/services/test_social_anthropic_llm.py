import asyncio
from types import SimpleNamespace

import anthropic
import httpx2
import pytest

from app.services.llm.llm_service import LLMError, LLMPreDispatchError, LLMProviderRejectedError
from app.services.social_anthropic_llm import AnthropicMeteredLLM
from app.services.social_extraction_service import EXTRACTION_SCHEMA

MESSAGES = [{"role": "system", "content": "rules"}, {"role": "user", "content": '{"posts": []}'}]


def llm_with(create):
    llm = AnthropicMeteredLLM(api_key="test-key")
    llm.client = SimpleNamespace(messages=SimpleNamespace(create=create))
    return llm


def call(llm):
    return asyncio.run(llm.completion(
        messages=MESSAGES, model="anthropic/claude-haiku-5-5", max_tokens=8192, timeout=120,
        temperature=0, response_format={"type": "json_object"}, allow_fallbacks=False, num_retries=0, metered=True,
    ))


def status_error(status, message="refused"):
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    return anthropic.APIStatusError(message, response=httpx2.Response(status, request=request), body=None)


def test_sends_one_schema_constrained_request_and_returns_the_ledger_shape():
    sent = {}

    async def create(**kwargs):
        sent.update(kwargs)
        return SimpleNamespace(
            id="msg_1", model="claude-haiku-5-5",
            content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text='{"posts": []}')],
            usage=SimpleNamespace(input_tokens=120, output_tokens=40),
        )

    response = call(llm_with(create))

    assert sent == {
        "model": "claude-haiku-5-5", "max_tokens": 8192, "system": "rules",
        "messages": [{"role": "user", "content": '{"posts": []}'}], "timeout": 120,
        "output_config": {"effort": "medium", "format": {"type": "json_schema", "schema": EXTRACTION_SCHEMA}},
    }
    assert (response.id, response.model, response.choices[0].message.content) == ("msg_1", "claude-haiku-5-5", '{"posts": []}')
    assert (response.usage.prompt_tokens, response.usage.completion_tokens) == (120, 40)
    assert response._hidden_params == {"custom_llm_provider": "anthropic"}


@pytest.mark.parametrize(("error", "expected"), [
    (status_error(401), LLMProviderRejectedError),
    (status_error(400, "Your credit balance is too low to access the Anthropic API."), LLMProviderRejectedError),
    (status_error(400, "invalid request"), LLMProviderRejectedError),
    (status_error(429), LLMError),
    (status_error(529), LLMError),
    (anthropic.APIConnectionError(request=httpx2.Request("POST", "https://api.anthropic.com")), LLMError),
])
def test_maps_provider_errors_to_the_metered_contract(error, expected):
    async def create(**_kwargs):
        raise error

    with pytest.raises(expected) as raised:
        call(llm_with(create))
    if expected is LLMProviderRejectedError:
        message = str(error.message).lower()
        assert raised.value.status_code == (402 if "credit balance" in message else error.status_code)
    else:
        assert not isinstance(raised.value, LLMProviderRejectedError)


def test_missing_key_fails_before_dispatch(monkeypatch):
    from app.services import social_anthropic_llm
    monkeypatch.setattr(social_anthropic_llm.settings, "anthropic_api_key", "")
    with pytest.raises(LLMPreDispatchError):
        AnthropicMeteredLLM()
