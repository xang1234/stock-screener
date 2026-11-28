from __future__ import annotations

import json
from decimal import Decimal
from types import SimpleNamespace

import pytest

from app.database import SessionLocal
from app.models.app_settings import AppSetting
from app.services import economic_taxonomy_llm_provider as provider_module
from app.services.economic_exposure_extraction import RetryableProviderError
from app.services.economic_taxonomy_llm_provider import (
    EconomicTaxonomyLLMProvider,
    RoutedEconomicReservationManager,
    build_economic_taxonomy_provider,
)
from app.services.llm.llm_service import LLMPreDispatchError, LLMService
from app.services.social_anthropic_llm import AnthropicMeteredLLM

HAIKU = "anthropic/claude-haiku-5-5"


class _LLM:
    def __init__(self, result):
        self.result = result
        self.calls = []

    async def completion(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def _response(payload: str):
    return SimpleNamespace(
        id="request-1",
        choices=[SimpleNamespace(message=SimpleNamespace(content=payload))],
    )


def _credentials(monkeypatch, *, anthropic="", minimax="", zai="", opencode=""):
    settings = provider_module.settings
    monkeypatch.setattr(settings, "anthropic_api_key", anthropic)
    monkeypatch.setattr(settings, "minimax_api_key", minimax)
    monkeypatch.setattr(settings, "zai_api_key", zai)
    monkeypatch.setattr(settings, "zai_api_keys", "")
    monkeypatch.setattr(settings, "opencode_go_api_key", opencode)


def test_provider_uses_single_metered_json_dispatch():
    llm = _LLM(_response('{"status":"successful_empty","candidates":[]}'))

    result = EconomicTaxonomyLLMProvider(llm, model="minimax/MiniMax-M2.7").extract(
        evidence={"original_text": "No economic theme."},
        policy_version="economic-extraction-v1",
    )

    assert result == {
        "status": "successful_empty",
        "candidates": [],
        "provider_request_id": "request-1",
        "actual_cost": 0,
    }
    assert len(llm.calls) == 1
    assert llm.calls[0]["metered"] is True
    assert llm.calls[0]["allow_fallbacks"] is False
    assert llm.calls[0]["num_retries"] == 0


def test_provider_configuration_failure_is_retryable_before_dispatch():
    provider = EconomicTaxonomyLLMProvider(
        _LLM(LLMPreDispatchError("missing credentials")), model="minimax/MiniMax-M2.7"
    )

    with pytest.raises(RetryableProviderError) as error:
        provider.extract(evidence={}, policy_version="economic-extraction-v1")

    assert error.value.dispatch_confirmed is False


@pytest.mark.parametrize(
    ("credentials", "expected"),
    [
        ({"anthropic": "a", "minimax": "m", "zai": "z", "opencode": "o"}, HAIKU),
        ({"minimax": "m", "zai": "z", "opencode": "o"}, "minimax/MiniMax-M2.7"),
        ({"zai": "z", "opencode": "o"}, "openai/glm-4.7-flash"),
        ({"opencode": "o"}, "opencode-go/deepseek-v4-flash"),
        ({"anthropic": "  "}, None),
    ],
)
def test_first_configured_credential_selects_the_provider(
    monkeypatch, credentials, expected
):
    _credentials(monkeypatch, **credentials)

    provider, reservations = build_economic_taxonomy_provider(SessionLocal)

    if expected is None:
        assert (provider, reservations) == (None, None)
        return
    assert provider.model == reservations.model == expected
    assert isinstance(provider.llm, AnthropicMeteredLLM if expected == HAIKU else LLMService)


def test_haiku_dispatch_sends_free_json_and_settles_social_cost(monkeypatch):
    _credentials(monkeypatch, anthropic="test-key", minimax="m")
    with SessionLocal.begin() as db:
        db.add(
            AppSetting(
                key="social_llm_pricing",
                category="social",
                value=json.dumps(
                    {
                        "version": "haiku-v1",
                        "models": {
                            HAIKU: {
                                "provider": "anthropic",
                                "actual_models": ["claude-haiku-5-5"],
                                "input_usd_per_million": "0.10",
                                "output_usd_per_million": "0.50",
                            }
                        },
                    }
                ),
            )
        )
    sent = {}

    async def create(**kwargs):
        sent.update(kwargs)
        return SimpleNamespace(
            id="msg_1",
            model="claude-haiku-5-5",
            content=[
                SimpleNamespace(type="thinking", thinking=""),
                SimpleNamespace(
                    type="text",
                    text='```json\n{"status":"successful_empty","candidates":[]}\n```',
                ),
            ],
            usage=SimpleNamespace(input_tokens=1000, output_tokens=200),
        )

    provider, _reservations = build_economic_taxonomy_provider(SessionLocal)
    provider.llm.client = SimpleNamespace(messages=SimpleNamespace(create=create))

    result = provider.extract(
        evidence={"original_text": "No economic theme."},
        policy_version="economic-extraction-v1",
    )

    assert sent["model"] == "claude-haiku-5-5"
    assert sent["max_tokens"] == RoutedEconomicReservationManager.OUTPUT_TOKEN_LIMIT
    assert sent["output_config"] == {"effort": "medium"}
    assert "temperature" not in sent and "response_format" not in sent
    assert result == {
        "status": "successful_empty",
        "candidates": [],
        "provider_request_id": "msg_1",
        "actual_cost": provider.budget.price(HAIKU).cost(1000, 200),
    }
    assert result["actual_cost"] == Decimal("0.0002")
