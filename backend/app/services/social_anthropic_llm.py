"""Metered social extraction through the Anthropic Messages API.

Returns the response shape the social budget ledger reads from LiteLLM, so the
backlog can bill and reconcile Claude calls like any other provider.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import anthropic

from app.config.settings import settings
from app.services.llm.llm_service import LLMError, LLMPreDispatchError, LLMProviderRejectedError
from app.services.social_extraction_service import EXTRACTION_SCHEMA

logger = logging.getLogger(__name__)

PROVIDER = "anthropic"
# On 40 live posts Claude Haiku 5.5 kept 11 claims at medium against 7-9 at low;
# high took ~41 s a call, close to the backlog's claim expiry for a 20-call run.
EFFORT = "medium"


class AnthropicMeteredLLM:
    def __init__(self, api_key: str | None = None):
        key = settings.anthropic_api_key if api_key is None else api_key
        if not key:
            raise LLMPreDispatchError("metered_provider_configuration_error")
        # One reserved dispatch per call: the backlog owns retries.
        self.client = anthropic.AsyncAnthropic(api_key=key, max_retries=0)

    async def completion(self, *, messages, model, max_tokens, timeout=None, **_ignored):
        # Sampling overrides (temperature) are rejected by current Claude models; the
        # schema replaces response_format, which has no Claude equivalent.
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        try:
            response = await self.client.messages.create(
                model=model.removeprefix(f"{PROVIDER}/"), max_tokens=max_tokens, system=system,
                messages=[m for m in messages if m["role"] != "system"], timeout=timeout,
                output_config={"effort": EFFORT, "format": {"type": "json_schema", "schema": EXTRACTION_SCHEMA}},
            )
        except anthropic.APIStatusError as exc:
            status = exc.status_code
            # An empty prepaid balance is a 400, but like 402 it refuses every request.
            if status == 400 and "credit balance" in str(exc.message).lower():
                status = 402
            logger.warning("Metered Anthropic dispatch failed: %s status=%s", type(exc).__name__, status)
            # A timeout (408) or rate limit (429) may still have been billed.
            if 400 <= status < 500 and status not in (408, 429):
                raise LLMProviderRejectedError("metered_provider_rejected", status) from None
            raise LLMError("metered_provider_error") from None
        except anthropic.APIError as exc:
            logger.warning("Metered Anthropic dispatch failed: %s", type(exc).__name__)
            raise LLMError("metered_provider_error") from None
        text = "".join(block.text for block in response.content if block.type == "text")
        return SimpleNamespace(
            id=response.id, model=response.model,
            choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
            usage=SimpleNamespace(prompt_tokens=response.usage.input_tokens,
                                  completion_tokens=response.usage.output_tokens),
            _hidden_params={"custom_llm_provider": PROVIDER},
        )
