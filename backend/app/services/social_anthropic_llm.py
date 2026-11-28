"""Metered social extraction through the Anthropic Messages API.

Returns the response shape the social budget ledger reads from LiteLLM, so the
backlog can bill and reconcile Claude calls like any other provider.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import anthropic

from app.config.settings import settings
from app.services.llm.llm_service import (
    LLMError,
    LLMPreDispatchError,
    LLMProviderRejectedError,
    LLMRateLimitError,
)
from app.services.social_extraction_service import EXTRACTION_SCHEMA

logger = logging.getLogger(__name__)

PROVIDER = "anthropic"
# On 40 live posts Claude Haiku 5.5 kept 11 claims at medium against 7-9 at low;
# high took ~41 s a call.
EFFORT = "medium"
_TRUNCATED = frozenset({"max_tokens", "model_context_window_exceeded"})


class AnthropicMeteredLLM:
    def __init__(self, api_key: str | None = None, *, output_schema: dict | None = EXTRACTION_SCHEMA):
        key = settings.anthropic_api_key if api_key is None else api_key
        if not key:
            raise LLMPreDispatchError("metered_provider_configuration_error")
        # One reserved dispatch per call: the backlog owns retries.
        self.client = anthropic.AsyncAnthropic(api_key=key, max_retries=0)
        self.output_config = {"effort": EFFORT}
        if output_schema is not None:
            self.output_config["format"] = {"type": "json_schema", "schema": output_schema}

    async def completion(self, *, messages, model, max_tokens, timeout=None, **_ignored):
        # Sampling overrides (temperature) are rejected by current Claude models; the
        # schema, when set, replaces response_format, which has no Claude equivalent.
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        try:
            response = await self.client.messages.create(
                model=model.removeprefix(f"{PROVIDER}/"), max_tokens=max_tokens, system=system,
                messages=[m for m in messages if m["role"] != "system"], timeout=timeout,
                output_config=self.output_config,
            )
        except anthropic.APIStatusError as exc:
            status = exc.status_code
            # An empty prepaid balance or a reached spend limit is a 400, but like 402
            # it refuses every request.
            message = str(exc.message).lower()
            if status == 400 and ("credit balance" in message or "usage limits" in message):
                status = 402
            logger.warning("Metered Anthropic dispatch failed: %s status=%s", type(exc).__name__, status)
            # A rate limit (429) is refused before inference; a timeout (408) may still have been billed.
            if status == 429:
                raise LLMRateLimitError("metered_provider_rate_limited") from None
            if 400 <= status < 500 and status != 408:
                raise LLMProviderRejectedError("metered_provider_rejected", status) from None
            raise LLMError("metered_provider_error") from None
        except anthropic.APIError as exc:
            logger.warning("Metered Anthropic dispatch failed: %s", type(exc).__name__)
            raise LLMError("metered_provider_error") from None
        text = "".join(block.text for block in response.content if block.type == "text")
        return SimpleNamespace(
            id=response.id, model=response.model,
            # "length" is LiteLLM's finish reason for a reply cut off at max_tokens.
            choices=[SimpleNamespace(
                message=SimpleNamespace(content=text),
                finish_reason="length" if response.stop_reason in _TRUNCATED else response.stop_reason,
            )],
            usage=SimpleNamespace(prompt_tokens=response.usage.input_tokens,
                                  completion_tokens=response.usage.output_tokens),
            _hidden_params={"custom_llm_provider": PROVIDER},
        )
