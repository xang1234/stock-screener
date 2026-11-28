"""Sanctioned LLM and Social-budget adapters for Economic Taxonomy workers."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from decimal import Decimal
from functools import partial
from typing import Any

from sqlalchemy import func, select

from app.config import settings
from app.models.economic_taxonomy_runtime import (
    EvidencePacket,
    ProcessingRequest,
    ProcessingRequestEvent,
)
from app.services.economic_exposure_extraction import (
    RetryableProviderError,
    TerminalProviderError,
    UncertainProviderError,
)
from app.services.llm.config import (
    MINIMAX_M27,
    OPENCODE_GO_DEEPSEEK_V4_FLASH,
    ZAI_GLM_47_FLASH,
    ModelConfig,
    ModelPreset,
)
from app.services.llm.llm_service import (
    LLMError,
    LLMPreDispatchError,
    LLMQuotaExceededError,
    LLMRateLimitError,
    LLMService,
)
from app.services.social_anthropic_llm import AnthropicMeteredLLM
from app.services.social_llm_budget_service import (
    SocialEconomicReservationManager,
    SocialLLMBudgetService,
)

_EXTRACTION_SYSTEM = """You extract investable economic themes from supplied evidence.
Return one JSON object only. Use status successful_empty with candidates [] when no
economic exposure is supported. Otherwise use status accepted_candidates and candidates
with: candidate_key, display_name, raw_facets, mechanism, evidence_spans,
relationship_evidence, exposure_support, development_support, candidate_kind, securities,
and source_membership_keys. For Social evidence, copy the relevant membership_key values
from source_metadata.social_memberships into each candidate's source_membership_keys.
Evidence spans must be verbatim substrings. Do not invent securities or facts."""

_REVIEW_SYSTEM = """You review extracted economic-theme candidates against their evidence.
Return one JSON object only with decisions, one per candidate. Each decision has
candidate_key, decision (accepted, held, or rejected), and reason. Accept only claims
whose theme, mechanism, facets, securities, and quoted evidence are mutually supported."""


CLAUDE_HAIKU_55 = ModelConfig(model_id="anthropic/claude-haiku-5-5")

# The first entry whose credential is configured serves every call.
_PROVIDERS = (
    (lambda: bool((settings.anthropic_api_key or "").strip()), CLAUDE_HAIKU_55),
    (lambda: bool((settings.minimax_api_key or "").strip()), MINIMAX_M27),
    (lambda: bool(settings.zai_api_keys_list), ZAI_GLM_47_FLASH),
    (lambda: bool((settings.opencode_go_api_key or "").strip()), OPENCODE_GO_DEEPSEEK_V4_FLASH),
)


def _configured_model() -> ModelConfig | None:
    return next((model for configured, model in _PROVIDERS if configured()), None)


def _json_payload(text: str) -> dict[str, Any]:
    value = text.strip()
    if "```" in value:
        value = value.split("```", 1)[1]
        if value.lstrip().startswith("json"):
            value = value.lstrip()[4:]
        value = value.split("```", 1)[0]
    parsed = json.loads(value.strip())
    if not isinstance(parsed, dict):
        raise TypeError("provider_response_must_be_object")
    return parsed


class EconomicTaxonomyLLMProvider:
    """One-dispatch JSON provider shared by extraction and claim review."""

    def __init__(
        self,
        llm: Callable[[], LLMService | AnthropicMeteredLLM],
        *,
        model: str,
        budget: SocialLLMBudgetService | None = None,
    ):
        self.llm = llm
        self.model = model
        self.budget = budget

    def extract(self, *, evidence: Mapping[str, Any], policy_version: str):
        return self._complete(
            system=_EXTRACTION_SYSTEM,
            payload={"policy_version": policy_version, "evidence": dict(evidence)},
        )

    def review(
        self,
        *,
        extraction: Mapping[str, Any],
        policy_version: str,
        facet_catalog_semantic_hash: str,
    ):
        return self._complete(
            system=_REVIEW_SYSTEM,
            payload={
                "policy_version": policy_version,
                "facet_catalog_semantic_hash": facet_catalog_semantic_hash,
                "extraction": dict(extraction),
            },
        )

    def _complete(self, *, system: str, payload: Mapping[str, Any]):
        try:
            # Each dispatch runs in its own event loop, and an async client stays
            # bound to the loop that first used it.
            response = asyncio.run(
                self.llm().completion(
                    messages=[
                        {"role": "system", "content": system},
                        {
                            "role": "user",
                            "content": json.dumps(payload, sort_keys=True, default=str),
                        },
                    ],
                    model=self.model,
                    allow_fallbacks=False,
                    response_format={"type": "json_object"},
                    temperature=0,
                    max_tokens=4000,
                    num_retries=0,
                    metered=True,
                )
            )
        except LLMPreDispatchError as exc:
            raise RetryableProviderError(
                "provider_configuration_error", dispatch_confirmed=False
            ) from exc
        except (LLMRateLimitError, LLMQuotaExceededError) as exc:
            raise RetryableProviderError(
                "provider_capacity_unavailable",
                actual_cost=Decimal(0),
                dispatch_confirmed=True,
            ) from exc
        except LLMError as exc:
            raise UncertainProviderError(
                "provider_outcome_uncertain", dispatch_confirmed=True
            ) from exc

        request_id, actual_cost = self._response_metadata(response)
        try:
            result = _json_payload(LLMService.extract_content(response))
        except (json.JSONDecodeError, TypeError) as exc:
            raise TerminalProviderError(
                "invalid_provider_json",
                provider_request_id=request_id,
                actual_cost=actual_cost,
                dispatch_confirmed=True,
            ) from exc
        result["provider_request_id"] = request_id
        result["actual_cost"] = actual_cost
        return result

    def _response_metadata(self, response) -> tuple[str | None, Decimal | None]:
        request_id = getattr(response, "id", None)
        if self.budget is None:
            return request_id, Decimal(0)
        price = self.budget.price(self.model)
        usage = getattr(response, "usage", None)
        inputs = getattr(usage, "prompt_tokens", None)
        outputs = getattr(usage, "completion_tokens", None)
        actual_model = getattr(response, "model", None)
        provider = (getattr(response, "_hidden_params", None) or {}).get(
            "custom_llm_provider"
        )
        if (
            price is None
            or provider != price.provider
            or actual_model not in price.actual_models
            or not isinstance(inputs, int)
            or not isinstance(outputs, int)
            or inputs < 0
            or outputs < 0
        ):
            return request_id, None
        return request_id, price.cost(inputs, outputs)


class _UnmeteredReservation:
    state = "reserved"

    def mark_dispatched(self) -> None:
        self.state = "dispatched"

    def release_pre_dispatch(self) -> None:
        self.state = "released"

    def reconcile(self, *, actual_cost, provider_request_id) -> None:
        self.state = "reconciled"


class RoutedEconomicReservationManager:
    """Use Social's durable budget only for evidence admitted from Social work."""

    INPUT_TOKEN_LIMIT = 20_000
    OUTPUT_TOKEN_LIMIT = 4_000

    def __init__(self, session_factory, *, budget: SocialLLMBudgetService, model: str):
        self.session_factory = session_factory
        self.budget = budget
        self.model = model

    def reserve(self, *, attempt_key, logical_request_id, operation_kind):
        with self.session_factory() as session:
            request = session.get(ProcessingRequest, logical_request_id)
            packet = (
                session.get(EvidencePacket, request.evidence_packet_id)
                if request is not None
                else None
            )
            social_work_id = (packet.source_metadata or {}).get("social_work_id") if packet else None
            # A terminal outcome closes the logical operation. A requeued request
            # reserves a new one, so an uncertain earlier attempt keeps its charge
            # without blocking the retry.
            requeues = session.scalar(
                select(func.count(ProcessingRequestEvent.id)).where(
                    ProcessingRequestEvent.processing_request_id == logical_request_id,
                    ProcessingRequestEvent.event_type == "terminal_failure",
                )
            )
        if social_work_id is None:
            return _UnmeteredReservation()
        if not isinstance(social_work_id, int) or isinstance(social_work_id, bool):
            return None
        price = self.budget.price(self.model)
        if price is None:
            return None
        return SocialEconomicReservationManager(
            self.budget,
            work_ids=(social_work_id,),
            maximum_usd=price.cost(self.INPUT_TOKEN_LIMIT, self.OUTPUT_TOKEN_LIMIT),
            now=datetime.now(timezone.utc),
            pricing_version=price.version,
            input_token_limit=self.INPUT_TOKEN_LIMIT,
            output_token_limit=self.OUTPUT_TOKEN_LIMIT,
        ).reserve(
            attempt_key=attempt_key,
            logical_request_id=(
                f"{logical_request_id}:requeue-{requeues}" if requeues else logical_request_id
            ),
            operation_kind=operation_kind,
        )


def build_economic_taxonomy_provider(session_factory):
    model = _configured_model()
    if model is None:
        return None, None
    budget = SocialLLMBudgetService(
        session_factory,
        daily_limit_usd=settings.social_llm_daily_budget_usd,
        budget_timezone=settings.social_llm_budget_timezone,
    )
    # Extraction and review each define their JSON shape in the prompt, so
    # Claude gets no Social schema.
    llm = (
        partial(AnthropicMeteredLLM, output_schema=None)
        if model.model_id.startswith("anthropic/")
        else partial(LLMService, use_case="extraction", preset=ModelPreset(primary=model))
    )
    provider = EconomicTaxonomyLLMProvider(llm, model=model.model_id, budget=budget)
    reservations = RoutedEconomicReservationManager(
        session_factory, budget=budget, model=provider.model
    )
    return provider, reservations
