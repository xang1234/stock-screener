"""Versioned screening-field capability registry by market/provider."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Mapping, Tuple

from . import provider_routing_policy as routing_policy
from .finviz_parser import FinvizParser
from .fundamentals_completeness import (
    ENHANCED_FIELDS,
    TECHNICAL_FIELDS,
    field_source_map,
    field_tier_map,
    screening_fields,
)

SUPPORT_STATE_SUPPORTED = "supported"
SUPPORT_STATE_COMPUTED = "computed"
SUPPORT_STATE_PARTIAL = "partial"
SUPPORT_STATE_UNSUPPORTED = "unsupported"
SUPPORT_STATE_MISSING = "missing"
SUPPORT_STATE_AVAILABLE = "available"

FALLBACK_BEHAVIOR_PRIMARY = "canonical_provider_primary"
FALLBACK_BEHAVIOR_FALLBACK = "canonical_provider_fallback"
FALLBACK_BEHAVIOR_POLICY_EXCLUDED = "market_policy_excludes_canonical_provider"
FALLBACK_BEHAVIOR_COMPUTED = "computed_locally_from_price_history"

SOURCE_TECHNICALS = "technicals"

REASON_CODE_POLICY_EXCLUDED = "unsupported_market_policy_excludes_canonical_provider"
REASON_CODE_NON_US_GAP = "unsupported_non_us_ownership_sentiment_data_unavailable"
REASON_CODE_MISSING_SUPPORTED = "missing_supported_field_value"

OWNERSHIP_SENTIMENT_FIELDS: Tuple[str, ...] = (
    "institutional_ownership",
    "insider_ownership",
    "short_interest",
)


@dataclass(frozen=True)
class FieldMarketCapability:
    market: str
    support_state: str
    fallback_behavior: str
    policy_provider_chain: Tuple[str, ...]
    canonical_provider: str
    canonical_provider_position: int | None
    providers_before_canonical: Tuple[str, ...]
    providers_after_canonical: Tuple[str, ...]
    provider_states: Mapping[str, str]


@dataclass(frozen=True)
class FieldCapabilityEntry:
    field: str
    tier: str
    canonical_source: str
    markets: Mapping[str, FieldMarketCapability]


class FieldCapabilityRegistryService:
    """Deterministic matrix for screening-field market/provider coverage."""

    REGISTRY_VERSION = "2026.04.12.1"
    MARKET_ORDER: Tuple[str, ...] = (
        routing_policy.MARKET_US,
        routing_policy.MARKET_HK,
        routing_policy.MARKET_JP,
        routing_policy.MARKET_TW,
    )
    PROVIDER_ORDER: Tuple[str, ...] = (
        routing_policy.PROVIDER_FINVIZ,
        routing_policy.PROVIDER_YFINANCE,
        routing_policy.PROVIDER_ALPHAVANTAGE,
        SOURCE_TECHNICALS,
    )
    SUPPORT_STATES: Tuple[str, ...] = (
        SUPPORT_STATE_SUPPORTED,
        SUPPORT_STATE_COMPUTED,
        SUPPORT_STATE_PARTIAL,
        SUPPORT_STATE_UNSUPPORTED,
    )

    def __init__(self) -> None:
        self._provider_supported_fields = self._build_provider_supported_fields()

    @staticmethod
    def _finviz_supported_fields() -> frozenset[str]:
        all_finviz_fields = set(FinvizParser.FUNDAMENTAL_FIELD_MAP.values())
        all_finviz_fields.update(FinvizParser.GROWTH_FIELD_MAP.values())
        return frozenset(all_finviz_fields & set(screening_fields()))

    def _build_provider_supported_fields(self) -> Dict[str, frozenset[str]]:
        fields = set(screening_fields())
        technical_fields = set(TECHNICAL_FIELDS)
        finviz_only_fields = set(ENHANCED_FIELDS)
        return {
            routing_policy.PROVIDER_FINVIZ: self._finviz_supported_fields(),
            # yfinance supports the baseline screening surface; enhanced
            # finviz-only and local technicals are excluded.
            routing_policy.PROVIDER_YFINANCE: frozenset(
                fields - finviz_only_fields - technical_fields
            ),
            # Keep alphavantage fail-closed until field-level support is
            # explicitly mapped and integrated in pipeline flows.
            routing_policy.PROVIDER_ALPHAVANTAGE: frozenset(),
            SOURCE_TECHNICALS: frozenset(technical_fields),
        }

    def _provider_state_for_market(
        self,
        field: str,
        market: str,
        provider: str,
        canonical_provider: str,
        canonical_support_state: str,
    ) -> str:
        if field not in self._provider_supported_fields[provider]:
            return SUPPORT_STATE_UNSUPPORTED

        if provider == SOURCE_TECHNICALS:
            return SUPPORT_STATE_COMPUTED

        policy_chain = routing_policy.providers_for(market)
        if provider not in policy_chain:
            return SUPPORT_STATE_UNSUPPORTED

        if provider == canonical_provider:
            return canonical_support_state

        return (
            SUPPORT_STATE_SUPPORTED
            if policy_chain.index(provider) == 0
            else SUPPORT_STATE_PARTIAL
        )

    def _provider_states(
        self,
        field: str,
        market: str,
        canonical_provider: str,
        canonical_support_state: str,
    ) -> Dict[str, str]:
        return {
            provider: self._provider_state_for_market(
                field=field,
                market=market,
                provider=provider,
                canonical_provider=canonical_provider,
                canonical_support_state=canonical_support_state,
            )
            for provider in self.PROVIDER_ORDER
        }

    @staticmethod
    def _is_present(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, str) and value == "":
            return False
        if isinstance(value, float) and math.isnan(value):
            return False
        return True

    def _market_capability(
        self,
        field: str,
        market: str,
        canonical_provider: str,
    ) -> FieldMarketCapability:
        policy_chain = routing_policy.providers_for(market)

        if canonical_provider == SOURCE_TECHNICALS:
            return FieldMarketCapability(
                market=market,
                support_state=SUPPORT_STATE_COMPUTED,
                fallback_behavior=FALLBACK_BEHAVIOR_COMPUTED,
                policy_provider_chain=policy_chain,
                canonical_provider=canonical_provider,
                canonical_provider_position=None,
                providers_before_canonical=tuple(),
                providers_after_canonical=tuple(),
                provider_states=self._provider_states(
                    field=field,
                    market=market,
                    canonical_provider=canonical_provider,
                    canonical_support_state=SUPPORT_STATE_COMPUTED,
                ),
            )

        if canonical_provider not in policy_chain:
            return FieldMarketCapability(
                market=market,
                support_state=SUPPORT_STATE_UNSUPPORTED,
                fallback_behavior=FALLBACK_BEHAVIOR_POLICY_EXCLUDED,
                policy_provider_chain=policy_chain,
                canonical_provider=canonical_provider,
                canonical_provider_position=None,
                providers_before_canonical=tuple(),
                providers_after_canonical=tuple(),
                provider_states=self._provider_states(
                    field=field,
                    market=market,
                    canonical_provider=canonical_provider,
                    canonical_support_state=SUPPORT_STATE_UNSUPPORTED,
                ),
            )

        position = policy_chain.index(canonical_provider)
        if position == 0:
            support_state = SUPPORT_STATE_SUPPORTED
            fallback_behavior = FALLBACK_BEHAVIOR_PRIMARY
        else:
            support_state = SUPPORT_STATE_PARTIAL
            fallback_behavior = FALLBACK_BEHAVIOR_FALLBACK

        return FieldMarketCapability(
            market=market,
            support_state=support_state,
            fallback_behavior=fallback_behavior,
            policy_provider_chain=policy_chain,
            canonical_provider=canonical_provider,
            canonical_provider_position=position,
            providers_before_canonical=policy_chain[:position],
            providers_after_canonical=policy_chain[position + 1 :],
            provider_states=self._provider_states(
                field=field,
                market=market,
                canonical_provider=canonical_provider,
                canonical_support_state=support_state,
            ),
        )

    def entries(self) -> Tuple[FieldCapabilityEntry, ...]:
        tiers = field_tier_map()
        sources = field_source_map()
        field_names = tuple(sorted(screening_fields()))

        entries = []
        for field in field_names:
            tier = tiers.get(field)
            source = sources.get(field)
            if tier is None or source is None:
                continue

            market_caps = {
                market: self._market_capability(field, market, source)
                for market in self.MARKET_ORDER
            }
            entries.append(
                FieldCapabilityEntry(
                    field=field,
                    tier=tier,
                    canonical_source=source,
                    markets=market_caps,
                )
            )

        return tuple(entries)

    def artifact(self) -> Dict[str, object]:
        entries = []
        for entry in self.entries():
            markets = {}
            for market in self.MARKET_ORDER:
                cap = entry.markets[market]
                markets[market] = {
                    "support_state": cap.support_state,
                    "fallback_behavior": cap.fallback_behavior,
                    "policy_provider_chain": list(cap.policy_provider_chain),
                    "canonical_provider": cap.canonical_provider,
                    "canonical_provider_position": cap.canonical_provider_position,
                    "providers_before_canonical": list(cap.providers_before_canonical),
                    "providers_after_canonical": list(cap.providers_after_canonical),
                    "provider_states": dict(cap.provider_states),
                }

            entries.append(
                {
                    "field": entry.field,
                    "tier": entry.tier,
                    "canonical_source": entry.canonical_source,
                    "markets": markets,
                }
            )

        return {
            "registry_version": self.REGISTRY_VERSION,
            "routing_policy_version": routing_policy.policy_version(),
            "markets": list(self.MARKET_ORDER),
            "providers": list(self.PROVIDER_ORDER),
            "support_states": list(self.SUPPORT_STATES),
            "field_count": len(entries),
            "fields": entries,
        }

    def derive_ownership_sentiment_availability(
        self,
        data: Mapping[str, Any] | None,
        market: str | None,
    ) -> Dict[str, Dict[str, Any]]:
        """Return status + explicit reason codes for ownership/sentiment fields.

        This powers graceful-degrade transparency for non-US markets where
        ownership/sentiment sources can be missing or policy-excluded.
        """
        resolved_market = routing_policy.normalize_market(market)
        by_field = {entry.field: entry for entry in self.entries()}
        payload = data or {}

        result: Dict[str, Dict[str, Any]] = {}
        for field in OWNERSHIP_SENTIMENT_FIELDS:
            entry = by_field.get(field)
            if entry is None:
                continue
            cap = entry.markets[resolved_market]
            present = self._is_present(payload.get(field))

            if present:
                result[field] = {
                    "status": SUPPORT_STATE_AVAILABLE,
                    "reason_code": None,
                    "canonical_provider": cap.canonical_provider,
                    "support_state": cap.support_state,
                }
                continue

            if cap.support_state == SUPPORT_STATE_UNSUPPORTED:
                reason_code = REASON_CODE_POLICY_EXCLUDED
                status = SUPPORT_STATE_UNSUPPORTED
            elif (
                resolved_market in (
                    routing_policy.MARKET_HK,
                    routing_policy.MARKET_JP,
                    routing_policy.MARKET_TW,
                )
            ):
                reason_code = REASON_CODE_NON_US_GAP
                status = SUPPORT_STATE_UNSUPPORTED
            else:
                reason_code = REASON_CODE_MISSING_SUPPORTED
                status = SUPPORT_STATE_MISSING

            result[field] = {
                "status": status,
                "reason_code": reason_code,
                "canonical_provider": cap.canonical_provider,
                "support_state": cap.support_state,
            }

        return result


field_capability_registry = FieldCapabilityRegistryService()
