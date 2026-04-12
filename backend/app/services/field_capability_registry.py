"""Versioned screening-field capability registry by market/provider."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Tuple

from . import provider_routing_policy as routing_policy
from .fundamentals_completeness import field_source_map, field_tier_map, screening_fields

SUPPORT_STATE_SUPPORTED = "supported"
SUPPORT_STATE_COMPUTED = "computed"
SUPPORT_STATE_PARTIAL = "partial"
SUPPORT_STATE_UNSUPPORTED = "unsupported"

FALLBACK_BEHAVIOR_PRIMARY = "canonical_provider_primary"
FALLBACK_BEHAVIOR_FALLBACK = "canonical_provider_fallback"
FALLBACK_BEHAVIOR_POLICY_EXCLUDED = "market_policy_excludes_canonical_provider"
FALLBACK_BEHAVIOR_COMPUTED = "computed_locally_from_price_history"

SOURCE_TECHNICALS = "technicals"


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

    def _provider_states(
        self,
        canonical_provider: str,
        support_state: str,
    ) -> Dict[str, str]:
        provider_states = {
            provider: SUPPORT_STATE_UNSUPPORTED for provider in self.PROVIDER_ORDER
        }
        provider_states[canonical_provider] = support_state
        return provider_states

    def _market_capability(
        self,
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
                    canonical_provider=canonical_provider,
                    support_state=SUPPORT_STATE_COMPUTED,
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
                    canonical_provider=canonical_provider,
                    support_state=SUPPORT_STATE_UNSUPPORTED,
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
                canonical_provider=canonical_provider,
                support_state=support_state,
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
                market: self._market_capability(market, source)
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


field_capability_registry = FieldCapabilityRegistryService()
