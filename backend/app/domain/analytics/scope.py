"""Analytics market-scope policy (T6.4 / E6).

Contract
--------
Some analytics services (theme discovery, breadth
snapshot) currently operate on a US-only universe — either because
their data sources are US-biased (English-language content feeds) or
because their source datasets are still US-only. Rather
than let that assumption live silently inside each service, this module
centralises two concerns:

1. **Explicit scope tagging.** Every analytics response that is
   US-scoped today carries a ``market_scope`` field (via
   :func:`us_only_tag`) so the scope is observable at the HTTP layer
   and in logs — not buried in source comments. Market-aware analytics
   use :func:`market_scope_tag` for the same reason.

2. **Guarded entry points.** :func:`require_us_scope` rejects any
   attempt to pass a non-US market to a currently US-only analytics
   feature. Market-aware analytics bypass this guard and tag their
   responses with the requested market instead.

When a feature becomes market-aware (e.g. breadth gains per-market
series), the fix is to replace the ``us_only_tag`` call with the
actual market and remove the guard at the endpoint — not to lie about
scope.

Policy version
--------------
``POLICY_VERSION`` bumps whenever a feature changes scope (US → market-
aware, or vice versa). Pure additions (new US-only service) do not
require a bump.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

POLICY_VERSION: str = "2026.04.26.1"

_US_MARKET: str = "US"


class AnalyticsFeature(str, Enum):
    """Analytics features whose scope is tracked by this policy."""

    THEME_DISCOVERY = "theme_discovery"
    IBD_GROUP_RANK = "ibd_group_rank"
    BREADTH_SNAPSHOT = "breadth_snapshot"


# Current scope for each feature. When a feature is generalised the
# entry moves to None (or a list of supported markets) and the service
# stops using us_only_tag / require_us_scope.
_US_ONLY_FEATURES: dict[AnalyticsFeature, str] = {
    AnalyticsFeature.THEME_DISCOVERY: (
        "theme content sources are English-language biased; no non-US coverage"
    ),
}


class UnsupportedMarketError(ValueError):
    """Raised when a caller passes a non-US market to a US-only analytics feature."""


def policy_version() -> str:
    """Accessor for the active policy version.

    Mirrors ``provider_routing_policy.policy_version()`` and
    ``mixed_market_policy.policy_version()`` so downstream consumers can
    import a consistent symbol from any policy module.
    """
    return POLICY_VERSION


def us_only_tag(feature: AnalyticsFeature) -> dict[str, str]:
    """Return the ``{market_scope, scope_reason}`` tag to embed in analytics responses.

    Callers spread this dict into their response payload. The returned
    keys are stable; values evolve with ``POLICY_VERSION`` (exposed
    separately via :func:`policy_version`). Consumers should treat
    ``market_scope == "US"`` as the feature's current scope.
    """
    return market_scope_tag(_US_MARKET, reason=_US_ONLY_FEATURES.get(feature))


def market_scope_tag(
    market: Optional[str],
    *,
    reason: str | None = None,
) -> dict[str, str]:
    """Return explicit scope metadata for market-aware analytics responses."""
    if market is None:
        normalized = _US_MARKET
    elif isinstance(market, str):
        normalized = market.strip().upper() or _US_MARKET
    else:
        normalized = str(market)
    tag = {"market_scope": normalized}
    if reason:
        tag["scope_reason"] = reason
    return tag


def require_us_scope(
    market: Optional[str],
    feature: AnalyticsFeature,
) -> None:
    """Guard that rejects non-US markets for currently US-only analytics features.

    ``None`` / empty / whitespace is treated as the US default (preserves
    legacy call sites that never pass a market). Non-string inputs and
    any other market raise :class:`UnsupportedMarketError` with a
    descriptive reason.
    """
    if market is None:
        return
    if not isinstance(market, str):
        # Match provider_routing_policy.normalize_market's defensive shape:
        # a non-string market is not US, so reject loudly.
        raise UnsupportedMarketError(
            f"{feature.value}: non-string market {market!r} (type {type(market).__name__}) not supported"
        )
    canonical = market.strip().upper()
    if canonical == "" or canonical == _US_MARKET:
        return

    reason = _US_ONLY_FEATURES.get(feature)
    if reason is None:
        return
    raise UnsupportedMarketError(
        f"{feature.value}: market '{market}' not supported — {reason}"
    )


def describe_policy() -> dict:
    """Stable snapshot for API/UI surfacing."""
    return {
        "policy_version": POLICY_VERSION,
        "us_only_features": {f.value: reason for f, reason in _US_ONLY_FEATURES.items()},
    }


__all__ = [
    "POLICY_VERSION",
    "AnalyticsFeature",
    "UnsupportedMarketError",
    "market_scope_tag",
    "policy_version",
    "us_only_tag",
    "require_us_scope",
    "describe_policy",
]
