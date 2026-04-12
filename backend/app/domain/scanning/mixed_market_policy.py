"""Mixed-market scan normalization policy (T6.3 / E6).

Policy contract
---------------
A scan universe may span one or multiple markets (US/HK/JP/TW/...). To
produce deterministic, fair rankings across markets, two distinct
dimensions are normalized — and each is orthogonal to the other:

1. **Percentiles — market-scoped.** RS percentiles are computed within
   each market's peer set, not globally. Implemented upstream in
   ``data_preparation._compute_market_rs_universe_performances``
   (T6.2); this module documents the contract but does not compute
   percentiles itself.

2. **Liquidity / cap constraints — USD-scoped in mixed mode.** When a
   scan spans >1 market, filters like ``market_cap_min`` / ``volume_min``
   are interpreted as USD and evaluated against the FX-normalized
   columns ``market_cap_usd`` and ``adv_usd`` populated by T5.3. Rows
   whose USD columns are ``None`` (FX unavailable) are excluded — a
   deterministic "fail closed" posture that prevents silent unit drift
   when an HKD market cap is compared against a USD threshold.

Single-market scans fall back to native-currency columns
(``market_cap`` and the 20-day price-data volume average) to preserve
pre-existing semantics. Users running a single-market HK scan expect
their ``market_cap_min`` to be in HKD.

Stability
---------
``is_mixed_market`` is computed once per scan at data-preparation time
and attached to each :class:`StockData` instance. Downstream filter
logic reads the flag instead of re-inferring it, so a late-arriving
symbol with a new market cannot flip the policy mid-scan.

Policy version
--------------
``POLICY_VERSION`` is bumped whenever the *semantics* change (e.g. if we
decide to downgrade rather than exclude rows without FX). Adding a new
USD-normalized filter field does not require a version bump.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional

POLICY_VERSION: str = "2026.04.13.1"

# Field names populated by T5.3's FX normalization enrichment hook.
USD_CAP_FIELD: str = "market_cap_usd"
USD_ADV_FIELD: str = "adv_usd"
# Native-currency cap field, populated on every StockFundamental row.
NATIVE_CAP_FIELD: str = "market_cap"

US_MARKET: str = "US"


def is_mixed_market(markets: Iterable[Optional[str]]) -> bool:
    """Return True when the scan universe spans more than one market.

    Any ``None`` market is normalised to ``"US"`` to match the convention
    used by ``data_preparation`` — legacy rows without a market tag are
    treated as US stocks (the historical default).
    """
    seen = {(m or US_MARKET) for m in markets}
    return len(seen) > 1


def resolve_cap_for_filter(
    fundamentals: Optional[Mapping[str, Any]],
    *,
    mixed_market: bool,
) -> Optional[float]:
    """Return the cap value for ``market_cap_min/max`` filter evaluation.

    - Mixed-market: ``market_cap_usd`` only. ``None`` when the USD column
      is missing so the caller fails the filter closed (no silent
      native-vs-USD mixing).
    - Single-market: ``market_cap`` (native currency).
    """
    if fundamentals is None:
        return None
    key = USD_CAP_FIELD if mixed_market else NATIVE_CAP_FIELD
    value = fundamentals.get(key)
    return float(value) if value is not None else None


def resolve_adv_for_filter(
    fundamentals: Optional[Mapping[str, Any]],
    native_avg_volume: Optional[float],
    *,
    mixed_market: bool,
) -> Optional[float]:
    """Return the ADV value for ``volume_min`` filter evaluation.

    - Mixed-market: ``adv_usd`` (USD notional ≈ shares × price × fx) from
      fundamentals. ``None`` when missing. Callers must interpret the
      ``volume_min`` threshold as USD in this mode.
    - Single-market: ``native_avg_volume`` (the 20-day average share
      volume computed from price data — the legacy contract).
    """
    if mixed_market:
        if fundamentals is None:
            return None
        value = fundamentals.get(USD_ADV_FIELD)
        return float(value) if value is not None else None
    return float(native_avg_volume) if native_avg_volume is not None else None


def describe_policy() -> dict:
    """Stable snapshot of the active policy for API/UI surfacing.

    Consumers (e.g. scan result metadata, frontend badges) should treat
    keys as stable; values may evolve when ``POLICY_VERSION`` bumps.
    """
    return {
        "policy_version": POLICY_VERSION,
        "percentile_scope": "per-market",
        "liquidity_cap_scope": {
            "mixed_market": "usd_normalized",
            "single_market": "native_currency",
        },
        "missing_fx_behaviour": "exclude",
        "usd_cap_field": USD_CAP_FIELD,
        "usd_adv_field": USD_ADV_FIELD,
    }


__all__ = [
    "POLICY_VERSION",
    "USD_CAP_FIELD",
    "USD_ADV_FIELD",
    "NATIVE_CAP_FIELD",
    "US_MARKET",
    "is_mixed_market",
    "resolve_cap_for_filter",
    "resolve_adv_for_filter",
    "describe_policy",
]
