"""
Fundamentals Completeness & Field-Level Provenance
==================================================

Market-aware completeness score (0-100) and per-field provenance for
``StockFundamental`` rows. Designed to pair with the T1 routing policy
(``provider_routing_policy``): fields that routing skips for a market
must not count against that market's completeness score.

Rationale
---------
An HK stock fetched via yfinance only should not be penalised for
missing finviz-only fields — those fields were deliberately not
attempted (see ``provider_routing_policy.py``). Without market-awareness
every non-US stock would score ~50%, making the score useless as a
data-quality signal.

Tiers
-----
Fields are grouped by *importance for scanners*, each tier weighted:

- ``CORE`` (weight 3): scanners cannot reliably rank without these
  (sector, industry, market_cap, ipo_date, pe_ratio, avg_volume,
  eps_rating).
- ``STANDARD`` (weight 1): the bulk of yfinance-fillable fundamentals.
  Expected for every market.
- ``ENHANCED`` (weight 1): finviz-only fields (short interest,
  insider/institutional transactions, forward estimates, price-to-cash,
  price-to-fcf, long-term debt ratios). Expected for US only; not
  counted against HK/JP/TW.

Score
-----
``score = round(100 * earned / possible)`` where

- ``possible = sum(weight * 1 for each *expected* field in the market)``
- ``earned   = sum(weight * 1 for each *present*  field in the market)``

A field is "present" iff ``data[field]`` is not ``None`` / empty string /
NaN. A field is "expected" iff the market's routing policy can populate
it (via any of its approved providers).

Provenance
----------
``derive_field_provenance`` returns ``{field_name: source_name}`` for
each populated field. The source is inferred from a static
``_FIELD_SOURCE`` map (stable knowledge about which provider supplies
which field in the hybrid pipeline) intersected with the market's
allowed providers. Fields missing from the source map are tagged
``"unknown"`` rather than dropped, so downstream consumers can still
inspect them.

Scope (T2)
----------
This module *computes* scores and provenance. Persistence is handled
by ``FundamentalsCacheService._store_in_database`` which writes the
values to ``stock_fundamentals.field_completeness_score`` and
``field_provenance``.

Consumers of the score (e.g., T4 quality-aware fallback, T6 ownership
degrade policy) read it from the DB column; they should not recompute
it on every access.
"""
from __future__ import annotations

import math
from typing import Any, Dict, FrozenSet, Mapping, Tuple

from . import provider_routing_policy as routing_policy


# --- Scoring weights --------------------------------------------------------

CORE_WEIGHT = 3
STANDARD_WEIGHT = 1
ENHANCED_WEIGHT = 1


# --- Tier definitions -------------------------------------------------------
# Kept as frozensets so callers can't mutate them at runtime.

CORE_FIELDS: FrozenSet[str] = frozenset({
    "sector",
    "industry",
    "market_cap",
    "ipo_date",
    "pe_ratio",
    "avg_volume",
    "eps_rating",
})

# Fields expected from yfinance + technical calculator. Covers valuation,
# growth, profitability, technicals, 52w range, performance — the fields
# that every market should populate via yfinance-or-better.
STANDARD_FIELDS: FrozenSet[str] = frozenset({
    # Market data
    "shares_outstanding",
    # Valuation (yfinance-fillable)
    "forward_pe",
    "peg_ratio",
    "price_to_book",
    "price_to_sales",
    "ev_ebitda",
    "ev_sales",
    "target_price",
    # Earnings / Growth
    "eps_current",
    "eps_growth_qq",
    "eps_growth_yy",
    "sales_growth_qq",
    "sales_growth_yy",
    "revenue_current",
    "revenue_growth",
    # Profitability
    "profit_margin",
    "operating_margin",
    "gross_margin",
    "roe",
    "roa",
    # Financial health
    "current_ratio",
    "quick_ratio",
    "debt_to_equity",
    # Ownership (yfinance exposes these broadly)
    "insider_ownership",
    "institutional_ownership",
    # Technicals (from local technical_calculator_service)
    "beta",
    "rsi_14",
    "atr_14",
    "sma_20",
    "sma_50",
    "sma_200",
    "volatility_week",
    "volatility_month",
    # Performance
    "perf_week",
    "perf_month",
    "perf_quarter",
    "perf_half_year",
    "perf_year",
    "perf_ytd",
    # 52-week range
    "week_52_high",
    "week_52_low",
    "week_52_high_distance",
    "week_52_low_distance",
    # Dividend
    "dividend_yield",
})

# Fields computed locally from cached price history rather than fetched
# directly from a remote fundamentals provider.
TECHNICAL_FIELDS: FrozenSet[str] = frozenset({
    "rsi_14",
    "atr_14",
    "sma_20",
    "sma_50",
    "sma_200",
    "volatility_week",
    "volatility_month",
    "perf_week",
    "perf_month",
    "perf_quarter",
    "perf_half_year",
    "perf_year",
    "perf_ytd",
    "week_52_high",
    "week_52_low",
    "week_52_high_distance",
    "week_52_low_distance",
})

# Fields only the finviz provider supplies in the hybrid pipeline.
# Not expected for markets where routing excludes finviz.
ENHANCED_FIELDS: FrozenSet[str] = frozenset({
    "short_float",
    "short_ratio",
    "short_interest",
    "insider_transactions",
    "institutional_transactions",
    "eps_next_y",
    "eps_next_5y",
    "eps_next_q",
    "lt_debt_to_equity",
    "roic",
    "price_to_cash",
    "price_to_fcf",
})

# Scan-critical fields that are not part of completeness scoring tiers.
# These are still screening fields and must be represented in capability
# artifacts used by transparency/governance surfaces.
AUXILIARY_FIELDS: FrozenSet[str] = frozenset({
    "first_trade_date",
    "recent_quarter_date",
    "previous_quarter_date",
})


# --- Static field -> source map --------------------------------------------
# Best-known provider that contributes each field in the current hybrid
# pipeline. Used to derive field-level provenance without refactoring the
# fetch path. If a field can come from multiple providers, the primary is
# listed (e.g., finviz wins over yfinance for shared fields on US because
# DataSourceService prefers finviz).

_SOURCE_TECHNICALS = "technicals"


def _build_field_source_map() -> Mapping[str, str]:
    mapping: Dict[str, str] = {}
    for field in ENHANCED_FIELDS:
        mapping[field] = routing_policy.PROVIDER_FINVIZ
    # Technical-calculator fields.
    for field in TECHNICAL_FIELDS:
        mapping[field] = _SOURCE_TECHNICALS
    # Everything else in CORE/STANDARD that isn't technicals comes from
    # yfinance as the baseline provider (finviz may overwrite some in US,
    # but we tag the *canonical* source — the one that guarantees the
    # field exists across markets).
    for field in CORE_FIELDS | STANDARD_FIELDS:
        mapping.setdefault(field, routing_policy.PROVIDER_YFINANCE)
    for field in AUXILIARY_FIELDS:
        mapping.setdefault(field, routing_policy.PROVIDER_YFINANCE)
    return mapping


_FIELD_SOURCE: Mapping[str, str] = _build_field_source_map()


def _build_field_tier_map() -> Mapping[str, str]:
    mapping: Dict[str, str] = {}
    for field in CORE_FIELDS:
        mapping[field] = "core"
    for field in STANDARD_FIELDS:
        mapping[field] = "standard"
    for field in ENHANCED_FIELDS:
        mapping[field] = "enhanced"
    for field in AUXILIARY_FIELDS:
        mapping[field] = "auxiliary"
    return mapping


_FIELD_TIER: Mapping[str, str] = _build_field_tier_map()


# --- Helpers ---------------------------------------------------------------

def _is_present(value: Any) -> bool:
    """Return True if ``value`` is a populated field value.

    Treats ``None``, empty string, and float NaN as missing. Zero and
    ``False`` are present (they are legitimate fundamental values).
    """
    if value is None:
        return False
    if isinstance(value, str) and value == "":
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    return True


def _expected_fields(market: str | None) -> Tuple[FrozenSet[str], FrozenSet[str], FrozenSet[str]]:
    """Return (core, standard, enhanced) field sets expected for ``market``.

    The enhanced tier is empty for markets whose routing policy excludes
    finviz.
    """
    normalized_market = routing_policy.normalize_market(market)
    finviz_allowed = routing_policy.is_supported(
        normalized_market, routing_policy.PROVIDER_FINVIZ
    )
    enhanced = ENHANCED_FIELDS if finviz_allowed else frozenset()
    return CORE_FIELDS, STANDARD_FIELDS, enhanced


# --- Public API ------------------------------------------------------------

def expected_fields(market: str | None) -> FrozenSet[str]:
    """Return the union of all fields ``market`` is expected to populate."""
    core, std, enh = _expected_fields(market)
    return core | std | enh


def screening_fields() -> FrozenSet[str]:
    """Return all fundamentals screening fields across all tiers."""
    return frozenset(_FIELD_TIER.keys())


def field_source_map() -> Mapping[str, str]:
    """Return the canonical field->source map used for provenance."""
    return _FIELD_SOURCE


def field_tier_map() -> Mapping[str, str]:
    """Return ``{field_name: tier_name}`` for all screening fields."""
    return _FIELD_TIER


def compute_completeness_score(
    data: Mapping[str, Any] | None,
    market: str | None,
) -> int:
    """Return a 0-100 completeness score for ``data`` in ``market``.

    Formula: ``round(100 * earned / possible)`` over weighted tiers. See
    module docstring for full semantics. Returns 0 for an empty payload.
    """
    if not data:
        return 0
    core, std, enh = _expected_fields(market)

    possible = (
        len(core) * CORE_WEIGHT
        + len(std) * STANDARD_WEIGHT
        + len(enh) * ENHANCED_WEIGHT
    )
    if possible == 0:
        # No expected fields defined for this market — degenerate case.
        return 0

    earned = 0
    for field in core:
        if _is_present(data.get(field)):
            earned += CORE_WEIGHT
    for field in std:
        if _is_present(data.get(field)):
            earned += STANDARD_WEIGHT
    for field in enh:
        if _is_present(data.get(field)):
            earned += ENHANCED_WEIGHT

    return round(100 * earned / possible)


def derive_field_provenance(
    data: Mapping[str, Any] | None,
    market: str | None,
) -> Dict[str, str]:
    """Return ``{field_name: source_name}`` for every populated field.

    Source is looked up in the static field->provider map intersected
    with the market's routing policy. Fields not in the static map are
    tagged ``"unknown"`` (informational; they still count towards score
    if they're in an expected tier).
    """
    if not data:
        return {}
    normalized_market = routing_policy.normalize_market(market)
    expected = expected_fields(normalized_market)
    allowed = set(routing_policy.providers_for(normalized_market))
    provenance: Dict[str, str] = {}
    for field, value in data.items():
        if field not in expected:
            continue
        if not _is_present(value):
            continue
        source = _FIELD_SOURCE.get(field)
        if source is None:
            provenance[field] = "unknown"
        elif source == _SOURCE_TECHNICALS:
            # Technicals are locally computed, always available.
            provenance[field] = _SOURCE_TECHNICALS
        elif source in allowed:
            provenance[field] = source
        else:
            # Field value present but its canonical source isn't in this
            # market's policy — likely from a fallback provider. Record
            # as "unknown" rather than misattributing.
            provenance[field] = "unknown"
    return provenance
