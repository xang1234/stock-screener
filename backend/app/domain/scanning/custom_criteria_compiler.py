"""Compile a custom-screener criteria dict into a feature-store FilterSpec.

A "custom" scan is essentially a parameterised query: price ranges, volume
thresholds, sector lists, MA alignment flags, etc. Almost every one of these
criteria corresponds directly to a field already pre-computed by the daily
feature snapshot (see ``app.use_cases.feature_store.build_daily_snapshot``
and the JSON field map in ``app.infra.query.feature_store_query``). When
that's the case the scan can be served by a single SQL query against an
already-published feature run, without recomputing per-symbol scores.

This module is the translation layer:

    custom_filters dict --(compile)--> CompiledCustomCriteria
                                            ├── filter_spec: FilterSpec
                                            ├── score_field: str | None
                                            ├── min_score: float | None
                                            └── unrepresentable_keys

The use case decides what to do based on ``unrepresentable_keys``:
  * empty       → fully feature-store-answerable, instant scan
  * non-empty   → fall back to async chunked compute
"""

from __future__ import annotations

from dataclasses import dataclass

from app.domain.common.query import FilterMode, FilterSpec


# Markets where the feature store's USD-normalised ``market_cap_usd``
# column matches the unit a user is most likely to type into the custom
# market-cap filter. ``CustomScanner._check_market_cap`` reads native
# market_cap in single-market mode and ``market_cap_usd`` in mixed-market
# mode (see ``app.domain.scanning.mixed_market_policy.resolve_cap_for_filter``);
# for US the native column already holds USD so the values agree, but
# every other single-market universe (HK, JP, TW…) holds native currency
# and would silently disagree with our USD column. Volume has no
# analogous mapping — single-market mode evaluates ``volume_min`` against
# share count, which has no column in ``stock_feature_daily``, so volume
# is *only* representable in mixed-market mode regardless of currency.
_USD_MARKET_CAP_COMPATIBLE_SINGLE_MARKETS: frozenset[str] = frozenset({"US"})


@dataclass(frozen=True)
class CompiledCustomCriteria:
    """Result of compiling a custom-criteria dict against the feature store."""

    filter_spec: FilterSpec
    score_field: str | None
    min_score: float | None
    unrepresentable_keys: tuple[str, ...]
    representable_keys: tuple[str, ...]

    @property
    def is_fully_representable(self) -> bool:
        return not self.unrepresentable_keys

    @property
    def has_constraints(self) -> bool:
        spec = self.filter_spec
        return bool(
            spec.range_filters
            or spec.categorical_filters
            or spec.boolean_filters
            or spec.text_searches
        ) or self.min_score is not None


def _flatten_filters(criteria: dict | None) -> dict:
    """Mirror ``CustomScanner._get_filters_config``: support nested + top-level.

    Top-level keys are the legacy form; the modern form nests under
    ``custom_filters``. Nested wins on conflict.
    """
    if not criteria:
        return {}
    base = dict(criteria)
    nested = base.pop("custom_filters", None) or {}
    base.pop("min_score", None)
    base.update(nested)
    return base


def _is_mixed_market(universe_market: str | None) -> bool:
    """Whether the universe runs in mixed-market mode (no single-market pin).

    ``CustomScanner`` derives mixed-vs-single from the resolved StockData
    flag, but at scan-creation time we only have the universe metadata;
    ``universe_type ∈ {ALL, MARKET}`` (gated upstream) plus
    ``universe_market is None`` is the unambiguous mixed-market signal.
    """
    return universe_market is None


def _is_market_cap_usd_compatible(universe_market: str | None) -> bool:
    """Whether the user's market-cap threshold can be evaluated against
    ``market_cap_usd`` without a unit mismatch — true for mixed-market and
    explicit USD markets only.
    """
    if _is_mixed_market(universe_market):
        return True
    return (
        str(universe_market).strip().upper()
        in _USD_MARKET_CAP_COMPATIBLE_SINGLE_MARKETS
    )


def compile_custom_criteria(
    criteria: dict | None,
    *,
    screeners: list[str] | None = None,
    universe_market: str | None = None,
) -> CompiledCustomCriteria:
    """Translate a custom-criteria dict into a feature-store ``FilterSpec``.

    Args:
        criteria: The scan request's ``criteria`` dict. May contain nested
            ``custom_filters`` and a ``min_score`` gate, plus legacy
            top-level filter keys (``CustomScanner`` accepts both forms).
        screeners: Screener list from the scan request. The score gate is
            only set for single-screener custom scans (``["custom"]``);
            multi-screener composites need extra logic to decide which
            score field gates passing.
        universe_market: Single-market universe code, or ``None`` for
            mixed-market. Determines whether USD-normalised columns are
            unit-compatible with the user's volume/market-cap thresholds.

    Returns:
        A ``CompiledCustomCriteria`` summarising what's representable.
    """
    flat = _flatten_filters(criteria)
    spec = FilterSpec()
    representable: list[str] = []
    unrepresentable: list[str] = []

    mixed_market = _is_mixed_market(universe_market)
    market_cap_compatible = _is_market_cap_usd_compatible(universe_market)

    # Price range -> JSON current_price
    p_min = flat.get("price_min")
    p_max = flat.get("price_max")
    if p_min is not None or p_max is not None:
        spec.add_range("current_price", min_value=p_min, max_value=p_max)
        if p_min is not None:
            representable.append("price_min")
        if p_max is not None:
            representable.append("price_max")

    # Volume -> StockFundamental.adv_usd (USD notional). Only correct in
    # mixed-market mode; single-market mode evaluates ``volume_min`` as a
    # *share* threshold against price-data volume which has no column in
    # ``stock_feature_daily``.
    v_min = flat.get("volume_min")
    if v_min is not None and v_min > 0:
        if mixed_market:
            spec.add_range("adv_usd", min_value=v_min)
            representable.append("volume_min")
        else:
            unrepresentable.append("volume_min")

    # Market cap -> StockFundamental.market_cap_usd. Mixed-market and US
    # single-market resolve to USD; other single-markets hold native
    # currency and would silently disagree with the column.
    mc_min = flat.get("market_cap_min")
    mc_max = flat.get("market_cap_max")
    if mc_min is not None or mc_max is not None:
        if market_cap_compatible:
            spec.add_range("market_cap_usd", min_value=mc_min, max_value=mc_max)
            if mc_min is not None:
                representable.append("market_cap_min")
            if mc_max is not None:
                representable.append("market_cap_max")
        else:
            if mc_min is not None:
                unrepresentable.append("market_cap_min")
            if mc_max is not None:
                unrepresentable.append("market_cap_max")

    # RS rating -> JSON rs_rating
    rs_min = flat.get("rs_rating_min")
    if rs_min is not None:
        spec.add_range("rs_rating", min_value=rs_min)
        representable.append("rs_rating_min")

    # Quarterly EPS growth -> JSON eps_growth_qq
    eps_min = flat.get("eps_growth_min")
    if eps_min is not None:
        spec.add_range("eps_growth_qq", min_value=eps_min)
        representable.append("eps_growth_min")

    # Quarterly sales growth -> JSON sales_growth_qq
    sales_min = flat.get("sales_growth_min")
    if sales_min is not None:
        spec.add_range("sales_growth_qq", min_value=sales_min)
        representable.append("sales_growth_min")

    # MA alignment -> JSON ma_alignment (boolean)
    ma_aligned = flat.get("ma_alignment")
    if ma_aligned is True:
        spec.add_boolean("ma_alignment", True)
        representable.append("ma_alignment")
    # ma_alignment == False is the no-op default; nothing to do.

    # 52-week high proximity -> JSON week_52_high_distance (% from high)
    near_high = flat.get("near_52w_high")
    if near_high is not None:
        spec.add_range("week_52_high_distance", max_value=near_high)
        representable.append("near_52w_high")

    # Sector inclusion -> JSON gics_sector
    sectors = flat.get("sectors")
    if sectors:
        spec.add_categorical(
            "gics_sector", tuple(sectors), mode=FilterMode.INCLUDE
        )
        representable.append("sectors")

    # Industry exclusion -> JSON gics_industry
    excluded = flat.get("exclude_industries")
    if excluded:
        spec.add_categorical(
            "gics_industry", tuple(excluded), mode=FilterMode.EXCLUDE
        )
        representable.append("exclude_industries")

    # Filters that have no feature-store representation today.
    if flat.get("debt_to_equity_max") is not None:
        unrepresentable.append("debt_to_equity_max")

    # Anything we don't recognise is unrepresentable so user intent is never
    # silently dropped.
    handled = {
        "price_min", "price_max", "volume_min",
        "market_cap_min", "market_cap_max",
        "rs_rating_min", "eps_growth_min", "sales_growth_min",
        "ma_alignment", "near_52w_high",
        "sectors", "exclude_industries",
        "debt_to_equity_max",
    }
    for key, val in flat.items():
        if key in handled or val is None:
            continue
        unrepresentable.append(key)

    # Score gate. Only applicable for single-screener custom scans, where
    # composite_score == custom_score by construction; we filter on the
    # JSON field directly to avoid depending on composite-method invariants.
    score_field: str | None = None
    normalized_screeners = sorted({
        str(s).strip().lower() for s in (screeners or []) if str(s).strip()
    })
    if normalized_screeners == ["custom"]:
        score_field = "custom_score"

    raw_min_score = (criteria or {}).get("min_score")
    if isinstance(raw_min_score, (int, float)) and not isinstance(
        raw_min_score, bool
    ):
        min_score_val: float | None = float(raw_min_score)
    elif raw_min_score is None:
        # CustomScanner default — applies whenever a score gate is in play.
        min_score_val = 70.0 if score_field is not None else None
    else:
        min_score_val = None

    return CompiledCustomCriteria(
        filter_spec=spec,
        score_field=score_field,
        min_score=min_score_val,
        unrepresentable_keys=tuple(unrepresentable),
        representable_keys=tuple(representable),
    )


__all__ = [
    "CompiledCustomCriteria",
    "compile_custom_criteria",
]
