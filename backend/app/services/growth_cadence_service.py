"""Cadence-aware growth extraction for mixed reporting markets."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import pandas as pd

from . import provider_routing_policy as routing_policy
from .field_capability_registry import (
    SUPPORT_STATE_AVAILABLE,
    SUPPORT_STATE_COMPUTED,
    SUPPORT_STATE_UNSUPPORTED,
    field_capability_registry,
)

CADENCE_QUARTERLY = "quarterly"
CADENCE_SEMIANNUAL = "semiannual"
CADENCE_ANNUAL = "annual"
CADENCE_UNKNOWN = "unknown"
CADENCE_INSUFFICIENT = "insufficient_history"

BASIS_QOQ = "quarterly_qoq"
BASIS_COMPARABLE_YOY = "comparable_period_yoy"
BASIS_UNAVAILABLE = "unavailable"

_MARKETS_COMPARABLE_PERIOD_PRIMARY = frozenset(
    {routing_policy.MARKET_HK, routing_policy.MARKET_JP}
)


def _normalize_market(market: str | None) -> str:
    return routing_policy.normalize_market(market)


def _to_timestamp(value: Any) -> Optional[pd.Timestamp]:
    try:
        timestamp = pd.Timestamp(value)
        if pd.isna(timestamp):
            return None
        return timestamp
    except Exception:
        return None


def _compute_growth(recent: Any, baseline: Any, *, min_abs_baseline: float = 0.0) -> Optional[float]:
    if pd.isna(recent) or pd.isna(baseline) or baseline == 0:
        return None
    if abs(float(baseline)) <= min_abs_baseline:
        return None
    growth = ((float(recent) - float(baseline)) / abs(float(baseline))) * 100.0
    return round(float(growth), 2)


def _infer_cadence(reference_gap_days: Optional[int]) -> str:
    if reference_gap_days is None:
        return CADENCE_UNKNOWN
    if 70 <= reference_gap_days <= 130:
        return CADENCE_QUARTERLY
    if 131 <= reference_gap_days <= 230:
        return CADENCE_SEMIANNUAL
    if reference_gap_days >= 231:
        return CADENCE_ANNUAL
    return CADENCE_UNKNOWN


def _find_metric_rows(quarterly_income: pd.DataFrame) -> Tuple[Any | None, Any | None]:
    eps_row = None
    revenue_row = None
    for idx in quarterly_income.index:
        idx_str = str(idx).lower().replace(" ", "")
        if eps_row is None and ("dilutedeps" in idx_str or "basiceps" in idx_str):
            eps_row = idx
        if revenue_row is None and ("totalrevenue" in idx_str or ("revenue" in idx_str and "total" in idx_str)):
            revenue_row = idx
        if eps_row is not None and revenue_row is not None:
            break
    if revenue_row is None:
        for idx in quarterly_income.index:
            idx_str = str(idx).lower()
            if "revenue" in idx_str:
                revenue_row = idx
                break
    return eps_row, revenue_row


def _find_comparable_period_column(
    columns: list[Any],
    recent_ts: Optional[pd.Timestamp],
) -> tuple[Any | None, Optional[int]]:
    if recent_ts is None:
        return None, None
    best_col = None
    best_gap = None
    best_distance = None
    for col in columns[1:]:
        col_ts = _to_timestamp(col)
        if col_ts is None:
            continue
        gap = abs(int((recent_ts - col_ts).days))
        if gap < 280 or gap > 460:
            continue
        distance = abs(gap - 365)
        if best_distance is None or distance < best_distance:
            best_col = col
            best_gap = gap
            best_distance = distance
    return best_col, best_gap


def compute_cadence_aware_growth(
    quarterly_income: pd.DataFrame | None,
    *,
    market: str | None = None,
) -> Dict[str, Any]:
    """Return growth metrics with cadence-aware semantics.

    For HK/JP with non-quarterly cadence, ``*_growth_qq`` is set from
    comparable-period YoY (same period prior year) to avoid fabricated
    quarter-over-quarter comparisons.
    """
    result: Dict[str, Any] = {
        "eps_growth_qq": None,
        "sales_growth_qq": None,
        "eps_growth_yy": None,
        "sales_growth_yy": None,
        "recent_quarter_date": None,
        "previous_quarter_date": None,
        "growth_reporting_cadence": CADENCE_INSUFFICIENT,
        "growth_metric_basis": BASIS_UNAVAILABLE,
        "growth_comparable_period_date": None,
        "growth_reference_gap_days": None,
    }

    if quarterly_income is None or quarterly_income.shape[1] < 2:
        return result

    columns = list(quarterly_income.columns)
    recent_col = columns[0]
    previous_col = columns[1]
    result["recent_quarter_date"] = str(recent_col)
    result["previous_quarter_date"] = str(previous_col)

    recent_ts = _to_timestamp(recent_col)
    previous_ts = _to_timestamp(previous_col)
    reference_gap_days = None
    if recent_ts is not None and previous_ts is not None:
        reference_gap_days = abs(int((recent_ts - previous_ts).days))
    result["growth_reference_gap_days"] = reference_gap_days

    cadence = _infer_cadence(reference_gap_days)
    result["growth_reporting_cadence"] = cadence

    eps_row, revenue_row = _find_metric_rows(quarterly_income)

    qoq_eps = None
    qoq_sales = None
    if eps_row is not None:
        qoq_eps = _compute_growth(
            quarterly_income.loc[eps_row, recent_col],
            quarterly_income.loc[eps_row, previous_col],
            min_abs_baseline=0.05,
        )
    if revenue_row is not None:
        qoq_sales = _compute_growth(
            quarterly_income.loc[revenue_row, recent_col],
            quarterly_income.loc[revenue_row, previous_col],
        )

    comparable_col, _ = _find_comparable_period_column(columns, recent_ts)
    comparable_eps = None
    comparable_sales = None
    if comparable_col is not None:
        result["growth_comparable_period_date"] = str(comparable_col)
        if eps_row is not None:
            comparable_eps = _compute_growth(
                quarterly_income.loc[eps_row, recent_col],
                quarterly_income.loc[eps_row, comparable_col],
                min_abs_baseline=0.05,
            )
        if revenue_row is not None:
            comparable_sales = _compute_growth(
                quarterly_income.loc[revenue_row, recent_col],
                quarterly_income.loc[revenue_row, comparable_col],
            )

    result["eps_growth_yy"] = comparable_eps
    result["sales_growth_yy"] = comparable_sales

    resolved_market = _normalize_market(market)
    if cadence == CADENCE_QUARTERLY:
        result["growth_metric_basis"] = BASIS_QOQ
        result["eps_growth_qq"] = qoq_eps
        result["sales_growth_qq"] = qoq_sales
        return result

    if resolved_market in _MARKETS_COMPARABLE_PERIOD_PRIMARY:
        result["growth_metric_basis"] = BASIS_COMPARABLE_YOY
        result["eps_growth_qq"] = comparable_eps
        result["sales_growth_qq"] = comparable_sales
        return result

    # For markets where comparable-period YoY is not the primary growth
    # contract, keep QoQ fields unpopulated instead of implicitly remapping
    # them. Comparable-period values remain available via *_yy fields.
    result["growth_metric_basis"] = BASIS_UNAVAILABLE
    return result


REASON_INSUFFICIENT_HISTORY = "insufficient_history"
REASON_COMPARABLE_YOY_FALLBACK = "comparable_period_yoy_fallback"

_GROWTH_FIELDS_AFFECTED_BY_CADENCE: Tuple[str, ...] = (
    "eps_growth_qq",
    "sales_growth_qq",
)


def derive_growth_availability(
    growth_metric_basis: Optional[str],
    growth_reporting_cadence: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """Emit ``field_availability`` entries for growth fields based on cadence state.

    Mirrors :func:`field_capability_registry.derive_ownership_sentiment_availability`
    but for growth metrics, so a single merged ``field_availability`` dict on the
    scan-result response can carry both concerns.

    Rules:
        - ``BASIS_UNAVAILABLE``: QoQ growth fields are unavailable; reason is
          ``insufficient_history`` (the only way we land here is either a
          too-short statement history or a market whose cadence isn't QoQ
          and isn't in the comparable-period primary set).
        - ``BASIS_COMPARABLE_YOY``: the QoQ fields carry a comparable-period
          YoY fallback, not genuine QoQ. Flag ``computed`` so clients can
          surface the synthesis (e.g. show a "computed" chip).
        - ``BASIS_QOQ`` / ``None``: empty dict — normal case needs no entry.

    Only emits entries for non-available states to keep the response dict
    compact; callers merging with the ownership helper should filter that
    output similarly so a "no entries" dict means "nothing to surface".
    """
    if growth_metric_basis == BASIS_UNAVAILABLE:
        return {
            field: {
                "status": SUPPORT_STATE_UNSUPPORTED,
                "reason_code": REASON_INSUFFICIENT_HISTORY,
                "support_state": SUPPORT_STATE_UNSUPPORTED,
                "cadence": growth_reporting_cadence,
            }
            for field in _GROWTH_FIELDS_AFFECTED_BY_CADENCE
        }
    if growth_metric_basis == BASIS_COMPARABLE_YOY:
        return {
            field: {
                "status": SUPPORT_STATE_COMPUTED,
                "reason_code": REASON_COMPARABLE_YOY_FALLBACK,
                "support_state": SUPPORT_STATE_COMPUTED,
                "cadence": growth_reporting_cadence,
            }
            for field in _GROWTH_FIELDS_AFFECTED_BY_CADENCE
        }
    return {}


def build_row_field_availability(
    *,
    market: Optional[str],
    institutional_ownership: Any,
    insider_ownership: Any,
    short_interest: Any,
    growth_metric_basis: Optional[str],
    growth_reporting_cadence: Optional[str],
) -> Optional[Dict[str, Dict[str, Any]]]:
    """Shared builder for scan-result-row ``field_availability``.

    Merges ownership/sentiment entries from
    :func:`field_capability_registry.derive_ownership_sentiment_availability`
    with growth-cadence entries from :func:`derive_growth_availability`,
    filters to only non-available entries (so US-quarterly rows return
    ``None`` rather than a dict full of ``status=available`` noise), and
    returns the merged dict or ``None`` when nothing needs surfacing.

    Callers: ``_unpack_joined_row`` in the legacy scan_result repo and
    ``_unpack_feature_joined_row`` in the feature-store repo. Keeping the
    merge here rather than duplicating it at each call site avoids the
    "non-available filter" invariant silently diverging between paths.
    """
    ownership = field_capability_registry.derive_ownership_sentiment_availability(
        {
            "institutional_ownership": institutional_ownership,
            "insider_ownership": insider_ownership,
            "short_interest": short_interest,
        },
        market,
    )
    growth = derive_growth_availability(growth_metric_basis, growth_reporting_cadence)
    merged = {
        field: entry
        for field, entry in {**ownership, **growth}.items()
        if entry.get("status") != SUPPORT_STATE_AVAILABLE
    }
    return merged or None
