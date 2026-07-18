"""Pure normalization rules for the bootstrap cache-coverage gate."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from app.domain.markets.price_coverage import (
    PriceCoveragePolicy,
    price_coverage_policy_for_market,
)


MISSING_SYMBOL_PREVIEW_LIMIT = 20


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _report_meets_policy(
    payload: Mapping[str, Any],
    policy: PriceCoveragePolicy,
) -> bool:
    price_ratio = _optional_float(payload.get("price_coverage_ratio"))
    fundamentals_ratio = _optional_float(payload.get("fundamentals_coverage_ratio"))
    if price_ratio is None or fundamentals_ratio is None:
        return False
    return (
        price_ratio >= policy.price_min_coverage
        and fundamentals_ratio >= policy.fundamentals_min_coverage
    )


def normalize_bootstrap_gate_report(
    *,
    market: str | None,
    report: Mapping[str, Any] | None,
    unsupported_symbols: Sequence[str],
) -> dict[str, Any]:
    payload = dict(report or {})
    policy = price_coverage_policy_for_market(market)
    eligible = _report_meets_policy(payload, policy)
    payload.update(
        {
            "eligible": eligible,
            "threshold": policy.price_min_coverage,
            "price_threshold": policy.price_min_coverage,
            "fundamentals_threshold": policy.fundamentals_min_coverage,
            "mode": "cache_only" if eligible else "waiting_for_cache_coverage",
            "unsupported_skipped_count": len(unsupported_symbols) if eligible else 0,
            "unsupported_symbols_preview": (
                list(unsupported_symbols[:MISSING_SYMBOL_PREVIEW_LIMIT])
                if eligible
                else []
            ),
        }
    )
    return payload


__all__ = ["MISSING_SYMBOL_PREVIEW_LIMIT", "normalize_bootstrap_gate_report"]
