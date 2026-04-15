"""Versioned payload builders for per-market telemetry (bead asia.10.1).

Every metric category has a single payload-builder function that returns a
JSON-serializable dict tagged with ``schema_version``. Bump the version when
adding required fields, then teach readers to handle both old and new shapes.
The dict is what lands in ``MarketTelemetryEvent.payload``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

# Bump when payload shape changes in a non-additive way. Additive (new
# optional fields) does NOT require a bump — readers must tolerate missing
# optional fields. Non-additive changes (renaming, removing, type change)
# require both a bump and a reader-side handler for the prior version(s).
SCHEMA_VERSION: int = 1


# Metric-key constants — single source of truth for the `metric_key` column.
class MetricKey:
    FRESHNESS_LAG = "freshness_lag"
    UNIVERSE_DRIFT = "universe_drift"
    BENCHMARK_AGE = "benchmark_age"
    EXTRACTION_SUCCESS = "extraction_success"
    COMPLETENESS_DISTRIBUTION = "completeness_distribution"
    FIELD_COVERAGE = "field_coverage"


def freshness_lag_payload(
    *, last_refresh_at_epoch: float, source: str, symbols_refreshed: int
) -> Dict[str, Any]:
    """Recorded after a successful price/fundamentals refresh.

    Lag at read time = ``now - last_refresh_at_epoch``. Storing the absolute
    timestamp (not the lag itself) keeps the metric monotonic and lets us
    answer "lag right now" without re-emitting.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "last_refresh_at_epoch": float(last_refresh_at_epoch),
        "source": source,                        # "prices" | "fundamentals"
        "symbols_refreshed": int(symbols_refreshed),
    }


def universe_drift_payload(
    *, current_size: int, prior_size: Optional[int]
) -> Dict[str, Any]:
    """Recorded after a universe sync. ``prior_size`` is None on first ever sync."""
    delta = (current_size - prior_size) if prior_size is not None else 0
    return {
        "schema_version": SCHEMA_VERSION,
        "current_size": int(current_size),
        "prior_size": int(prior_size) if prior_size is not None else None,
        "delta": int(delta),
    }


def benchmark_age_payload(
    *, last_warmed_at_epoch: float, benchmark_symbol: str
) -> Dict[str, Any]:
    """Recorded after the SPY (or per-market benchmark) cache warm completes."""
    return {
        "schema_version": SCHEMA_VERSION,
        "last_warmed_at_epoch": float(last_warmed_at_epoch),
        "benchmark_symbol": benchmark_symbol,
    }


def extraction_success_payload(
    *, language: str, success: bool, latency_ms: Optional[int] = None,
    provider: Optional[str] = None,
) -> Dict[str, Any]:
    """Recorded after each theme-extraction LLM call."""
    return {
        "schema_version": SCHEMA_VERSION,
        "language": language,
        "success": bool(success),
        "latency_ms": int(latency_ms) if latency_ms is not None else None,
        "provider": provider,
    }


# Completeness buckets — fixed at v1 so SQL aggregations and readers stay stable.
COMPLETENESS_BUCKETS = ("0-25", "25-50", "50-75", "75-90", "90-100")


def completeness_bucket_for(score: float) -> str:
    """Map a 0..100 completeness score to one of the v1 buckets."""
    if score < 25:
        return "0-25"
    if score < 50:
        return "25-50"
    if score < 75:
        return "50-75"
    if score < 90:
        return "75-90"
    return "90-100"


def low_completeness_ratio(payload: Dict[str, Any]) -> Optional[float]:
    """Return fraction of a completeness payload's universe in the 0-25 bucket.

    Shared reader helper for the alert evaluator (live breach check) and the
    weekly audit (historical regression signal) — both interpret the same
    payload shape the same way, so the extraction keeps their semantics in
    lockstep when the payload schema evolves.
    """
    total = payload.get("symbols_total") or 0
    if total <= 0:
        return None
    buckets = payload.get("bucket_counts") or {}
    return float(buckets.get("0-25", 0)) / float(total)


def field_coverage_payload(
    *,
    total_fields: int,
    support_state_counts: Dict[str, int],
    unsupported_field_names: tuple,
    computed_field_names: tuple,
    cadence_counts: Dict[str, int],
    cadence_eligible_universe: int,
) -> Dict[str, Any]:
    """Recorded after a per-market field-coverage snapshot (bead asia.10.5).

    Static part (from ``field_capability_registry``): how many screening fields
    are supported/computed/partial/unsupported for this market, plus the
    concrete field names in each degraded bucket so runbooks can diagnose
    without a registry lookup.

    Dynamic part (from ``stock_fundamentals.growth_reporting_cadence``):
    per-symbol cadence distribution — the ``comparable_period_yoy`` count
    over a non-zero ``cadence_eligible_universe`` is the cadence-fallback
    signal operators watch.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "total_fields": int(total_fields),
        "support_state_counts": {str(k): int(v) for k, v in support_state_counts.items()},
        "unsupported_field_names": list(unsupported_field_names),
        "computed_field_names": list(computed_field_names),
        "cadence_counts": {str(k): int(v) for k, v in cadence_counts.items()},
        "cadence_eligible_universe": int(cadence_eligible_universe),
    }


# The fallback-basis key name. String literal rather than an import from
# growth_cadence_service to keep schema.py free of the growth-pipeline
# dependency graph (schema is a foundation module; growth_cadence_service
# depends on the capability registry which depends on routing policy).
_BASIS_COMPARABLE_YOY = "comparable_period_yoy"


def cadence_fallback_ratio(payload: Dict[str, Any]) -> Optional[float]:
    """Return fraction of universe on comparable-period YoY fallback.

    Shared reader so the alert evaluator and weekly audit agree on the
    definition. ``comparable_period_yoy`` is the cadence-fallback marker
    (see ``growth_cadence_service.BASIS_COMPARABLE_YOY``).
    """
    eligible = payload.get("cadence_eligible_universe") or 0
    if eligible <= 0:
        return None
    fallbacks = (payload.get("cadence_counts") or {}).get(_BASIS_COMPARABLE_YOY, 0)
    return float(fallbacks) / float(eligible)


def unsupported_field_ratio(payload: Dict[str, Any]) -> Optional[float]:
    """Return fraction of screening fields in the ``unsupported`` state."""
    total = payload.get("total_fields") or 0
    if total <= 0:
        return None
    unsupported = (payload.get("support_state_counts") or {}).get("unsupported", 0)
    return float(unsupported) / float(total)


def completeness_distribution_payload(
    *, bucket_counts: Dict[str, int], symbols_total: int
) -> Dict[str, Any]:
    """Recorded after fundamentals completeness scoring for a market.

    ``bucket_counts`` keys must match :data:`COMPLETENESS_BUCKETS`. Missing
    buckets are stored as zero.
    """
    normalized = {b: int(bucket_counts.get(b, 0)) for b in COMPLETENESS_BUCKETS}
    return {
        "schema_version": SCHEMA_VERSION,
        "bucket_counts": normalized,
        "symbols_total": int(symbols_total),
    }
