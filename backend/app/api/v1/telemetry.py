"""REST API for per-market telemetry (bead asia.10.1).

Endpoints:
- ``GET /v1/telemetry/markets`` — summary across all supported markets
- ``GET /v1/telemetry/markets/{market}`` — detailed gauges + today's counters
- ``GET /v1/telemetry/markets/{market}/{metric_key}`` — recent raw events
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query

from ...services.telemetry import get_telemetry
from ...services.telemetry.schema import MetricKey
from ...tasks.market_queues import SUPPORTED_MARKETS, normalize_market

router = APIRouter(prefix="/telemetry")


_ALLOWED_METRIC_KEYS = {
    MetricKey.FRESHNESS_LAG,
    MetricKey.UNIVERSE_DRIFT,
    MetricKey.BENCHMARK_AGE,
    MetricKey.COMPLETENESS_DISTRIBUTION,
    # EXTRACTION_SUCCESS deliberately omitted: it's Redis-only (counter
    # increments per LLM call), so no PG rows exist for the history endpoint
    # to return. Live counters surface via /v1/telemetry/markets[/{m}].
}


@router.get("/markets")
async def list_market_summaries() -> Dict[str, Any]:
    """Return latest gauges + today's counters for every supported market."""
    telemetry = get_telemetry()
    summaries = [telemetry.market_summary(m) for m in telemetry.list_markets()]
    return {"markets": summaries}


@router.get("/markets/{market}")
async def market_detail(market: str) -> Dict[str, Any]:
    """Return the full telemetry summary for a single market."""
    normalized = normalize_market(market)
    if normalized not in SUPPORTED_MARKETS:
        raise HTTPException(status_code=404, detail=f"Unknown market: {market}")
    return get_telemetry().market_summary(normalized)


@router.get("/markets/{market}/{metric_key}")
async def market_metric_history(
    market: str,
    metric_key: str,
    days: int = Query(default=15, ge=1, le=15),
    limit: int = Query(default=200, ge=1, le=1000),
) -> Dict[str, Any]:
    """Return raw event rows for one metric in one market over the last N days.

    Capped at 15 days (the retention window) and 1000 rows. Used by dashboards
    (10.2) for per-metric trend graphs.
    """
    normalized = normalize_market(market)
    if normalized not in SUPPORTED_MARKETS:
        raise HTTPException(status_code=404, detail=f"Unknown market: {market}")
    if metric_key not in _ALLOWED_METRIC_KEYS:
        raise HTTPException(status_code=400, detail=f"Unknown metric_key: {metric_key}")

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    # Inline DB read — keeps the API endpoint thin and self-contained.
    from ...database import SessionLocal
    from ...models.market_telemetry import MarketTelemetryEvent

    db = SessionLocal()
    try:
        rows = (
            db.query(MarketTelemetryEvent)
            .filter(
                MarketTelemetryEvent.market == normalized,
                MarketTelemetryEvent.metric_key == metric_key,
                MarketTelemetryEvent.recorded_at >= cutoff,
            )
            .order_by(MarketTelemetryEvent.recorded_at.desc())
            .limit(limit)
            .all()
        )
    finally:
        db.close()

    return {
        "market": normalized,
        "metric_key": metric_key,
        "events": [
            {
                "recorded_at": r.recorded_at.isoformat() if r.recorded_at else None,
                "schema_version": r.schema_version,
                "payload": r.payload,
            }
            for r in rows
        ],
    }
