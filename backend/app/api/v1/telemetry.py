"""REST API for per-market telemetry (beads asia.10.1 + 10.2).

Endpoints:
- ``GET /v1/telemetry/markets`` — summary across all supported markets
- ``GET /v1/telemetry/markets/{market}`` — detailed gauges + today's counters
- ``GET /v1/telemetry/markets/{market}/{metric_key}`` — recent raw events
- ``GET /v1/telemetry/alerts`` — evaluate thresholds and return active alerts (10.2)
- ``POST /v1/telemetry/alerts/{id}/acknowledge`` — acknowledge an open alert (10.2)
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from ...database import get_db
from ...services.telemetry import get_telemetry
from ...services.telemetry.alert_evaluator import (
    acknowledge_alert,
    evaluate_all,
    list_active_alerts,
)
from ...services.telemetry.schema import MetricKey
from ...tasks.market_queues import SHARED_SENTINEL, SUPPORTED_MARKETS, normalize_market

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
    try:
        normalized = normalize_market(market)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Unknown market: {market}")
    if normalized not in SUPPORTED_MARKETS:
        raise HTTPException(status_code=404, detail=f"Unknown market: {market}")
    return get_telemetry().market_summary(normalized)


@router.get("/markets/{market}/{metric_key}")
async def market_metric_history(
    market: str,
    metric_key: str,
    days: int = Query(default=15, ge=1, le=15),
    limit: int = Query(default=200, ge=1, le=1000),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Return raw event rows for one metric in one market over the last N days.

    Capped at 15 days (the retention window) and 1000 rows. Used by dashboards
    (10.2) for per-metric trend graphs.
    """
    try:
        normalized = normalize_market(market)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Unknown market: {market}")
    if normalized not in SUPPORTED_MARKETS:
        raise HTTPException(status_code=404, detail=f"Unknown market: {market}")
    if metric_key not in _ALLOWED_METRIC_KEYS:
        raise HTTPException(status_code=400, detail=f"Unknown metric_key: {metric_key}")

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    from ...models.market_telemetry import MarketTelemetryEvent

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


# ---------------------------------------------------------------------------
# Alerts (bead asia.10.2)
# ---------------------------------------------------------------------------
def _serialize_alert(alert) -> Dict[str, Any]:
    """Render an alert row for API consumers (frontend table)."""
    return {
        "id": alert.id,
        "market": alert.market,
        "metric_key": alert.metric_key,
        "severity": alert.severity,
        "state": alert.state,
        "owner": alert.owner,
        "title": alert.title,
        "description": alert.description,
        "metrics": alert.metrics,
        "opened_at": alert.opened_at.isoformat() if alert.opened_at else None,
        "acknowledged_at": alert.acknowledged_at.isoformat() if alert.acknowledged_at else None,
        "acknowledged_by": alert.acknowledged_by,
        "closed_at": alert.closed_at.isoformat() if alert.closed_at else None,
    }


@router.get("/alerts")
async def get_alerts(
    evaluate: bool = Query(default=True),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Return active (open + acknowledged) telemetry alerts + the per-market
    summaries used to evaluate them.

    The summaries are returned alongside the alerts so a single round-trip
    feeds both the dashboard widgets and the alert table — the eval already
    builds them, and the frontend would otherwise poll ``/markets`` separately
    for the same data.

    By default, evaluates current thresholds before returning so callers see
    fresh state. Pass ``?evaluate=false`` to skip eval and only read existing
    alert rows (useful for cheap UI refreshes that piggyback the next eval).
    """
    telemetry = get_telemetry()
    # Always build summaries — they're returned to the client whether or not
    # we re-evaluate. The SHARED summary is appended so the extraction-success
    # rule (SHARED-scoped) gets evaluated when ``evaluate=true``.
    summaries = [telemetry.market_summary(m) for m in telemetry.list_markets()]
    shared_summary = telemetry.market_summary(SHARED_SENTINEL)

    if evaluate:
        alerts = evaluate_all(db, [*summaries, shared_summary])
    else:
        alerts = list_active_alerts(db)

    return {
        "alerts": [_serialize_alert(a) for a in alerts],
        "summaries": summaries,
    }


@router.post("/alerts/{alert_id}/acknowledge")
async def post_acknowledge(
    alert_id: int,
    body: Dict[str, Any] = Body(default_factory=dict),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Acknowledge an open alert.

    Body: ``{"acknowledged_by": "username"}``. Acking an already-acknowledged
    alert is a no-op (returns the unchanged row at 200). Acking a closed alert
    is a lifecycle error and returns 409.
    """
    from ...models.market_telemetry_alert import AlertState

    acknowledged_by = (body.get("acknowledged_by") or "").strip() or "anonymous"
    alert = acknowledge_alert(db, alert_id, acknowledged_by)
    if alert is None:
        raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")
    if alert.state == AlertState.CLOSED:
        raise HTTPException(status_code=409, detail="Cannot acknowledge a closed alert")
    return _serialize_alert(alert)
