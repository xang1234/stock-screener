"""Per-market alert evaluator (bead asia.10.2).

Stateless eval (no class state) that compares current gauges against
thresholds and writes new/updated alerts to ``market_telemetry_alerts``.

Hysteresis rules (enforced by partial unique index ``ux_telemetry_alerts_active``):
- Breach + no active alert → INSERT new alert (state=open)
- Breach + active alert at same severity → no-op (don't re-fire)
- Breach + active alert at lower severity → UPDATE severity (warning→critical)
- Recovery + active alert → UPDATE state=closed, set closed_at

Acknowledged alerts behave like open for hysteresis (we don't re-fire) but a
recovery still closes them.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from sqlalchemy.exc import IntegrityError

from ...models.market_telemetry_alert import (
    AlertSeverity,
    AlertState,
    MarketTelemetryAlert,
)
from ...tasks.market_queues import SHARED_SENTINEL, SUPPORTED_MARKETS
from .alert_thresholds import Levels, owner_for, thresholds_for
from .schema import MetricKey

logger = logging.getLogger(__name__)


# Gauge → (value extractor, comparison direction) for each metric category.
# direction="hi" means "bigger value crosses the threshold first" (e.g. lag);
# direction="lo" means "smaller value crosses first" (e.g. success rate).
#
# Freshness/age extractors read the precomputed ``lag_seconds`` / ``age_seconds``
# fields that ``market_summary`` already derives — keeps the "lag" definition in
# one place rather than recomputing from the raw epoch here.
def _freshness_value(payload: Dict[str, Any]) -> Optional[float]:
    return payload.get("lag_seconds")


def _benchmark_age_value(payload: Dict[str, Any]) -> Optional[float]:
    return payload.get("age_seconds")


def _completeness_value(payload: Dict[str, Any]) -> Optional[float]:
    """Return fraction of universe in the 0-25 completeness bucket."""
    total = payload.get("symbols_total") or 0
    if total <= 0:
        return None
    bucket_counts = payload.get("bucket_counts") or {}
    return float(bucket_counts.get("0-25", 0)) / float(total)


def _drift_value(payload: Dict[str, Any]) -> Optional[float]:
    """|delta|/prior_size, capped at 1.0 when prior was zero."""
    prior = payload.get("prior_size")
    delta = payload.get("delta")
    if prior is None or delta is None:
        return None
    if prior <= 0:
        return 1.0 if delta != 0 else 0.0
    return abs(float(delta)) / float(prior)


# Metric → (extractor, direction). EXTRACTION_SUCCESS handled separately —
# it's not a single gauge but a derived ratio over today's counters.
_METRIC_RULES: Dict[str, Tuple[Callable[[Dict[str, Any]], Optional[float]], str]] = {
    MetricKey.FRESHNESS_LAG: (_freshness_value, "hi"),
    MetricKey.BENCHMARK_AGE: (_benchmark_age_value, "hi"),
    MetricKey.COMPLETENESS_DISTRIBUTION: (_completeness_value, "hi"),
    MetricKey.UNIVERSE_DRIFT: (_drift_value, "hi"),
}


def _classify(value: float, levels: Levels, direction: str) -> Optional[str]:
    """Return ``critical``, ``warning``, or None based on direction + thresholds."""
    crit = levels.get("critical")
    warn = levels.get("warning")
    if direction == "hi":
        if crit is not None and value >= crit:
            return AlertSeverity.CRITICAL
        if warn is not None and value >= warn:
            return AlertSeverity.WARNING
    else:  # "lo"
        if crit is not None and value <= crit:
            return AlertSeverity.CRITICAL
        if warn is not None and value <= warn:
            return AlertSeverity.WARNING
    return None


def _format_alert(
    market: str, metric_key: str, severity: str, value: float, levels: Levels,
) -> Tuple[str, str]:
    """Build (title, description) for a new alert."""
    title = f"[{severity.upper()}] {metric_key} on {market}"
    threshold = levels.get(severity)
    description = (
        f"{metric_key} = {value:.2f} crossed the {severity} threshold "
        f"({threshold}) on market {market}."
    )
    return title, description


_SEVERITY_RANK = {AlertSeverity.WARNING: 1, AlertSeverity.CRITICAL: 2}


def _evaluate_one(
    db,
    *,
    market: str,
    metric_key: str,
    value: Optional[float],
    active: Optional[MarketTelemetryAlert] = None,
) -> Optional[MarketTelemetryAlert]:
    """Run hysteresis logic for one (market, metric) pair.

    ``active`` is the prefetched active alert for this (market, metric) pair
    (or None). Caller is expected to commit; this function only stages
    INSERT/UPDATE on the session.
    """
    levels = thresholds_for(metric_key, market)
    if not levels:
        return None

    if metric_key in _METRIC_RULES:
        direction = _METRIC_RULES[metric_key][1]
    elif metric_key == MetricKey.EXTRACTION_SUCCESS:
        direction = "lo"
    else:
        return None

    if value is None:
        # No gauge data — Redis outage, newly onboarded market, or malformed
        # payload. We can't distinguish "recovered" from "still breached", so
        # preserving the current alert state is the safe choice. Silently
        # closing alerts during a Redis outage would create a blind spot.
        return active  # None when no active alert; unchanged row when active

    severity = _classify(value, levels, direction)

    if severity is None:
        # Real recovery: gauge is present and below all thresholds.
        if active is None:
            return None
        active.state = AlertState.CLOSED
        active.closed_at = datetime.now(timezone.utc)
        logger.info(
            "telemetry alert closed: market=%s metric=%s id=%s",
            market, metric_key, active.id,
        )
        return active

    if active is None:
        title, description = _format_alert(market, metric_key, severity, value, levels)
        new_alert = MarketTelemetryAlert(
            market=market,
            metric_key=metric_key,
            severity=severity,
            state=AlertState.OPEN,
            owner=owner_for(market),
            title=title,
            description=description,
            metrics={"value": value, "thresholds": dict(levels)},
        )
        db.add(new_alert)
        logger.warning(
            "telemetry alert opened: market=%s metric=%s severity=%s value=%.4f",
            market, metric_key, severity, value,
        )
        return new_alert

    if _SEVERITY_RANK[severity] > _SEVERITY_RANK[active.severity]:
        old_severity = active.severity
        active.severity = severity
        active.title, active.description = _format_alert(
            market, metric_key, severity, value, levels,
        )
        active.metrics = {"value": value, "thresholds": dict(levels)}
        logger.warning(
            "telemetry alert upgraded: market=%s metric=%s %s→%s id=%s",
            market, metric_key, old_severity, severity, active.id,
        )
    return active


def evaluate_all(db, summaries: List[Dict[str, Any]]) -> List[MarketTelemetryAlert]:
    """Evaluate every (market, metric) from a list of ``market_summary`` outputs.

    Returns active alerts (open + acknowledged) after evaluation, ordered by
    open time descending. Single SELECT prefetches all active alerts; single
    commit at the end keeps the whole evaluation transactional.
    """
    # Prefetch all currently-active alerts in one query (replaces N=20 LIMIT 1
    # SELECTs). The partial unique index ux_telemetry_alerts_active guarantees
    # at most one active row per (market, metric_key), so the dict is unambiguous.
    active_by_key: Dict[Tuple[str, str], MarketTelemetryAlert] = {
        (a.market, a.metric_key): a for a in list_active_alerts(db)
    }

    for summary in summaries:
        market = summary.get("market") or SHARED_SENTINEL

        for metric_key, (extractor, _direction) in _METRIC_RULES.items():
            payload = summary.get(metric_key)
            value = extractor(payload) if isinstance(payload, dict) else None
            _evaluate_one(
                db,
                market=market,
                metric_key=metric_key,
                value=value,
                active=active_by_key.get((market, metric_key)),
            )

        if market == SHARED_SENTINEL:
            extraction = summary.get("extraction_today") or {}
            ratio = _extraction_success_ratio(extraction)
            _evaluate_one(
                db,
                market=SHARED_SENTINEL,
                metric_key=MetricKey.EXTRACTION_SUCCESS,
                value=ratio,
                active=active_by_key.get((SHARED_SENTINEL, MetricKey.EXTRACTION_SUCCESS)),
            )

    # Single commit keeps the whole eval pass transactional. If two workers
    # evaluate concurrently, the partial unique index ux_telemetry_alerts_active
    # will raise IntegrityError on the duplicate INSERT.
    #
    # Trade-off: the rollback discards the WHOLE batch (any unrelated UPDATEs
    # to other (market, metric) pairs included), not just the racing INSERT.
    # The next /alerts poll (~30s later) re-evaluates and re-applies them.
    # Acceptable because the race window is small (commit takes ms) and at our
    # scale concurrent eval calls are rare; if this becomes a real source of
    # flapping, switch to per-pair savepoints (db.begin_nested()).
    try:
        db.commit()
    except IntegrityError as exc:
        logger.info("telemetry alert eval lost race to concurrent worker (%s); rolling back", exc)
        db.rollback()

    return list_active_alerts(db)


def _extraction_success_ratio(extraction_today: Dict[str, Any]) -> Optional[float]:
    """Aggregate per-language counters into a single success rate for the day."""
    by_lang = extraction_today.get("by_language") or {}
    total = sum((bucket.get("total") or 0) for bucket in by_lang.values())
    success = sum((bucket.get("success") or 0) for bucket in by_lang.values())
    if total <= 0:
        return None
    return float(success) / float(total)


def list_active_alerts(db) -> List[MarketTelemetryAlert]:
    """Return all open + acknowledged alerts, newest first."""
    return (
        db.query(MarketTelemetryAlert)
        .filter(
            MarketTelemetryAlert.state.in_([AlertState.OPEN, AlertState.ACKNOWLEDGED]),
        )
        .order_by(MarketTelemetryAlert.opened_at.desc())
        .all()
    )


def acknowledge_alert(db, alert_id: int, acknowledged_by: str) -> Optional[MarketTelemetryAlert]:
    """Mark an open alert as acknowledged. Closed/already-acked alerts are unchanged."""
    alert = db.query(MarketTelemetryAlert).filter(MarketTelemetryAlert.id == alert_id).first()
    if alert is None:
        return None
    if alert.state != AlertState.OPEN:
        return alert  # idempotent
    alert.state = AlertState.ACKNOWLEDGED
    alert.acknowledged_at = datetime.now(timezone.utc)
    alert.acknowledged_by = acknowledged_by
    db.commit()
    return alert
