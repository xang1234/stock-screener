"""Weekly governance audit of per-market telemetry (bead asia.10.4).

Reads the last 7 days of ``market_telemetry_events`` + ``market_telemetry_alerts``
and produces a tamper-evident governance report (JSON + Markdown + SHA-256) for
historical traceability.

Design:
- **Pure aggregation**: no I/O other than the SQL reads passed in via the
  session. The Celery task wrapper handles filesystem + scheduling.
- **Self-contained snapshot per run**: the raw event log has 15d retention,
  so each weekly report must stand alone — no reliance on prior reports.
- **Signed artifact = SHA-256 over canonical JSON**: cryptographic signing with
  keypairs would require key-management infrastructure we don't have. A content
  hash gives tamper-evidence, which is what "historical traceability" needs.
- **Report is point-in-time**: all thresholds, owners, and metric definitions
  are captured at generation time so a future threshold change doesn't silently
  re-interpret old reports.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy.orm import Session

from ...models.market_telemetry import MarketTelemetryEvent
from ...models.market_telemetry_alert import MarketTelemetryAlert, AlertState
from ...tasks.market_queues import SHARED_SENTINEL, SUPPORTED_MARKETS
from .alert_thresholds import OWNERS, THRESHOLDS
from .schema import MetricKey, SCHEMA_VERSION, low_completeness_ratio

# Window the weekly report covers. Fits comfortably inside the 15d retention
# window of market_telemetry_events so week-on-week gaps don't silently lose data.
AUDIT_WINDOW_DAYS = 7

# Report format version. Bump on non-additive changes (field rename / removal /
# type change). Reader tools should assert this matches a supported version.
REPORT_SCHEMA_VERSION = 1


@dataclass
class MarketMetricSummary:
    """Per-(market, metric) rollup over the audit window."""
    market: str
    metric_key: str
    event_count: int
    first_recorded_at: Optional[str]
    last_recorded_at: Optional[str]
    # Metric-specific rollup; shape varies by metric_key and is documented in
    # the render code. Keeping it open-ended avoids one dataclass per metric.
    rollup: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AlertRollup:
    """Per-market alert lifecycle counts over the window."""
    market: str
    opened: int
    closed: int
    still_active: int
    by_severity: Dict[str, int]


@dataclass
class GovernanceReport:
    report_schema_version: int
    payload_schema_version: int
    generated_at: str
    window_start: str
    window_end: str
    markets: List[str]
    metrics: List[MarketMetricSummary]
    alerts: List[AlertRollup]
    thresholds_snapshot: Dict[str, Dict[str, Dict[str, float]]]
    owners_snapshot: Dict[str, str]
    content_hash: Optional[str] = None


def run_weekly_audit(
    db: Session,
    *,
    now: Optional[datetime] = None,
) -> GovernanceReport:
    """Produce a ``GovernanceReport`` for the 7 days ending at ``now``.

    The function reads from the passed session and does not commit. ``now``
    is injectable so tests can pin the window deterministically.
    """
    generated_at = now or datetime.now(timezone.utc)
    window_start = generated_at - timedelta(days=AUDIT_WINDOW_DAYS)

    markets = [*SUPPORTED_MARKETS, SHARED_SENTINEL]

    metrics: List[MarketMetricSummary] = []
    for market in markets:
        for metric_key in _METRIC_KEYS:
            metrics.append(
                _summarize_metric(db, market, metric_key, window_start, generated_at)
            )

    alerts = [_summarize_alerts(db, m, window_start, generated_at) for m in markets]

    report = GovernanceReport(
        report_schema_version=REPORT_SCHEMA_VERSION,
        payload_schema_version=SCHEMA_VERSION,
        generated_at=generated_at.isoformat(),
        window_start=window_start.isoformat(),
        window_end=generated_at.isoformat(),
        markets=list(markets),
        metrics=metrics,
        alerts=alerts,
        thresholds_snapshot=_serialize_thresholds(),
        owners_snapshot=dict(OWNERS),
    )
    report.content_hash = _content_hash(report)
    return report


_METRIC_KEYS: Tuple[str, ...] = (
    MetricKey.FRESHNESS_LAG,
    MetricKey.UNIVERSE_DRIFT,
    MetricKey.BENCHMARK_AGE,
    MetricKey.EXTRACTION_SUCCESS,
    MetricKey.COMPLETENESS_DISTRIBUTION,
)


def _summarize_metric(
    db: Session,
    market: str,
    metric_key: str,
    window_start: datetime,
    window_end: datetime,
) -> MarketMetricSummary:
    rows = (
        db.query(MarketTelemetryEvent)
        .filter(
            MarketTelemetryEvent.market == market,
            MarketTelemetryEvent.metric_key == metric_key,
            MarketTelemetryEvent.recorded_at >= window_start,
            MarketTelemetryEvent.recorded_at <= window_end,
        )
        .order_by(MarketTelemetryEvent.recorded_at.asc())
        .all()
    )

    if not rows:
        return MarketMetricSummary(
            market=market,
            metric_key=metric_key,
            event_count=0,
            first_recorded_at=None,
            last_recorded_at=None,
            rollup={},
        )

    rollup = _rollup_for_metric(metric_key, rows, window_end)

    return MarketMetricSummary(
        market=market,
        metric_key=metric_key,
        event_count=len(rows),
        first_recorded_at=rows[0].recorded_at.isoformat() if rows[0].recorded_at else None,
        last_recorded_at=rows[-1].recorded_at.isoformat() if rows[-1].recorded_at else None,
        rollup=rollup,
    )


def _rollup_for_metric(
    metric_key: str, rows: List[MarketTelemetryEvent], window_end: datetime,
) -> Dict[str, Any]:
    """Compute a metric-specific rollup. Each branch is documented inline."""
    if metric_key == MetricKey.FRESHNESS_LAG:
        # Two signals worth tracking in the governance window:
        #   1. Freshness at report time: window_end - latest refresh epoch.
        #      Tells you "how stale is the data right now?"
        #   2. Max gap between consecutive refreshes. A large gap (e.g. Celery
        #      stalled for 6h) is invisible from (1) if the pipeline recovered
        #      by week's end — but it's the real governance red flag.
        latest_epoch = (rows[-1].payload or {}).get("last_refresh_at_epoch")
        latest_lag = (
            max(0.0, window_end.timestamp() - float(latest_epoch))
            if latest_epoch else None
        )
        epochs = sorted(
            float(ts) for ts in (
                (r.payload or {}).get("last_refresh_at_epoch") for r in rows
            ) if ts is not None
        )
        gaps = [b - a for a, b in zip(epochs, epochs[1:])]
        return {
            "freshness_at_report_seconds": latest_lag,
            "max_gap_between_refreshes_seconds": max(gaps) if gaps else None,
            "refresh_events_with_symbols": sum(
                int((r.payload or {}).get("symbols_refreshed") or 0) for r in rows
            ),
        }

    if metric_key == MetricKey.UNIVERSE_DRIFT:
        # |delta|/prior_size captures both growth and shrink. Max value over
        # the window = worst drift burst; sum |delta| = cumulative churn.
        max_ratio = 0.0
        cumulative_abs_delta = 0
        for r in rows:
            p = r.payload or {}
            delta = p.get("delta")
            prior = p.get("prior_size")
            if delta is None:
                continue
            cumulative_abs_delta += abs(int(delta))
            if prior is None or prior <= 0:
                ratio = 1.0 if delta else 0.0
            else:
                ratio = abs(float(delta)) / float(prior)
            if ratio > max_ratio:
                max_ratio = ratio
        return {
            "max_drift_ratio": max_ratio,
            "cumulative_abs_delta": cumulative_abs_delta,
        }

    if metric_key == MetricKey.BENCHMARK_AGE:
        # Report the most recent warm and the implied age at report time. If no
        # warm landed in the window, the gap itself is the governance signal.
        latest = rows[-1].payload or {}
        last_warmed = latest.get("last_warmed_at_epoch")
        implied_age = (
            max(0.0, window_end.timestamp() - float(last_warmed))
            if last_warmed
            else None
        )
        return {
            "latest_benchmark_symbol": latest.get("benchmark_symbol"),
            "latest_warmed_at_epoch": last_warmed,
            "implied_age_seconds": implied_age,
        }

    if metric_key == MetricKey.EXTRACTION_SUCCESS:
        # Aggregate by language: total calls, success count, ratio. SHARED scope
        # only, but the aggregator doesn't filter on market — it just rolls up
        # whatever rows were pulled (0 rows for non-SHARED markets, by design).
        by_lang: Dict[str, Dict[str, int]] = {}
        for r in rows:
            p = r.payload or {}
            lang = p.get("language") or "unknown"
            bucket = by_lang.setdefault(lang, {"total": 0, "success": 0})
            bucket["total"] += 1
            if p.get("success"):
                bucket["success"] += 1
        total = sum(v["total"] for v in by_lang.values())
        success = sum(v["success"] for v in by_lang.values())
        return {
            "overall_total": total,
            "overall_success": success,
            "overall_success_ratio": (success / total) if total > 0 else None,
            "by_language": {
                lang: {
                    "total": v["total"],
                    "success": v["success"],
                    "success_ratio": (v["success"] / v["total"]) if v["total"] else None,
                }
                for lang, v in sorted(by_lang.items())
            },
        }

    if metric_key == MetricKey.COMPLETENESS_DISTRIBUTION:
        # Regression detection: compare first vs last snapshot's 0-25 fraction.
        # A growing 0-25 bucket means provenance quality eroded over the week.
        first_p = rows[0].payload or {}
        last_p = rows[-1].payload or {}
        first_ratio = low_completeness_ratio(first_p)
        last_ratio = low_completeness_ratio(last_p)
        return {
            "first_snapshot_low_bucket_ratio": first_ratio,
            "last_snapshot_low_bucket_ratio": last_ratio,
            "low_bucket_ratio_delta": (
                (last_ratio - first_ratio)
                if first_ratio is not None and last_ratio is not None
                else None
            ),
            "last_snapshot_symbols_total": last_p.get("symbols_total"),
        }

    return {}


def _as_aware_utc(dt: Optional[datetime]) -> Optional[datetime]:
    if dt is None or dt.tzinfo is not None:
        return dt
    return dt.replace(tzinfo=timezone.utc)


def _summarize_alerts(
    db: Session, market: str, window_start: datetime, window_end: datetime,
) -> AlertRollup:
    """Count alerts touched during the window for one market.

    "Opened in window": opened_at inside [start, end].
    "Closed in window": closed_at inside [start, end].
    "Still active at report time": state != CLOSED and opened_at <= end.

    Implementation: single query for all rows with ``opened_at <= window_end``
    (superset of the three categories), then classify in Python. Collapses
    the former 3 queries per market into 1 while keeping the semantics above.
    """
    rows = (
        db.query(MarketTelemetryAlert)
        .filter(
            MarketTelemetryAlert.market == market,
            MarketTelemetryAlert.opened_at <= window_end,
        )
        .all()
    )

    opened_in_window = 0
    closed_in_window = 0
    still_active = 0
    by_severity: Dict[str, int] = {}
    for a in rows:
        # SQLite returns naive datetimes even for `DateTime(timezone=True)`;
        # Postgres returns aware. Coerce to UTC-aware so Python comparisons
        # don't raise `can't compare offset-naive and offset-aware`.
        opened_at = _as_aware_utc(a.opened_at)
        closed_at = _as_aware_utc(a.closed_at)
        if opened_at is not None and opened_at >= window_start:
            opened_in_window += 1
            by_severity[a.severity] = by_severity.get(a.severity, 0) + 1
        if closed_at is not None and window_start <= closed_at <= window_end:
            closed_in_window += 1
        if a.state != AlertState.CLOSED:
            still_active += 1

    return AlertRollup(
        market=market,
        opened=opened_in_window,
        closed=closed_in_window,
        still_active=still_active,
        by_severity=by_severity,
    )


def _serialize_thresholds() -> Dict[str, Dict[str, Dict[str, float]]]:
    """Capture the active threshold map point-in-time.

    The evaluator resolves `(metric, market)` → Levels at alert-fire time.
    Snapshotting it into the report lets future readers re-interpret an old
    report against old thresholds even if the code has since moved on.
    """
    snapshot: Dict[str, Dict[str, Dict[str, float]]] = {}
    for metric, by_market in THRESHOLDS.items():
        snapshot[metric] = {m: dict(levels) for m, levels in by_market.items()}
    return snapshot


def _content_hash(report: GovernanceReport) -> str:
    """SHA-256 over canonical JSON (sorted keys, no whitespace).

    The hash excludes its own field — otherwise the report would need to
    re-hash itself. We set the field to None, serialize, hash, then assign.
    """
    as_dict = asdict(report)
    as_dict["content_hash"] = None
    blob = json.dumps(as_dict, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def render_json(report: GovernanceReport) -> str:
    """Return canonical JSON (sorted, indented for human review)."""
    return json.dumps(asdict(report), sort_keys=True, indent=2, default=str)


def render_markdown(report: GovernanceReport) -> str:
    """Render a human-readable Markdown view.

    The Markdown is lossy compared to the JSON — operators read this; tools
    should consume the JSON. The content_hash is printed at both the top and
    the bottom so it's hard to tamper with the visible header alone.
    """
    lines: List[str] = []
    lines.append(f"# Weekly Telemetry Governance Report — {report.generated_at}")
    lines.append("")
    lines.append(f"- Report schema version: {report.report_schema_version}")
    lines.append(f"- Payload schema version: {report.payload_schema_version}")
    lines.append(f"- Window: {report.window_start} → {report.window_end}")
    lines.append(f"- Content hash (SHA-256): `{report.content_hash}`")
    lines.append("")

    lines.append("## Alerts")
    lines.append("")
    lines.append("| Market | Opened | Closed | Still active | Severity breakdown |")
    lines.append("|---|---:|---:|---:|---|")
    for a in report.alerts:
        sev = ", ".join(f"{s}={n}" for s, n in sorted(a.by_severity.items())) or "—"
        lines.append(f"| {a.market} | {a.opened} | {a.closed} | {a.still_active} | {sev} |")
    lines.append("")

    lines.append("## Metrics")
    lines.append("")
    for metric_key in _METRIC_KEYS:
        lines.append(f"### {metric_key}")
        lines.append("")
        lines.append("| Market | Events | First | Last | Rollup |")
        lines.append("|---|---:|---|---|---|")
        for m in report.metrics:
            if m.metric_key != metric_key:
                continue
            rollup = json.dumps(m.rollup, sort_keys=True, default=str) if m.rollup else "—"
            lines.append(
                f"| {m.market} | {m.event_count} | "
                f"{m.first_recorded_at or '—'} | {m.last_recorded_at or '—'} | "
                f"`{rollup}` |"
            )
        lines.append("")

    lines.append("## Thresholds snapshot (point-in-time)")
    lines.append("")
    lines.append(f"Owners: `{json.dumps(report.owners_snapshot, sort_keys=True)}`")
    lines.append("")
    lines.append(
        f"Thresholds JSON: `{json.dumps(report.thresholds_snapshot, sort_keys=True)}`"
    )
    lines.append("")
    lines.append("---")
    lines.append(f"Content hash (SHA-256): `{report.content_hash}`")
    lines.append("")
    return "\n".join(lines)
