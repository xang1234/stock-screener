"""Unit tests for weekly telemetry governance audit (bead asia.10.4).

Covers:
- Aggregation correctness for each metric_key rollup
- Alert lifecycle counts (opened/closed/still_active)
- Content hash determinism (same input → same hash)
- Content hash sensitivity (modifying any field changes the hash)
- Threshold snapshot captured into the report
- Rendering to JSON (valid, parseable) and Markdown (hash present)
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
import app.models.market_telemetry  # noqa: F401
import app.models.market_telemetry_alert  # noqa: F401
from app.models.market_telemetry import MarketTelemetryEvent
from app.models.market_telemetry_alert import (
    AlertSeverity,
    AlertState,
    MarketTelemetryAlert,
)
from app.services.telemetry.schema import (
    MetricKey,
    SCHEMA_VERSION,
    completeness_distribution_payload,
    extraction_success_payload,
    freshness_lag_payload,
    universe_drift_payload,
    benchmark_age_payload,
)
from app.services.telemetry.weekly_audit import (
    AUDIT_WINDOW_DAYS,
    REPORT_SCHEMA_VERSION,
    _content_hash,
    render_json,
    render_markdown,
    run_weekly_audit,
)


@pytest.fixture
def telemetry_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    sess = sessionmaker(bind=engine)()
    try:
        yield sess
    finally:
        sess.close()
        engine.dispose()


_event_id_counter = [0]
_alert_id_counter = [0]


def _event(market: str, metric_key: str, payload: dict, recorded_at: datetime):
    # Explicit IDs: BigInteger primary keys don't alias to SQLite's rowid, so
    # autoincrement is inert under :memory: SQLite. Postgres in production
    # handles autoincrement itself; this explicit ID is test-only.
    _event_id_counter[0] += 1
    return MarketTelemetryEvent(
        id=_event_id_counter[0],
        market=market,
        metric_key=metric_key,
        schema_version=SCHEMA_VERSION,
        payload=payload,
        recorded_at=recorded_at,
    )


def _alert(**kwargs):
    _alert_id_counter[0] += 1
    return MarketTelemetryAlert(id=_alert_id_counter[0], **kwargs)


_NOW = datetime(2026, 4, 15, 12, 0, 0, tzinfo=timezone.utc)


class TestRollupFreshnessLag:
    def test_freshness_at_report_and_max_gap(self, telemetry_session):
        # Three refresh events at T-5h, T-3h, T-1h. Latest = T-1h → freshness
        # at report time = 1h. Gaps between consecutive refreshes: 2h, 2h →
        # max gap 2h = 7200s.
        events = [
            _event(
                "HK",
                MetricKey.FRESHNESS_LAG,
                freshness_lag_payload(
                    last_refresh_at_epoch=(_NOW - timedelta(hours=5)).timestamp(),
                    source="prices",
                    symbols_refreshed=500,
                ),
                _NOW - timedelta(hours=5),
            ),
            _event(
                "HK",
                MetricKey.FRESHNESS_LAG,
                freshness_lag_payload(
                    last_refresh_at_epoch=(_NOW - timedelta(hours=3)).timestamp(),
                    source="prices",
                    symbols_refreshed=100,
                ),
                _NOW - timedelta(hours=3),
            ),
            _event(
                "HK",
                MetricKey.FRESHNESS_LAG,
                freshness_lag_payload(
                    last_refresh_at_epoch=(_NOW - timedelta(hours=1)).timestamp(),
                    source="prices",
                    symbols_refreshed=200,
                ),
                _NOW - timedelta(hours=1),
            ),
        ]
        telemetry_session.add_all(events)
        telemetry_session.commit()

        report = run_weekly_audit(telemetry_session, now=_NOW)

        hk_fresh = next(
            m for m in report.metrics
            if m.market == "HK" and m.metric_key == MetricKey.FRESHNESS_LAG
        )
        assert hk_fresh.event_count == 3
        # Latest refresh is 1h before _NOW → freshness at report = 3600s.
        assert abs(hk_fresh.rollup["freshness_at_report_seconds"] - 3600) < 1
        # Max gap between refreshes = 2h = 7200s.
        assert abs(hk_fresh.rollup["max_gap_between_refreshes_seconds"] - 7200) < 1
        assert hk_fresh.rollup["refresh_events_with_symbols"] == 800


class TestRollupUniverseDrift:
    def test_max_ratio_and_cumulative(self, telemetry_session):
        events = [
            _event(
                "JP",
                MetricKey.UNIVERSE_DRIFT,
                universe_drift_payload(current_size=480, prior_size=500),  # delta=-20, ratio=0.04
                _NOW - timedelta(days=5),
            ),
            _event(
                "JP",
                MetricKey.UNIVERSE_DRIFT,
                universe_drift_payload(current_size=400, prior_size=480),  # delta=-80, ratio≈0.1667
                _NOW - timedelta(days=2),
            ),
        ]
        telemetry_session.add_all(events)
        telemetry_session.commit()

        report = run_weekly_audit(telemetry_session, now=_NOW)
        jp_drift = next(
            m for m in report.metrics
            if m.market == "JP" and m.metric_key == MetricKey.UNIVERSE_DRIFT
        )
        assert jp_drift.event_count == 2
        assert jp_drift.rollup["max_drift_ratio"] == pytest.approx(80 / 480)
        assert jp_drift.rollup["cumulative_abs_delta"] == 100


class TestRollupCompleteness:
    def test_low_bucket_ratio_delta(self, telemetry_session):
        # First snapshot: 10/100 in 0-25 → 0.10. Last: 40/100 → 0.40. Delta=+0.30.
        events = [
            _event(
                "TW",
                MetricKey.COMPLETENESS_DISTRIBUTION,
                completeness_distribution_payload(
                    bucket_counts={"0-25": 10, "90-100": 90}, symbols_total=100,
                ),
                _NOW - timedelta(days=6),
            ),
            _event(
                "TW",
                MetricKey.COMPLETENESS_DISTRIBUTION,
                completeness_distribution_payload(
                    bucket_counts={"0-25": 40, "90-100": 60}, symbols_total=100,
                ),
                _NOW - timedelta(days=1),
            ),
        ]
        telemetry_session.add_all(events)
        telemetry_session.commit()

        report = run_weekly_audit(telemetry_session, now=_NOW)
        tw_comp = next(
            m for m in report.metrics
            if m.market == "TW" and m.metric_key == MetricKey.COMPLETENESS_DISTRIBUTION
        )
        assert tw_comp.rollup["first_snapshot_low_bucket_ratio"] == pytest.approx(0.10)
        assert tw_comp.rollup["last_snapshot_low_bucket_ratio"] == pytest.approx(0.40)
        assert tw_comp.rollup["low_bucket_ratio_delta"] == pytest.approx(0.30)


class TestRollupExtraction:
    def test_per_language_aggregation(self, telemetry_session):
        events = [
            _event(
                "SHARED",
                MetricKey.EXTRACTION_SUCCESS,
                extraction_success_payload(language="en", success=True),
                _NOW - timedelta(hours=h),
            )
            for h in (10, 8, 6)
        ]
        events.append(
            _event(
                "SHARED",
                MetricKey.EXTRACTION_SUCCESS,
                extraction_success_payload(language="en", success=False),
                _NOW - timedelta(hours=4),
            )
        )
        events.append(
            _event(
                "SHARED",
                MetricKey.EXTRACTION_SUCCESS,
                extraction_success_payload(language="ja", success=True),
                _NOW - timedelta(hours=2),
            )
        )
        telemetry_session.add_all(events)
        telemetry_session.commit()

        report = run_weekly_audit(telemetry_session, now=_NOW)
        shared_extraction = next(
            m for m in report.metrics
            if m.market == "SHARED" and m.metric_key == MetricKey.EXTRACTION_SUCCESS
        )
        # en: 3 success / 4 total = 0.75. ja: 1/1 = 1.0. Overall: 4/5 = 0.80.
        assert shared_extraction.rollup["overall_total"] == 5
        assert shared_extraction.rollup["overall_success"] == 4
        assert shared_extraction.rollup["overall_success_ratio"] == pytest.approx(0.80)
        assert shared_extraction.rollup["by_language"]["en"]["success_ratio"] == pytest.approx(0.75)
        assert shared_extraction.rollup["by_language"]["ja"]["success_ratio"] == 1.0


class TestRollupBenchmarkAge:
    def test_latest_warm_only(self, telemetry_session):
        # Two warms; only the latest should appear in the rollup.
        events = [
            _event(
                "US",
                MetricKey.BENCHMARK_AGE,
                benchmark_age_payload(
                    last_warmed_at_epoch=(_NOW - timedelta(days=3)).timestamp(),
                    benchmark_symbol="SPY",
                ),
                _NOW - timedelta(days=3),
            ),
            _event(
                "US",
                MetricKey.BENCHMARK_AGE,
                benchmark_age_payload(
                    last_warmed_at_epoch=(_NOW - timedelta(hours=6)).timestamp(),
                    benchmark_symbol="SPY",
                ),
                _NOW - timedelta(hours=6),
            ),
        ]
        telemetry_session.add_all(events)
        telemetry_session.commit()

        report = run_weekly_audit(telemetry_session, now=_NOW)
        us_bench = next(
            m for m in report.metrics
            if m.market == "US" and m.metric_key == MetricKey.BENCHMARK_AGE
        )
        assert us_bench.rollup["latest_benchmark_symbol"] == "SPY"
        assert abs(us_bench.rollup["implied_age_seconds"] - 21600) < 1  # 6h = 21600s


class TestAlertRollup:
    def test_counts_opened_closed_and_still_active(self, telemetry_session):
        # 2 opened in window; 1 of them closed in window; 1 still open; 1
        # was opened BEFORE window but still open → counts as still_active
        # (opened_at <= window_end and state != CLOSED).
        telemetry_session.add_all([
            _alert(
                market="HK",
                metric_key=MetricKey.FRESHNESS_LAG,
                severity=AlertSeverity.WARNING,
                state=AlertState.CLOSED,
                title="closed in window",
                opened_at=_NOW - timedelta(days=3),
                closed_at=_NOW - timedelta(days=1),
            ),
            _alert(
                market="HK",
                metric_key=MetricKey.UNIVERSE_DRIFT,
                severity=AlertSeverity.CRITICAL,
                state=AlertState.OPEN,
                title="opened in window, still open",
                opened_at=_NOW - timedelta(days=2),
            ),
            _alert(
                market="HK",
                metric_key=MetricKey.BENCHMARK_AGE,
                severity=AlertSeverity.WARNING,
                state=AlertState.OPEN,
                title="opened before window, still open",
                opened_at=_NOW - timedelta(days=10),
            ),
        ])
        telemetry_session.commit()

        report = run_weekly_audit(telemetry_session, now=_NOW)
        hk_alerts = next(a for a in report.alerts if a.market == "HK")
        assert hk_alerts.opened == 2  # first two, inside window
        assert hk_alerts.closed == 1  # only the first
        assert hk_alerts.still_active == 2  # second and third
        assert hk_alerts.by_severity == {
            AlertSeverity.WARNING: 1,
            AlertSeverity.CRITICAL: 1,
        }


class TestContentHash:
    def test_same_input_same_hash(self, telemetry_session):
        # Deterministic seed + pinned `now` must produce identical hashes.
        telemetry_session.add(_event(
            "US",
            MetricKey.FRESHNESS_LAG,
            freshness_lag_payload(
                last_refresh_at_epoch=(_NOW - timedelta(hours=1)).timestamp(),
                source="prices",
                symbols_refreshed=1,
            ),
            _NOW - timedelta(minutes=30),
        ))
        telemetry_session.commit()

        r1 = run_weekly_audit(telemetry_session, now=_NOW)
        r2 = run_weekly_audit(telemetry_session, now=_NOW)
        assert r1.content_hash == r2.content_hash
        assert len(r1.content_hash) == 64  # SHA-256 hex

    def test_hash_changes_when_payload_changes(self, telemetry_session):
        r1 = run_weekly_audit(telemetry_session, now=_NOW)
        telemetry_session.add(_event(
            "US",
            MetricKey.UNIVERSE_DRIFT,
            universe_drift_payload(current_size=100, prior_size=99),
            _NOW - timedelta(hours=1),
        ))
        telemetry_session.commit()
        r2 = run_weekly_audit(telemetry_session, now=_NOW)
        assert r1.content_hash != r2.content_hash

    def test_hash_is_over_canonical_json(self, telemetry_session):
        # External verification path: recompute the hash from the JSON file
        # (with content_hash nulled) and assert it matches.
        report = run_weekly_audit(telemetry_session, now=_NOW)
        blob = json.loads(render_json(report))
        assert blob["content_hash"] == report.content_hash
        blob["content_hash"] = None
        recomputed = hashlib.sha256(
            json.dumps(blob, sort_keys=True, separators=(",", ":"), default=str).encode()
        ).hexdigest()
        assert recomputed == report.content_hash


class TestThresholdSnapshot:
    def test_thresholds_captured_point_in_time(self, telemetry_session):
        report = run_weekly_audit(telemetry_session, now=_NOW)
        # Sanity: each metric_key present, each entry is a dict of market→levels.
        assert MetricKey.FRESHNESS_LAG in report.thresholds_snapshot
        us_levels = report.thresholds_snapshot[MetricKey.FRESHNESS_LAG]["US"]
        assert us_levels["warning"] == 7200
        assert us_levels["critical"] == 21600
        # Owners snapshot present so reviewers can map alerts to teams.
        assert report.owners_snapshot["HK"] == "asia-ops"


class TestRendering:
    def test_json_roundtrip(self, telemetry_session):
        report = run_weekly_audit(telemetry_session, now=_NOW)
        parsed = json.loads(render_json(report))
        assert parsed["report_schema_version"] == REPORT_SCHEMA_VERSION
        assert parsed["window_start"] < parsed["window_end"]

    def test_markdown_includes_hash(self, telemetry_session):
        report = run_weekly_audit(telemetry_session, now=_NOW)
        md = render_markdown(report)
        # Hash appears twice (top + bottom) to resist single-field tampering.
        assert md.count(report.content_hash) == 2


class TestWindowRespected:
    def test_events_outside_window_are_ignored(self, telemetry_session):
        # Inside: 6 days ago. Outside: 10 days ago (> AUDIT_WINDOW_DAYS).
        telemetry_session.add_all([
            _event(
                "US",
                MetricKey.UNIVERSE_DRIFT,
                universe_drift_payload(current_size=100, prior_size=99),
                _NOW - timedelta(days=6),
            ),
            _event(
                "US",
                MetricKey.UNIVERSE_DRIFT,
                universe_drift_payload(current_size=200, prior_size=100),
                _NOW - timedelta(days=AUDIT_WINDOW_DAYS + 3),
            ),
        ])
        telemetry_session.commit()

        report = run_weekly_audit(telemetry_session, now=_NOW)
        us_drift = next(
            m for m in report.metrics
            if m.market == "US" and m.metric_key == MetricKey.UNIVERSE_DRIFT
        )
        assert us_drift.event_count == 1
