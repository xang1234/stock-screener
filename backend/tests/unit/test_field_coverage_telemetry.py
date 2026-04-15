"""Unit tests for field-coverage telemetry (bead asia.10.5).

Covers:
- Schema helpers (field_coverage_payload, unsupported_field_ratio,
  cadence_fallback_ratio) handle empty / zero cases without divide-by-zero
- PerMarketTelemetry.record_field_coverage_from_registry reads the registry,
  emits the correct support-state counts, no-ops for SHARED
- Weekly audit rolls up the new metric including worst-ratio window signals
- Alert evaluator classifies FIELD_COVERAGE breaches per per-market thresholds
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
import app.models.market_telemetry  # noqa: F401
import app.models.market_telemetry_alert  # noqa: F401
from app.models.market_telemetry import MarketTelemetryEvent
from app.services.telemetry.schema import (
    MetricKey,
    SCHEMA_VERSION,
    cadence_fallback_ratio,
    field_coverage_payload,
    unsupported_field_ratio,
)
from app.services.telemetry.alert_evaluator import _classify, _METRIC_RULES
from app.services.telemetry.alert_thresholds import thresholds_for
from app.models.market_telemetry_alert import AlertSeverity


class TestSchemaHelpers:
    def test_payload_normalizes_types(self):
        p = field_coverage_payload(
            total_fields=50,
            support_state_counts={"supported": 45, "unsupported": 5},
            unsupported_field_names=("institutional_ownership", "short_interest"),
            computed_field_names=("rs_rating",),
            cadence_counts={"quarterly_qoq": 400, "comparable_period_yoy": 100},
            cadence_eligible_universe=500,
        )
        assert p["schema_version"] == SCHEMA_VERSION
        assert p["total_fields"] == 50
        assert p["support_state_counts"] == {"supported": 45, "unsupported": 5}
        assert p["cadence_eligible_universe"] == 500
        # Tuples convert to lists for JSON serialization.
        assert p["unsupported_field_names"] == ["institutional_ownership", "short_interest"]

    def test_unsupported_ratio_computes_fraction(self):
        p = field_coverage_payload(
            total_fields=50,
            support_state_counts={"supported": 45, "unsupported": 5},
            unsupported_field_names=(),
            computed_field_names=(),
            cadence_counts={},
            cadence_eligible_universe=0,
        )
        assert unsupported_field_ratio(p) == pytest.approx(5 / 50)

    def test_unsupported_ratio_zero_total_returns_none(self):
        p = field_coverage_payload(
            total_fields=0,
            support_state_counts={},
            unsupported_field_names=(),
            computed_field_names=(),
            cadence_counts={},
            cadence_eligible_universe=0,
        )
        assert unsupported_field_ratio(p) is None

    def test_cadence_fallback_ratio(self):
        p = field_coverage_payload(
            total_fields=50,
            support_state_counts={},
            unsupported_field_names=(),
            computed_field_names=(),
            cadence_counts={"quarterly_qoq": 400, "comparable_period_yoy": 100},
            cadence_eligible_universe=500,
        )
        assert cadence_fallback_ratio(p) == pytest.approx(0.20)

    def test_cadence_fallback_zero_universe_returns_none(self):
        # A market with no fundamentals (yet) must not trip a cadence alert.
        p = field_coverage_payload(
            total_fields=50,
            support_state_counts={},
            unsupported_field_names=(),
            computed_field_names=(),
            cadence_counts={},
            cadence_eligible_universe=0,
        )
        assert cadence_fallback_ratio(p) is None


class TestEvaluatorIntegration:
    def test_field_coverage_registered_with_hi_direction(self):
        extractor, direction = _METRIC_RULES[MetricKey.FIELD_COVERAGE]
        assert direction == "hi"  # bigger ratio = worse
        # Extractor is the schema helper.
        p = field_coverage_payload(
            total_fields=50,
            support_state_counts={"supported": 40, "unsupported": 10},
            unsupported_field_names=(),
            computed_field_names=(),
            cadence_counts={},
            cadence_eligible_universe=0,
        )
        assert extractor(p) == pytest.approx(0.20)

    def test_thresholds_fire_warning_on_hk(self):
        levels = thresholds_for(MetricKey.FIELD_COVERAGE, "HK")
        assert levels == {"warning": 0.10, "critical": 0.25}
        # HK warns at 0.10, critical at 0.25 — a 0.15 ratio should warn.
        assert _classify(0.15, levels, "hi") == AlertSeverity.WARNING
        assert _classify(0.30, levels, "hi") == AlertSeverity.CRITICAL
        assert _classify(0.05, levels, "hi") is None

    def test_us_has_tighter_thresholds(self):
        # US has all supported fields; any unsupported regression is a big deal.
        levels = thresholds_for(MetricKey.FIELD_COVERAGE, "US")
        assert levels == {"warning": 0.05, "critical": 0.15}


class TestRecordFromRegistry:
    def test_shared_sentinel_noops(self):
        from app.services.telemetry.per_market_telemetry import PerMarketTelemetry

        t = PerMarketTelemetry(redis_client_factory=lambda: None, session_factory=None)
        t._emit_pg = MagicMock()
        t._set_gauge = MagicMock()
        t.record_field_coverage_from_registry("SHARED")
        t._emit_pg.assert_not_called()
        t._set_gauge.assert_not_called()

    def test_per_market_emits_registry_snapshot(self, monkeypatch):
        from app.services.telemetry.per_market_telemetry import PerMarketTelemetry
        from app.services.telemetry import per_market_telemetry as pmt_module

        fake_entry = MagicMock()
        fake_entry.field = "institutional_ownership"
        fake_entry.markets = {
            "HK": MagicMock(support_state="unsupported"),
        }
        fake_entry_supported = MagicMock()
        fake_entry_supported.field = "price"
        fake_entry_supported.markets = {
            "HK": MagicMock(support_state="supported"),
        }
        fake_registry = MagicMock()
        fake_registry.entries.return_value = (fake_entry, fake_entry_supported)

        monkeypatch.setattr(
            "app.services.field_capability_registry.field_capability_registry",
            fake_registry,
        )

        t = PerMarketTelemetry(redis_client_factory=lambda: None, session_factory=None)
        # Stub DB cadence query via internal helper
        t._read_cadence_counts = lambda market: ({"quarterly_qoq": 400}, 400)
        t._emit_pg = MagicMock()
        t._set_gauge = MagicMock()

        t.record_field_coverage_from_registry("HK")

        assert t._set_gauge.call_count == 1
        metric_key, market, payload = t._set_gauge.call_args[0]
        assert metric_key == MetricKey.FIELD_COVERAGE
        assert market == "HK"
        assert payload["support_state_counts"] == {"unsupported": 1, "supported": 1}
        assert payload["unsupported_field_names"] == ["institutional_ownership"]
        assert payload["cadence_eligible_universe"] == 400


_event_id_counter = [1000]
_NOW = datetime(2026, 4, 15, 12, 0, 0, tzinfo=timezone.utc)


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


def _cov_event(market: str, payload: dict, recorded_at: datetime):
    _event_id_counter[0] += 1
    return MarketTelemetryEvent(
        id=_event_id_counter[0],
        market=market,
        metric_key=MetricKey.FIELD_COVERAGE,
        schema_version=SCHEMA_VERSION,
        payload=payload,
        recorded_at=recorded_at,
    )


class TestWeeklyAuditRollup:
    def test_latest_wins_and_worst_ratios_recorded(self, telemetry_session):
        from app.services.telemetry.weekly_audit import run_weekly_audit

        # Two snapshots: mid-week regression (worst) then recovery (latest).
        telemetry_session.add_all([
            _cov_event(
                "HK",
                field_coverage_payload(
                    total_fields=40,
                    support_state_counts={"supported": 20, "unsupported": 20},
                    unsupported_field_names=tuple(f"f{i}" for i in range(20)),
                    computed_field_names=(),
                    cadence_counts={"quarterly_qoq": 200, "comparable_period_yoy": 300},
                    cadence_eligible_universe=500,
                ),
                _NOW - timedelta(days=3),
            ),
            _cov_event(
                "HK",
                field_coverage_payload(
                    total_fields=40,
                    support_state_counts={"supported": 38, "unsupported": 2},
                    unsupported_field_names=("institutional_ownership", "short_interest"),
                    computed_field_names=(),
                    cadence_counts={"quarterly_qoq": 400, "comparable_period_yoy": 100},
                    cadence_eligible_universe=500,
                ),
                _NOW - timedelta(days=1),
            ),
        ])
        telemetry_session.commit()

        report = run_weekly_audit(telemetry_session, now=_NOW)
        hk_cov = next(
            m for m in report.metrics
            if m.market == "HK" and m.metric_key == MetricKey.FIELD_COVERAGE
        )
        # Latest snapshot drives the "state" section.
        assert hk_cov.rollup["latest_support_state_counts"] == {"supported": 38, "unsupported": 2}
        assert hk_cov.rollup["latest_cadence_eligible_universe"] == 500
        # Worst-window signals capture the mid-week regression.
        assert hk_cov.rollup["worst_unsupported_ratio"] == pytest.approx(20 / 40)
        assert hk_cov.rollup["worst_cadence_fallback_ratio"] == pytest.approx(300 / 500)
