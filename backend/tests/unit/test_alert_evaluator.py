"""Unit tests for the per-market alert evaluator (bead asia.10.2).

Covers:
- Open on first breach
- No-op while breach persists (hysteresis)
- Severity upgrade (warning → critical)
- Recovery closes the active alert
- Acknowledge doesn't re-fire
- Owner from market→owner map
- Extraction success aggregation across languages
"""
from __future__ import annotations

import time
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest
from sqlalchemy.exc import IntegrityError

from app.models.market_telemetry_alert import AlertSeverity, AlertState, MarketTelemetryAlert
from app.services.telemetry import alert_evaluator
from app.services.telemetry.alert_evaluator import (
    _classify,
    _evaluate_one,
    _extraction_success_ratio,
    acknowledge_alert,
    evaluate_all,
)
from app.services.telemetry.alert_thresholds import OWNERS, owner_for
from app.services.telemetry.schema import MetricKey


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------
class TestClassify:
    def test_hi_direction_warning(self):
        assert _classify(7500, {"warning": 7200, "critical": 21600}, "hi") == AlertSeverity.WARNING

    def test_hi_direction_critical(self):
        assert _classify(30000, {"warning": 7200, "critical": 21600}, "hi") == AlertSeverity.CRITICAL

    def test_hi_below_thresholds_returns_none(self):
        assert _classify(100, {"warning": 7200, "critical": 21600}, "hi") is None

    def test_lo_direction_warning(self):
        assert _classify(0.80, {"warning": 0.85, "critical": 0.70}, "lo") == AlertSeverity.WARNING

    def test_lo_direction_critical(self):
        assert _classify(0.50, {"warning": 0.85, "critical": 0.70}, "lo") == AlertSeverity.CRITICAL

    def test_lo_above_threshold_returns_none(self):
        assert _classify(0.99, {"warning": 0.85, "critical": 0.70}, "lo") is None


class TestExtractionRatio:
    def test_success_ratio(self):
        extraction = {"by_language": {"en": {"total": 10, "success": 9}, "ja": {"total": 5, "success": 4}}}
        assert _extraction_success_ratio(extraction) == pytest.approx(13 / 15)

    def test_zero_total_returns_none(self):
        assert _extraction_success_ratio({"by_language": {}}) is None
        assert _extraction_success_ratio({"by_language": {"en": {"total": 0}}}) is None

    def test_missing_success_treated_as_zero(self):
        extraction = {"by_language": {"en": {"total": 10}}}
        assert _extraction_success_ratio(extraction) == 0.0


class TestOwnerMap:
    def test_known_market_returns_owner(self):
        assert owner_for("US") == "us-ops"
        assert owner_for("HK") == "asia-ops"

    def test_unknown_market_falls_back_to_shared(self):
        # Unknown market should fall back to SHARED owner.
        assert owner_for("XX") == OWNERS["SHARED"]


# ---------------------------------------------------------------------------
# Hysteresis lifecycle (using an in-memory mock session)
# ---------------------------------------------------------------------------
class _FakeDB:
    """Minimal SQLAlchemy session stand-in for ``_evaluate_one`` unit tests.

    The evaluator now receives the prefetched ``active`` alert as a kwarg, so
    the fake only needs to record ``add()`` calls — no chainable query API.
    """

    def __init__(self):
        self.added = []

    def add(self, obj):
        if getattr(obj, "id", None) is None:
            obj.id = len(self.added) + 1
        self.added.append(obj)


class TestHysteresis:
    """Hysteresis lifecycle tests.

    ``_evaluate_one`` no longer commits — that moved to ``evaluate_all`` so
    the whole eval pass is one transaction. These tests assert the *staged*
    state on the row and the session, not commit counts.
    """

    def test_breach_with_no_active_opens_alert(self):
        db = _FakeDB()
        # freshness_lag = 30000s, US thresholds: warn=7200, crit=21600 → critical
        result = _evaluate_one(
            db, market="US", metric_key=MetricKey.FRESHNESS_LAG, value=30000.0,
            active=None,
        )
        assert result is not None
        assert result.state == AlertState.OPEN
        assert result.severity == AlertSeverity.CRITICAL
        assert result.owner == "us-ops"
        assert result.metrics["value"] == 30000.0
        assert len(db.added) == 1

    def test_repeated_breach_at_same_severity_is_noop(self):
        existing = MarketTelemetryAlert(
            id=1, market="US", metric_key=MetricKey.FRESHNESS_LAG,
            severity=AlertSeverity.WARNING, state=AlertState.OPEN, owner="us-ops",
            title="x", description="x", metrics={},
        )
        db = _FakeDB()
        _evaluate_one(
            db, market="US", metric_key=MetricKey.FRESHNESS_LAG, value=10000.0,
            active=existing,
        )
        assert db.added == []
        assert existing.severity == AlertSeverity.WARNING
        assert existing.state == AlertState.OPEN

    def test_warning_upgrades_to_critical(self):
        existing = MarketTelemetryAlert(
            id=1, market="US", metric_key=MetricKey.FRESHNESS_LAG,
            severity=AlertSeverity.WARNING, state=AlertState.OPEN, owner="us-ops",
            title="warn-title", description="warn-desc", metrics={"value": 10000},
        )
        db = _FakeDB()
        _evaluate_one(
            db, market="US", metric_key=MetricKey.FRESHNESS_LAG, value=30000.0,
            active=existing,
        )
        assert existing.severity == AlertSeverity.CRITICAL
        assert "CRITICAL" in existing.title
        assert db.added == []

    def test_critical_does_not_downgrade_to_warning(self):
        existing = MarketTelemetryAlert(
            id=1, market="US", metric_key=MetricKey.FRESHNESS_LAG,
            severity=AlertSeverity.CRITICAL, state=AlertState.OPEN, owner="us-ops",
            title="crit-title", description="crit-desc", metrics={"value": 30000},
        )
        db = _FakeDB()
        _evaluate_one(
            db, market="US", metric_key=MetricKey.FRESHNESS_LAG, value=10000.0,
            active=existing,
        )
        assert existing.severity == AlertSeverity.CRITICAL
        assert existing.state == AlertState.OPEN

    def test_recovery_closes_active_alert(self):
        existing = MarketTelemetryAlert(
            id=1, market="US", metric_key=MetricKey.FRESHNESS_LAG,
            severity=AlertSeverity.CRITICAL, state=AlertState.OPEN, owner="us-ops",
            title="x", description="x", metrics={},
        )
        db = _FakeDB()
        _evaluate_one(
            db, market="US", metric_key=MetricKey.FRESHNESS_LAG, value=100.0,
            active=existing,
        )
        assert existing.state == AlertState.CLOSED
        assert existing.closed_at is not None

    def test_recovery_with_no_active_is_noop(self):
        db = _FakeDB()
        result = _evaluate_one(
            db, market="US", metric_key=MetricKey.FRESHNESS_LAG, value=100.0,
            active=None,
        )
        assert result is None
        assert db.added == []

    def test_acknowledged_alert_does_not_refire(self):
        acked = MarketTelemetryAlert(
            id=1, market="HK", metric_key=MetricKey.BENCHMARK_AGE,
            severity=AlertSeverity.WARNING, state=AlertState.ACKNOWLEDGED, owner="asia-ops",
            title="x", description="x", metrics={},
        )
        db = _FakeDB()
        _evaluate_one(
            db, market="HK", metric_key=MetricKey.BENCHMARK_AGE, value=200000.0,
            active=acked,
        )
        assert db.added == []
        assert acked.state == AlertState.ACKNOWLEDGED

    def test_recovery_closes_acknowledged_alert(self):
        # Recovery still closes even if the alert was ACKNOWLEDGED — the
        # underlying condition is gone, so the lifecycle ends.
        acked = MarketTelemetryAlert(
            id=1, market="HK", metric_key=MetricKey.BENCHMARK_AGE,
            severity=AlertSeverity.WARNING, state=AlertState.ACKNOWLEDGED, owner="asia-ops",
            title="x", description="x", metrics={},
        )
        db = _FakeDB()
        _evaluate_one(
            db, market="HK", metric_key=MetricKey.BENCHMARK_AGE, value=100.0,
            active=acked,
        )
        assert acked.state == AlertState.CLOSED
        assert acked.closed_at is not None


# ---------------------------------------------------------------------------
# Acknowledge endpoint logic
# ---------------------------------------------------------------------------
class TestEvaluateAll:
    """End-to-end smoke tests for evaluate_all (prefetch + dispatch + commit)."""

    def _make_db_with_summaries(self, monkeypatch, prefetched_active):
        """Build a fake db that records add+commit and stub list_active_alerts."""
        db = _FakeDB()
        db.commits = 0
        db.rolledback = 0

        def _commit():
            db.commits += 1

        def _rollback():
            db.rolledback += 1

        db.commit = _commit
        db.rollback = _rollback

        # Patch list_active_alerts at the module level so both the prefetch
        # call and the post-eval return call get the same controlled rows.
        monkeypatch.setattr(
            alert_evaluator, "list_active_alerts",
            lambda _db: list(prefetched_active),
        )
        return db

    def test_breach_in_summary_inserts_new_alert_and_commits_once(self, monkeypatch):
        db = self._make_db_with_summaries(monkeypatch, prefetched_active=[])

        # US freshness lag = 30000s → critical (US warn=7200, crit=21600)
        summaries = [
            {
                "market": "US",
                "freshness_lag": {"lag_seconds": 30000.0, "last_refresh_at_epoch": 0},
                "benchmark_age": None,
                "completeness_distribution": None,
                "universe_drift": None,
            },
        ]
        evaluate_all(db, summaries)

        assert db.commits == 1
        assert db.rolledback == 0
        # Exactly one INSERT staged for the breached pair.
        assert len(db.added) == 1
        new_alert = db.added[0]
        assert new_alert.market == "US"
        assert new_alert.metric_key == MetricKey.FRESHNESS_LAG
        assert new_alert.severity == AlertSeverity.CRITICAL

    def test_extraction_eval_only_runs_for_shared_summary(self, monkeypatch):
        db = self._make_db_with_summaries(monkeypatch, prefetched_active=[])

        # Per-market summary with extraction_today populated — must NOT eval
        # extraction (it's SHARED-scoped only).
        summaries = [
            {
                "market": "US",
                "freshness_lag": None,
                "benchmark_age": None,
                "completeness_distribution": None,
                "universe_drift": None,
                "extraction_today": {"by_language": {"en": {"total": 100, "success": 1}}},
            },
        ]
        evaluate_all(db, summaries)
        # No extraction alert opened despite the per-market summary having
        # a low success ratio — because eval only fires for SHARED scope.
        assert db.added == []

    def test_integrity_error_on_commit_rolls_back(self, monkeypatch):
        db = self._make_db_with_summaries(monkeypatch, prefetched_active=[])

        def _raising_commit():
            raise IntegrityError("dup", {}, Exception("dup"))

        db.commit = _raising_commit

        summaries = [
            {
                "market": "US",
                "freshness_lag": {"lag_seconds": 30000.0},
                "benchmark_age": None,
                "completeness_distribution": None,
                "universe_drift": None,
            },
        ]
        # Should swallow IntegrityError and call rollback.
        evaluate_all(db, summaries)
        assert db.rolledback == 1


class TestAcknowledge:
    def test_open_alert_becomes_acknowledged(self):
        alert = MarketTelemetryAlert(
            id=42, market="US", metric_key=MetricKey.FRESHNESS_LAG,
            severity=AlertSeverity.WARNING, state=AlertState.OPEN, owner="us-ops",
            title="x", description="x", metrics={},
        )
        db = MagicMock()
        db.query.return_value.filter.return_value.first.return_value = alert

        result = acknowledge_alert(db, 42, "alice")

        assert result is alert
        assert alert.state == AlertState.ACKNOWLEDGED
        assert alert.acknowledged_by == "alice"
        assert alert.acknowledged_at is not None
        db.commit.assert_called_once()

    def test_already_acknowledged_is_idempotent(self):
        alert = MarketTelemetryAlert(
            id=42, market="US", metric_key=MetricKey.FRESHNESS_LAG,
            severity=AlertSeverity.WARNING, state=AlertState.ACKNOWLEDGED,
            owner="us-ops", title="x", description="x", metrics={},
            acknowledged_by="bob",
        )
        db = MagicMock()
        db.query.return_value.filter.return_value.first.return_value = alert

        result = acknowledge_alert(db, 42, "alice")

        assert result is alert
        # Doesn't overwrite the original ack metadata.
        assert alert.acknowledged_by == "bob"
        db.commit.assert_not_called()

    def test_unknown_alert_returns_none(self):
        db = MagicMock()
        db.query.return_value.filter.return_value.first.return_value = None
        assert acknowledge_alert(db, 999, "alice") is None
