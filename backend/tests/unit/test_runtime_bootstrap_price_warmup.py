"""Regression tests for the runtime bootstrap price-warmup barrier."""

from __future__ import annotations

from datetime import date

import pytest
from celery.exceptions import Retry


class _FakeSession:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class _FakeCalendarService:
    def last_completed_trading_day(self, market):
        assert market in {"US", "HK"}
        return date(2026, 6, 8)


def _patch_price_barrier_dependencies(
    monkeypatch,
    module,
    *,
    session,
    warmup_metadata,
    coverage_report,
):
    class _FakePriceCache:
        def get_warmup_metadata(self, *, market=None):
            return warmup_metadata

    monkeypatch.setattr(module, "SessionLocal", lambda: session)
    monkeypatch.setattr("app.wiring.bootstrap.get_price_cache", lambda: _FakePriceCache())
    monkeypatch.setattr(
        "app.wiring.bootstrap.get_market_calendar_service",
        lambda: _FakeCalendarService(),
    )
    monkeypatch.setattr(
        module,
        "evaluate_bootstrap_price_readiness",
        lambda _db, *, market, as_of_date: coverage_report,
        raising=False,
    )


def test_wait_for_bootstrap_price_warmup_returns_ready_for_completed_metadata(monkeypatch):
    from app.tasks import runtime_bootstrap_tasks as module

    session = _FakeSession()
    calls = {}
    _patch_price_barrier_dependencies(
        monkeypatch,
        module,
        session=session,
        warmup_metadata={
            "status": "completed",
            "count": 249,
            "total": 249,
        },
        coverage_report={
            "eligible": True,
            "price_covered_symbols": 249,
            "price_total_symbols": 249,
            "price_missing_symbols": 0,
            "price_coverage_ratio": 1.0,
            "threshold": 0.95,
        },
    )
    monkeypatch.setattr(
        module,
        "mark_market_activity_completed",
        lambda _db, **kwargs: calls.setdefault("completed", kwargs),
    )

    result = module.wait_for_bootstrap_price_warmup.run(market="us")

    assert result["status"] == "ready"
    assert result["market"] == "US"
    assert result["coverage"]["price_covered_symbols"] == 249
    assert calls["completed"]["market"] == "US"
    assert calls["completed"]["stage_key"] == "prices"
    assert calls["completed"]["message"] == "Price cache coverage ready"
    assert session.closed is True


def test_wait_for_bootstrap_price_warmup_uses_coverage_not_stale_warmup_metadata(
    monkeypatch,
):
    from app.tasks import runtime_bootstrap_tasks as module

    session = _FakeSession()
    calls = {}
    _patch_price_barrier_dependencies(
        monkeypatch,
        module,
        session=session,
        warmup_metadata={
            "status": "partial",
            "count": 19,
            "total": 31,
            "completed_at": "2026-06-09T08:00:00",
        },
        coverage_report={
            "eligible": True,
            "price_covered_symbols": 31,
            "price_total_symbols": 31,
            "price_missing_symbols": 0,
            "price_coverage_ratio": 1.0,
            "threshold": 0.95,
        },
    )
    monkeypatch.setattr(
        module,
        "mark_market_activity_completed",
        lambda _db, **kwargs: calls.setdefault("completed", kwargs),
    )

    result = module.wait_for_bootstrap_price_warmup.run(market="HK")

    assert result["status"] == "ready"
    assert result["market"] == "HK"
    assert result["warmup"]["status"] == "partial"
    assert result["coverage"]["eligible"] is True
    assert calls["completed"]["message"] == "Price cache coverage ready"


def test_wait_for_bootstrap_price_warmup_retries_partial_coverage(monkeypatch):
    from app.tasks import runtime_bootstrap_tasks as module

    session = _FakeSession()
    progress_calls = []
    retry_calls = []
    _patch_price_barrier_dependencies(
        monkeypatch,
        module,
        session=session,
        warmup_metadata={
            "status": "completed",
            "count": 31,
            "total": 31,
            "completed_at": "2026-06-09T08:00:00",
        },
        coverage_report={
            "eligible": False,
            "price_covered_symbols": 19,
            "price_total_symbols": 31,
            "price_missing_symbols": 12,
            "price_coverage_ratio": 19 / 31,
            "threshold": 0.95,
        },
    )

    def fake_retry(*, exc=None, countdown=None, max_retries=None):
        retry_calls.append(
            {
                "exc": exc,
                "countdown": countdown,
                "max_retries": max_retries,
            }
        )
        raise Retry("retry")

    monkeypatch.setattr(
        module,
        "mark_market_activity_progress",
        lambda _db, **kwargs: progress_calls.append(kwargs),
    )
    monkeypatch.setattr(module.wait_for_bootstrap_price_warmup, "retry", fake_retry)
    module.wait_for_bootstrap_price_warmup.request.id = "wait-task-123"
    module.wait_for_bootstrap_price_warmup.request.retries = 0

    with pytest.raises(Retry):
        module.wait_for_bootstrap_price_warmup.run(market="HK")

    assert progress_calls == [
        {
            "market": "HK",
            "stage_key": "prices",
            "lifecycle": "bootstrap",
            "task_name": "app.tasks.runtime_bootstrap_tasks.wait_for_bootstrap_price_warmup",
            "task_id": "wait-task-123",
            "percent": pytest.approx(61.2903),
            "current": 19,
            "total": 31,
            "message": "Waiting for price cache coverage: 19/31 (61.3%, threshold=95.0%; warmup=completed, 31/31)",
        }
    ]
    assert len(retry_calls) == 1
    assert "waiting_for_bootstrap_price_coverage:HK" in str(retry_calls[0]["exc"])
    assert retry_calls[0]["countdown"] == 30
    assert retry_calls[0]["max_retries"] == 120


def test_wait_for_bootstrap_price_warmup_marks_failed_when_retries_exhausted(
    monkeypatch,
):
    from app.tasks import runtime_bootstrap_tasks as module

    session = _FakeSession()
    failed_calls = []
    _patch_price_barrier_dependencies(
        monkeypatch,
        module,
        session=session,
        warmup_metadata={
            "status": "completed",
            "count": 31,
            "total": 31,
            "completed_at": "2026-06-09T08:00:00",
        },
        coverage_report={
            "eligible": False,
            "price_covered_symbols": 19,
            "price_total_symbols": 31,
            "price_missing_symbols": 12,
            "price_coverage_ratio": 19 / 31,
            "threshold": 0.95,
        },
    )
    monkeypatch.setattr(
        module,
        "mark_market_activity_failed",
        lambda _db, **kwargs: failed_calls.append(kwargs),
    )
    module.wait_for_bootstrap_price_warmup.request.id = "wait-task-456"
    module.wait_for_bootstrap_price_warmup.request.retries = (
        module.BOOTSTRAP_PRICE_WARMUP_MAX_RETRIES
    )

    with pytest.raises(RuntimeError) as exc_info:
        module.wait_for_bootstrap_price_warmup.run(market="HK")

    assert "Price cache coverage incomplete for HK" in str(exc_info.value)
    assert failed_calls == [
        {
            "market": "HK",
            "stage_key": "prices",
            "lifecycle": "bootstrap",
            "task_name": "app.tasks.runtime_bootstrap_tasks.wait_for_bootstrap_price_warmup",
            "task_id": "wait-task-456",
            "message": (
                "Price cache warmup unavailable: Price cache coverage incomplete "
                "for HK: 19/31 (61.3%, threshold=95.0%, missing=12)"
            ),
        }
    ]
    assert session.closed is True
