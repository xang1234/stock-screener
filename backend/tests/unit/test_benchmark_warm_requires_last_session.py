"""A benchmark warm counts only when a candidate reaches the market's last
completed session; a fetch ending on an earlier session is a failed warm."""

from __future__ import annotations

from datetime import date

import pandas as pd

import app.tasks.cache_tasks as cache_tasks
from app.services.benchmark_cache_service import BenchmarkCacheService
from app.services.cache_manager import BenchmarkWarmResult, CacheManager


def _ohlcv_frame(days: list[str]) -> pd.DataFrame:
    closes = [100.0 + index for index in range(len(days))]
    return pd.DataFrame(
        {"Open": closes, "High": closes, "Low": closes, "Close": closes, "Volume": [1_000_000] * len(days)},
        index=pd.to_datetime(days),
    )


class _FakeCalendar:
    def __init__(self, last_completed: date):
        self._last_completed = last_completed

    def last_completed_trading_day(self, _market):
        return self._last_completed


def _cache_manager(monkeypatch, *, fetched: pd.DataFrame, last_completed: date) -> CacheManager:
    service = BenchmarkCacheService(redis_client=None, session_factory=lambda: None)
    service._redis_client = None
    service._get_from_database = lambda **_kwargs: None  # type: ignore[assignment]
    service._fetch_from_yfinance = lambda benchmark_symbol, period: fetched  # type: ignore[assignment]
    service._store_in_database = lambda **_kwargs: None  # type: ignore[assignment]
    monkeypatch.setattr(
        "app.wiring.bootstrap.get_market_calendar_service", lambda: _FakeCalendar(last_completed),
    )
    manager = object.__new__(CacheManager)
    manager.benchmark_cache = service
    return manager


def test_warm_rejects_a_fetch_that_ends_before_the_last_completed_session(monkeypatch):
    manager = _cache_manager(
        monkeypatch,
        fetched=_ohlcv_frame(["2026-03-12", "2026-03-13"]),
        last_completed=date(2026, 3, 16),
    )

    assert manager.warm_benchmark_cache(period="2y", market="US") == BenchmarkWarmResult(
        ok=False,
        required_date=date(2026, 3, 16),
        latest_date=date(2026, 3, 13),
        reason="benchmark_not_current",
    )


def test_warm_accepts_a_fetch_that_ends_on_the_last_completed_session(monkeypatch):
    manager = _cache_manager(
        monkeypatch,
        fetched=_ohlcv_frame(["2026-03-13", "2026-03-16"]),
        last_completed=date(2026, 3, 16),
    )

    assert manager.warm_benchmark_cache(period="2y", market="US") == BenchmarkWarmResult(
        ok=True,
        required_date=date(2026, 3, 16),
        latest_date=date(2026, 3, 16),
        reason=None,
    )


def test_warm_mid_session_requires_only_the_prior_session(monkeypatch):
    # The 2026-03-17 session is in progress, so the calendar's last completed
    # session is 2026-03-16: a frame carrying that bar (and today's partial
    # bar) warms, one that stops a session earlier does not.
    with_partial_bar = _cache_manager(
        monkeypatch,
        fetched=_ohlcv_frame(["2026-03-16", "2026-03-17"]),
        last_completed=date(2026, 3, 16),
    )
    a_session_short = _cache_manager(
        monkeypatch,
        fetched=_ohlcv_frame(["2026-03-12", "2026-03-13"]),
        last_completed=date(2026, 3, 16),
    )

    assert with_partial_bar.warm_benchmark_cache(period="1y", market="US") == BenchmarkWarmResult(
        ok=True,
        required_date=date(2026, 3, 16),
        latest_date=date(2026, 3, 16),
        reason=None,
    )
    assert a_session_short.warm_benchmark_cache(period="1y", market="US").ok is False


def test_warm_reports_a_calendar_failure_instead_of_warming_blind(monkeypatch):
    class _BrokenCalendar:
        def last_completed_trading_day(self, _market):
            raise RuntimeError("coverage expired")

    manager = _cache_manager(
        monkeypatch,
        fetched=_ohlcv_frame(["2026-03-16"]),
        last_completed=date(2026, 3, 16),
    )
    monkeypatch.setattr("app.wiring.bootstrap.get_market_calendar_service", lambda: _BrokenCalendar())

    assert manager.warm_benchmark_cache(period="2y", market="US") == BenchmarkWarmResult(
        ok=False,
        required_date=None,
        latest_date=None,
        reason="error: coverage expired",
    )


class _FakeRedis:
    def __init__(self):
        self.stored: dict[str, bytes] = {}

    def set(self, *_args, **_kwargs):
        return True

    def delete(self, _key):
        return 1

    def setex(self, key, _ttl, value):
        self.stored[key] = value


def test_a_fetch_short_of_the_required_date_is_not_cached_in_redis():
    stale, current = _FakeRedis(), _FakeRedis()
    for redis, fetched in (
        (stale, _ohlcv_frame(["2026-03-12", "2026-03-13"])),
        (current, _ohlcv_frame(["2026-03-13", "2026-03-16"])),
    ):
        service = BenchmarkCacheService(redis_client=redis, session_factory=lambda: None)
        service._fetch_from_yfinance = lambda benchmark_symbol, period, frame=fetched: frame  # type: ignore[assignment]
        service._store_in_database = lambda **_kwargs: None  # type: ignore[assignment]
        service._fetch_and_cache_benchmark("SPY", "US", "2y", required_as_of_date=date(2026, 3, 16))

    assert stale.stored == {}
    assert list(current.stored) == ["benchmark:US:SPY:2y"]


def _warm_spy_cache_with(monkeypatch, outcome: BenchmarkWarmResult):
    class _FakeCacheManager:
        @staticmethod
        def warm_benchmark_cache(*, period, market):
            return outcome

    class _FakeSession:
        @staticmethod
        def close():
            return None

    recorded: list[tuple[str, str]] = []

    class _FakeTelemetry:
        def record_benchmark_age(self, market, *, benchmark_symbol):
            recorded.append((market, benchmark_symbol))

    monkeypatch.setattr(cache_tasks, "SessionLocal", lambda: _FakeSession())
    monkeypatch.setattr(cache_tasks, "CacheManager", lambda: _FakeCacheManager())
    monkeypatch.setattr(cache_tasks, "_active_benchmark_markets", lambda _db, scope_market=None: ["US"])
    monkeypatch.setattr(cache_tasks, "format_market_status", lambda: "closed")
    monkeypatch.setattr("app.services.telemetry.get_telemetry", lambda: _FakeTelemetry())
    return cache_tasks.warm_spy_cache(market="US"), recorded


def test_warm_spy_cache_reports_the_stale_reason_and_keeps_benchmark_age(monkeypatch):
    result, recorded = _warm_spy_cache_with(
        monkeypatch,
        BenchmarkWarmResult(
            ok=False,
            required_date=date(2026, 3, 16),
            latest_date=date(2026, 3, 13),
            reason="benchmark_not_current",
        ),
    )

    assert result["by_market"] == {"US": {"2y": False, "1y": False}}
    assert result["error"] == (
        "Benchmark warm failed for "
        "US:2y (benchmark_not_current: latest 2026-03-13, required 2026-03-16), "
        "US:1y (benchmark_not_current: latest 2026-03-13, required 2026-03-16)"
    )
    assert recorded == []


def test_warm_spy_cache_records_benchmark_age_after_a_current_warm(monkeypatch):
    result, recorded = _warm_spy_cache_with(
        monkeypatch,
        BenchmarkWarmResult(ok=True, required_date=date(2026, 3, 16), latest_date=date(2026, 3, 16)),
    )

    assert result["by_market"] == {"US": {"2y": True, "1y": True}}
    assert "error" not in result
    assert recorded == [("US", "SPY")]
