from app.tasks.cache_tasks import _active_benchmark_markets
import app.tasks.cache_tasks as cache_tasks


class _FakeQuery:
    def __init__(self, rows):
        self._rows = rows

    def filter(self, *_args, **_kwargs):
        return self

    def all(self):
        return self._rows


class _FakeDB:
    def __init__(self, rows):
        self._rows = rows

    def query(self, *_args, **_kwargs):
        return _FakeQuery(self._rows)


def test_active_benchmark_markets_infers_non_us_when_market_blank():
    db = _FakeDB(
        [
            ("", "XHKG", "0700"),
            (None, "XTKS", "9984"),
            ("US", "NASDAQ", "AAPL"),
        ]
    )

    markets = _active_benchmark_markets(db)

    assert markets == ["HK", "JP", "US"]


def test_active_benchmark_markets_always_includes_us():
    db = _FakeDB([("HK", "XHKG", "0700.HK")])

    markets = _active_benchmark_markets(db)

    assert markets == ["HK", "US"]


def test_warm_spy_cache_returns_error_when_any_market_warm_fails(monkeypatch):
    from datetime import date

    from app.services.cache_manager import BenchmarkWarmResult

    class _FakeCacheManager:
        @staticmethod
        def warm_benchmark_cache(*, period, market):
            if market == "HK" and period == "1y":
                return BenchmarkWarmResult(
                    ok=False, required_date=date(2026, 3, 16), reason="no_benchmark_data",
                )
            return BenchmarkWarmResult(ok=True, required_date=date(2026, 3, 16), latest_date=date(2026, 3, 16))

    class _FakeSession:
        @staticmethod
        def close():
            return None

    monkeypatch.setattr(cache_tasks, "SessionLocal", lambda: _FakeSession())
    monkeypatch.setattr(cache_tasks, "CacheManager", lambda: _FakeCacheManager())
    monkeypatch.setattr(cache_tasks, "_active_benchmark_markets", lambda _db, scope_market=None: ["HK", "US"])
    monkeypatch.setattr(cache_tasks, "format_market_status", lambda: "closed")

    result = cache_tasks.warm_spy_cache()

    assert result["error"] == "Benchmark warm failed for HK:1y (no_benchmark_data: latest none, required 2026-03-16)"
    assert "by_market" in result
    assert result["by_market"]["US"]["2y"] is True
