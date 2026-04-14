from app.tasks.cache_tasks import _active_benchmark_markets, _benchmark_markets_for_symbols
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


def test_benchmark_markets_for_symbols_infers_market_from_symbol_suffix():
    db = _FakeDB(
        [
            ("", None, "3008.TWO"),
            ("", None, "2330.TW"),
        ]
    )

    markets = _benchmark_markets_for_symbols(db, symbols=["3008.TWO", "2330.TW"])

    assert markets == ["TW"]


def test_benchmark_markets_for_symbols_infers_market_when_db_has_no_rows():
    db = _FakeDB([])

    markets = _benchmark_markets_for_symbols(db, symbols=["0700.HK"])

    assert markets == ["HK"]


def test_active_benchmark_markets_always_includes_us():
    db = _FakeDB([("HK", "XHKG", "0700.HK")])

    markets = _active_benchmark_markets(db)

    assert markets == ["HK", "US"]


def test_warm_spy_cache_returns_error_when_any_market_warm_fails(monkeypatch):
    class _FakeCacheManager:
        @staticmethod
        def warm_benchmark_cache(*, period, market):
            return not (market == "HK" and period == "1y")

    class _FakeSession:
        @staticmethod
        def close():
            return None

    monkeypatch.setattr(cache_tasks, "SessionLocal", lambda: _FakeSession())
    monkeypatch.setattr(cache_tasks, "CacheManager", lambda: _FakeCacheManager())
    monkeypatch.setattr(cache_tasks, "_active_benchmark_markets", lambda _db, scope_market=None: ["HK", "US"])
    monkeypatch.setattr(cache_tasks, "format_market_status", lambda: "closed")

    result = cache_tasks.warm_spy_cache()

    assert "error" in result
    assert "HK:1y" in result["error"]
    assert "by_market" in result
    assert result["by_market"]["US"]["2y"] is True
