import pandas as pd
from datetime import datetime

from app.services.benchmark_cache_service import BenchmarkCacheService
import app.services.benchmark_cache_service as benchmark_cache_module


def test_get_benchmark_symbol_supports_all_markets():
    service = BenchmarkCacheService(redis_client=None, session_factory=lambda: None)

    assert service.get_benchmark_symbol("US") == "SPY"
    assert service.get_benchmark_symbol("HK") == "^HSI"
    assert service.get_benchmark_symbol("JP") == "^N225"
    assert service.get_benchmark_symbol("TW") == "^TWII"


def test_non_us_candidates_do_not_include_spy():
    service = BenchmarkCacheService(redis_client=None, session_factory=lambda: None)

    assert "SPY" not in service.get_benchmark_candidates("HK")
    assert "SPY" not in service.get_benchmark_candidates("JP")
    assert "SPY" not in service.get_benchmark_candidates("TW")


def test_get_spy_data_delegates_to_market_api():
    service = BenchmarkCacheService(redis_client=None, session_factory=lambda: None)

    captured = {}

    def fake_get_benchmark_data(*, market, period, force_refresh):
        captured["market"] = market
        captured["period"] = period
        captured["force_refresh"] = force_refresh
        return "ok"

    service.get_benchmark_data = fake_get_benchmark_data  # type: ignore[assignment]
    result = service.get_spy_data(period="1y", force_refresh=True)

    assert result == "ok"
    assert captured == {"market": "US", "period": "1y", "force_refresh": True}


def test_market_scoped_redis_keys_are_deterministic():
    service = BenchmarkCacheService(redis_client=None, session_factory=lambda: None)

    data_key = service._redis_data_key("^N225", "2y", market="JP")
    lock_key = service._redis_lock_key("^N225", "2y", market="JP")

    assert data_key == "benchmark:JP:^N225:2y"
    assert lock_key == "benchmark:JP:^N225:2y:lock"


def test_fetch_and_cache_benchmark_without_redis_fetches_directly_and_persists():
    service = BenchmarkCacheService(redis_client=None, session_factory=lambda: None)
    service._redis_client = None

    calls = {"wait": 0, "store_db": 0}
    data = pd.DataFrame({"Close": [100.0]}, index=pd.to_datetime(["2026-04-10"]))

    def fail_if_wait(*args, **kwargs):
        calls["wait"] += 1
        raise AssertionError("wait path must not run when redis is unavailable")

    def fake_store_db(*, benchmark_symbol, data):
        calls["store_db"] += 1
        assert benchmark_symbol == "^HSI"
        assert not data.empty

    service._wait_for_cache = fail_if_wait  # type: ignore[assignment]
    service._store_in_database = fake_store_db  # type: ignore[assignment]
    service._fetch_from_yfinance = lambda benchmark_symbol, period: data  # type: ignore[assignment]

    result = service._fetch_and_cache_benchmark("^HSI", "HK", "2y")

    assert result is data
    assert calls["wait"] == 0
    assert calls["store_db"] == 1


def test_get_benchmark_data_uses_fallback_when_primary_fails():
    service = BenchmarkCacheService(redis_client=None, session_factory=lambda: None)
    service._redis_client = None

    calls = []
    fallback_df = pd.DataFrame({"Close": [1.0]}, index=pd.to_datetime(["2026-04-10"]))

    def fake_fetch(symbol, period):
        calls.append(symbol)
        if symbol == "^HSI":
            return pd.DataFrame()
        if symbol == "2800.HK":
            return fallback_df
        return pd.DataFrame()

    service._fetch_from_yfinance = fake_fetch  # type: ignore[assignment]
    service._store_in_database = lambda **kwargs: None  # type: ignore[assignment]

    result = service.get_benchmark_data(market="HK", period="2y", force_refresh=True)

    assert calls[:2] == ["^HSI", "2800.HK"]
    assert result is fallback_df


def test_get_benchmark_data_prefers_cached_fallback_before_primary_network_fetch():
    service = BenchmarkCacheService(redis_client=None, session_factory=lambda: None)
    service._redis_client = None

    fallback_df = pd.DataFrame({"Close": [1.0]}, index=pd.to_datetime(["2026-04-10"]))
    calls = []

    def fake_get_from_db(*, benchmark_symbol, period, market):
        if benchmark_symbol == "2800.HK":
            return fallback_df
        return None

    def fake_fetch(*, benchmark_symbol, market, period):
        calls.append(benchmark_symbol)
        return pd.DataFrame()

    service._get_from_database = fake_get_from_db  # type: ignore[assignment]
    service._is_data_fresh = lambda data, market="US", max_age_hours=24: True  # type: ignore[assignment]
    service._fetch_and_cache_benchmark = fake_fetch  # type: ignore[assignment]

    result = service.get_benchmark_data(market="HK", period="2y", force_refresh=False)

    assert result is fallback_df
    assert calls == []


def test_get_benchmark_data_skips_stale_redis_hit():
    service = BenchmarkCacheService(redis_client=None, session_factory=lambda: None)
    service._redis_client = None

    stale_df = pd.DataFrame({"Close": [1.0]}, index=pd.to_datetime(["2026-04-01"]))
    fresh_df = pd.DataFrame({"Close": [2.0]}, index=pd.to_datetime(["2026-04-10"]))

    def fake_get_from_redis(*, benchmark_symbol, period, market="US"):
        assert market == "HK"
        if benchmark_symbol == "^HSI":
            return stale_df
        return None

    def fake_get_from_db(*, benchmark_symbol, period, market):
        if benchmark_symbol == "2800.HK":
            return fresh_df
        return None

    def fake_is_fresh(data, market="US", max_age_hours=24):
        return data is fresh_df

    service._get_from_redis = fake_get_from_redis  # type: ignore[assignment]
    service._get_from_database = fake_get_from_db  # type: ignore[assignment]
    service._is_data_fresh = fake_is_fresh  # type: ignore[assignment]

    result = service.get_benchmark_data(market="HK", period="2y", force_refresh=False)

    assert result is fresh_df


def test_is_data_fresh_fallback_allows_weekend_without_calendar(monkeypatch):
    service = BenchmarkCacheService(redis_client=None, session_factory=lambda: None)
    data = pd.DataFrame({"Close": [100.0]}, index=pd.to_datetime(["2026-04-10"]))  # Friday
    service._market_calendar.last_completed_trading_day = lambda market: None  # type: ignore[method-assign]

    monkeypatch.setattr(
        pd.Timestamp,
        "utcnow",
        lambda: pd.Timestamp(datetime(2026, 4, 12, 12, 0), tz="UTC"),
    )

    assert service._is_data_fresh(data, market="HK") is True


def test_is_data_fresh_uses_us_market_hours_fallback_when_calendar_unavailable(monkeypatch):
    service = BenchmarkCacheService(redis_client=None, session_factory=lambda: None)
    data = pd.DataFrame({"Close": [100.0]}, index=pd.to_datetime(["2026-04-10"]))  # Friday
    service._market_calendar.last_completed_trading_day = lambda market: (_ for _ in ()).throw(RuntimeError("no calendar"))  # type: ignore[method-assign]

    monkeypatch.setattr(
        benchmark_cache_module,
        "get_eastern_now",
        lambda: datetime.fromisoformat("2026-04-13T10:00:00-04:00"),  # Monday, market open
    )
    monkeypatch.setattr(benchmark_cache_module, "is_market_open", lambda _dt=None: True)
    monkeypatch.setattr(
        benchmark_cache_module,
        "get_last_trading_day",
        lambda d=None: pd.Timestamp("2026-04-10").date(),
    )
    monkeypatch.setattr(benchmark_cache_module, "is_trading_day", lambda d=None: True)

    assert service._is_data_fresh(data, market="US") is True


def test_is_data_fresh_us_fallback_premarket_uses_previous_trading_day(monkeypatch):
    service = BenchmarkCacheService(redis_client=None, session_factory=lambda: None)
    data = pd.DataFrame({"Close": [100.0]}, index=pd.to_datetime(["2026-04-10"]))  # Friday close
    service._market_calendar.last_completed_trading_day = lambda market: (_ for _ in ()).throw(RuntimeError("no calendar"))  # type: ignore[method-assign]

    monkeypatch.setattr(
        benchmark_cache_module,
        "get_eastern_now",
        lambda: datetime.fromisoformat("2026-04-13T08:00:00-04:00"),  # Monday pre-market
    )
    monkeypatch.setattr(benchmark_cache_module, "is_market_open", lambda _dt=None: False)
    monkeypatch.setattr(benchmark_cache_module, "is_trading_day", lambda d=None: True)
    monkeypatch.setattr(
        benchmark_cache_module,
        "get_last_trading_day",
        lambda d=None: pd.Timestamp("2026-04-10").date(),
    )

    assert service._is_data_fresh(data, market="US") is True


def test_is_data_fresh_us_fallback_after_close_requires_same_day(monkeypatch):
    service = BenchmarkCacheService(redis_client=None, session_factory=lambda: None)
    stale_data = pd.DataFrame({"Close": [100.0]}, index=pd.to_datetime(["2026-04-10"]))  # Friday close
    service._market_calendar.last_completed_trading_day = lambda market: (_ for _ in ()).throw(RuntimeError("no calendar"))  # type: ignore[method-assign]

    monkeypatch.setattr(
        benchmark_cache_module,
        "get_eastern_now",
        lambda: datetime.fromisoformat("2026-04-13T17:30:00-04:00"),  # Monday after close buffer
    )
    monkeypatch.setattr(benchmark_cache_module, "is_market_open", lambda _dt=None: False)
    monkeypatch.setattr(benchmark_cache_module, "is_trading_day", lambda d=None: True)
    monkeypatch.setattr(
        benchmark_cache_module,
        "get_last_trading_day",
        lambda d=None: pd.Timestamp("2026-04-10").date(),
    )

    assert service._is_data_fresh(stale_data, market="US") is False
