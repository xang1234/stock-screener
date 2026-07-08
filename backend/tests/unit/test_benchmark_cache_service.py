import pandas as pd
from datetime import date, datetime
from pathlib import Path

from app.services.benchmark_cache_service import (
    BenchmarkCacheService,
    BenchmarkCandidateOutcome,
    BenchmarkCandidateSource,
    BenchmarkFallbackPolicy,
)
import app.services.benchmark_cache_service as benchmark_cache_module
from app.services.benchmark_registry_service import benchmark_registry
from app.services.benchmark_resolution import BenchmarkResolver


def test_benchmark_cache_service_stays_below_giant_file_threshold():
    service_path = Path(benchmark_cache_module.__file__)

    assert len(service_path.read_text(encoding="utf-8").splitlines()) < 1_000


def _ohlcv_frame(closes: list[float], days: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Open": closes,
            "High": closes,
            "Low": closes,
            "Close": closes,
            "Volume": [1_000_000] * len(closes),
        },
        index=pd.to_datetime(days),
    )


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
    data = _ohlcv_frame([100.0], ["2026-04-10"])

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


def test_fetch_and_cache_benchmark_cleans_non_finite_rows_before_cache_db_and_return():
    class FakeRedis:
        def __init__(self):
            self.deleted = []

        def set(self, *_args, **_kwargs):
            return True

        def delete(self, key):
            self.deleted.append(key)

    service = BenchmarkCacheService(redis_client=FakeRedis(), session_factory=lambda: None)
    raw = _ohlcv_frame([100.0, float("nan")], ["2026-04-10", "2026-04-11"])
    captured = {}

    service._fetch_from_yfinance = lambda benchmark_symbol, period: raw  # type: ignore[assignment]
    service._store_in_redis = lambda **kwargs: captured.setdefault("redis", kwargs["data"])  # type: ignore[assignment]
    service._store_in_database = lambda **kwargs: captured.setdefault("db", kwargs["data"])  # type: ignore[assignment]

    result = service._fetch_and_cache_benchmark("SPY", "US", "2y")

    assert result is not None
    assert result["Close"].tolist() == [100.0]
    assert captured["redis"]["Close"].tolist() == [100.0]
    assert captured["db"]["Close"].tolist() == [100.0]


def test_get_benchmark_data_uses_fallback_when_primary_fails():
    service = BenchmarkCacheService(redis_client=None, session_factory=lambda: None)
    service._redis_client = None

    calls = []
    fallback_df = _ohlcv_frame([1.0], ["2026-04-10"])

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


def test_get_benchmark_bundle_uses_current_fallback_when_primary_fetch_is_stale():
    service = BenchmarkCacheService(redis_client=None, session_factory=lambda: None)
    service._redis_client = None

    calls = []
    stale_primary_df = _ohlcv_frame([1.0], ["2026-07-03"])
    current_fallback_df = _ohlcv_frame([2.0], ["2026-07-07"])

    def fake_fetch(symbol, period):
        calls.append(symbol)
        if symbol == "000300.SS":
            return stale_primary_df
        if symbol == "000001.SS":
            return current_fallback_df
        return pd.DataFrame()

    service._fetch_from_yfinance = fake_fetch  # type: ignore[assignment]
    service._store_in_database = lambda **kwargs: None  # type: ignore[assignment]

    bundle = service.get_benchmark_bundle(
        market="CN",
        period="2y",
        force_refresh=True,
        required_as_of_date=date(2026, 7, 7),
    )

    assert calls == ["000300.SS", "000001.SS"]
    assert bundle is not None
    assert bundle.benchmark_symbol == "000001.SS"
    assert bundle.benchmark_role == "fallback"
    assert bundle.data is current_fallback_df


def test_resolve_benchmark_bundle_reports_typed_stale_candidates_without_side_channel():
    service = BenchmarkCacheService(redis_client=None, session_factory=lambda: None)
    service._redis_client = None

    stale_primary_df = _ohlcv_frame([1.0], ["2026-07-03"])

    def fake_fetch(symbol, period):
        if symbol == "000300.SS":
            return stale_primary_df
        return pd.DataFrame()

    service._fetch_from_yfinance = fake_fetch  # type: ignore[assignment]
    service._store_in_database = lambda **kwargs: None  # type: ignore[assignment]

    resolution = service.resolve_benchmark_bundle(
        market="CN",
        period="2y",
        force_refresh=True,
        required_as_of_date=date(2026, 7, 7),
    )

    assert resolution.bundle is None
    assert resolution.error == "benchmark_not_current"
    assert not hasattr(service, "last_candidate_statuses")
    assert resolution.candidate_statuses[0].symbol == "000300.SS"
    assert resolution.candidate_statuses[0].source is BenchmarkCandidateSource.FETCH
    assert resolution.candidate_statuses[0].outcome is BenchmarkCandidateOutcome.STALE_REQUIRED_DATE
    assert resolution.candidate_statuses[0].as_diagnostic() == {
        "symbol": "000300.SS",
        "role": "primary",
        "source": "fetch",
        "status": "stale_required_date",
        "latest_date": "2026-07-03",
    }


def test_benchmark_resolver_uses_public_adapter_contract():
    fallback_df = _ohlcv_frame([1.0], ["2026-04-10"])

    class PublicOnlyAdapter:
        def __init__(self):
            self.fetch_calls = []
            self.redis_writes = []

        def load_benchmark_from_redis(self, *, benchmark_symbol, period, market):
            return None

        def load_benchmark_from_database(self, *, benchmark_symbol, period, market):
            if benchmark_symbol == "2800.HK":
                return fallback_df
            return None

        def benchmark_data_is_fresh(self, data, market="US", max_age_hours=24):
            return True

        def store_benchmark_in_redis(self, *, benchmark_symbol, period, data, market):
            self.redis_writes.append((benchmark_symbol, period, data, market))

        def fetch_and_cache_benchmark(self, *, benchmark_symbol, market, period):
            self.fetch_calls.append(benchmark_symbol)
            return pd.DataFrame()

    adapter = PublicOnlyAdapter()

    resolution = BenchmarkResolver(adapter=adapter, registry=benchmark_registry).resolve(
        market="HK",
        period="2y",
        force_refresh=False,
    )

    assert resolution.bundle is not None
    assert resolution.bundle.benchmark_symbol == "2800.HK"
    assert resolution.bundle.data is fallback_df
    assert adapter.redis_writes == [("2800.HK", "2y", fallback_df, "HK")]
    assert adapter.fetch_calls == []


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


def test_get_benchmark_bundle_primary_only_policy_fetches_primary():
    service = BenchmarkCacheService(redis_client=None, session_factory=lambda: None)
    service._redis_client = None

    cached_fallback_df = pd.DataFrame({"Close": [1.0]}, index=pd.to_datetime(["2026-04-10"]))
    fetched_primary_df = pd.DataFrame({"Close": [2.0]}, index=pd.to_datetime(["2026-04-10"]))
    fetch_calls = []

    def fake_get_from_db(*, benchmark_symbol, period, market):
        if benchmark_symbol == "IVV":
            return cached_fallback_df
        return None

    def fake_fetch(*, benchmark_symbol, market, period):
        fetch_calls.append(benchmark_symbol)
        if benchmark_symbol == "SPY":
            return fetched_primary_df
        return pd.DataFrame()

    service._get_from_database = fake_get_from_db  # type: ignore[assignment]
    service._is_data_fresh = lambda data, market="US", max_age_hours=24: True  # type: ignore[assignment]
    service._fetch_and_cache_benchmark = fake_fetch  # type: ignore[assignment]

    bundle = service.get_benchmark_bundle(
        market="US",
        period="2y",
        fallback_policy=BenchmarkFallbackPolicy.PRIMARY_ONLY,
    )

    assert bundle is not None
    assert bundle.benchmark_symbol == "SPY"
    assert bundle.benchmark_role == "primary"
    assert bundle.candidate_symbols == ("SPY",)
    assert bundle.data is fetched_primary_df
    assert fetch_calls == ["SPY"]


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
