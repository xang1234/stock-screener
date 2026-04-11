from app.services.benchmark_cache_service import BenchmarkCacheService


def test_get_benchmark_symbol_supports_all_markets():
    service = BenchmarkCacheService(redis_client=None, session_factory=lambda: None)

    assert service.get_benchmark_symbol("US") == "SPY"
    assert service.get_benchmark_symbol("HK") == "^HSI"
    assert service.get_benchmark_symbol("JP") == "^N225"
    assert service.get_benchmark_symbol("TW") == "^TWII"


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

    data_key = service._redis_data_key("^N225", "2y")
    lock_key = service._redis_lock_key("^N225", "2y")

    assert data_key == "benchmark:^N225:2y"
    assert lock_key == "benchmark:^N225:2y:lock"
