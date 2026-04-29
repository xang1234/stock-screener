from app.services.benchmark_registry_service import benchmark_registry


def test_registry_versioned_mapping_exists_for_all_markets():
    table = benchmark_registry.mapping_table()

    assert benchmark_registry.TABLE_VERSION == "2026-04-29.v1"
    assert set(table.keys()) == {"US", "HK", "IN", "JP", "KR", "TW"}


def test_non_us_candidates_have_no_spy_leakage():
    for market in ("HK", "IN", "JP", "KR", "TW"):
        candidates = benchmark_registry.get_candidate_symbols(market)
        assert "SPY" not in candidates
        assert len(candidates) >= 1


def test_india_registry_uses_nifty_index_with_etf_fallback():
    entry = benchmark_registry.get_entry("IN")

    assert entry.primary_symbol == "^NSEI"
    assert entry.fallback_symbol == "NIFTYBEES.NS"


def test_korea_registry_uses_kospi_index_with_etf_fallback():
    entry = benchmark_registry.get_entry("KR")

    assert entry.primary_symbol == "^KS11"
    assert entry.fallback_symbol == "069500.KS"
