from app.services.benchmark_registry_service import benchmark_registry


def test_registry_versioned_mapping_exists_for_all_markets():
    table = benchmark_registry.mapping_table()

    assert benchmark_registry.TABLE_VERSION == "2026-04-11.v1"
    assert set(table.keys()) == {"US", "HK", "JP", "TW"}


def test_non_us_candidates_have_no_spy_leakage():
    for market in ("HK", "JP", "TW"):
        candidates = benchmark_registry.get_candidate_symbols(market)
        assert "SPY" not in candidates
        assert len(candidates) >= 1
