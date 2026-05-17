from app.domain.markets.registry import market_registry
from app.domain.common.benchmarks import supported_benchmark_markets
from app.services.benchmark_registry_service import benchmark_registry


def test_registry_versioned_mapping_exists_for_all_markets():
    table = benchmark_registry.mapping_table()

    assert benchmark_registry.TABLE_VERSION == "2026-05-17.v1"
    assert set(table.keys()) == set(market_registry.supported_market_codes())


def test_supported_benchmark_markets_match_market_registry():
    assert tuple(supported_benchmark_markets()) == market_registry.supported_market_codes()


def test_non_us_candidates_have_no_spy_leakage():
    for market in ("HK", "IN", "JP", "KR", "TW", "CN", "SG"):
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


def test_china_registry_uses_csi_300_with_shanghai_composite_fallback():
    entry = benchmark_registry.get_entry("CN")

    assert entry.primary_symbol == "000300.SS"
    assert entry.fallback_symbol == "000001.SS"
