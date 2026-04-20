"""E6 Market Normalization regression harness (T6.5).

Two axes of guarantee:

1. **US parity** — the expansion to HK/JP/TW must not change any US-path
   behaviour. Existing US call sites resolve to the same benchmark, the
   same RS semantics, and the same scope tags they produced before E6.
2. **Non-US correctness** — HK/JP/TW call sites reach the correct
   market benchmark, produce per-market RS percentiles, and have their
   USD-normalised liquidity filters applied consistently.

Implemented as *golden fixtures* (exact-match on stable strings) plus
numeric tolerances (``pytest.approx``) where computed floats are
involved. If an E-series change moves a benchmark symbol, rewrites a
scope reason, or re-partitions RS universes, this harness fires first.
"""
from __future__ import annotations

import pandas as pd
import pytest

from app.domain.analytics.scope import (
    POLICY_VERSION as ANALYTICS_POLICY_VERSION,
    AnalyticsFeature,
    UnsupportedMarketError,
    market_scope_tag,
    require_us_scope,
    us_only_tag,
)
from app.domain.scanning.mixed_market_policy import (
    POLICY_VERSION as MIXED_MARKET_POLICY_VERSION,
    REASON_MISSING_CAP_USD,
    UNIT_NATIVE,
    UNIT_SHARES,
    UNIT_USD,
    is_mixed_market,
    resolve_adv_for_filter,
    resolve_cap_for_filter,
)
from app.scanners.base_screener import StockData
from app.scanners.custom_scanner import CustomScanner
from app.scanners.data_preparation import DataPreparationLayer
from app.services.benchmark_registry_service import benchmark_registry


# ---------------------------------------------------------------------------
# Golden fixtures — pin the cross-market contract to stable values.
# ---------------------------------------------------------------------------

# Benchmark symbol table: source of truth for market → primary ETF/index.
EXPECTED_PRIMARY_BENCHMARK = {
    "US": "SPY",
    "HK": "^HSI",
    "JP": "^N225",
    "TW": "^TWII",
}

# Scope reasons per feature. Exact string match — any rewording must be
# deliberate and bumps the policy version.
EXPECTED_SCOPE_REASON = {
    AnalyticsFeature.THEME_DISCOVERY:
        "theme content sources are English-language biased; no non-US coverage",
    AnalyticsFeature.BREADTH_SNAPSHOT:
        "breadth indicators are computed from the US universe only",
}


# ---------------------------------------------------------------------------
# US parity — pre-E6 behaviour must be unchanged.
# ---------------------------------------------------------------------------


class TestUSParityBenchmark:
    """T6.1 / T3.2: US resolves to SPY; ``normalize_market(None)`` stays US."""

    def test_us_primary_is_spy(self):
        assert benchmark_registry.get_primary_symbol("US") == "SPY"

    def test_none_defaults_to_us(self):
        assert benchmark_registry.normalize_market(None) == "US"

    def test_us_candidates_are_stable(self):
        # Frontend charts and cache keys depend on this ordering.
        assert benchmark_registry.get_candidate_symbols("US") == ["SPY", "IVV"]


class TestUSParityMixedMarketPolicy:
    """T6.3: single-market US scans use native (==USD) columns, no FX drift."""

    def test_all_us_is_not_mixed(self):
        assert is_mixed_market(["US", "US", None]) is False

    def test_single_market_cap_uses_native_column(self):
        # For a US row, native == USD numerically; policy must still
        # select the native column in single-market mode (legacy contract).
        us_fundamentals = {"market_cap": 500_000_000_000, "market_cap_usd": 500_000_000_000}
        assert resolve_cap_for_filter(us_fundamentals, mixed_market=False) == pytest.approx(500_000_000_000)

    def test_single_market_adv_uses_share_volume(self):
        adv = resolve_adv_for_filter(
            {"adv_usd": 9_999},  # ignored in single-market mode
            native_avg_volume=1_000_000,
            mixed_market=False,
        )
        assert adv == pytest.approx(1_000_000)

    def test_scanner_us_only_marks_not_mixed(self):
        # _detect_and_set_mixed_market_flag runs unconditionally (even
        # when needs_benchmark=False) and must mark pure-US scans as
        # is_mixed_market=False (pre-E6 parity contract).
        prep = DataPreparationLayer.__new__(DataPreparationLayer)
        results = {
            s: StockData(
                symbol=s,
                price_data=pd.DataFrame(),
                benchmark_data=pd.DataFrame(),
                market="US",
            )
            for s in ("AAPL", "MSFT", "NVDA")
        }
        prep._detect_and_set_mixed_market_flag(results)
        assert all(item.is_mixed_market is False for item in results.values())


class TestUSParityAnalyticsScope:
    """T6.4: US-only analytics features carry stable scope tags."""

    @pytest.mark.parametrize(
        "feature",
        [AnalyticsFeature.THEME_DISCOVERY, AnalyticsFeature.BREADTH_SNAPSHOT],
    )
    def test_us_tag_shape_is_stable(self, feature: AnalyticsFeature):
        tag = us_only_tag(feature)
        assert tag["market_scope"] == "US"
        assert tag["scope_reason"] == EXPECTED_SCOPE_REASON[feature]
        # policy_version must live on its own accessor, not inside the tag,
        # so the tag spreads cleanly into Pydantic response models.
        assert "policy_version" not in tag

    @pytest.mark.parametrize("market", [None, "US", "us", " US "])
    def test_us_markets_pass_require_scope(self, market):
        # Defensive: US variants (case, whitespace, None) must not raise.
        for feature in (AnalyticsFeature.THEME_DISCOVERY, AnalyticsFeature.BREADTH_SNAPSHOT):
            require_us_scope(market, feature)

    def test_group_rank_scope_is_market_aware(self):
        tag = market_scope_tag("HK")
        assert tag == {"market_scope": "HK"}
        require_us_scope("HK", AnalyticsFeature.IBD_GROUP_RANK)


class TestUSParityScanner:
    """Custom scanner's cap/volume filters treat US as native (legacy)."""

    def test_us_cap_filter_uses_market_cap(self):
        scanner = CustomScanner()
        data = _build_stock_data(
            symbol="AAPL",
            market="US",
            is_mixed=False,
            fundamentals={"market_cap": 2_500_000_000_000, "market_cap_usd": 2_500_000_000_000},
            volume=50_000_000,
        )
        result = scanner.scan_stock("AAPL", data, {"custom_filters": {"market_cap_min": 1_000_000_000}})
        cap = result.details["filter_results"]["market_cap"]
        assert cap["passes"] is True
        assert cap["unit"] == UNIT_NATIVE

    def test_us_volume_filter_uses_shares(self):
        scanner = CustomScanner()
        data = _build_stock_data(
            symbol="AAPL",
            market="US",
            is_mixed=False,
            fundamentals={"adv_usd": 1},  # ignored for single-market US
            volume=50_000_000,
        )
        result = scanner.scan_stock("AAPL", data, {"custom_filters": {"volume_min": 10_000_000}})
        vol = result.details["filter_results"]["volume"]
        assert vol["passes"] is True
        assert vol["unit"] == UNIT_SHARES


# ---------------------------------------------------------------------------
# Non-US correctness — HK/JP/TW resolve and compute correctly.
# ---------------------------------------------------------------------------


class TestNonUSBenchmark:
    """T6.1 / T3.2: HK/JP/TW resolve to expected market benchmarks."""

    @pytest.mark.parametrize("market,expected", [
        ("HK", EXPECTED_PRIMARY_BENCHMARK["HK"]),
        ("JP", EXPECTED_PRIMARY_BENCHMARK["JP"]),
        ("TW", EXPECTED_PRIMARY_BENCHMARK["TW"]),
    ])
    def test_primary_symbol(self, market, expected):
        assert benchmark_registry.get_primary_symbol(market) == expected

    def test_each_non_us_market_has_fallback(self):
        # T3.2 contract: every non-US market has a fallback so a temporary
        # index feed outage doesn't break downstream RS calculations.
        for market in ("HK", "JP", "TW"):
            entry = benchmark_registry.get_entry(market)
            assert entry.fallback_symbol is not None, f"{market} missing fallback"

    def test_unsupported_market_raises(self):
        # Regression: a silently-accepted unknown market would let scans
        # pick an arbitrary default benchmark.
        with pytest.raises(ValueError):
            benchmark_registry.get_primary_symbol("ZZ")


class TestNonUSMixedMarketPolicy:
    """T6.3: mixed scans switch to USD-normalised columns, fail closed on missing FX."""

    def test_cross_market_is_mixed(self):
        assert is_mixed_market(["US", "HK"]) is True
        assert is_mixed_market(["JP", "TW"]) is True

    def test_mixed_cap_picks_usd_column(self):
        hk_fundamentals = {"market_cap": 10_000_000_000, "market_cap_usd": 1_280_000_000}
        assert resolve_cap_for_filter(hk_fundamentals, mixed_market=True) == pytest.approx(1_280_000_000)

    def test_mixed_cap_missing_usd_fails_closed(self):
        # The keystone fail-safe: a mixed-market HK row without FX data
        # must NOT be compared to a USD threshold using its HKD value.
        hk_fundamentals = {"market_cap": 10_000_000_000, "market_cap_usd": None}
        assert resolve_cap_for_filter(hk_fundamentals, mixed_market=True) is None

    def test_mixed_adv_picks_usd_column(self):
        adv = resolve_adv_for_filter(
            {"adv_usd": 6_400_000},
            native_avg_volume=500_000,  # would be "share count" — must be ignored
            mixed_market=True,
        )
        assert adv == pytest.approx(6_400_000)

    def test_scanner_detects_mixed_scan(self):
        prep = DataPreparationLayer.__new__(DataPreparationLayer)
        results = {
            "AAPL": StockData(
                symbol="AAPL", price_data=pd.DataFrame(), benchmark_data=pd.DataFrame(),
                market="US",
            ),
            "0700.HK": StockData(
                symbol="0700.HK", price_data=pd.DataFrame(), benchmark_data=pd.DataFrame(),
                market="HK",
            ),
        }
        prep._detect_and_set_mixed_market_flag(results)
        assert all(item.is_mixed_market is True for item in results.values())


class TestNonUSRSUniverse:
    """T6.2: RS universe partitions by market; US and HK don't share a bucket."""

    def test_market_scoped_rs_universe(self):
        # Structural partitioning: 3 US inputs and 2 HK inputs must land
        # in separate market buckets, each with its own performance list.
        # If the computation ever pooled markets we'd see only one key
        # or a single merged list of length 5.
        prep = DataPreparationLayer.__new__(DataPreparationLayer)
        universe = prep._compute_market_rs_universe_performances([
            _make_stock_with_benchmark("AAPL", market="US"),
            _make_stock_with_benchmark("MSFT", market="US"),
            _make_stock_with_benchmark("NVDA", market="US"),
            _make_stock_with_benchmark("0700.HK", market="HK"),
            _make_stock_with_benchmark("9988.HK", market="HK"),
        ])
        assert set(universe.keys()) == {"US", "HK"}
        assert "weighted" in universe["US"] and "weighted" in universe["HK"]
        # Bucket cardinality proves partitioning without relying on
        # numeric coincidence: US has 3 inputs, HK has 2.
        assert len(universe["US"]["weighted"]) == 3
        assert len(universe["HK"]["weighted"]) == 2


class TestNonUSAnalyticsScope:
    """T6.4: only the still-US-scoped analytics features are rejected."""

    @pytest.mark.parametrize("market", ["HK", "JP", "TW", "hk", " JP ", "eu"])
    @pytest.mark.parametrize(
        "feature",
        [AnalyticsFeature.THEME_DISCOVERY, AnalyticsFeature.BREADTH_SNAPSHOT],
    )
    def test_non_us_scope_is_rejected(self, market, feature):
        with pytest.raises(UnsupportedMarketError) as exc:
            require_us_scope(market, feature)
        assert feature.value in str(exc.value)

    @pytest.mark.parametrize("market", ["HK", "JP", "TW", "hk", " JP ", "eu"])
    def test_group_rank_scope_is_not_rejected(self, market):
        require_us_scope(market, AnalyticsFeature.IBD_GROUP_RANK)


class TestNonUSScanner:
    """Custom scanner enforces mixed-market semantics for non-US rows."""

    def test_mixed_hk_row_uses_usd_cap(self):
        scanner = CustomScanner()
        data = _build_stock_data(
            symbol="0700.HK",
            market="HK",
            is_mixed=True,
            fundamentals={"market_cap": 10_000_000_000, "market_cap_usd": 1_280_000_000},
            volume=500_000,
        )
        result = scanner.scan_stock("0700.HK", data, {"custom_filters": {"market_cap_min": 1_000_000_000}})
        cap = result.details["filter_results"]["market_cap"]
        assert cap["passes"] is True
        assert cap["unit"] == UNIT_USD
        # Golden: the resolved cap value is market_cap_usd, not market_cap.
        assert cap["market_cap"] == 1_280_000_000

    def test_mixed_hk_missing_fx_fails_closed(self):
        scanner = CustomScanner()
        data = _build_stock_data(
            symbol="0700.HK",
            market="HK",
            is_mixed=True,
            fundamentals={"market_cap": 10_000_000_000, "market_cap_usd": None},
            volume=500_000,
        )
        result = scanner.scan_stock("0700.HK", data, {"custom_filters": {"market_cap_min": 1}})
        cap = result.details["filter_results"]["market_cap"]
        assert cap["passes"] is False
        assert cap["reason"] == REASON_MISSING_CAP_USD


# ---------------------------------------------------------------------------
# Policy versions — snapshot for release-note audits.
# ---------------------------------------------------------------------------


class TestPolicyVersions:
    """Pin active policy versions so a silent semantics change bumps the harness."""

    def test_mixed_market_policy_version(self):
        assert MIXED_MARKET_POLICY_VERSION == "2026.04.13.1"

    def test_analytics_scope_policy_version(self):
        assert ANALYTICS_POLICY_VERSION == "2026.04.20.1"


# ---------------------------------------------------------------------------
# Wire-contract pins — ensure the constants themselves haven't been renamed.
# ---------------------------------------------------------------------------


class TestWireContract:
    """Pin the raw string values behind the mixed-market constants.

    The scanner-behaviour tests assert against imported constants (drift-
    safe), but a silent rename of the constant *value* would still change
    the JSON payload downstream. These tests catch that case so
    frontends reading ``cap["unit"]`` aren't surprised.
    """

    def test_unit_values(self):
        assert UNIT_USD == "usd"
        assert UNIT_NATIVE == "native"
        assert UNIT_SHARES == "shares"

    def test_reason_values(self):
        assert REASON_MISSING_CAP_USD == "missing_market_cap_usd"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_stock_data(
    *,
    symbol: str,
    market: str,
    is_mixed: bool,
    fundamentals: dict,
    volume: int,
    days: int = 260,
) -> StockData:
    """Build a scanner-ready StockData with a flat-close OHLCV frame."""
    idx = pd.date_range("2024-01-01", periods=days, freq="B")
    closes = pd.Series([100.0] * days, index=idx)
    price = pd.DataFrame(
        {
            "Open": closes, "High": closes * 1.01, "Low": closes * 0.99,
            "Close": closes, "Volume": [volume] * days,
        },
        index=idx,
    )
    return StockData(
        symbol=symbol,
        price_data=price,
        benchmark_data=pd.DataFrame(),
        fundamentals=fundamentals,
        market=market,
        is_mixed_market=is_mixed,
    )


def _make_stock_with_benchmark(symbol: str, *, market: str) -> StockData:
    """Build StockData + benchmark for the RS-universe partitioning test.

    Shape of the price / benchmark series is not load-bearing — the
    partitioning test only checks that entries land in the correct
    market bucket. The series needs enough history for the RS calc's
    longest lookback (252 days) and some non-zero return so the weighted
    performance is recorded.
    """
    days = 260
    idx = pd.date_range("2024-01-01", periods=days, freq="B")
    closes_stock = [100.0 * (1.0015 ** i) for i in range(days)]   # ~47% over 260 days
    closes_bench = [100.0 * (1.0005 ** i) for i in range(days)]   # ~14% over 260 days
    stock_df = pd.DataFrame(
        {"Open": closes_stock, "High": closes_stock, "Low": closes_stock,
         "Close": closes_stock, "Volume": [1_000_000] * days},
        index=idx,
    )
    bench_df = pd.DataFrame(
        {"Open": closes_bench, "High": closes_bench, "Low": closes_bench,
         "Close": closes_bench, "Volume": [1_000_000] * days},
        index=idx,
    )
    return StockData(
        symbol=symbol,
        price_data=stock_df,
        benchmark_data=bench_df,
        market=market,
    )
