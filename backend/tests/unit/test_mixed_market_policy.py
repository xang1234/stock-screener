"""Unit tests for the mixed-market scan normalization policy (T6.3)."""
from __future__ import annotations

import pytest

from app.domain.scanning.mixed_market_policy import (
    POLICY_VERSION,
    describe_policy,
    is_mixed_market,
    resolve_adv_for_filter,
    resolve_cap_for_filter,
)


class TestIsMixedMarket:
    def test_single_market_is_not_mixed(self):
        assert is_mixed_market(["US", "US", "US"]) is False

    def test_two_markets_are_mixed(self):
        assert is_mixed_market(["US", "HK"]) is True

    def test_none_is_treated_as_us(self):
        # Legacy rows without a market tag are historically US stocks.
        assert is_mixed_market([None, "US"]) is False
        assert is_mixed_market([None, "HK"]) is True

    def test_empty_is_not_mixed(self):
        assert is_mixed_market([]) is False

    def test_casing_and_whitespace_are_folded(self):
        # Upstream producers already normalise, but the policy must not
        # treat casing/whitespace as distinct markets (defensive).
        assert is_mixed_market(["US", "us", " US "]) is False
        assert is_mixed_market(["US", "  ", None]) is False  # blanks → US

    def test_short_circuits_on_second_distinct_market(self):
        # Implementation detail, but worth pinning: an iterable that would
        # raise after the 2nd element is never consumed past it.
        def gen():
            yield "US"
            yield "HK"
            raise AssertionError("should not iterate past 2nd market")

        assert is_mixed_market(gen()) is True


class TestResolveCapForFilter:
    def test_single_market_uses_native_cap(self):
        f = {"market_cap": 1_500_000_000, "market_cap_usd": 192_000_000}
        assert resolve_cap_for_filter(f, mixed_market=False) == 1_500_000_000

    def test_mixed_market_uses_usd_cap(self):
        f = {"market_cap": 1_500_000_000, "market_cap_usd": 192_000_000}
        assert resolve_cap_for_filter(f, mixed_market=True) == 192_000_000

    def test_mixed_market_missing_usd_returns_none(self):
        # Fail closed: don't silently compare HKD to USD threshold.
        f = {"market_cap": 1_500_000_000, "market_cap_usd": None}
        assert resolve_cap_for_filter(f, mixed_market=True) is None

    def test_none_fundamentals_returns_none(self):
        assert resolve_cap_for_filter(None, mixed_market=True) is None
        assert resolve_cap_for_filter(None, mixed_market=False) is None


class TestResolveAdvForFilter:
    def test_single_market_uses_native_volume(self):
        adv = resolve_adv_for_filter({"adv_usd": 1_000}, 500_000, mixed_market=False)
        assert adv == 500_000

    def test_mixed_market_uses_adv_usd(self):
        adv = resolve_adv_for_filter({"adv_usd": 6_400_000}, 500_000, mixed_market=True)
        assert adv == 6_400_000

    def test_mixed_market_missing_adv_usd_returns_none(self):
        adv = resolve_adv_for_filter({"adv_usd": None}, 500_000, mixed_market=True)
        assert adv is None

    def test_single_market_none_volume_returns_none(self):
        # Defensive: scanner should never pass None, but policy must not crash.
        adv = resolve_adv_for_filter({}, None, mixed_market=False)
        assert adv is None


class TestDescribePolicy:
    def test_snapshot_has_stable_shape(self):
        snap = describe_policy()
        assert snap["policy_version"] == POLICY_VERSION
        assert snap["percentile_scope"] == "per-market"
        assert snap["missing_fx_behaviour"] == "exclude"
        assert snap["liquidity_cap_scope"]["mixed_market"] == "usd_normalized"
        assert snap["liquidity_cap_scope"]["single_market"] == "native_currency"


class TestFairnessAndStability:
    """Policy-level fairness/stability checks (bead acceptance criterion)."""

    def test_mixed_market_equal_threshold_applied_uniformly(self):
        # A US mega-cap ($500B) and an HK mid-cap (HK$1B ≈ USD$128M) with a
        # $200M cap_min: the HK stock is excluded because 128M < 200M, the US
        # stock passes at 500B. No native/USD mixing would flip this.
        us = {"market_cap": 500_000_000_000, "market_cap_usd": 500_000_000_000}
        hk = {"market_cap": 1_000_000_000, "market_cap_usd": 128_000_000}
        cap_min = 200_000_000
        assert resolve_cap_for_filter(us, mixed_market=True) >= cap_min
        assert resolve_cap_for_filter(hk, mixed_market=True) < cap_min

    def test_policy_stable_regardless_of_market_order(self):
        # Iteration order of markets doesn't change the mixed-market flag.
        assert is_mixed_market(["HK", "US"]) == is_mixed_market(["US", "HK"])
        assert is_mixed_market(["US", "US", "HK"]) == is_mixed_market(["HK", "US", "US"])
