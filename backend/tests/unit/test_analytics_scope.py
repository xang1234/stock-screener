"""Unit tests for analytics market-scope policy (T6.4)."""
from __future__ import annotations

import pytest

from app.domain.analytics.scope import (
    POLICY_VERSION,
    AnalyticsFeature,
    UnsupportedMarketError,
    describe_policy,
    market_scope_tag,
    policy_version,
    require_us_scope,
    us_only_tag,
)


class TestPolicyVersion:
    def test_accessor_returns_constant(self):
        # Parity with provider_routing_policy.policy_version() and
        # mixed_market_policy.policy_version().
        assert policy_version() == POLICY_VERSION


class TestUsOnlyTag:
    def test_tag_shape_is_stable(self):
        tag = us_only_tag(AnalyticsFeature.THEME_DISCOVERY)
        assert tag["market_scope"] == "US"
        assert "S&P" in tag["scope_reason"] or "US" in tag["scope_reason"]
        # policy_version is exposed via policy_version(), not inside the
        # tag dict — keeps the tag spreadable into Pydantic models without
        # rejected-field errors.
        assert "policy_version" not in tag

    def test_breadth_snapshot_is_market_aware(self):
        themes = us_only_tag(AnalyticsFeature.THEME_DISCOVERY)
        breadth = market_scope_tag("HK")

        assert themes["market_scope"] == "US"
        assert "scope_reason" in themes
        assert breadth == {"market_scope": "HK"}


class TestRequireUsScope:
    @pytest.mark.parametrize("market", [None, "", "  ", "US", "us", " us ", "Us"])
    def test_us_or_blank_is_accepted(self, market):
        # All of these should be treated as US (or absent, which defaults to US).
        require_us_scope(market, AnalyticsFeature.THEME_DISCOVERY)

    @pytest.mark.parametrize(
        "market,feature",
        [
            ("JP", AnalyticsFeature.THEME_DISCOVERY),
            ("hk", AnalyticsFeature.THEME_DISCOVERY),  # case-folded
        ],
    )
    def test_non_us_markets_raise(self, market, feature):
        with pytest.raises(UnsupportedMarketError) as exc:
            require_us_scope(market, feature)
        # Error must carry both the feature name and the specific market so
        # operators can find the call site from logs.
        assert feature.value in str(exc.value)
        assert market in str(exc.value)

    def test_non_string_market_raises_loudly(self):
        # Defensive: a non-string market (e.g. accidental int from an ORM
        # integer column) should not crash with AttributeError — that
        # would masquerade as a bug elsewhere.
        with pytest.raises(UnsupportedMarketError) as exc:
            require_us_scope(123, AnalyticsFeature.THEME_DISCOVERY)  # type: ignore[arg-type]
        assert "non-string" in str(exc.value)

    def test_market_aware_features_bypass_us_only_guard(self):
        require_us_scope("HK", AnalyticsFeature.IBD_GROUP_RANK)
        require_us_scope("TW", AnalyticsFeature.IBD_GROUP_RANK)
        require_us_scope("TW", AnalyticsFeature.BREADTH_SNAPSHOT)


class TestMarketScopeTag:
    def test_market_scope_tag_normalizes_requested_market(self):
        tag = market_scope_tag(" hk ")
        assert tag == {"market_scope": "HK"}

    def test_market_scope_tag_accepts_optional_reason(self):
        tag = market_scope_tag("JP", reason="computed from JP feature runs")
        assert tag["market_scope"] == "JP"
        assert tag["scope_reason"] == "computed from JP feature runs"


class TestDescribePolicy:
    def test_snapshot_lists_current_us_only_features(self):
        snap = describe_policy()
        assert snap["policy_version"] == POLICY_VERSION
        us_only = snap["us_only_features"]
        for feature in (AnalyticsFeature.THEME_DISCOVERY,):
            assert feature.value in us_only
            assert isinstance(us_only[feature.value], str)
        assert AnalyticsFeature.IBD_GROUP_RANK.value not in us_only
        assert AnalyticsFeature.BREADTH_SNAPSHOT.value not in us_only
