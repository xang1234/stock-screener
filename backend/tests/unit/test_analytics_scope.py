"""Unit tests for analytics market-scope policy (T6.4)."""
from __future__ import annotations

import pytest

from app.domain.analytics.scope import (
    POLICY_VERSION,
    AnalyticsFeature,
    UnsupportedMarketError,
    describe_policy,
    require_us_scope,
    us_only_tag,
)


class TestUsOnlyTag:
    def test_tag_shape_is_stable(self):
        tag = us_only_tag(AnalyticsFeature.IBD_GROUP_RANK)
        assert tag["market_scope"] == "US"
        assert tag["policy_version"] == POLICY_VERSION
        assert "S&P" in tag["scope_reason"] or "US" in tag["scope_reason"]

    def test_each_feature_has_its_own_reason(self):
        themes = us_only_tag(AnalyticsFeature.THEME_DISCOVERY)
        groups = us_only_tag(AnalyticsFeature.IBD_GROUP_RANK)
        breadth = us_only_tag(AnalyticsFeature.BREADTH_SNAPSHOT)

        # Reasons must differ — each feature has a distinct justification.
        assert themes["scope_reason"] != groups["scope_reason"]
        assert groups["scope_reason"] != breadth["scope_reason"]
        assert themes["scope_reason"] != breadth["scope_reason"]


class TestRequireUsScope:
    @pytest.mark.parametrize("market", [None, "", "  ", "US", "us", " us ", "Us"])
    def test_us_or_blank_is_accepted(self, market):
        # All of these should be treated as US (or absent, which defaults to US).
        require_us_scope(market, AnalyticsFeature.THEME_DISCOVERY)

    @pytest.mark.parametrize(
        "market,feature",
        [
            ("HK", AnalyticsFeature.IBD_GROUP_RANK),
            ("JP", AnalyticsFeature.THEME_DISCOVERY),
            ("TW", AnalyticsFeature.BREADTH_SNAPSHOT),
            ("hk", AnalyticsFeature.IBD_GROUP_RANK),  # case-folded
        ],
    )
    def test_non_us_markets_raise(self, market, feature):
        with pytest.raises(UnsupportedMarketError) as exc:
            require_us_scope(market, feature)
        # Error must carry both the feature name and the specific market so
        # operators can find the call site from logs.
        assert feature.value in str(exc.value)
        assert market in str(exc.value)


class TestDescribePolicy:
    def test_snapshot_lists_all_features(self):
        snap = describe_policy()
        assert snap["policy_version"] == POLICY_VERSION
        us_only = snap["us_only_features"]
        for feature in AnalyticsFeature:
            assert feature.value in us_only
            assert isinstance(us_only[feature.value], str)
