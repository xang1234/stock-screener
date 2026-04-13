"""Unit tests: analytics responses carry the US-only scope tag (T6.4).

Drives the response-assembly code paths with in-memory stubs to confirm
that every US-only analytics surface (groups endpoint, theme price
metrics) emits a ``market_scope`` field that downstream consumers can
observe. All dependencies are synthetic; no server or database required.
"""
from __future__ import annotations

from app.domain.analytics.scope import AnalyticsFeature, us_only_tag


class TestGroupsEndpointResponseCarriesScope:
    """The Pydantic schema should serialize market_scope when the endpoint tags it."""

    def test_rankings_response_includes_scope(self):
        from app.schemas.groups import GroupRankingsResponse

        scope = us_only_tag(AnalyticsFeature.IBD_GROUP_RANK)
        resp = GroupRankingsResponse(
            date="2026-04-11",
            total_groups=0,
            rankings=[],
            market_scope=scope["market_scope"],
            scope_reason=scope.get("scope_reason"),
        )
        dumped = resp.model_dump()
        assert dumped["market_scope"] == "US"
        assert "S&P" in dumped["scope_reason"] or "US" in dumped["scope_reason"]

    def test_movers_response_includes_scope(self):
        from app.schemas.groups import MoversResponse

        scope = us_only_tag(AnalyticsFeature.IBD_GROUP_RANK)
        resp = MoversResponse(
            period="1w",
            gainers=[],
            losers=[],
            market_scope=scope["market_scope"],
            scope_reason=scope.get("scope_reason"),
        )
        dumped = resp.model_dump()
        assert dumped["market_scope"] == "US"

    def test_rankings_response_backcompat_when_not_tagged(self):
        """Older callers that don't tag the response still work (optional field)."""
        from app.schemas.groups import GroupRankingsResponse

        resp = GroupRankingsResponse(
            date="2026-04-11", total_groups=0, rankings=[],
        )
        assert resp.market_scope is None


class TestThemePriceMetricsCarriesScope:
    def test_empty_metrics_tagged(self):
        # Exercise the cheap _empty_price_metrics path (no DB needed).
        from app.services.theme_discovery_service import ThemeDiscoveryService

        # Construct without __init__ to skip DB wiring — we only test the
        # pure metrics shape.
        svc = ThemeDiscoveryService.__new__(ThemeDiscoveryService)
        metrics = svc._empty_price_metrics()

        assert metrics["market_scope"] == "US"
        assert "scope_reason" in metrics
