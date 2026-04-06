"""Route-registration smoke tests for slash-compatible collection endpoints."""

from app.main import app


def test_collection_routes_support_slash_and_no_slash_forms():
    paths = {
        getattr(route, "path", None)
        for route in app.router.routes
        if getattr(route, "path", None)
    }

    assert "/api/v1/user-watchlists" in paths
    assert "/api/v1/user-watchlists/" in paths
    assert "/api/v1/user-themes" in paths
    assert "/api/v1/user-themes/" in paths
    assert "/api/v1/filter-presets" in paths
    assert "/api/v1/filter-presets/" in paths
    assert "/api/v1/strategy-profiles" in paths
    assert "/api/v1/strategy-profiles/" in paths
