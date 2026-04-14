"""warm_spy_cache market scoping (bead asia.9.2 — fix for 9.1 inherited issue #3).

Pre-9.2: warm_spy_cache() warmed benchmarks for ALL markets regardless of
caller scope, so 4 parallel weekly_full_refresh tasks did 4× redundant work.
With 9.2's market kwarg, each market's refresh warms only its own benchmark.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from app.tasks.cache_tasks import _active_benchmark_markets


class TestActiveBenchmarkMarketsScoping:
    def test_no_scope_returns_all_active_markets(self):
        """Legacy behavior preserved when scope_market is None."""
        db = MagicMock()
        rows = [
            ("US", "NASDAQ", "AAPL"),
            ("HK", "SEHK", "0700.HK"),
            ("JP", "TSE", "7203.T"),
        ]
        db.query.return_value.filter.return_value.all.return_value = rows
        with patch("app.tasks.cache_tasks.benchmark_registry") as registry, \
             patch("app.tasks.cache_tasks.security_master_resolver") as resolver:
            registry.supported_markets.return_value = ["US", "HK", "JP", "TW"]
            resolver.normalize_market.side_effect = lambda m: m
            resolver.infer_market.return_value = None
            result = _active_benchmark_markets(db)
        assert set(result) == {"US", "HK", "JP"}

    def test_scope_market_us_returns_only_us(self):
        db = MagicMock()
        with patch("app.tasks.cache_tasks.benchmark_registry") as registry:
            registry.supported_markets.return_value = ["US", "HK", "JP", "TW"]
            result = _active_benchmark_markets(db, scope_market="US")
        # US scope -> only US benchmark, no HK/JP/TW work
        assert result == ["US"]

    def test_scope_market_hk_returns_only_hk(self):
        db = MagicMock()
        with patch("app.tasks.cache_tasks.benchmark_registry") as registry:
            registry.supported_markets.return_value = ["US", "HK", "JP", "TW"]
            result = _active_benchmark_markets(db, scope_market="HK")
        assert result == ["HK"]

    def test_scope_market_unknown_falls_back_to_us(self):
        """Defensive: unknown market scope warms US benchmark only (safe default)."""
        db = MagicMock()
        with patch("app.tasks.cache_tasks.benchmark_registry") as registry:
            registry.supported_markets.return_value = ["US", "HK", "JP", "TW"]
            result = _active_benchmark_markets(db, scope_market="CN")
        assert result == ["US"]

    def test_scope_skips_db_query_entirely(self):
        """Per-market scope should not hit the DB at all (no n+1, no full scan)."""
        db = MagicMock()
        with patch("app.tasks.cache_tasks.benchmark_registry") as registry:
            registry.supported_markets.return_value = ["US", "HK", "JP", "TW"]
            _active_benchmark_markets(db, scope_market="HK")
        db.query.assert_not_called()
