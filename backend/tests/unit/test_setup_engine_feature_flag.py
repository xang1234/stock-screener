"""Tests for the setup_engine feature flag in the scan orchestrator.

Covers:
- Orchestrator silently filters out setup_engine when flag is disabled
- Orchestrator includes setup_engine when flag is enabled (default)
- Empty screener list after filtering returns descriptive error
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.scanners.scan_orchestrator import ScanOrchestrator


@pytest.fixture
def orchestrator():
    """Build an orchestrator with mock dependencies."""
    data_provider = MagicMock()
    registry = MagicMock()
    return ScanOrchestrator(data_provider=data_provider, registry=registry)


class TestFeatureFlagDisabled:
    """When setup_engine_enabled=False, setup_engine should be silently filtered."""

    @patch("app.scanners.scan_orchestrator.settings")
    def test_filters_setup_engine_from_names(self, mock_settings, orchestrator):
        """setup_engine should be removed; other screeners should remain."""
        mock_settings.setup_engine_enabled = False

        # Mock registry to return screeners for the remaining names
        mock_screener = MagicMock()
        mock_result = MagicMock()
        mock_result.score = 50.0
        mock_result.passes = True
        mock_result.rating = "Watch"
        mock_result.breakdown = {}
        mock_result.details = {}
        mock_screener.scan_stock.return_value = mock_result

        orchestrator._registry.get_multiple.return_value = {"minervini": mock_screener}

        # Mock stock data
        mock_data = MagicMock()
        mock_data.has_sufficient_data.return_value = True
        mock_data.get_current_price.return_value = 100.0
        mock_data.fetch_errors = None
        mock_data.fundamentals = None
        mock_data.quarterly_growth = None
        mock_data.benchmark_data.empty = True
        orchestrator._data_provider.prepare_data.return_value = mock_data

        result = orchestrator.scan_stock_multi(
            "TEST",
            ["minervini", "setup_engine"],
        )

        # Registry should only receive ["minervini"]
        call_args = orchestrator._registry.get_multiple.call_args[0][0]
        assert "setup_engine" not in call_args
        assert "minervini" in call_args

    @patch("app.scanners.scan_orchestrator.settings")
    def test_all_screeners_disabled_returns_error(self, mock_settings, orchestrator):
        """If only setup_engine was requested and it's disabled, return error."""
        mock_settings.setup_engine_enabled = False

        result = orchestrator.scan_stock_multi(
            "TEST",
            ["setup_engine"],
        )

        assert result["error"] == "All requested screeners are disabled"
        assert result["rating"] == "Error"
        assert result["composite_score"] == 0
        assert result["screeners_run"] == []


class TestFeatureFlagEnabled:
    """When setup_engine_enabled=True (default), setup_engine should be included."""

    @patch("app.scanners.scan_orchestrator.settings")
    def test_includes_setup_engine(self, mock_settings, orchestrator):
        """setup_engine should not be filtered when flag is enabled."""
        mock_settings.setup_engine_enabled = True

        # Mock registry
        mock_screener = MagicMock()
        mock_result = MagicMock()
        mock_result.score = 50.0
        mock_result.passes = True
        mock_result.rating = "Watch"
        mock_result.breakdown = {}
        mock_result.details = {}
        mock_screener.scan_stock.return_value = mock_result

        orchestrator._registry.get_multiple.return_value = {"setup_engine": mock_screener}

        # Mock stock data
        mock_data = MagicMock()
        mock_data.has_sufficient_data.return_value = True
        mock_data.get_current_price.return_value = 100.0
        mock_data.fetch_errors = None
        mock_data.fundamentals = None
        mock_data.quarterly_growth = None
        mock_data.benchmark_data.empty = True
        orchestrator._data_provider.prepare_data.return_value = mock_data

        result = orchestrator.scan_stock_multi(
            "TEST",
            ["setup_engine"],
        )

        # Registry should receive ["setup_engine"]
        call_args = orchestrator._registry.get_multiple.call_args[0][0]
        assert "setup_engine" in call_args
