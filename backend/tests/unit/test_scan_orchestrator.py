"""Unit tests for ScanOrchestrator with injected fakes.

Verifies:
- Single and multi-screener composite scoring
- All composite methods (weighted_average, maximum, minimum)
- Defensive handling of unknown composite_method strings
- Rating thresholds and pass-rate adjustments
- Insufficient data and all-screeners-fail error paths
- Pre-fetched data skips the provider
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.domain.scanning.ports import StockDataProvider
from app.scanners.base_screener import (
    BaseStockScreener,
    DataRequirements,
    ScreenerResult,
    StockData,
)
from app.scanners.scan_orchestrator import ScanOrchestrator
from app.scanners.screener_registry import ScreenerRegistry


# ── Fakes ───────────────────────────────────────────────────────────


class FakeDataProvider(StockDataProvider):
    """Returns pre-built StockData from a dict keyed by symbol."""

    def __init__(self, stock_data_map: dict[str, StockData]):
        self._map = stock_data_map
        self.prepare_data_called = False

    def prepare_data(self, symbol: str, requirements: object) -> StockData:
        self.prepare_data_called = True
        return self._map[symbol]

    def prepare_data_bulk(
        self, symbols: list[str], requirements: object
    ) -> dict[str, object]:
        return {s: self._map[s] for s in symbols if s in self._map}


def _make_stock_data(symbol: str = "TEST", n_days: int = 200) -> StockData:
    """Build a StockData with n_days of synthetic price data."""
    dates = pd.date_range(end="2026-01-15", periods=n_days, freq="B")
    df = pd.DataFrame(
        {
            "Open": 100.0,
            "High": 105.0,
            "Low": 95.0,
            "Close": 100.0,
            "Volume": 1_000_000,
        },
        index=dates,
    )
    benchmark = df.copy()
    return StockData(
        symbol=symbol,
        price_data=df,
        benchmark_data=benchmark,
    )


def make_fake_screener_class(
    name: str, score: float, passes: bool
) -> type[BaseStockScreener]:
    """Factory that creates a BaseStockScreener subclass with fixed results."""

    class FakeScreener(BaseStockScreener):
        @property
        def screener_name(self) -> str:
            return name

        def get_data_requirements(self, criteria=None) -> DataRequirements:
            return DataRequirements()

        def scan_stock(self, symbol, data, criteria=None) -> ScreenerResult:
            rating = "Strong Buy" if passes else "Pass"
            return ScreenerResult(
                score=score,
                passes=passes,
                rating=rating,
                breakdown={"test": score},
                details={"test_detail": True},
                screener_name=name,
            )

        def calculate_rating(self, score, details) -> str:
            return "Strong Buy" if score >= 80 else "Pass"

    # Give each fake a unique class name for registry compatibility
    FakeScreener.__name__ = f"FakeScreener_{name}"
    FakeScreener.__qualname__ = f"FakeScreener_{name}"
    return FakeScreener


def _build_orchestrator(
    screener_configs: list[tuple[str, float, bool]],
    n_days: int = 200,
) -> tuple[ScanOrchestrator, FakeDataProvider, ScreenerRegistry]:
    """Helper to build an orchestrator with fake screeners and data.

    Args:
        screener_configs: List of (name, score, passes) tuples.
        n_days: Number of days of synthetic price data.

    Returns:
        (orchestrator, data_provider, registry)
    """
    stock_data = _make_stock_data("TEST", n_days=n_days)
    provider = FakeDataProvider({"TEST": stock_data})
    registry = ScreenerRegistry()

    for name, score, passes in screener_configs:
        cls = make_fake_screener_class(name, score, passes)
        registry.register(cls)

    orchestrator = ScanOrchestrator(data_provider=provider, registry=registry)
    return orchestrator, provider, registry


# ── Tests ───────────────────────────────────────────────────────────


class TestScanOrchestratorScoring:
    def test_single_screener_scoring(self):
        """Single screener: composite score equals the screener's score."""
        orch, _, _ = _build_orchestrator([("alpha", 75.0, True)])
        result = orch.scan_stock_multi("TEST", ["alpha"], composite_method="weighted_average")

        assert result["composite_score"] == 75.0
        assert result["alpha_score"] == 75.0
        assert result["screeners_passed"] == 1

    def test_weighted_average_two_screeners(self):
        """Two screeners: weighted_average is the arithmetic mean."""
        orch, _, _ = _build_orchestrator([
            ("alpha", 80.0, True),
            ("beta", 60.0, True),
        ])
        result = orch.scan_stock_multi("TEST", ["alpha", "beta"], composite_method="weighted_average")

        assert result["composite_score"] == 70.0

    def test_maximum_composite_method(self):
        """'maximum' string selects the highest score."""
        orch, _, _ = _build_orchestrator([
            ("alpha", 90.0, True),
            ("beta", 60.0, True),
        ])
        result = orch.scan_stock_multi("TEST", ["alpha", "beta"], composite_method="maximum")

        assert result["composite_score"] == 90.0

    def test_minimum_composite_method(self):
        """'minimum' string selects the lowest score."""
        orch, _, _ = _build_orchestrator([
            ("alpha", 90.0, True),
            ("beta", 60.0, True),
        ])
        result = orch.scan_stock_multi("TEST", ["alpha", "beta"], composite_method="minimum")

        assert result["composite_score"] == 60.0

    def test_unknown_method_defaults_to_weighted_average(self):
        """Unknown composite_method string falls back to weighted_average."""
        orch, _, _ = _build_orchestrator([
            ("alpha", 80.0, True),
            ("beta", 60.0, True),
        ])
        result = orch.scan_stock_multi("TEST", ["alpha", "beta"], composite_method="some_nonsense")

        # Falls back to weighted_average = (80 + 60) / 2 = 70
        assert result["composite_score"] == 70.0


class TestScanOrchestratorRating:
    def test_rating_strong_buy(self):
        """Composite >= 80 with all passing → 'Strong Buy'."""
        orch, _, _ = _build_orchestrator([("alpha", 85.0, True)])
        result = orch.scan_stock_multi("TEST", ["alpha"], composite_method="weighted_average")

        assert result["rating"] == "Strong Buy"

    def test_rating_downgrade_low_pass_rate(self):
        """Score in Buy range (70-80) but < half pass → downgraded to Watch."""
        orch, _, _ = _build_orchestrator([
            ("alpha", 95.0, True),
            ("beta", 70.0, False),
            ("gamma", 70.0, False),
        ])
        result = orch.scan_stock_multi(
            "TEST", ["alpha", "beta", "gamma"], composite_method="weighted_average"
        )

        # Average: (95 + 70 + 70) / 3 ≈ 78.33 → base = "Buy" (70-80 range)
        # 1 of 3 passed = less than half → downgrade to "Watch"
        assert result["rating"] == "Watch"

    def test_rating_pass_when_none_pass(self):
        """No screeners pass → 'Pass' regardless of score."""
        orch, _, _ = _build_orchestrator([
            ("alpha", 90.0, False),
            ("beta", 85.0, False),
        ])
        result = orch.scan_stock_multi("TEST", ["alpha", "beta"], composite_method="weighted_average")

        assert result["rating"] == "Pass"


class TestScanOrchestratorErrorPaths:
    def test_insufficient_data_returns_error(self):
        """StockData with < 100 days → insufficient data result."""
        orch, _, _ = _build_orchestrator([("alpha", 85.0, True)], n_days=50)
        result = orch.scan_stock_multi("TEST", ["alpha"], composite_method="weighted_average")

        assert result["rating"] == "Insufficient Data"
        assert result["composite_score"] == 0

    def test_all_screeners_fail_returns_error(self):
        """When all screeners raise exceptions → error result."""
        stock_data = _make_stock_data("TEST", n_days=200)
        provider = FakeDataProvider({"TEST": stock_data})
        registry = ScreenerRegistry()

        # Create a screener that always raises
        class FailingScreener(BaseStockScreener):
            @property
            def screener_name(self) -> str:
                return "failing"

            def get_data_requirements(self, criteria=None) -> DataRequirements:
                return DataRequirements()

            def scan_stock(self, symbol, data, criteria=None) -> ScreenerResult:
                raise RuntimeError("Boom!")

            def calculate_rating(self, score, details) -> str:
                return "Pass"

        registry.register(FailingScreener)
        orch = ScanOrchestrator(data_provider=provider, registry=registry)

        result = orch.scan_stock_multi("TEST", ["failing"], composite_method="weighted_average")

        assert result["rating"] == "Error"
        assert "All screeners failed" in result.get("error", "")


class TestScanOrchestratorDataFlow:
    def test_pre_fetched_data_skips_provider(self):
        """When pre_fetched_data is passed, provider.prepare_data is not called."""
        stock_data = _make_stock_data("TEST", n_days=200)
        provider = FakeDataProvider({"TEST": stock_data})
        registry = ScreenerRegistry()

        cls = make_fake_screener_class("alpha", 75.0, True)
        registry.register(cls)

        orch = ScanOrchestrator(data_provider=provider, registry=registry)

        result = orch.scan_stock_multi(
            "TEST", ["alpha"],
            composite_method="weighted_average",
            pre_fetched_data=stock_data,
        )

        assert provider.prepare_data_called is False
        assert result["composite_score"] == 75.0
