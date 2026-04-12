"""Integration tests: T4 quality-aware fallback flows through ScanOrchestrator.

Verifies that the orchestrator reads ``field_completeness_score`` from
the stock's fundamentals and applies the T4 policy to the combined
result, surfacing the reason and the score for downstream consumers.
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


class _FakeProvider(StockDataProvider):
    def __init__(self, stock_data):
        self._stock_data = stock_data

    def prepare_data(self, symbol, requirements):
        return self._stock_data

    def prepare_data_bulk(
        self,
        symbols,
        requirements,
        *,
        allow_partial: bool = True,
        batch_only_prices: bool = False,
        batch_only_fundamentals: bool = False,
    ):
        return {s: self._stock_data for s in symbols}


def _make_stock_data_with_completeness(completeness):
    dates = pd.date_range(end="2026-01-15", periods=200, freq="B")
    df = pd.DataFrame(
        {"Open": 100.0, "High": 105.0, "Low": 95.0, "Close": 100.0, "Volume": 1_000_000},
        index=dates,
    )
    return StockData(
        symbol="TEST",
        price_data=df,
        benchmark_data=df.copy(),
        fundamentals={
            "market_cap": 1_000_000_000,
            "field_completeness_score": completeness,
        } if completeness is not None else {"market_cap": 1_000_000_000},
    )


def _make_strong_buy_screener():
    class _S(BaseStockScreener):
        @property
        def screener_name(self):
            return "alpha"

        def get_data_requirements(self, criteria=None):
            return DataRequirements()

        def scan_stock(self, symbol, data, criteria=None):
            return ScreenerResult(
                score=90.0, passes=True, rating="Strong Buy",
                breakdown={}, details={}, screener_name="alpha",
            )

        def calculate_rating(self, score, details):
            return "Strong Buy"
    _S.__name__ = "FakeScreener_alpha"
    return _S


def _build(completeness):
    registry = ScreenerRegistry()
    registry.register(_make_strong_buy_screener())
    provider = _FakeProvider(_make_stock_data_with_completeness(completeness))
    return ScanOrchestrator(data_provider=provider, registry=registry)


class TestOrchestratorAppliesQualityPolicy:
    def test_high_completeness_preserves_strong_buy(self):
        orch = _build(completeness=85)
        result = orch.scan_stock_multi("TEST", ["alpha"], composite_method="weighted_average")
        assert result["rating"] == "Strong Buy"
        assert result["quality_downgrade_reason"] is None
        assert result["field_completeness_score"] == 85

    def test_mid_completeness_downgrades_to_buy(self):
        orch = _build(completeness=45)
        result = orch.scan_stock_multi("TEST", ["alpha"], composite_method="weighted_average")
        assert result["rating"] == "Buy"  # downgraded from Strong Buy
        assert result["quality_downgrade_reason"] is not None
        assert "45" in result["quality_downgrade_reason"]

    def test_low_completeness_forces_pass(self):
        orch = _build(completeness=10)
        result = orch.scan_stock_multi("TEST", ["alpha"], composite_method="weighted_average")
        assert result["rating"] == "Pass"
        assert result["quality_downgrade_reason"] is not None
        assert "exclusion" in result["quality_downgrade_reason"]

    def test_missing_completeness_passes_through(self):
        """Legacy rows with no completeness score keep their rating (avoids
        regressing existing scan results when the T2 migration hasn't yet
        run everywhere)."""
        orch = _build(completeness=None)
        result = orch.scan_stock_multi("TEST", ["alpha"], composite_method="weighted_average")
        assert result["rating"] == "Strong Buy"
        assert result["quality_downgrade_reason"] is None
        assert result["field_completeness_score"] is None

    def test_composite_score_is_not_mutated(self):
        """Design decision: rating carries the quality signal, composite_score
        stays honest. Downgraded rows still show the raw screener score."""
        orch = _build(completeness=45)
        result = orch.scan_stock_multi("TEST", ["alpha"], composite_method="weighted_average")
        assert result["composite_score"] == 90.0  # unchanged by downgrade
        assert result["rating"] == "Buy"  # but rating reflects the quality hit

    def test_reason_lives_at_top_level_only(self):
        """The reason is a primary signal (like ``rating``), so it lives at
        the top level where API consumers find it without drilling into
        ``details``. Keeping a second copy inside ``details`` risked drift
        between the two and bloated the serialised payload."""
        orch = _build(completeness=10)
        result = orch.scan_stock_multi("TEST", ["alpha"], composite_method="weighted_average")
        assert result["quality_downgrade_reason"] is not None
        assert "quality_downgrade_reason" not in result["details"]
