from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.domain.relative_strength import (
    BALANCED_RS_FORMULA_VERSION,
    LEGACY_RS_FORMULA_VERSION,
)
from app.domain.scanning.ports import MarketRsResolution
from app.scanners.base_screener import StockData
from app.scanners.canslim_scanner import CANSLIMScanner
from app.scanners.custom_scanner import CustomScanner
from app.scanners.criteria.rs_resolution import (
    CanonicalStockRsUnavailable,
    resolve_stock_rs,
)
from app.scanners.minervini_scanner import MinerviniScanner
from app.scanners.data_preparation import DataPreparationLayer
from app.scanners.partial_history_metrics import partial_history_metrics
from app.scanners.scan_orchestrator import _build_precomputed_scan_context
from app.scanners.scan_orchestrator import ScanOrchestrator
from app.scanners.screener_registry import ScreenerRegistry
from app.scanners.setup_engine_screener import SetupEngineScanner


def _stock_data() -> StockData:
    dates = pd.bdate_range(end="2026-07-17", periods=320)
    close = np.linspace(50.0, 160.0, len(dates))
    benchmark_close = np.linspace(300.0, 360.0, len(dates))
    return StockData(
        symbol="TEST",
        market="US",
        price_data=pd.DataFrame(
            {
                "Open": close,
                "High": close * 1.01,
                "Low": close * 0.99,
                "Close": close,
                "Volume": np.full(len(dates), 1_000_000),
            },
            index=dates,
        ),
        benchmark_data=pd.DataFrame(
            {
                "Open": benchmark_close,
                "High": benchmark_close * 1.01,
                "Low": benchmark_close * 0.99,
                "Close": benchmark_close,
                "Volume": np.full(len(dates), 10_000_000),
            },
            index=dates,
        ),
    )


def _canonical_stock_data() -> StockData:
    data = _stock_data()
    data.rs_formula_version = BALANCED_RS_FORMULA_VERSION
    data.canonical_rs_ratings = {
        "rs_rating": 87,
        "rs_rating_1m": 42,
        "rs_rating_3m": 91,
        "rs_rating_12m": 98,
    }
    return data


def test_balanced_mode_returns_canonical_ratings_without_calling_legacy():
    data = _canonical_stock_data()

    def legacy_factory():
        raise AssertionError("balanced mode must not call the legacy calculator")

    assert resolve_stock_rs(data, legacy_factory) == data.canonical_rs_ratings


def test_balanced_mode_fails_closed_for_stock_outside_eligible_set():
    data = _stock_data()
    data.rs_formula_version = BALANCED_RS_FORMULA_VERSION

    with pytest.raises(CanonicalStockRsUnavailable, match="TEST"):
        resolve_stock_rs(data, lambda: {"rs_rating": 99})


def test_legacy_mode_calls_legacy_factory():
    data = _stock_data()
    data.rs_formula_version = LEGACY_RS_FORMULA_VERSION
    expected = {
        "rs_rating": 73,
        "rs_rating_1m": 80,
        "rs_rating_3m": 76,
        "rs_rating_12m": 69,
    }

    assert resolve_stock_rs(data, lambda: expected) == expected


def test_precomputed_context_uses_canonical_balanced_values(monkeypatch):
    data = _canonical_stock_data()

    def legacy_calculation(*args, **kwargs):
        raise AssertionError("balanced precomputation must not invoke legacy RS")

    monkeypatch.setattr(
        "app.scanners.scan_orchestrator.RelativeStrengthCalculator.calculate_all_rs_ratings",
        legacy_calculation,
    )

    context = _build_precomputed_scan_context(data)

    assert context is not None
    assert context.rs_ratings == data.canonical_rs_ratings


def test_precomputed_context_rejects_balanced_ineligible_stock(monkeypatch):
    data = _stock_data()
    data.rs_formula_version = BALANCED_RS_FORMULA_VERSION

    def legacy_calculation(*args, **kwargs):
        raise AssertionError("balanced precomputation must not invoke legacy RS")

    monkeypatch.setattr(
        "app.scanners.scan_orchestrator.RelativeStrengthCalculator.calculate_all_rs_ratings",
        legacy_calculation,
    )

    with pytest.raises(CanonicalStockRsUnavailable, match="TEST"):
        _build_precomputed_scan_context(data)


def test_minervini_direct_path_uses_canonical_ratings(monkeypatch):
    data = _canonical_stock_data()
    scanner = MinerviniScanner()
    monkeypatch.setattr(
        scanner.rs_calc,
        "calculate_all_rs_ratings",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("legacy RS must not run")
        ),
    )

    result = scanner.scan_stock("TEST", data, criteria={"include_vcp": False})

    assert result.rating != "Error"
    assert result.details["rs_rating"] == 87
    assert result.details["rs_rating_1m"] == 42
    assert result.details["rs_rating_3m"] == 91
    assert result.details["rs_rating_12m"] == 98


def test_canslim_direct_path_uses_canonical_ratings(monkeypatch):
    data = _canonical_stock_data()
    scanner = CANSLIMScanner()
    monkeypatch.setattr(
        scanner.rs_calc,
        "calculate_rs_rating",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("legacy RS must not run")
        ),
    )
    monkeypatch.setattr(
        scanner.rs_calc,
        "calculate_all_rs_ratings",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("legacy RS must not run")
        ),
    )

    result = scanner.scan_stock("TEST", data)

    assert result.rating != "Error"
    assert result.details["rs_rating"] == 87
    assert result.details["rs_rating_1m"] == 42
    assert result.details["rs_rating_3m"] == 91
    assert result.details["rs_rating_12m"] == 98


def test_custom_direct_path_uses_canonical_rating(monkeypatch):
    data = _canonical_stock_data()
    scanner = CustomScanner()
    monkeypatch.setattr(
        scanner.rs_calc,
        "calculate_rs_rating",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("legacy RS must not run")
        ),
    )

    result = scanner.scan_stock(
        "TEST",
        data,
        criteria={"custom_filters": {"rs_rating_min": 80}},
    )

    assert result.details["filter_results"]["rs_rating"]["rs_rating"] == 87


def test_setup_engine_direct_path_uses_canonical_rating(monkeypatch):
    data = _canonical_stock_data()
    scanner = SetupEngineScanner()
    monkeypatch.setattr(
        scanner._rs_calc,
        "calculate_rs_rating",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("legacy RS must not run")
        ),
    )

    result = scanner.scan_stock("TEST", data)

    assert result.rating != "Error"
    assert result.details["setup_engine"]["rs_rating"] == 87


def test_partial_history_uses_all_canonical_rating_fields(monkeypatch):
    data = _canonical_stock_data()
    monkeypatch.setattr(
        "app.scanners.partial_history_metrics.RelativeStrengthCalculator.calculate_period_rs_rating",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("legacy RS must not run")
        ),
    )

    metrics = partial_history_metrics(data)

    assert metrics["rs_rating"] == 87
    assert metrics["rs_rating_1m"] == 42
    assert metrics["rs_rating_3m"] == 91
    assert metrics["rs_rating_12m"] == 98


class _DirectProvider:
    def __init__(self, data: StockData) -> None:
        self.data = data

    def prepare_data(self, symbol, requirements, *, allow_partial=True):
        return self.data

    def prepare_data_bulk(self, symbols, requirements, **kwargs):
        return {symbol: self.data for symbol in symbols}


class _CanonicalReader:
    def __init__(self, *, include_symbol: bool = True) -> None:
        self.include_symbol = include_symbol
        self.calls: list[dict] = []

    def get(self, **kwargs) -> MarketRsResolution:
        self.calls.append(kwargs)
        ratings = (
            {
                "TEST": {
                    "rs_rating": 87,
                    "rs_rating_1m": 42,
                    "rs_rating_3m": 91,
                    "rs_rating_12m": 98,
                }
            }
            if self.include_symbol
            else {}
        )
        return MarketRsResolution(
            market="US",
            as_of_date=pd.Timestamp("2026-07-17").date(),
            formula_version=BALANCED_RS_FORMULA_VERSION,
            mode="canonical",
            run_id=42,
            universe_size=5000,
            ratings_by_symbol=ratings,
        )


def _minervini_registry() -> tuple[ScreenerRegistry, MinerviniScanner]:
    registry = ScreenerRegistry()
    registry.register(MinerviniScanner)
    scanner = registry.get("minervini")
    return registry, scanner


def test_single_stock_orchestrator_hydrates_latest_canonical_rs(monkeypatch):
    data = _stock_data()
    reader = _CanonicalReader()
    registry, scanner = _minervini_registry()
    monkeypatch.setattr(
        scanner.rs_calc,
        "calculate_all_rs_ratings",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("legacy RS must not run")
        ),
    )
    orchestrator = ScanOrchestrator(
        data_provider=_DirectProvider(data),
        registry=registry,
        market_rs_reader=reader,
    )

    result = orchestrator.scan_stock_multi(
        "TEST",
        ["minervini"],
        criteria={"include_vcp": False},
    )

    assert reader.calls == [
        {
            "market": "US",
            "symbols": ("TEST",),
            "as_of_date": None,
            "formula_version": None,
        }
    ]
    assert result["rs_rating"] == 87
    assert result["rs_rating_1m"] == 42
    assert result["rs_rating_3m"] == 91
    assert result["rs_rating_12m"] == 98
    assert result["rs_formula_version"] == BALANCED_RS_FORMULA_VERSION
    assert result["market_rs_run_id"] == 42
    assert result["rs_universe_size"] == 5000
    assert result["details"]["rs_formula_version"] == BALANCED_RS_FORMULA_VERSION


def test_orchestrator_maps_balanced_ineligible_stock_to_insufficient_history():
    data = _stock_data()
    reader = _CanonicalReader(include_symbol=False)
    registry, _ = _minervini_registry()
    orchestrator = ScanOrchestrator(
        data_provider=_DirectProvider(data),
        registry=registry,
        market_rs_reader=reader,
    )

    result = orchestrator.scan_stock_multi("TEST", ["minervini"])

    assert result["rating"] == "Insufficient Data"
    assert result["result_status"] == "insufficient_history"
    assert result["rs_formula_version"] == BALANCED_RS_FORMULA_VERSION
    assert "eligible universe" in result["reason"]


def test_balanced_resolution_bypasses_scan_local_percentile_universe(monkeypatch):
    layer = DataPreparationLayer.__new__(DataPreparationLayer)
    data = _stock_data()
    calls = []
    monkeypatch.setattr(
        layer,
        "_attach_market_rs_universe_performances",
        lambda results: calls.append(results),
    )
    resolution = _CanonicalReader().get(
        market="US",
        symbols=("TEST",),
        as_of_date=None,
        formula_version=None,
    )

    layer.apply_market_rs_resolution({"TEST": data}, resolution)

    assert calls == []
    assert data.canonical_rs_ratings["rs_rating"] == 87
    assert data.rs_formula_version == BALANCED_RS_FORMULA_VERSION


def test_legacy_resolution_retains_scan_local_percentile_universe(monkeypatch):
    layer = DataPreparationLayer.__new__(DataPreparationLayer)
    data = _stock_data()
    calls = []
    monkeypatch.setattr(
        layer,
        "_attach_market_rs_universe_performances",
        lambda results: calls.append(results),
    )
    resolution = MarketRsResolution(
        market="US",
        as_of_date=None,
        formula_version=LEGACY_RS_FORMULA_VERSION,
        mode="legacy",
        run_id=None,
        universe_size=None,
        ratings_by_symbol={},
    )

    layer.apply_market_rs_resolution({"TEST": data}, resolution)

    assert calls == [{"TEST": data}]
    assert data.canonical_rs_ratings is None
    assert data.rs_formula_version == LEGACY_RS_FORMULA_VERSION
