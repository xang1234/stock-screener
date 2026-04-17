from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from app.scanners.base_screener import BaseStockScreener, DataRequirements, ScreenerResult, StockData
from app.scanners.canslim_scanner import CANSLIMScanner
from app.scanners.criteria.relative_strength import RelativeStrengthCalculator
from app.scanners.minervini_scanner import MinerviniScanner
from app.scanners.scan_orchestrator import ScanOrchestrator
from app.scanners.screener_registry import ScreenerRegistry
from app.scanners.setup_engine_screener import SetupEngineScanner


def _make_price_data(num_days: int, *, start_price: float = 100.0) -> pd.DataFrame:
    dates = pd.bdate_range(end=pd.Timestamp("2026-02-27"), periods=num_days)
    close = np.linspace(start_price, start_price + (num_days - 1) * 0.4, num_days)
    return pd.DataFrame(
        {
            "Open": close * 0.995,
            "High": close * 1.02,
            "Low": close * 0.98,
            "Close": close,
            "Volume": np.linspace(1_000_000, 2_000_000, num_days),
        },
        index=dates,
    )


def _make_stock_data(symbol: str = "TEST", num_days: int = 320) -> StockData:
    return StockData(
        symbol=symbol,
        price_data=_make_price_data(num_days, start_price=75.0),
        benchmark_data=_make_price_data(num_days, start_price=300.0),
        fundamentals={
            "institutional_ownership": 72.0,
        },
        quarterly_growth={
            "eps_growth_qq": 35.0,
            "eps_growth_yy": 28.0,
            "sales_growth_qq": 22.0,
            "sales_growth_yy": 18.0,
        },
    )


def _manual_precomputed_context(data: StockData) -> SimpleNamespace:
    close_chrono = data.price_data["Close"].reset_index(drop=True)
    volume_chrono = data.price_data["Volume"].reset_index(drop=True)
    benchmark_close_chrono = data.benchmark_data["Close"].reset_index(drop=True)
    close_rev = close_chrono[::-1].reset_index(drop=True)
    volume_rev = volume_chrono[::-1].reset_index(drop=True)
    benchmark_close_rev = benchmark_close_chrono[::-1].reset_index(drop=True)
    ma_200_series = close_chrono.rolling(window=200, min_periods=200).mean()
    rs_ratings = RelativeStrengthCalculator().calculate_all_rs_ratings(
        data.symbol,
        close_rev,
        benchmark_close_rev,
        data.rs_universe_performances,
    )
    return SimpleNamespace(
        close_chrono=close_chrono,
        close_rev=close_rev,
        volume_chrono=volume_chrono,
        volume_rev=volume_rev,
        benchmark_close_chrono=benchmark_close_chrono,
        benchmark_close_rev=benchmark_close_rev,
        current_price=float(close_chrono.iloc[-1]),
        ma_50=float(close_chrono.rolling(window=50, min_periods=50).mean().iloc[-1]),
        ma_150=float(close_chrono.rolling(window=150, min_periods=150).mean().iloc[-1]),
        ma_200=float(ma_200_series.iloc[-1]),
        ma_200_month_ago=float(ma_200_series.iloc[-21]),
        ema_10=float(close_chrono.ewm(span=10, adjust=False).mean().iloc[-1]),
        ema_20=float(close_chrono.ewm(span=20, adjust=False).mean().iloc[-1]),
        ema_50=float(close_chrono.ewm(span=50, adjust=False).mean().iloc[-1]),
        high_52w=float(close_rev.max()),
        low_52w=float(close_rev.min()),
        rs_ratings=rs_ratings,
    )


class _FakeDataProvider:
    def __init__(self, stock_data: StockData):
        self.stock_data = stock_data

    def prepare_data(self, symbol: str, requirements: object) -> StockData:
        assert symbol == self.stock_data.symbol
        return self.stock_data

    def prepare_data_bulk(
        self,
        symbols: list[str],
        requirements: object,
        *,
        allow_partial: bool = True,
        batch_only_prices: bool = False,
        batch_only_fundamentals: bool = False,
    ) -> dict[str, StockData]:
        return {symbol: self.stock_data for symbol in symbols}


def _make_context_reporting_screener(name: str) -> type[BaseStockScreener]:
    class _ContextReportingScreener(BaseStockScreener):
        @property
        def screener_name(self) -> str:
            return name

        def get_data_requirements(self, criteria=None) -> DataRequirements:
            return DataRequirements(price_period="2y", needs_benchmark=True)

        def scan_stock(self, symbol, data, criteria=None) -> ScreenerResult:
            context = getattr(data, "precomputed_scan_context", None)
            return ScreenerResult(
                score=50.0,
                passes=True,
                rating="Watch",
                breakdown={"score": 50.0},
                details={
                    "has_precomputed": context is not None,
                    "context_id": id(context) if context is not None else None,
                    "ma_50": getattr(context, "ma_50", None),
                },
                screener_name=name,
            )

        def calculate_rating(self, score, details) -> str:
            return "Watch"

    _ContextReportingScreener.__name__ = f"ContextReportingScreener_{name}"
    _ContextReportingScreener.__qualname__ = f"ContextReportingScreener_{name}"
    return _ContextReportingScreener


def test_orchestrator_attaches_shared_precomputed_scan_context():
    stock_data = _make_stock_data()
    provider = _FakeDataProvider(stock_data)
    registry = ScreenerRegistry()
    registry.register(_make_context_reporting_screener("alpha"))
    registry.register(_make_context_reporting_screener("beta"))
    orchestrator = ScanOrchestrator(data_provider=provider, registry=registry)

    result = orchestrator.scan_stock_multi(
        stock_data.symbol,
        ["alpha", "beta"],
        composite_method="weighted_average",
    )

    context = getattr(stock_data, "precomputed_scan_context", None)
    expected_ma_50 = float(
        stock_data.price_data["Close"].reset_index(drop=True).rolling(window=50, min_periods=50).mean().iloc[-1]
    )
    alpha_details = result["details"]["screeners"]["alpha"]["details"]
    beta_details = result["details"]["screeners"]["beta"]["details"]

    assert context is not None
    assert context.ma_50 == pytest.approx(expected_ma_50)
    assert alpha_details["has_precomputed"] is True
    assert beta_details["has_precomputed"] is True
    assert alpha_details["context_id"] == beta_details["context_id"] == id(context)


def test_minervini_uses_precomputed_context_for_rs_ratings(monkeypatch):
    stock_data = _make_stock_data()
    stock_data.precomputed_scan_context = _manual_precomputed_context(stock_data)
    scanner = MinerviniScanner()

    def _raise_if_called(*args, **kwargs):
        raise AssertionError("relative strength should come from precomputed context")

    monkeypatch.setattr(scanner.rs_calc, "calculate_all_rs_ratings", _raise_if_called)

    result = scanner.scan_stock(stock_data.symbol, stock_data, criteria={"include_vcp": False})

    assert result.rating != "Error"
    assert result.details["rs_rating"] == stock_data.precomputed_scan_context.rs_ratings["rs_rating"]
    assert result.details["rs_rating_1m"] == stock_data.precomputed_scan_context.rs_ratings["rs_rating_1m"]
    assert result.details["rs_rating_3m"] == stock_data.precomputed_scan_context.rs_ratings["rs_rating_3m"]
    assert result.details["rs_rating_12m"] == stock_data.precomputed_scan_context.rs_ratings["rs_rating_12m"]


def test_canslim_uses_precomputed_context_for_rs_ratings(monkeypatch):
    stock_data = _make_stock_data()
    stock_data.precomputed_scan_context = _manual_precomputed_context(stock_data)
    scanner = CANSLIMScanner()

    def _raise_if_called(*args, **kwargs):
        raise AssertionError("relative strength should come from precomputed context")

    monkeypatch.setattr(scanner.rs_calc, "calculate_rs_rating", _raise_if_called)
    monkeypatch.setattr(scanner.rs_calc, "calculate_all_rs_ratings", _raise_if_called)

    result = scanner.scan_stock(stock_data.symbol, stock_data)

    assert result.details["rs_rating"] == stock_data.precomputed_scan_context.rs_ratings["rs_rating"]
    assert result.details["rs_rating_1m"] == stock_data.precomputed_scan_context.rs_ratings["rs_rating_1m"]
    assert result.details["rs_rating_3m"] == stock_data.precomputed_scan_context.rs_ratings["rs_rating_3m"]
    assert result.details["rs_rating_12m"] == stock_data.precomputed_scan_context.rs_ratings["rs_rating_12m"]


def test_setup_engine_uses_precomputed_context_for_rs_rating(monkeypatch):
    stock_data = _make_stock_data()
    stock_data.precomputed_scan_context = _manual_precomputed_context(stock_data)
    scanner = SetupEngineScanner()

    def _raise_if_called(*args, **kwargs):
        raise AssertionError("relative strength should come from precomputed context")

    monkeypatch.setattr(scanner._rs_calc, "calculate_rs_rating", _raise_if_called)

    result = scanner.scan_stock(stock_data.symbol, stock_data)

    assert result.rating != "Error"
    assert result.details["setup_engine"]["rs_rating"] == stock_data.precomputed_scan_context.rs_ratings["rs_rating"]
