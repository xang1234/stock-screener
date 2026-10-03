"""52-week high/low must come from the newest 252 bars (issue #411).

Scans load 2y or 5y of history depending on the screener mix and cache tier,
so a range computed over the whole series changes with context. Each fixture
puts an extreme spike and dip in the *older* history; the newest 252 closes
run from 100 to 200, so the correct 52-week range is (200, 100).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.scanners.base_screener import StockData
from app.scanners.canslim_scanner import CANSLIMScanner
from app.scanners.minervini_scanner import MinerviniScanner
from app.scanners.scan_orchestrator import _build_precomputed_scan_context

RECENT_HIGH = 200.0
RECENT_LOW = 100.0


def _frame(close: np.ndarray) -> pd.DataFrame:
    dates = pd.bdate_range(end=pd.Timestamp("2026-02-27"), periods=len(close))
    return pd.DataFrame(
        {
            "Open": close * 0.995,
            "High": close * 1.02,
            "Low": close * 0.98,
            "Close": close,
            "Volume": np.full(len(close), 1_500_000.0),
        },
        index=dates,
    )


def _stock_data(total_bars: int, *, older_spike: float = 1_000.0, older_dip: float = 1.0) -> StockData:
    recent = np.linspace(RECENT_LOW, RECENT_HIGH, 252)
    older = np.full(total_bars - 252, 150.0)
    older[len(older) // 3] = older_spike
    older[2 * len(older) // 3] = older_dip
    close = np.concatenate([older, recent])
    benchmark = np.linspace(300.0, 400.0, total_bars)
    return StockData(
        symbol="TEST",
        price_data=_frame(close),
        benchmark_data=_frame(benchmark),
        fundamentals={"institutional_ownership": 72.0},
        quarterly_growth={
            "eps_growth_qq": 35.0,
            "eps_growth_yy": 28.0,
            "sales_growth_qq": 22.0,
            "sales_growth_yy": 18.0,
        },
    )


@pytest.mark.parametrize("total_bars", [504, 1260])
def test_precomputed_context_range_uses_newest_252_bars(total_bars):
    context = _build_precomputed_scan_context(_stock_data(total_bars))

    assert context.high_52w == pytest.approx(RECENT_HIGH)
    assert context.low_52w == pytest.approx(RECENT_LOW)


def test_minervini_fallback_range_uses_newest_252_bars():
    data = _stock_data(504)
    assert data.precomputed_scan_context is None

    result = MinerviniScanner().scan_stock(data.symbol, data, criteria={"include_vcp": False})

    assert result.details["high_52w"] == pytest.approx(RECENT_HIGH)
    assert result.details["low_52w"] == pytest.approx(RECENT_LOW)


@pytest.mark.parametrize("with_precomputed", [False, True])
def test_canslim_new_highs_uses_newest_252_bars(with_precomputed):
    data = _stock_data(504)
    if with_precomputed:
        data.precomputed_scan_context = _build_precomputed_scan_context(data)

    result = CANSLIMScanner().scan_stock(data.symbol, data)

    n_result = result.details["full_analysis"]["N_new_highs"]
    assert n_result["high_52w"] == pytest.approx(RECENT_HIGH)
    assert result.details["from_52w_high_pct"] == pytest.approx(0.0)


def test_minervini_range_is_independent_of_history_length():
    """Same newest year, different older history (2y vs 5y) -> same 52-week verdict."""
    two_year = _stock_data(504, older_spike=500.0, older_dip=50.0)
    five_year = _stock_data(1260, older_spike=2_000.0, older_dip=5.0)
    for data in (two_year, five_year):
        data.precomputed_scan_context = _build_precomputed_scan_context(data)

    scanner = MinerviniScanner()
    keys = ("high_52w", "low_52w", "above_52w_low_pct", "from_52w_high_pct")
    two_year_details = scanner.scan_stock("TEST", two_year, criteria={"include_vcp": False}).details
    five_year_details = scanner.scan_stock("TEST", five_year, criteria={"include_vcp": False}).details

    assert {k: two_year_details[k] for k in keys} == {k: five_year_details[k] for k in keys}


def test_52_week_position_endpoint_uses_newest_252_bars(monkeypatch):
    from app.api.v1 import technical

    price_data = _stock_data(504).price_data

    class _StubYFinance:
        def get_historical_data(self, symbol, period):
            return price_data

    monkeypatch.setattr(technical, "get_yfinance_service", lambda: _StubYFinance())

    response = technical.get_52w_position("test")

    assert response["high_52w"] == pytest.approx(RECENT_HIGH)
    assert response["low_52w"] == pytest.approx(RECENT_LOW)
    assert response["meets_low_criteria"] is True


def test_short_history_uses_all_available_bars():
    close = np.linspace(50.0, 80.0, 120)
    data = StockData(symbol="IPO", price_data=_frame(close), benchmark_data=_frame(close))

    context = _build_precomputed_scan_context(data)

    assert context.high_52w == pytest.approx(80.0)
    assert context.low_52w == pytest.approx(50.0)
