from __future__ import annotations

from datetime import date
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd

from app.services.breadth_calculator_service import BreadthCalculatorService


def _make_price_df(end_date: date, base_close: float = 100.0) -> pd.DataFrame:
    index = pd.bdate_range(end=end_date, periods=80)
    closes = [base_close + i for i in range(len(index))]
    return pd.DataFrame(
        {
            "Open": closes,
            "High": [c + 1 for c in closes],
            "Low": [c - 1 for c in closes],
            "Close": closes,
            "Volume": [1_000_000] * len(index),
        },
        index=index,
    )


def test_calculate_daily_breadth_uses_bulk_cached_prices(monkeypatch):
    db = MagicMock()
    db.query.return_value.filter.return_value.all.return_value = [
        SimpleNamespace(symbol="AAA"),
        SimpleNamespace(symbol="BBB"),
    ]

    price_cache = MagicMock()
    price_cache.get_many_cached_only.return_value = {
        "AAA": _make_price_df(date(2026, 3, 20), 100.0),
        "BBB": _make_price_df(date(2026, 3, 20), 200.0),
    }
    price_cache.get_historical_data.side_effect = AssertionError(
        "breadth should not use per-symbol historical fetches"
    )

    monkeypatch.setattr(
        "app.services.breadth_calculator_service.PriceCacheService.get_instance",
        staticmethod(lambda: price_cache),
    )

    calculator = BreadthCalculatorService(db)
    monkeypatch.setattr(
        calculator,
        "_calculate_ratios",
        lambda calculation_date: {"ratio_5day": 1.5, "ratio_10day": 2.5},
    )

    metrics = calculator.calculate_daily_breadth(date(2026, 3, 20))

    assert metrics["total_stocks_scanned"] == 2
    assert metrics["skipped_stocks"] == 0
    assert metrics["ratio_5day"] == 1.5
    assert metrics["ratio_10day"] == 2.5
    price_cache.get_many_cached_only.assert_called_once_with(["AAA", "BBB"], period="2y")
    price_cache.get_historical_data.assert_not_called()


def test_calculate_stock_metrics_reads_cached_only(monkeypatch):
    db = MagicMock(spec=[])
    price_cache = MagicMock()
    price_cache.get_cached_only.return_value = _make_price_df(date(2026, 3, 20), 150.0)
    price_cache.get_historical_data.side_effect = AssertionError(
        "breadth should not fall back to historical Yahoo fetches"
    )

    monkeypatch.setattr(
        "app.services.breadth_calculator_service.PriceCacheService.get_instance",
        staticmethod(lambda: price_cache),
    )

    calculator = BreadthCalculatorService(db)
    metrics = calculator._calculate_stock_metrics("AAA", date(2026, 3, 20))

    assert metrics is not None
    assert set(metrics) == {"pct_change_1d", "pct_change_21d", "pct_change_34d", "pct_change_63d"}
    price_cache.get_cached_only.assert_called_once_with("AAA", period="2y")
    price_cache.get_historical_data.assert_not_called()
