import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from app.scanners.base_screener import StockData
from app.scanners.data_preparation import DataPreparationLayer


def _make_price_df(days: int, start_price: float, end_price: float) -> pd.DataFrame:
    dates = pd.date_range(end=pd.Timestamp.today(), periods=days, freq="B")
    close = np.linspace(start_price, end_price, days)
    open_ = close * 0.995
    high = close * 1.01
    low = close * 0.99
    volume = np.linspace(1_500_000, 2_000_000, days).astype(int)

    return pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        },
        index=dates,
    )


def _make_stub_stock_data(symbol: str) -> StockData:
    price_data = _make_price_df(days=300, start_price=100.0, end_price=200.0)
    benchmark_data = _make_price_df(days=300, start_price=100.0, end_price=130.0)

    fundamentals = {
        "market_cap": 50_000_000_000,
        "avg_volume": 1_800_000,
        "institutional_ownership": 65.0,
        "debt_to_equity": 0.2,
        "sector": "Technology",
        "industry": "Software",
        "eps_growth_qq": 35.0,
        "sales_growth_qq": 20.0,
        "eps_growth_yy": 25.0,
        "sales_growth_yy": 18.0,
        "recent_quarter_date": "2025-12-31",
        "previous_quarter_date": "2025-09-30",
        "first_trade_date": (datetime.now() - timedelta(days=365)).timestamp(),
        "ipo_date": (datetime.now() - timedelta(days=365)).date(),
    }

    quarterly_growth = {
        "eps_growth_qq": fundamentals["eps_growth_qq"],
        "sales_growth_qq": fundamentals["sales_growth_qq"],
        "eps_growth_yy": fundamentals["eps_growth_yy"],
        "sales_growth_yy": fundamentals["sales_growth_yy"],
        "recent_quarter_date": fundamentals["recent_quarter_date"],
        "previous_quarter_date": fundamentals["previous_quarter_date"],
    }

    return StockData(
        symbol=symbol,
        price_data=price_data,
        benchmark_data=benchmark_data,
        fundamentals=fundamentals,
        quarterly_growth=quarterly_growth,
        earnings_history=None,
        fetch_errors={},
    )


@pytest.fixture(autouse=True)
def _stub_external_data(monkeypatch):
    """
    Stub external data sources for unit tests to avoid network/Redis access.
    """

    def _prepare_data_stub(self, symbol, requirements, *, allow_partial=True):
        return _make_stub_stock_data(symbol)

    def _prepare_data_bulk_stub(self, symbols, requirements, *, allow_partial=True):
        return {symbol: _make_stub_stock_data(symbol) for symbol in symbols}

    monkeypatch.setattr(DataPreparationLayer, "prepare_data", _prepare_data_stub, raising=True)
    monkeypatch.setattr(DataPreparationLayer, "prepare_data_bulk", _prepare_data_bulk_stub, raising=True)

    # Avoid Redis connection attempts in unit tests
    from app.services import redis_pool

    monkeypatch.setattr(redis_pool, "get_redis_pool", lambda: None, raising=True)
    monkeypatch.setattr(redis_pool, "get_redis_client", lambda: None, raising=True)
    monkeypatch.setattr(redis_pool, "get_bulk_redis_pool", lambda: None, raising=True)
    monkeypatch.setattr(redis_pool, "get_bulk_redis_client", lambda: None, raising=True)
    redis_pool._pool = None
    redis_pool._bulk_pool = None

    yield
