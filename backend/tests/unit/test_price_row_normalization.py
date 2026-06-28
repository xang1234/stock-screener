from __future__ import annotations

from datetime import date

import pandas as pd

from app.services.price_row_normalization import (
    drop_non_finite_close_rows,
    normalize_price_batch,
    normalize_price_frame,
    stock_price_row_from_ohlcv,
)


def _ohlcv_frame(closes: list[float], days: list[date]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Open": closes,
            "High": closes,
            "Low": closes,
            "Close": closes,
            "Volume": [1_000_000] * len(closes),
        },
        index=pd.to_datetime(days),
    )


def test_drop_non_finite_close_rows_removes_unusable_market_prices():
    payload = _ohlcv_frame(
        [101.0, float("nan"), float("inf")],
        [date(2026, 6, 24), date(2026, 6, 25), date(2026, 6, 26)],
    )

    cleaned = drop_non_finite_close_rows(payload)

    assert cleaned is not None
    assert cleaned["Close"].tolist() == [101.0]
    assert cleaned.index.tolist() == [pd.Timestamp(date(2026, 6, 24))]


def test_drop_non_finite_close_rows_treats_missing_close_as_empty_price_frame():
    payload = pd.DataFrame(
        {"Open": [100.0], "Volume": [1_000_000]},
        index=pd.to_datetime([date(2026, 6, 24)]),
    )

    cleaned = drop_non_finite_close_rows(payload)

    assert cleaned is not None
    assert cleaned.empty
    assert list(cleaned.columns) == ["Open", "Volume"]


def test_stock_price_row_from_ohlcv_skips_rows_without_finite_close():
    row = pd.Series({"Open": 100.0, "Close": float("nan"), "Volume": 1_000_000})

    assert stock_price_row_from_ohlcv(symbol="SPY", row_date=date(2026, 6, 24), row=row) is None


def test_stock_price_row_from_ohlcv_skips_rows_without_complete_finite_ohlc():
    row = pd.Series(
        {
            "Open": 100.0,
            "High": float("nan"),
            "Low": 99.0,
            "Close": 101.0,
            "Volume": 1_000_000,
        }
    )

    assert stock_price_row_from_ohlcv(symbol="SPY", row_date=date(2026, 6, 24), row=row) is None


def test_normalize_price_frame_enforces_min_rows_after_filtering():
    payload = _ohlcv_frame([101.0, float("nan")], [date(2026, 6, 24), date(2026, 6, 25)])

    assert normalize_price_frame(payload, min_rows=2) is None

    cleaned = normalize_price_frame(payload, min_rows=1)

    assert cleaned is not None
    assert cleaned["Close"].tolist() == [101.0]


def test_normalize_price_frame_removes_rows_with_incomplete_ohlc_values():
    payload = pd.DataFrame(
        {
            "Open": [100.0, float("nan")],
            "High": [102.0, 103.0],
            "Low": [99.0, 100.0],
            "Close": [101.0, 102.0],
            "Volume": [1_000_000, 1_000_000],
        },
        index=pd.to_datetime([date(2026, 6, 24), date(2026, 6, 25)]),
    )

    cleaned = normalize_price_frame(payload)

    assert cleaned is not None
    assert cleaned["Close"].tolist() == [101.0]
    assert cleaned.index.tolist() == [pd.Timestamp(date(2026, 6, 24))]


def test_normalize_price_batch_filters_symbols_with_insufficient_clean_rows():
    enough_rows = _ohlcv_frame([101.0, 102.0], [date(2026, 6, 24), date(2026, 6, 25)])
    insufficient_after_filter = _ohlcv_frame([103.0, float("nan")], [date(2026, 6, 24), date(2026, 6, 25)])
    no_close = pd.DataFrame(
        {"Open": [100.0, 101.0], "Volume": [1_000_000, 1_000_000]},
        index=pd.to_datetime([date(2026, 6, 24), date(2026, 6, 25)]),
    )

    cleaned = normalize_price_batch(
        {"AAPL": enough_rows, "MSFT": insufficient_after_filter, "BAD": no_close},
        min_rows=2,
    )

    assert list(cleaned) == ["AAPL"]
    assert cleaned["AAPL"]["Close"].tolist() == [101.0, 102.0]
