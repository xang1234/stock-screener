"""Shared normalization for persisted OHLCV price rows."""

from __future__ import annotations

from datetime import date
from typing import Any, Mapping

import pandas as pd

from app.infra.serialization import finite_float_or_none

OHLC_COLUMNS = ("Open", "High", "Low", "Close")


def drop_non_finite_close_rows(data: pd.DataFrame | None) -> pd.DataFrame | None:
    """Remove rows whose OHLC values cannot be safely treated as market prices."""
    if data is None or data.empty:
        return data
    if any(column not in data.columns for column in OHLC_COLUMNS):
        return data.iloc[0:0].copy()
    keep_mask = pd.Series(True, index=data.index)
    for column in OHLC_COLUMNS:
        keep_mask &= data[column].map(finite_float_or_none).notna()
    if bool(keep_mask.all()):
        return data
    return data.loc[keep_mask].copy()


def normalize_price_frame(
    data: pd.DataFrame | None,
    *,
    min_rows: int = 1,
) -> pd.DataFrame | None:
    """Return a finite-close OHLCV frame that satisfies the row-count contract."""
    cleaned = drop_non_finite_close_rows(data)
    if cleaned is None or cleaned.empty:
        return None
    if len(cleaned) < min_rows:
        return None
    return cleaned


def normalize_price_batch(
    batch_data: Mapping[str, pd.DataFrame | None],
    *,
    min_rows: int = 1,
) -> dict[str, pd.DataFrame]:
    """Normalize a symbol->price-frame batch and drop unusable symbols."""
    normalized: dict[str, pd.DataFrame] = {}
    for symbol, data in batch_data.items():
        cleaned = normalize_price_frame(data, min_rows=min_rows)
        if cleaned is not None:
            normalized[symbol] = cleaned
    return normalized


def _volume_or_zero(value: Any) -> int:
    number = finite_float_or_none(value)
    if number is None:
        return 0
    return int(number)


def stock_price_row_from_ohlcv(
    *,
    symbol: str,
    row_date: date,
    row: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Build a StockPrice mapping, skipping rows without complete finite OHLC."""
    open_ = finite_float_or_none(row.get("Open"))
    high = finite_float_or_none(row.get("High"))
    low = finite_float_or_none(row.get("Low"))
    close = finite_float_or_none(row.get("Close"))
    if open_ is None or high is None or low is None or close is None:
        return None
    adj_close = finite_float_or_none(row.get("Adj Close"))
    return {
        "symbol": symbol,
        "date": row_date,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": _volume_or_zero(row.get("Volume")),
        "adj_close": adj_close if adj_close is not None else close,
    }
