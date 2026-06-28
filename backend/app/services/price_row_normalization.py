"""Shared normalization for persisted OHLCV price rows."""

from __future__ import annotations

from datetime import date
from typing import Any, Mapping

import pandas as pd

from app.infra.serialization import finite_float_or_none


def drop_non_finite_close_rows(data: pd.DataFrame | None) -> pd.DataFrame | None:
    """Remove rows whose close cannot be safely treated as a market price."""
    if data is None or data.empty:
        return data
    if "Close" not in data.columns:
        return data.iloc[0:0].copy()
    keep_mask = data["Close"].map(finite_float_or_none).notna()
    if bool(keep_mask.all()):
        return data
    return data.loc[keep_mask].copy()


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
    """Build a StockPrice mapping, skipping rows with no finite close."""
    close = finite_float_or_none(row.get("Close"))
    if close is None:
        return None
    adj_close = finite_float_or_none(row.get("Adj Close"))
    return {
        "symbol": symbol,
        "date": row_date,
        "open": finite_float_or_none(row.get("Open")),
        "high": finite_float_or_none(row.get("High")),
        "low": finite_float_or_none(row.get("Low")),
        "close": close,
        "volume": _volume_or_zero(row.get("Volume")),
        "adj_close": adj_close if adj_close is not None else close,
    }
