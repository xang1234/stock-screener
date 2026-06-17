"""Market-context card entries (close history + multi-period changes) from the database."""

from __future__ import annotations

from collections import defaultdict
from datetime import date, timedelta
from typing import Any

from sqlalchemy.orm import Session

from app.domain.markets.market_context import market_context_instruments
from app.models.stock import StockPrice

MARKET_CONTEXT_HISTORY_POINTS = 64

# Roughly 3 months of trading days
MARKET_CONTEXT_HISTORY_CALENDAR_DAYS = 120

def _pct_change(
    history: list[dict[str, Any]],
    periods_back: int,
) -> float | None:
    """
    Calculate percent change from N trading periods ago.
    """

    if len(history) <= periods_back:
        return None

    latest = history[-1]
    reference = history[-(periods_back + 1)]

    latest_close = latest.get("close")
    reference_close = reference.get("close")

    if (
        latest_close is None
        or reference_close in (None, 0)
    ):
        return None

    return round(
        ((latest_close - reference_close) / reference_close) * 100,
        2,
    )


def build_market_context_entries(
    db: Session,
    market: str | None = None,
    *,
    points: int = MARKET_CONTEXT_HISTORY_POINTS,
) -> list[dict[str, Any]]:
    instruments = market_context_instruments(market)

    if not instruments:
        return []

    data_symbols = [
        instrument.data_symbol
        for instrument in instruments
    ]

    cutoff = date.today() - timedelta(
        days=MARKET_CONTEXT_HISTORY_CALENDAR_DAYS
    )

    rows = (
        db.query(
            StockPrice.symbol,
            StockPrice.date,
            StockPrice.close,
        )
        .filter(
            StockPrice.symbol.in_(data_symbols),
            StockPrice.date >= cutoff,
        )
        .order_by(
            StockPrice.symbol.asc(),
            StockPrice.date.asc(),
        )
        .all()
    )

    history_by_symbol: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for symbol, row_date, close in rows:
        history_by_symbol[str(symbol).upper()].append(
            {
                "date": row_date.isoformat(),
                "close": close,
            }
        )

    entries: list[dict[str, Any]] = []

    for instrument in instruments:
        history = history_by_symbol.get(
            instrument.data_symbol.upper(),
            [],
        )[-points:]

        latest = history[-1] if history else None

        entries.append(
            {
                "category": instrument.category,
                "symbol": instrument.display_symbol,
                "data_symbol": instrument.data_symbol,
                "display_name": instrument.display_name,
                "currency": instrument.currency,
                "latest_close": (
                    latest["close"]
                    if latest is not None
                    else None
                ),
                "latest_date": (
                    latest["date"]
                    if latest is not None
                    else None
                ),

                # Market Cartographer metrics
                "change_1d": _pct_change(history, 1),
                "change_5d": _pct_change(history, 5),
                "change_21d": _pct_change(history, 21),
                "change_63d": _pct_change(history, 63),

                # Raw history retained for charts
                "history": history,
            }
        )

    return entries