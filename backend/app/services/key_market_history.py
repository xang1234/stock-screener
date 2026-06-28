"""Key-market card entries (close history + 1-day change) from the database.

Canonical builder for the Daily Snapshot key-market cards, shared by the
server-mode snapshot endpoint and tests that exercise the same shape the
static-site exporter publishes in ``home.json``.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date, timedelta
from typing import Any

from sqlalchemy.orm import Session

from app.domain.markets.key_markets import key_market_instruments
from app.infra.serialization import finite_float_or_none
from app.models.stock import StockPrice

KEY_MARKET_HISTORY_POINTS = 30
# Calendar window wide enough to cover 30 trading days across holidays.
KEY_MARKET_HISTORY_CALENDAR_DAYS = 60


def build_key_market_entries(
    db: Session,
    market: str,
    *,
    points: int = KEY_MARKET_HISTORY_POINTS,
) -> list[dict[str, Any]]:
    instruments = key_market_instruments(market)
    if not instruments:
        return []
    data_symbols = [instrument.data_symbol for instrument in instruments]
    cutoff = date.today() - timedelta(days=KEY_MARKET_HISTORY_CALENDAR_DAYS)
    rows = (
        db.query(StockPrice.symbol, StockPrice.date, StockPrice.close)
        .filter(
            StockPrice.symbol.in_(data_symbols),
            StockPrice.date >= cutoff,
        )
        .order_by(StockPrice.symbol.asc(), StockPrice.date.asc())
        .all()
    )
    history_by_symbol: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for symbol, row_date, close in rows:
        history_by_symbol[str(symbol).upper()].append(
            {"date": row_date.isoformat(), "close": finite_float_or_none(close)}
        )

    entries: list[dict[str, Any]] = []
    for instrument in instruments:
        history = history_by_symbol.get(instrument.data_symbol.upper(), [])[-points:]
        # The latest row is reported as-is: a null close stays null rather
        # than falling back to an older close, so no change is fabricated.
        latest = history[-1] if history else None
        previous = history[-2] if len(history) > 1 else None
        change_1d = None
        if (
            latest is not None
            and previous is not None
            and latest["close"] is not None
            and previous["close"] not in (None, 0)
        ):
            change_1d = round(
                ((latest["close"] - previous["close"]) / previous["close"]) * 100, 2
            )
        entries.append(
            {
                "symbol": instrument.display_symbol,
                "display_name": instrument.display_name,
                "currency": instrument.currency,
                "latest_close": latest["close"] if latest is not None else None,
                "latest_date": latest["date"] if latest is not None else None,
                "change_1d": change_1d,
                "history": history,
            }
        )
    return entries
