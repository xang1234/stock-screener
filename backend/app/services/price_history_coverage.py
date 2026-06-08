"""Coverage classification for persisted daily price history."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

from sqlalchemy import func
from sqlalchemy.orm import Session

from ..models.stock import StockPrice


@dataclass(frozen=True)
class PriceHistoryCoverage:
    fresh: tuple[str, ...] = ()
    stale: tuple[str, ...] = ()
    no_history: tuple[str, ...] = ()

    @property
    def refresh_symbols(self) -> tuple[str, ...]:
        return self.stale + self.no_history


def _normalize_symbols(symbols: Sequence[str]) -> tuple[str, ...]:
    return tuple(str(symbol).upper() for symbol in symbols)


def classify_price_history(
    db: Session,
    *,
    symbols: Sequence[str],
    as_of_date: date,
) -> PriceHistoryCoverage:
    """Split symbols by whether persisted prices already cover ``as_of_date``."""
    normalized_symbols = _normalize_symbols(symbols)
    latest_by_symbol: dict[str, date | None] = {}
    for chunk_start in range(0, len(normalized_symbols), 500):
        chunk_symbols = normalized_symbols[chunk_start:chunk_start + 500]
        rows = (
            db.query(StockPrice.symbol, func.max(StockPrice.date))
            .filter(StockPrice.symbol.in_(chunk_symbols))
            .group_by(StockPrice.symbol)
            .all()
        )
        latest_by_symbol.update(
            {str(symbol).upper(): latest_date for symbol, latest_date in rows}
        )

    fresh_symbols: list[str] = []
    stale_symbols: list[str] = []
    no_history_symbols: list[str] = []
    for symbol in normalized_symbols:
        latest_date = latest_by_symbol.get(symbol)
        if latest_date is None:
            no_history_symbols.append(symbol)
        elif latest_date < as_of_date:
            stale_symbols.append(symbol)
        else:
            fresh_symbols.append(symbol)

    return PriceHistoryCoverage(
        fresh=tuple(fresh_symbols),
        stale=tuple(stale_symbols),
        no_history=tuple(no_history_symbols),
    )
