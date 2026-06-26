"""Coverage classification for persisted daily price history."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

from sqlalchemy import and_, func
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


def _has_positive_volume(value: object) -> bool:
    if value is None:
        return False
    try:
        return float(value) > 0
    except (TypeError, ValueError):
        return False


def _record_latest_dates(
    db: Session,
    *,
    chunk_symbols: Sequence[str],
    latest_by_symbol: dict[str, date | None],
) -> None:
    rows = (
        db.query(StockPrice.symbol, func.max(StockPrice.date))
        .filter(StockPrice.symbol.in_(chunk_symbols))
        .group_by(StockPrice.symbol)
        .all()
    )
    latest_by_symbol.update(
        {str(symbol).upper(): latest_date for symbol, latest_date in rows}
    )


def _record_latest_rows(
    db: Session,
    *,
    chunk_symbols: Sequence[str],
    latest_by_symbol: dict[str, date | None],
    latest_volume_by_symbol: dict[str, object],
) -> None:
    latest_dates = (
        db.query(
            StockPrice.symbol.label("symbol"),
            func.max(StockPrice.date).label("latest_date"),
        )
        .filter(StockPrice.symbol.in_(chunk_symbols))
        .group_by(StockPrice.symbol)
        .subquery()
    )
    rows = (
        db.query(StockPrice.symbol, StockPrice.date, StockPrice.volume)
        .join(
            latest_dates,
            and_(
                StockPrice.symbol == latest_dates.c.symbol,
                StockPrice.date == latest_dates.c.latest_date,
            ),
        )
        .all()
    )
    for symbol, latest_date, volume in rows:
        key = str(symbol).upper()
        latest_by_symbol[key] = latest_date
        latest_volume_by_symbol[key] = volume


def classify_price_history(
    db: Session,
    *,
    symbols: Sequence[str],
    as_of_date: date,
    require_positive_volume: bool = False,
    symbols_requiring_positive_volume: Sequence[str] | None = None,
) -> PriceHistoryCoverage:
    """Split symbols by whether persisted prices already cover ``as_of_date``."""
    normalized_symbols = _normalize_symbols(symbols)
    if require_positive_volume:
        positive_volume_symbols = set(normalized_symbols)
    else:
        positive_volume_symbols = set(
            _normalize_symbols(symbols_requiring_positive_volume or ())
        )
    latest_by_symbol: dict[str, date | None] = {}
    latest_volume_by_symbol: dict[str, object] = {}
    for chunk_start in range(0, len(normalized_symbols), 500):
        chunk_symbols = normalized_symbols[chunk_start:chunk_start + 500]
        if positive_volume_symbols:
            _record_latest_rows(
                db,
                chunk_symbols=chunk_symbols,
                latest_by_symbol=latest_by_symbol,
                latest_volume_by_symbol=latest_volume_by_symbol,
            )
        else:
            _record_latest_dates(
                db,
                chunk_symbols=chunk_symbols,
                latest_by_symbol=latest_by_symbol,
            )

    fresh_symbols: list[str] = []
    stale_symbols: list[str] = []
    no_history_symbols: list[str] = []
    for symbol in normalized_symbols:
        latest_date = latest_by_symbol.get(symbol)
        latest_volume = latest_volume_by_symbol.get(symbol)
        if latest_date is None:
            no_history_symbols.append(symbol)
        elif latest_date < as_of_date:
            stale_symbols.append(symbol)
        elif symbol in positive_volume_symbols and not _has_positive_volume(latest_volume):
            stale_symbols.append(symbol)
        else:
            fresh_symbols.append(symbol)

    return PriceHistoryCoverage(
        fresh=tuple(fresh_symbols),
        stale=tuple(stale_symbols),
        no_history=tuple(no_history_symbols),
    )
