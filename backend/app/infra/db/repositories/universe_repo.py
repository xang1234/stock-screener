"""SQLAlchemy implementation of UniverseRepository."""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.domain.scanning.ports import UniverseRepository
from app.models.stock_universe import StockUniverse
from app.services.universe_resolver import resolve_symbols as _resolve


class SqlUniverseRepository(UniverseRepository):
    """Resolve universe symbols using the existing universe_resolver service."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def resolve_symbols(self, universe_def: object) -> list[str]:
        return _resolve(self._session, universe_def)

    def resolve_markets(self, symbols: list[str]) -> dict[str, str]:
        normalized = sorted({str(s).strip().upper() for s in symbols if str(s).strip()})
        if not normalized:
            return {}
        rows = (
            self._session.query(StockUniverse.symbol, StockUniverse.market)
            .filter(StockUniverse.symbol.in_(normalized))
            .all()
        )
        return {
            symbol: str(market).strip().upper()
            for symbol, market in rows
            if market and str(market).strip()
        }
