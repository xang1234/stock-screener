"""Point-in-time reconstruction of Market universe membership."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
import hashlib
from zoneinfo import ZoneInfo

from sqlalchemy.orm import Session

from app.domain.markets.catalog import MarketCatalog, get_market_catalog
from app.models.stock_universe import (
    StockUniverse,
    StockUniverseStatusEvent,
    UNIVERSE_EVENT_STATUS_CHANGED,
    UNIVERSE_STATUS_ACTIVE,
)
from app.services.market_calendar_service import MarketCalendarService


@dataclass(frozen=True)
class PointInTimeUniverse:
    market: str
    as_of_date: date
    symbols: tuple[str, ...]
    universe_hash: str


class PointInTimeUniverseUnavailable(RuntimeError):
    """Raised when historical lifecycle evidence cannot reproduce membership."""


class PointInTimeUniverseService:
    def __init__(
        self,
        *,
        market_calendar: MarketCalendarService | None = None,
        market_catalog: MarketCatalog | None = None,
    ) -> None:
        self._market_calendar = market_calendar or MarketCalendarService()
        self._market_catalog = market_catalog or get_market_catalog()

    @staticmethod
    def _universe_hash(symbols: tuple[str, ...]) -> str:
        payload = "".join(f"{symbol}\n" for symbol in symbols).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _snapshot(
        self,
        *,
        market: str,
        as_of_date: date,
        symbols: tuple[str, ...],
    ) -> PointInTimeUniverse:
        return PointInTimeUniverse(
            market=market,
            as_of_date=as_of_date,
            symbols=symbols,
            universe_hash=self._universe_hash(symbols),
        )

    def resolve(
        self,
        db: Session,
        *,
        market: str,
        as_of_date: date,
    ) -> PointInTimeUniverse:
        normalized = self._market_calendar.normalize_market(market)
        if as_of_date == self._market_calendar.market_now(normalized).date():
            symbols = tuple(
                row[0]
                for row in db.query(StockUniverse.symbol)
                .filter(
                    StockUniverse.market == normalized,
                    StockUniverse.active_filter(),
                )
                .order_by(StockUniverse.symbol.asc())
                .all()
            )
            return self._snapshot(
                market=normalized,
                as_of_date=as_of_date,
                symbols=symbols,
            )

        market_timezone = ZoneInfo(
            self._market_catalog.get(normalized).display_timezone
        )
        cutoff = datetime.combine(
            as_of_date + timedelta(days=1),
            time.min,
            tzinfo=market_timezone,
        ).astimezone(timezone.utc)

        candidates = tuple(
            row[0]
            for row in db.query(StockUniverse.symbol)
            .filter(
                StockUniverse.market == normalized,
                StockUniverse.first_seen_at < cutoff,
            )
            .order_by(StockUniverse.symbol.asc())
            .all()
        )
        if not candidates:
            return self._snapshot(
                market=normalized,
                as_of_date=as_of_date,
                symbols=(),
            )

        events = (
            db.query(StockUniverseStatusEvent)
            .filter(
                StockUniverseStatusEvent.symbol.in_(candidates),
                StockUniverseStatusEvent.event_type
                == UNIVERSE_EVENT_STATUS_CHANGED,
                StockUniverseStatusEvent.created_at < cutoff,
            )
            .order_by(
                StockUniverseStatusEvent.symbol.asc(),
                StockUniverseStatusEvent.created_at.desc(),
                StockUniverseStatusEvent.id.desc(),
            )
            .all()
        )
        latest_by_symbol: dict[str, StockUniverseStatusEvent] = {}
        for event in events:
            latest_by_symbol.setdefault(event.symbol, event)

        missing = tuple(
            symbol for symbol in candidates if symbol not in latest_by_symbol
        )
        if missing:
            raise PointInTimeUniverseUnavailable(
                f"{normalized} historical universe for {as_of_date.isoformat()} "
                f"is missing lifecycle events for: {', '.join(missing)}"
            )

        symbols = tuple(
            symbol
            for symbol in candidates
            if latest_by_symbol[symbol].new_status == UNIVERSE_STATUS_ACTIVE
        )
        return self._snapshot(
            market=normalized,
            as_of_date=as_of_date,
            symbols=symbols,
        )
