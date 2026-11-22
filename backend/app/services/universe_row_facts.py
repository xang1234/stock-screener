"""StockUniverse row facts used by market-aware enrichment paths."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from sqlalchemy.orm import Session

from ..domain.markets.catalog import get_market_catalog
from ..models.stock_universe import StockUniverse

logger = logging.getLogger(__name__)


def normalize_universe_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().upper()
    return normalized or None


@dataclass(frozen=True, slots=True)
class UniverseRowFacts:
    market: str | None = None
    currency: str | None = None


class UniverseRowFactsResolver:
    """Resolve row-level market facts without coupling callers to StockUniverse."""

    def __init__(self, session_factory: Callable[[], Session]) -> None:
        self._session_factory = session_factory

    def resolve(self, symbol: str) -> UniverseRowFacts:
        db = None
        try:
            db = self._session_factory()
            row = (
                db.query(StockUniverse.market, StockUniverse.currency)
                .filter(StockUniverse.symbol == symbol)
                .first()
            )
            if not row:
                return UniverseRowFacts()
            return UniverseRowFacts(
                market=_row_text(row, 0, "market"),
                currency=_row_text(row, 1, "currency"),
            )
        except Exception as exc:  # pragma: no cover - DB hiccup shouldn't block fetch
            logger.warning("Universe row fact lookup failed for %s (%s)", symbol, exc)
            return UniverseRowFacts()
        finally:
            if db is not None:
                db.close()

    def resolve_for_payload(
        self,
        symbol: str,
        payload: Mapping[str, Any],
        *,
        market: str | None = None,
        currency: str | None = None,
        fallback: UniverseRowFacts | None = None,
    ) -> UniverseRowFacts:
        payload_market = normalize_universe_text(market) or normalize_universe_text(
            payload.get("market")
        )
        payload_currency = normalize_universe_text(currency) or normalize_universe_text(
            payload.get("currency")
        )
        if payload_market and payload_currency:
            return UniverseRowFacts(market=payload_market, currency=payload_currency)

        row_facts = fallback or self.resolve(symbol)
        return UniverseRowFacts(
            market=payload_market or row_facts.market,
            currency=payload_currency or row_facts.currency,
        )


def active_universe_currency_drift(db: Session) -> list[dict[str, object]]:
    catalog = get_market_catalog()
    drift: list[dict[str, object]] = []
    rows = (
        db.query(StockUniverse.symbol, StockUniverse.market, StockUniverse.currency)
        .filter(StockUniverse.active_filter())
        .all()
    )
    for symbol, market, currency in rows:
        market_code = normalize_universe_text(market)
        row_currency = normalize_universe_text(currency)
        entry = catalog.get(market_code)
        if row_currency not in entry.supported_currencies:
            drift.append(
                {
                    "symbol": symbol,
                    "market": market_code,
                    "currency": row_currency,
                    "supported_currencies": entry.supported_currencies,
                }
            )
    return drift


def active_universe_timezone_drift(db: Session) -> list[dict[str, object]]:
    catalog = get_market_catalog()
    drift: list[dict[str, object]] = []
    rows = (
        db.query(
            StockUniverse.symbol,
            StockUniverse.market,
            StockUniverse.exchange,
            StockUniverse.timezone,
        )
        .filter(StockUniverse.active_filter())
        .all()
    )
    for symbol, market, mic, timezone in rows:
        market_code = normalize_universe_text(market)
        row_mic = normalize_universe_text(mic)
        expected_timezone = catalog.get(market_code).mic_facts_for(row_mic).timezone
        if timezone != expected_timezone:
            drift.append(
                {
                    "symbol": symbol,
                    "market": market_code,
                    "mic": row_mic,
                    "timezone": timezone,
                    "expected_timezone": expected_timezone,
                }
            )
    return drift


def _row_text(row: Any, index: int, attr: str) -> str | None:
    value = getattr(row, attr, None)
    if value is None:
        try:
            value = row[index]
        except (IndexError, KeyError, TypeError):
            return None
    return normalize_universe_text(value)
