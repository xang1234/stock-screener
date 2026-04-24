"""Symbol-scoped staleness check for manual scans.

Returns a 409 detail payload when any symbol the scan will process lacks cached
prices through its market's last completed trading day. Scoped to the resolved
universe, so unrelated data-quality issues on other symbols don't false-positive.
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional

from sqlalchemy import func

from ..database import SessionLocal
from ..models.stock import StockPrice
from ..models.stock_universe import StockUniverse
from ..wiring.bootstrap import get_market_calendar_service

logger = logging.getLogger(__name__)


def check_symbol_freshness(symbols: Iterable[str]) -> Optional[dict]:
    """Return a 409 detail dict when any symbol lacks fresh cached prices.

    Groups the requested symbols by their universe market, compares each group
    to that market's last completed trading day, and returns a structured
    stale-markets payload if any group fails. Returns None when every market
    covered by the universe is fresh.
    """
    normalized = sorted({s.upper() for s in symbols if s})
    if not normalized:
        return None

    session = SessionLocal()
    try:
        rows = (
            session.query(
                StockUniverse.symbol.label("symbol"),
                StockUniverse.market.label("market"),
                func.max(StockPrice.date).label("last_date"),
            )
            .outerjoin(StockPrice, StockPrice.symbol == StockUniverse.symbol)
            .filter(StockUniverse.symbol.in_(normalized))
            .group_by(StockUniverse.symbol, StockUniverse.market)
            .all()
        )
    finally:
        session.close()

    per_market: dict[str, list] = {}
    for row in rows:
        per_market.setdefault(row.market, []).append(row)

    calendar = get_market_calendar_service()
    stale_markets: list[dict] = []
    for market, market_rows in sorted(per_market.items()):
        try:
            expected_date = calendar.last_completed_trading_day(market)
        except Exception:
            logger.warning(
                "Could not resolve last completed trading day for market=%s; skipping staleness check",
                market,
                exc_info=True,
            )
            continue

        uncovered = sum(1 for r in market_rows if r.last_date is None)
        covered_dates = [r.last_date for r in market_rows if r.last_date is not None]
        oldest = min(covered_dates) if covered_dates else None

        if uncovered > 0 or oldest is None or oldest < expected_date:
            stale_markets.append({
                "market": market,
                "total_symbols": len(market_rows),
                "covered_symbols": len(covered_dates),
                "uncovered_symbols": uncovered,
                "oldest_last_cached_date": str(oldest) if oldest is not None else None,
                "expected_date": str(expected_date),
            })

    if not stale_markets:
        return None

    summary = ", ".join(
        f"{m['market']} (oldest: {m['oldest_last_cached_date'] or 'never'}, expected: {m['expected_date']})"
        for m in stale_markets
    )
    return {
        "code": "market_data_stale",
        "message": (
            f"Price data is stale for: {summary}. "
            "Wait for the next scheduled refresh before starting a scan."
        ),
        "stale_markets": stale_markets,
    }
