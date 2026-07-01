"""Symbol-scoped staleness check for manual scans.

Returns a 409 detail payload when any symbol the scan will process lacks cached
prices through its market's last completed trading day. Scoped to the resolved
universe, so unrelated data-quality issues on other symbols don't false-positive.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Iterable, Optional

from sqlalchemy import func

from ..database import SessionLocal
from ..domain.scanning.models import FreshnessDecision, FreshnessOmissionWarning
from ..models.stock import StockPrice
from ..models.stock_universe import StockUniverse
from ..services.market_refresh_state_service import get_market_refresh_state
from ..wiring.bootstrap import get_market_calendar_service

logger = logging.getLogger(__name__)

SCAN_DEGRADED_MIN_FRESH_RATE = 0.99
SCAN_DEGRADED_MAX_OMITTED_SYMBOLS = 100
STALE_SYMBOL_SAMPLE_LIMIT = 20


def _parse_state_date(value: object) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(str(value))
    except ValueError:
        return None


def _ordered_unique_symbols(symbols: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(symbol).upper() for symbol in symbols if symbol))


def _ordered_row_symbols(ordered_symbols: tuple[str, ...], market_rows: list) -> list[str]:
    row_symbols = {row.symbol for row in market_rows}
    return [symbol for symbol in ordered_symbols if symbol in row_symbols]


def _stale_entry(
    *,
    market: str,
    market_rows: list,
    stale_symbols: list[str],
    covered_dates: list[date],
    uncovered: int,
    expected_date: date | None,
    oldest: date | None,
    reason: str | None = None,
    freshness_rate: float | None = None,
) -> dict:
    entry = {
        "market": market,
        "total_symbols": len(market_rows),
        "covered_symbols": len(covered_dates),
        "uncovered_symbols": uncovered,
        "oldest_last_cached_date": str(oldest) if oldest is not None else None,
        "expected_date": str(expected_date) if expected_date is not None else None,
        "stale_symbol_count": len(stale_symbols),
        "sample_stale_symbols": stale_symbols[:STALE_SYMBOL_SAMPLE_LIMIT],
        "omission_thresholds": {
            "min_freshness_rate": SCAN_DEGRADED_MIN_FRESH_RATE,
            "max_omitted_symbols": SCAN_DEGRADED_MAX_OMITTED_SYMBOLS,
        },
    }
    if reason is not None:
        entry["reason"] = reason
    if freshness_rate is not None:
        entry["freshness_rate"] = freshness_rate
    return entry


def _build_blocking_detail(
    *,
    stale_markets: list[dict],
    unresolved_symbols: list[str],
) -> dict:
    def _describe(market_entry: dict) -> str:
        if market_entry.get("reason") == "calendar_unavailable":
            return f"{market_entry['market']} (calendar unavailable — could not verify freshness)"
        return (
            f"{market_entry['market']} "
            f"(oldest: {market_entry['oldest_last_cached_date'] or 'never'}, "
            f"expected: {market_entry['expected_date']})"
        )

    fragments: list[str] = []
    if stale_markets:
        fragments.append(", ".join(_describe(m) for m in stale_markets))
    if unresolved_symbols:
        preview = ", ".join(unresolved_symbols[:5])
        more = "" if len(unresolved_symbols) <= 5 else f" (+{len(unresolved_symbols) - 5} more)"
        fragments.append(f"unknown symbols: {preview}{more}")

    return {
        "code": "market_data_stale",
        "message": (
            f"Price data is stale for: {'; '.join(fragments)}. "
            "Wait for the next scheduled refresh before starting a scan."
        ),
        "stale_markets": stale_markets,
        "unresolved_symbols": unresolved_symbols,
    }


def evaluate_symbol_freshness(
    symbols: Iterable[str],
    *,
    allow_stale_tail: bool = False,
) -> FreshnessDecision:
    """Return scan symbols, omission warnings, or a blocking freshness detail.

    Groups the requested symbols by their universe market, compares each group
    to that market's last completed trading day, and optionally allows broad
    scans to omit a small stale tail when the rest of the scan is provably
    fresh.
    """
    ordered_symbols = _ordered_unique_symbols(symbols)
    if not ordered_symbols:
        return FreshnessDecision(symbols_to_scan=())

    session = SessionLocal()
    try:
        rows = (
            session.query(
                StockUniverse.symbol.label("symbol"),
                StockUniverse.market.label("market"),
                func.max(StockPrice.date).label("last_date"),
            )
            .outerjoin(StockPrice, StockPrice.symbol == StockUniverse.symbol)
            .filter(StockUniverse.symbol.in_(ordered_symbols))
            .group_by(StockUniverse.symbol, StockUniverse.market)
            .all()
        )

        resolved_symbols = {row.symbol for row in rows}
        unresolved_symbols = sorted(set(ordered_symbols) - resolved_symbols)

        per_market: dict[str, list] = {}
        for row in rows:
            per_market.setdefault(row.market, []).append(row)

        calendar = get_market_calendar_service()
        stale_markets: list[dict] = []
        hard_block_reasons: list[str] = []
        omittable_symbols: set[str] = set()
        expected_dates: dict[str, str | None] = {}
        oldest_dates: dict[str, str | None] = {}
        for market, market_rows in sorted(per_market.items()):
            try:
                expected_date = calendar.last_completed_trading_day(market)
            except Exception:
                # Fail-closed: if we cannot verify freshness (unsupported market
                # code, calendar backend outage, schedule lookup error), the safe
                # default is to block the scan. Silently treating the market as
                # fresh defeats the entire purpose of the gate.
                logger.warning(
                    "Could not resolve last completed trading day for market=%s; treating as stale",
                    market,
                    exc_info=True,
                )
                covered_dates = [r.last_date for r in market_rows if r.last_date is not None]
                stale_markets.append(_stale_entry(
                    market=market,
                    market_rows=market_rows,
                    stale_symbols=_ordered_row_symbols(ordered_symbols, market_rows),
                    covered_dates=covered_dates,
                    uncovered=sum(1 for r in market_rows if r.last_date is None),
                    expected_date=None,
                    oldest=min(covered_dates) if covered_dates else None,
                    reason="calendar_unavailable",
                ))
                hard_block_reasons.append("calendar_unavailable")
                continue

            state = get_market_refresh_state(session, market)
            observed_date = _parse_state_date(
                state.get("last_refreshed_trading_day") if state else None
            )
            comparison_date = expected_date
            if state is not None and (state.get("status") != "completed" or observed_date is None):
                covered_dates = [r.last_date for r in market_rows if r.last_date is not None]
                stale_markets.append(_stale_entry(
                    market=market,
                    market_rows=market_rows,
                    stale_symbols=_ordered_row_symbols(ordered_symbols, market_rows),
                    covered_dates=covered_dates,
                    uncovered=sum(1 for r in market_rows if r.last_date is None),
                    expected_date=expected_date,
                    oldest=None,
                    reason="refresh_state_missing",
                ))
                hard_block_reasons.append("refresh_state_missing")
                continue
            if observed_date is not None and observed_date < expected_date:
                covered_dates = [r.last_date for r in market_rows if r.last_date is not None]
                stale_markets.append(_stale_entry(
                    market=market,
                    market_rows=market_rows,
                    stale_symbols=_ordered_row_symbols(ordered_symbols, market_rows),
                    covered_dates=covered_dates,
                    uncovered=sum(1 for r in market_rows if r.last_date is None),
                    expected_date=expected_date,
                    oldest=observed_date,
                    reason="refresh_state_stale",
                ))
                hard_block_reasons.append("refresh_state_stale")
                continue
            if observed_date is not None:
                comparison_date = observed_date

            uncovered = sum(1 for r in market_rows if r.last_date is None)
            covered_dates = [r.last_date for r in market_rows if r.last_date is not None]
            oldest = min(covered_dates) if covered_dates else None
            expected_dates[market] = str(comparison_date)
            oldest_dates[market] = str(oldest) if oldest is not None else None

            stale_symbol_set = {
                row.symbol
                for row in market_rows
                if row.last_date is None or row.last_date < comparison_date
            }
            stale_symbols = [
                symbol for symbol in ordered_symbols if symbol in stale_symbol_set
            ]

            if stale_symbols:
                if state is None:
                    hard_block_reasons.append("refresh_state_missing")
                omittable_symbols.update(stale_symbols)
                stale_markets.append(_stale_entry(
                    market=market,
                    market_rows=market_rows,
                    stale_symbols=stale_symbols,
                    covered_dates=covered_dates,
                    uncovered=uncovered,
                    expected_date=comparison_date,
                    oldest=oldest,
                    reason="refresh_state_missing" if state is None else None,
                ))
    finally:
        session.close()

    if not stale_markets and not unresolved_symbols:
        return FreshnessDecision(symbols_to_scan=ordered_symbols)

    omitted_symbols = tuple(
        symbol for symbol in ordered_symbols if symbol in omittable_symbols
    )
    omitted_count = len(omitted_symbols)
    total_symbols = len(ordered_symbols)
    fresh_count = total_symbols - omitted_count
    freshness_rate = round(fresh_count / total_symbols, 6) if total_symbols else 1.0

    for market_entry in stale_markets:
        market_entry.setdefault("freshness_rate", freshness_rate)

    can_omit_tail = (
        allow_stale_tail
        and not unresolved_symbols
        and not hard_block_reasons
        and omitted_count > 0
        and omitted_count <= SCAN_DEGRADED_MAX_OMITTED_SYMBOLS
        and freshness_rate >= SCAN_DEGRADED_MIN_FRESH_RATE
    )
    if can_omit_tail:
        symbols_to_scan = tuple(
            symbol for symbol in ordered_symbols if symbol not in omittable_symbols
        )
        warning = FreshnessOmissionWarning(
            code="market_data_stale_tail_omitted",
            message=(
                f"Omitted {omitted_count} stale symbols from this broad scan "
                f"({freshness_rate:.2%} fresh)."
            ),
            markets=tuple(sorted(expected_dates)),
            omitted_symbols=omitted_symbols,
            omitted_count=omitted_count,
            total_symbols=total_symbols,
            fresh_count=fresh_count,
            freshness_rate=freshness_rate,
            expected_dates=expected_dates,
            oldest_last_cached_dates=oldest_dates,
        )
        return FreshnessDecision(
            symbols_to_scan=symbols_to_scan,
            warnings=(warning,),
        )

    return FreshnessDecision(
        symbols_to_scan=ordered_symbols,
        blocking_detail=_build_blocking_detail(
            stale_markets=stale_markets,
            unresolved_symbols=unresolved_symbols,
        ),
    )


def check_symbol_freshness(symbols: Iterable[str]) -> Optional[dict]:
    """Return a 409 detail dict when any symbol lacks fresh cached prices."""
    decision = evaluate_symbol_freshness(symbols, allow_stale_tail=False)
    return decision.blocking_detail
