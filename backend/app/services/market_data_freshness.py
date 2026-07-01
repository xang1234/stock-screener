"""Symbol-scoped staleness check for manual scans.

Returns a 409 detail payload when any symbol the scan will process lacks cached
prices through its market's last completed trading day. Scoped to the resolved
universe, so unrelated data-quality issues on other symbols don't false-positive.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Any, Iterable, Optional

from sqlalchemy import func

from ..database import SessionLocal
from ..domain.scanning.models import (
    FreshnessDecision,
    FreshnessOmissionWarning,
    ScanFreshnessPolicy,
    ScanWarningCode,
)
from ..models.stock import StockPrice
from ..models.stock_universe import StockUniverse
from ..services.market_refresh_state_service import get_market_refresh_state
from ..wiring.bootstrap import get_market_calendar_service

logger = logging.getLogger(__name__)

_DEFAULT_DEGRADED_POLICY = ScanFreshnessPolicy.allowing_stale_tail()
SCAN_DEGRADED_MIN_FRESH_RATE = _DEFAULT_DEGRADED_POLICY.min_freshness_rate
SCAN_DEGRADED_MAX_OMITTED_SYMBOLS = _DEFAULT_DEGRADED_POLICY.max_omitted_symbols
STALE_SYMBOL_SAMPLE_LIMIT = _DEFAULT_DEGRADED_POLICY.stale_symbol_sample_limit


@dataclass(frozen=True)
class _MarketFreshnessAssessment:
    market: str
    rows: tuple[Any, ...]
    stale_symbols: tuple[str, ...] = ()
    covered_dates: tuple[date, ...] = ()
    uncovered_symbols: int = 0
    expected_date: date | None = None
    oldest_last_cached_date: date | None = None
    reason: str | None = None
    hard_block: bool = False

    @property
    def has_stale_signal(self) -> bool:
        return bool(self.stale_symbols) or self.reason is not None

    def to_stale_entry(
        self,
        *,
        policy: ScanFreshnessPolicy,
        freshness_rate: float | None = None,
    ) -> dict[str, Any]:
        entry = {
            "market": self.market,
            "total_symbols": len(self.rows),
            "covered_symbols": len(self.covered_dates),
            "uncovered_symbols": self.uncovered_symbols,
            "oldest_last_cached_date": (
                str(self.oldest_last_cached_date)
                if self.oldest_last_cached_date is not None
                else None
            ),
            "expected_date": (
                str(self.expected_date) if self.expected_date is not None else None
            ),
            "stale_symbol_count": len(self.stale_symbols),
            "sample_stale_symbols": list(
                self.stale_symbols[:policy.stale_symbol_sample_limit]
            ),
            "omission_thresholds": {
                "min_freshness_rate": policy.min_freshness_rate,
                "max_omitted_symbols": policy.max_omitted_symbols,
            },
        }
        if self.reason is not None:
            entry["reason"] = self.reason
        if freshness_rate is not None:
            entry["freshness_rate"] = freshness_rate
        return entry


def _parse_state_date(value: object) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(str(value))
    except ValueError:
        return None


def _ordered_unique_symbols(symbols: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(symbol).upper() for symbol in symbols if symbol))


def _ordered_row_symbols(
    ordered_symbols: tuple[str, ...],
    market_rows: Iterable[Any],
) -> list[str]:
    row_symbols = {row.symbol for row in market_rows}
    return [symbol for symbol in ordered_symbols if symbol in row_symbols]


def _row_coverage(market_rows: tuple[Any, ...]) -> tuple[tuple[date, ...], int]:
    covered_dates = tuple(row.last_date for row in market_rows if row.last_date is not None)
    uncovered = sum(1 for row in market_rows if row.last_date is None)
    return covered_dates, uncovered


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


def _resolve_policy(
    policy: ScanFreshnessPolicy | None,
    allow_stale_tail: bool | None,
) -> ScanFreshnessPolicy:
    if policy is not None:
        return policy
    if allow_stale_tail:
        return ScanFreshnessPolicy.allowing_stale_tail()
    return ScanFreshnessPolicy.strict()


def _assess_market_freshness(
    *,
    session: Any,
    calendar: Any,
    market: str,
    market_rows: list[Any],
    ordered_symbols: tuple[str, ...],
) -> _MarketFreshnessAssessment:
    rows = tuple(market_rows)
    covered_dates, uncovered = _row_coverage(rows)
    oldest = min(covered_dates) if covered_dates else None
    all_market_symbols = tuple(_ordered_row_symbols(ordered_symbols, rows))

    try:
        expected_date = calendar.last_completed_trading_day(market)
    except Exception:
        # Fail-closed: if we cannot verify freshness (unsupported market code,
        # calendar backend outage, schedule lookup error), the safe default is
        # to block the scan. Silently treating the market as fresh defeats the
        # entire purpose of the gate.
        logger.warning(
            "Could not resolve last completed trading day for market=%s; treating as stale",
            market,
            exc_info=True,
        )
        return _MarketFreshnessAssessment(
            market=market,
            rows=rows,
            stale_symbols=all_market_symbols,
            covered_dates=covered_dates,
            uncovered_symbols=uncovered,
            expected_date=None,
            oldest_last_cached_date=oldest,
            reason="calendar_unavailable",
            hard_block=True,
        )

    state = get_market_refresh_state(session, market)
    observed_date = _parse_state_date(
        state.get("last_refreshed_trading_day") if state else None
    )
    if state is not None and (state.get("status") != "completed" or observed_date is None):
        return _MarketFreshnessAssessment(
            market=market,
            rows=rows,
            stale_symbols=all_market_symbols,
            covered_dates=covered_dates,
            uncovered_symbols=uncovered,
            expected_date=expected_date,
            oldest_last_cached_date=None,
            reason="refresh_state_missing",
            hard_block=True,
        )
    if observed_date is not None and observed_date < expected_date:
        return _MarketFreshnessAssessment(
            market=market,
            rows=rows,
            stale_symbols=all_market_symbols,
            covered_dates=covered_dates,
            uncovered_symbols=uncovered,
            expected_date=expected_date,
            oldest_last_cached_date=observed_date,
            reason="refresh_state_stale",
            hard_block=True,
        )

    comparison_date = observed_date if observed_date is not None else expected_date
    stale_symbol_set = {
        row.symbol
        for row in rows
        if row.last_date is None or row.last_date < comparison_date
    }
    stale_symbols = tuple(
        symbol for symbol in ordered_symbols if symbol in stale_symbol_set
    )
    missing_state_with_stale_rows = bool(stale_symbols and state is None)
    return _MarketFreshnessAssessment(
        market=market,
        rows=rows,
        stale_symbols=stale_symbols,
        covered_dates=covered_dates,
        uncovered_symbols=uncovered,
        expected_date=comparison_date,
        oldest_last_cached_date=oldest,
        reason="refresh_state_missing" if missing_state_with_stale_rows else None,
        hard_block=missing_state_with_stale_rows,
    )


def _build_omission_warning(
    *,
    assessments: tuple[_MarketFreshnessAssessment, ...],
    omitted_symbols: tuple[str, ...],
    omitted_count: int,
    total_symbols: int,
    fresh_count: int,
    freshness_rate: float,
) -> FreshnessOmissionWarning:
    date_assessments = tuple(
        assessment for assessment in assessments if not assessment.hard_block
    )
    expected_dates = {
        assessment.market: (
            str(assessment.expected_date)
            if assessment.expected_date is not None
            else None
        )
        for assessment in date_assessments
    }
    oldest_dates = {
        assessment.market: (
            str(assessment.oldest_last_cached_date)
            if assessment.oldest_last_cached_date is not None
            else None
        )
        for assessment in date_assessments
    }
    return FreshnessOmissionWarning(
        code=ScanWarningCode.STALE_TAIL_OMITTED.value,
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


def _decide_scan_freshness(
    *,
    ordered_symbols: tuple[str, ...],
    unresolved_symbols: list[str],
    assessments: tuple[_MarketFreshnessAssessment, ...],
    policy: ScanFreshnessPolicy,
) -> FreshnessDecision:
    stale_assessments = tuple(
        assessment for assessment in assessments if assessment.has_stale_signal
    )
    if not stale_assessments and not unresolved_symbols:
        return FreshnessDecision(symbols_to_scan=ordered_symbols)

    stale_symbol_set = {
        symbol
        for assessment in stale_assessments
        for symbol in assessment.stale_symbols
    }
    omitted_symbols = tuple(
        symbol for symbol in ordered_symbols if symbol in stale_symbol_set
    )
    omitted_count = len(omitted_symbols)
    total_symbols = len(ordered_symbols)
    fresh_count = total_symbols - omitted_count
    freshness_rate = round(fresh_count / total_symbols, 6) if total_symbols else 1.0
    hard_block = any(assessment.hard_block for assessment in stale_assessments)

    can_omit_tail = (
        policy.allow_stale_tail
        and not unresolved_symbols
        and not hard_block
        and omitted_count > 0
        and omitted_count <= policy.max_omitted_symbols
        and freshness_rate >= policy.min_freshness_rate
    )
    if can_omit_tail:
        symbols_to_scan = tuple(
            symbol for symbol in ordered_symbols if symbol not in stale_symbol_set
        )
        return FreshnessDecision(
            symbols_to_scan=symbols_to_scan,
            warnings=(
                _build_omission_warning(
                    assessments=assessments,
                    omitted_symbols=omitted_symbols,
                    omitted_count=omitted_count,
                    total_symbols=total_symbols,
                    fresh_count=fresh_count,
                    freshness_rate=freshness_rate,
                ),
            ),
        )

    stale_markets = [
        assessment.to_stale_entry(
            policy=policy,
            freshness_rate=freshness_rate,
        )
        for assessment in stale_assessments
    ]
    return FreshnessDecision(
        symbols_to_scan=ordered_symbols,
        blocking_detail=_build_blocking_detail(
            stale_markets=stale_markets,
            unresolved_symbols=unresolved_symbols,
        ),
    )


def evaluate_symbol_freshness(
    symbols: Iterable[str],
    *,
    policy: ScanFreshnessPolicy | None = None,
    allow_stale_tail: bool | None = None,
) -> FreshnessDecision:
    """Return scan symbols, omission warnings, or a blocking freshness detail.

    Groups the requested symbols by their universe market, compares each group
    to that market's last completed trading day, then applies the supplied
    freshness policy.
    """
    policy = _resolve_policy(policy, allow_stale_tail)
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
        assessments = tuple(
            _assess_market_freshness(
                session=session,
                calendar=calendar,
                market=market,
                market_rows=market_rows,
                ordered_symbols=ordered_symbols,
            )
            for market, market_rows in sorted(per_market.items())
        )
    finally:
        session.close()

    return _decide_scan_freshness(
        ordered_symbols=ordered_symbols,
        unresolved_symbols=unresolved_symbols,
        assessments=assessments,
        policy=policy,
    )


def check_symbol_freshness(symbols: Iterable[str]) -> Optional[dict]:
    """Return a 409 detail dict when any symbol lacks fresh cached prices."""
    decision = evaluate_symbol_freshness(symbols, allow_stale_tail=False)
    return decision.blocking_detail
