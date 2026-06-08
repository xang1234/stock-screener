"""Price refresh planning for GitHub-seeded and live market refreshes."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from sqlalchemy.orm import Session

from .price_history_coverage import PriceHistoryCoverage, classify_price_history


STALE_PRICE_TOP_UP_PERIOD = "7d"
NO_HISTORY_PRICE_BOOTSTRAP_PERIOD = "2y"
LIVE_TOP_UP_MODES = frozenset({"bootstrap", "delta"})
GITHUB_SYNC_SUCCESS_STATUSES = frozenset({"success", "up_to_date"})


@dataclass(frozen=True)
class PriceRefreshJob:
    kind: str
    symbols: tuple[str, ...]
    period: str


@dataclass(frozen=True)
class PriceRefreshPlan:
    symbols: tuple[str, ...]
    jobs: tuple[PriceRefreshJob, ...] = ()
    github_sync: Mapping[str, Any] | None = None
    github_seed_used: bool = False
    completion_message: str | None = None

    @property
    def source(self) -> str:
        if self.github_seed_used:
            return "github+live" if self.jobs else "github"
        return "live"

    @property
    def used_github_seed(self) -> bool:
        return self.github_seed_used


def _normalize_symbols(symbols: Sequence[str]) -> tuple[str, ...]:
    return tuple(str(symbol).upper() for symbol in symbols)


def _parse_bundle_date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return datetime.fromisoformat(str(value)).date()
    except (TypeError, ValueError):
        return None


def build_top_up_jobs(coverage: PriceHistoryCoverage) -> tuple[PriceRefreshJob, ...]:
    jobs: list[PriceRefreshJob] = []
    if coverage.stale:
        jobs.append(
            PriceRefreshJob(
                kind="stale",
                symbols=coverage.stale,
                period=STALE_PRICE_TOP_UP_PERIOD,
            )
        )
    if coverage.no_history:
        jobs.append(
            PriceRefreshJob(
                kind="no_history",
                symbols=coverage.no_history,
                period=NO_HISTORY_PRICE_BOOTSTRAP_PERIOD,
            )
        )
    return tuple(jobs)


def _symbols_from_jobs(jobs: Sequence[PriceRefreshJob]) -> tuple[str, ...]:
    return tuple(symbol for job in jobs for symbol in job.symbols)


def _plan_live_full(symbols: tuple[str, ...]) -> PriceRefreshPlan:
    jobs = (
        PriceRefreshJob(
            kind="full",
            symbols=symbols,
            period=NO_HISTORY_PRICE_BOOTSTRAP_PERIOD,
        ),
    ) if symbols else ()
    return PriceRefreshPlan(symbols=symbols, jobs=jobs)


def _plan_live_auto(
    symbols: tuple[str, ...],
    *,
    recently_refreshed_filter: Callable[[Sequence[str]], Sequence[str]] | None,
) -> PriceRefreshPlan:
    refresh_symbols = (
        _normalize_symbols(recently_refreshed_filter(symbols))
        if recently_refreshed_filter is not None
        else symbols
    )
    jobs = (
        PriceRefreshJob(
            kind="auto",
            symbols=refresh_symbols,
            period=NO_HISTORY_PRICE_BOOTSTRAP_PERIOD,
        ),
    ) if refresh_symbols else ()
    return PriceRefreshPlan(symbols=refresh_symbols, jobs=jobs)


def _plan_live_top_up(
    db: Session,
    *,
    symbols: tuple[str, ...],
    effective_market: str,
    market_calendar_service,
    github_sync: Mapping[str, Any] | None = None,
) -> PriceRefreshPlan:
    target_as_of = market_calendar_service.last_completed_trading_day(effective_market)
    coverage = classify_price_history(db, symbols=symbols, as_of_date=target_as_of)
    jobs = build_top_up_jobs(coverage)
    return PriceRefreshPlan(
        symbols=_symbols_from_jobs(jobs),
        jobs=jobs,
        github_sync=github_sync,
    )


def _plan_github_top_up(
    db: Session,
    *,
    symbols: tuple[str, ...],
    effective_market: str,
    github_sync: Mapping[str, Any],
    market_calendar_service,
) -> PriceRefreshPlan:
    target_as_of = market_calendar_service.last_completed_trading_day(effective_market)
    github_as_of = _parse_bundle_date(github_sync.get("as_of_date"))
    coverage = classify_price_history(db, symbols=symbols, as_of_date=target_as_of)
    jobs = build_top_up_jobs(coverage)
    live_symbols = _symbols_from_jobs(jobs)
    completion_message = None
    if not live_symbols:
        completion_message = (
            "GitHub daily price bundle is current - no live fetch needed"
            if github_as_of == target_as_of
            else "All symbols already fresh - no live fetch needed"
        )
    return PriceRefreshPlan(
        symbols=live_symbols,
        jobs=jobs,
        github_sync=github_sync,
        github_seed_used=True,
        completion_message=completion_message,
    )


def plan_price_refresh(
    db: Session,
    *,
    all_symbols: Sequence[str],
    mode: str,
    effective_market: str,
    market_calendar_service,
    github_sync: Mapping[str, Any] | None = None,
    recently_refreshed_filter: Callable[[Sequence[str]], Sequence[str]] | None = None,
) -> PriceRefreshPlan:
    """Plan live price-fetch work without performing any fetches."""
    normalized_symbols = _normalize_symbols(all_symbols)
    if not normalized_symbols:
        return PriceRefreshPlan(
            symbols=(),
            jobs=(),
            completion_message="No active symbols found in universe",
        )

    if mode == "auto":
        return _plan_live_auto(
            normalized_symbols,
            recently_refreshed_filter=recently_refreshed_filter,
        )
    if mode == "full":
        return _plan_live_full(normalized_symbols)
    if mode not in LIVE_TOP_UP_MODES:
        raise ValueError(f"Unknown price refresh mode: {mode}")

    if github_sync and github_sync.get("status") in GITHUB_SYNC_SUCCESS_STATUSES:
        return _plan_github_top_up(
            db,
            symbols=normalized_symbols,
            effective_market=effective_market,
            github_sync=github_sync,
            market_calendar_service=market_calendar_service,
        )

    return _plan_live_top_up(
        db,
        symbols=normalized_symbols,
        effective_market=effective_market,
        market_calendar_service=market_calendar_service,
        github_sync=github_sync,
    )
