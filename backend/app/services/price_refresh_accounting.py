"""Shared completion accounting for market price refreshes."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from .price_refresh_activity import PriceRefreshFinalization, PriceRefreshOutcome
from .price_refresh_execution import PriceRefreshExecutionSummary
from .price_refresh_planning import (
    GitHubSeedOutcome,
    PriceRefreshMode,
    PriceRefreshPlan,
    PriceRefreshSource,
)


PRICE_REFRESH_COMPLETED_SUCCESS_RATE = 0.95


@dataclass(frozen=True)
class PriceRefreshAccounting:
    status: str
    source: PriceRefreshSource
    refreshed: int
    failed: int
    total: int
    message: str
    github_seed: GitHubSeedOutcome | None = None
    failed_symbols: tuple[str, ...] = ()
    heartbeat_status: str | None = "completed"
    market_success_rates: Mapping[str, tuple[Any, float]] = field(default_factory=dict)
    coverage_refreshed: int | None = None
    coverage_failed: int | None = None
    coverage_total: int | None = None
    coverage_success_rate: float | None = None
    already_fresh: int | None = None
    live_top_up_refreshed: int | None = None
    live_top_up_failed: int | None = None
    live_top_up_total: int | None = None
    unsupported_top_up_total: int | None = None

    def to_finalization(self) -> PriceRefreshFinalization:
        return PriceRefreshFinalization(
            metadata_status=self.status,
            metadata_refreshed=self.refreshed,
            metadata_total=self.total,
            activity_current=self.total,
            activity_total=self.total,
            message=self.message,
            heartbeat_status=self.heartbeat_status,
            market_success_rates=self.market_success_rates,
        )

    def to_outcome(self, *, mode: PriceRefreshMode) -> PriceRefreshOutcome:
        return PriceRefreshOutcome(
            status=self.status,
            source=self.source,
            mode=mode,
            message=self.message,
            refreshed=self.refreshed,
            failed=self.failed,
            total=self.total,
            failed_symbols=list(self.failed_symbols),
            github_seed=self.github_seed,
            coverage_refreshed=self.coverage_refreshed,
            coverage_failed=self.coverage_failed,
            coverage_total=self.coverage_total,
            coverage_success_rate=self.coverage_success_rate,
            already_fresh=self.already_fresh,
            live_top_up_refreshed=self.live_top_up_refreshed,
            live_top_up_failed=self.live_top_up_failed,
            live_top_up_total=self.live_top_up_total,
            unsupported_top_up_total=self.unsupported_top_up_total,
        )


def account_live_refresh(
    plan: PriceRefreshPlan,
    execution: PriceRefreshExecutionSummary,
    *,
    effective_market: str,
    last_completed_trading_day: Callable[[str], Any],
) -> PriceRefreshAccounting:
    live_total = len(plan.symbols)
    coverage = plan.coverage_summary
    use_seeded_coverage = (
        plan.used_github_seed
        and coverage is not None
        and coverage.universe_total > 0
    )

    if use_seeded_coverage and coverage is not None:
        coverage_total = coverage.universe_total
        coverage_refreshed = min(
            coverage_total,
            coverage.already_fresh + execution.refreshed,
        )
        coverage_failed = max(coverage_total - coverage_refreshed, 0)
        success_rate = coverage_refreshed / coverage_total
        market_success_rates = _coverage_market_success_rates(
            coverage_total_by_market=coverage.universe_total_by_market,
            already_fresh_by_market=coverage.already_fresh_by_market,
            refreshed_by_market=execution.refreshed_by_market,
            effective_market=effective_market,
            fallback_total=coverage_total,
            last_completed_trading_day=last_completed_trading_day,
        )
        status = _status_from_success_rate(success_rate)
        return PriceRefreshAccounting(
            status=status,
            source=PriceRefreshSource.GITHUB_AND_LIVE,
            refreshed=coverage_refreshed,
            failed=coverage_failed,
            total=coverage_total,
            message=f"Price refresh {status}",
            github_seed=plan.github_seed,
            failed_symbols=tuple(execution.failed_symbols),
            market_success_rates=market_success_rates,
            coverage_refreshed=coverage_refreshed,
            coverage_failed=coverage_failed,
            coverage_total=coverage_total,
            coverage_success_rate=success_rate,
            already_fresh=coverage.already_fresh,
            live_top_up_refreshed=execution.refreshed,
            live_top_up_failed=execution.failed,
            live_top_up_total=live_total,
            unsupported_top_up_total=coverage.unsupported_top_up_total,
        )

    success_rate = execution.refreshed / live_total if live_total > 0 else 0
    status = _status_from_success_rate(success_rate)
    return PriceRefreshAccounting(
        status=status,
        source=(
            PriceRefreshSource.GITHUB_AND_LIVE
            if plan.used_github_seed
            else PriceRefreshSource.LIVE
        ),
        refreshed=execution.refreshed,
        failed=execution.failed,
        total=live_total,
        message=f"Price refresh {status}",
        github_seed=plan.github_seed,
        failed_symbols=tuple(execution.failed_symbols),
        market_success_rates=_live_market_success_rates(
            symbols=plan.symbols,
            symbol_markets=plan.symbol_markets,
            refreshed_by_market=execution.refreshed_by_market,
            effective_market=effective_market,
            last_completed_trading_day=last_completed_trading_day,
        ),
    )


def account_terminal_refresh(
    plan: PriceRefreshPlan,
    *,
    mode: PriceRefreshMode,
    effective_market: str,
    last_completed_trading_day: Callable[[str], Any],
) -> PriceRefreshAccounting:
    if plan.source is PriceRefreshSource.GITHUB:
        message = (
            plan.completion_message
            or "GitHub daily price bundle is current - no live fetch needed"
        )
        symbol_count = _github_terminal_symbol_count(plan)
        trading_day = _completion_trading_day(
            plan.github_seed,
            effective_market,
            last_completed_trading_day=last_completed_trading_day,
        )
        return PriceRefreshAccounting(
            status="completed",
            source=PriceRefreshSource.GITHUB,
            refreshed=symbol_count,
            failed=0,
            total=symbol_count,
            message=message,
            github_seed=plan.github_seed,
            market_success_rates={effective_market: (trading_day, 1.0)},
            coverage_refreshed=symbol_count,
            coverage_failed=0,
            coverage_total=symbol_count,
            coverage_success_rate=1.0 if symbol_count > 0 else 0,
            already_fresh=(
                plan.coverage_summary.already_fresh
                if plan.coverage_summary is not None
                else symbol_count
            ),
            live_top_up_refreshed=0,
            live_top_up_failed=0,
            live_top_up_total=0,
            unsupported_top_up_total=(
                plan.coverage_summary.unsupported_top_up_total
                if plan.coverage_summary is not None
                else 0
            ),
        )

    return PriceRefreshAccounting(
        status="completed",
        source=plan.source,
        refreshed=0,
        failed=0,
        total=0,
        message=_empty_refresh_message(plan, mode),
        github_seed=plan.github_seed,
        heartbeat_status=None,
    )


def _status_from_success_rate(success_rate: float) -> str:
    if success_rate >= PRICE_REFRESH_COMPLETED_SUCCESS_RATE:
        return "completed"
    return "partial"


def _github_terminal_symbol_count(plan: PriceRefreshPlan) -> int:
    if plan.coverage_summary is not None and plan.coverage_summary.universe_total > 0:
        return plan.coverage_summary.universe_total
    return len(plan.all_symbols)


def _completion_trading_day(
    github_seed: GitHubSeedOutcome | None,
    effective_market: str,
    *,
    last_completed_trading_day: Callable[[str], Any],
) -> Any:
    if github_seed and github_seed.as_of_date is not None:
        return github_seed.as_of_date
    return last_completed_trading_day(effective_market)


def _empty_refresh_message(
    refresh_plan: PriceRefreshPlan,
    mode: PriceRefreshMode,
) -> str:
    if refresh_plan.completion_message:
        return refresh_plan.completion_message
    if mode is PriceRefreshMode.AUTO:
        return "All symbols recently refreshed - nothing to do"
    if mode in {PriceRefreshMode.BOOTSTRAP, PriceRefreshMode.DELTA}:
        return "All symbols already fresh - no live fetch needed"
    return "No active symbols found in universe"


def _live_market_success_rates(
    *,
    symbols: tuple[str, ...],
    symbol_markets: Mapping[str, str],
    refreshed_by_market: Mapping[str, int],
    effective_market: str,
    last_completed_trading_day: Callable[[str], Any],
) -> dict[str, tuple[Any, float]]:
    symbol_market_totals: Counter[str] = Counter(
        _market_for_symbol(
            symbol,
            symbol_markets=symbol_markets,
            effective_market=effective_market,
        )
        for symbol in symbols
    )
    return _success_rates_for_totals(
        total_by_market=symbol_market_totals,
        refreshed_by_market=refreshed_by_market,
        last_completed_trading_day=last_completed_trading_day,
    )


def _coverage_market_success_rates(
    *,
    coverage_total_by_market: Mapping[str, int],
    already_fresh_by_market: Mapping[str, int],
    refreshed_by_market: Mapping[str, int],
    effective_market: str,
    fallback_total: int,
    last_completed_trading_day: Callable[[str], Any],
) -> dict[str, tuple[Any, float]]:
    total_by_market = (
        coverage_total_by_market
        if coverage_total_by_market
        else {effective_market: fallback_total}
    )
    covered_by_market: dict[str, int] = {}
    for market, market_total in total_by_market.items():
        market_key = str(market).upper()
        covered_by_market[market_key] = min(
            market_total,
            _market_count(already_fresh_by_market, market_key)
            + _market_count(refreshed_by_market, market_key),
        )
    return _success_rates_for_totals(
        total_by_market={str(market).upper(): total for market, total in total_by_market.items()},
        refreshed_by_market=covered_by_market,
        last_completed_trading_day=last_completed_trading_day,
    )


def _success_rates_for_totals(
    *,
    total_by_market: Mapping[str, int],
    refreshed_by_market: Mapping[str, int],
    last_completed_trading_day: Callable[[str], Any],
) -> dict[str, tuple[Any, float]]:
    market_success_rates: dict[str, tuple[Any, float]] = {}
    for refresh_market, market_total in total_by_market.items():
        if market_total <= 0:
            continue
        market_key = str(refresh_market).upper()
        market_success_rate = _market_count(refreshed_by_market, market_key) / market_total
        if market_success_rate >= PRICE_REFRESH_COMPLETED_SUCCESS_RATE:
            market_success_rates[market_key] = (
                last_completed_trading_day(market_key),
                market_success_rate,
            )
    return market_success_rates


def _market_for_symbol(
    symbol: str,
    *,
    symbol_markets: Mapping[str, str],
    effective_market: str,
) -> str:
    return str(symbol_markets.get(str(symbol).upper(), effective_market)).upper()


def _market_count(counts: Mapping[str, int], market: str) -> int:
    market_key = str(market).upper()
    for key, value in counts.items():
        if str(key).upper() == market_key:
            return value
    return 0
