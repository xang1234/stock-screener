"""Terminal completion helpers for market price refreshes."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .price_refresh_accounting import account_terminal_refresh
from .price_refresh_activity import PriceRefreshFinalization, PriceRefreshOutcome
from .price_refresh_planning import (
    PriceRefreshMode,
    PriceRefreshPlan,
)


@dataclass(frozen=True)
class PriceRefreshTerminalCompletion:
    outcome: PriceRefreshOutcome
    finalization: PriceRefreshFinalization


def build_terminal_completion(
    *,
    mode: PriceRefreshMode,
    effective_market: str,
    plan: PriceRefreshPlan,
    last_completed_trading_day: Callable[[str], Any],
) -> PriceRefreshTerminalCompletion | None:
    if plan.symbols:
        return None

    accounting = account_terminal_refresh(
        plan,
        mode=mode,
        effective_market=effective_market,
        last_completed_trading_day=last_completed_trading_day,
    )
    return PriceRefreshTerminalCompletion(
        outcome=accounting.to_outcome(mode=mode),
        finalization=accounting.to_finalization(),
    )
