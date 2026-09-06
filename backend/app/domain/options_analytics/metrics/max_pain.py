"""Max Pain calculated from open interest across the full selected chain."""

from __future__ import annotations

import math
from collections.abc import Iterable

from ..models import MetricValue, NormalizedOptionContract, OptionSide


def _usable_notional_open_interest(
    contract: NormalizedOptionContract,
) -> float | None:
    if contract.open_interest is None or contract.multiplier is None:
        return None
    open_interest = float(contract.open_interest)
    multiplier = float(contract.multiplier)
    if (
        not math.isfinite(open_interest)
        or open_interest < 0
        or not math.isfinite(multiplier)
        or multiplier <= 0
    ):
        return None
    return open_interest * multiplier


def calculate_max_pain(contracts: Iterable[NormalizedOptionContract]) -> MetricValue:
    usable = [
        (contract, notional_open_interest)
        for contract in contracts
        if (notional_open_interest := _usable_notional_open_interest(contract))
        is not None
    ]
    if not usable or sum(weight for _, weight in usable) <= 0:
        return MetricValue(
            available=False,
            reason_codes=("open_interest_unavailable",),
        )

    settlement_strikes = sorted({contract.strike for contract, _ in usable})

    def payout(settlement: float) -> float:
        total = 0.0
        for contract, notional_open_interest in usable:
            if contract.side is OptionSide.CALL:
                intrinsic = max(settlement - contract.strike, 0.0)
            else:
                intrinsic = max(contract.strike - settlement, 0.0)
            total += intrinsic * notional_open_interest
        return total

    best_strike = min(settlement_strikes, key=lambda strike: (payout(strike), strike))
    return MetricValue(available=True, value=float(best_strike), label="Max Pain")
