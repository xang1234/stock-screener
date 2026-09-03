"""Max Pain calculated from open interest across the full selected chain."""

from __future__ import annotations

import math
from collections.abc import Iterable

from ..models import MetricValue, NormalizedOptionContract, OptionSide


def _usable_open_interest(contract: NormalizedOptionContract) -> float | None:
    value = contract.open_interest
    if value is None:
        return None
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0:
        return None
    return numeric


def calculate_max_pain(contracts: Iterable[NormalizedOptionContract]) -> MetricValue:
    usable = [
        (contract, open_interest)
        for contract in contracts
        if (open_interest := _usable_open_interest(contract)) is not None
    ]
    if not usable or sum(open_interest for _, open_interest in usable) <= 0:
        return MetricValue(
            available=False,
            reason_codes=("open_interest_unavailable",),
        )

    settlement_strikes = sorted({contract.strike for contract, _ in usable})

    def payout(settlement: float) -> float:
        total = 0.0
        for contract, open_interest in usable:
            if contract.side is OptionSide.CALL:
                intrinsic = max(settlement - contract.strike, 0.0)
            else:
                intrinsic = max(contract.strike - settlement, 0.0)
            total += intrinsic * open_interest
        return total

    best_strike = min(settlement_strikes, key=lambda strike: (payout(strike), strike))
    return MetricValue(available=True, value=float(best_strike), label="Max Pain")

