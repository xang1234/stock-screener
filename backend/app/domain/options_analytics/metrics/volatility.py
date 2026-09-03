"""Implied, realized, and relative volatility metrics."""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Sequence

from ..models import MetricValue, NormalizedOptionContract, OptionSide


def _valid_iv(value: float | None) -> bool:
    return value is not None and math.isfinite(float(value)) and float(value) > 0


def calculate_atm_iv(
    contracts: Iterable[NormalizedOptionContract], *, spot: float
) -> MetricValue:
    by_strike: dict[float, dict[OptionSide, float]] = {}
    for contract in contracts:
        if _valid_iv(contract.implied_volatility):
            by_strike.setdefault(contract.strike, {})[contract.side] = float(
                contract.implied_volatility
            )
    paired = [
        (strike, sides)
        for strike, sides in by_strike.items()
        if OptionSide.CALL in sides and OptionSide.PUT in sides
    ]
    if not paired or not math.isfinite(float(spot)):
        return MetricValue(
            available=False,
            reason_codes=("atm_two_sided_iv_unavailable",),
        )
    strike, sides = min(paired, key=lambda item: (abs(item[0] - spot), item[0]))
    return MetricValue(
        available=True,
        value=(sides[OptionSide.CALL] + sides[OptionSide.PUT]) / 2,
        evidence={"atm_strike": strike},
        label="ATM IV",
    )


def calculate_25_delta_skew(
    contracts: Iterable[NormalizedOptionContract],
) -> MetricValue:
    selected: dict[OptionSide, NormalizedOptionContract] = {}
    for side in (OptionSide.CALL, OptionSide.PUT):
        eligible = [
            contract
            for contract in contracts
            if contract.side is side
            and contract.delta is not None
            and math.isfinite(float(contract.delta))
            and 0.20 <= abs(float(contract.delta)) <= 0.30
            and _valid_iv(contract.implied_volatility)
        ]
        if eligible:
            selected[side] = min(
                eligible,
                key=lambda row: (abs(abs(float(row.delta)) - 0.25), row.strike),
            )
    if set(selected) != {OptionSide.CALL, OptionSide.PUT}:
        return MetricValue(
            available=False,
            reason_codes=("twenty_five_delta_pair_unavailable",),
        )
    value = float(selected[OptionSide.PUT].implied_volatility) - float(
        selected[OptionSide.CALL].implied_volatility
    )
    return MetricValue(available=True, value=value, label="25-Delta Put-Call IV Skew")


def calculate_realized_volatility(closes: Sequence[float | None]) -> MetricValue:
    recent = tuple(closes[-21:])
    if len(recent) != 21 or any(
        value is None
        or not math.isfinite(float(value))
        or float(value) <= 0
        for value in recent
    ):
        return MetricValue(
            available=False,
            reason_codes=("realized_volatility_history_unavailable",),
        )
    returns = [
        math.log(float(recent[index]) / float(recent[index - 1]))
        for index in range(1, len(recent))
    ]
    return MetricValue(
        available=True,
        value=statistics.stdev(returns) * math.sqrt(252),
        label="20-Return Realized Volatility",
    )


def calculate_volatility_risk_premium(
    *, atm_iv: float | None, realized_volatility: float | None
) -> MetricValue:
    if (
        atm_iv is None
        or realized_volatility is None
        or not math.isfinite(float(atm_iv))
        or not math.isfinite(float(realized_volatility))
    ):
        return MetricValue(
            available=False,
            reason_codes=("vrp_inputs_unavailable",),
        )
    return MetricValue(
        available=True,
        value=float(atm_iv) - float(realized_volatility),
        label="Volatility Risk Premium",
    )

