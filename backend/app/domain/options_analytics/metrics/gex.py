"""Assumption-labeled gamma exposure estimates."""

from __future__ import annotations

import math
from collections.abc import Iterable

from ..models import MetricValue, NormalizedOptionContract, OptionSide

DEALER_PROXY_SIGN = "calls_positive_puts_negative"


def _finite_positive(value: float | None) -> bool:
    return value is not None and math.isfinite(float(value)) and float(value) > 0


def black_scholes_unit_gamma(
    *,
    spot: float,
    strike: float,
    time_years: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
) -> float:
    if not all(
        math.isfinite(float(value))
        for value in (spot, strike, time_years, rate, dividend_yield, volatility)
    ):
        raise ValueError("Black-Scholes inputs must be finite")
    if spot <= 0 or strike <= 0 or time_years <= 0 or volatility <= 0:
        raise ValueError("Spot, strike, time, and volatility must be positive")
    sqrt_time = math.sqrt(time_years)
    d1 = (
        math.log(spot / strike)
        + (rate - dividend_yield + volatility * volatility / 2) * time_years
    ) / (volatility * sqrt_time)
    density = math.exp(-(d1 * d1) / 2) / math.sqrt(2 * math.pi)
    return math.exp(-dividend_yield * time_years) * density / (
        spot * volatility * sqrt_time
    )


def estimate_contract_gex(
    contract: NormalizedOptionContract,
    *,
    spot: float,
    time_years: float,
    rate: float,
    dividend_yield: float,
) -> MetricValue:
    if contract.contract_size != "REGULAR" or contract.multiplier != 100:
        return MetricValue(
            available=False,
            reason_codes=("contract_multiplier_unavailable",),
            label="Estimated GEX",
        )
    if (
        contract.open_interest is None
        or not math.isfinite(float(contract.open_interest))
        or contract.open_interest < 0
        or not _finite_positive(contract.implied_volatility)
    ):
        return MetricValue(
            available=False,
            reason_codes=("gex_inputs_unavailable",),
            label="Estimated GEX",
        )
    try:
        gamma = black_scholes_unit_gamma(
            spot=spot,
            strike=contract.strike,
            time_years=time_years,
            rate=rate,
            dividend_yield=dividend_yield,
            volatility=float(contract.implied_volatility),
        )
    except ValueError:
        return MetricValue(
            available=False,
            reason_codes=("gex_inputs_unavailable",),
            label="Estimated GEX",
        )
    unsigned = gamma * contract.open_interest * contract.multiplier * spot * spot * 0.01
    sign = 1.0 if contract.side is OptionSide.CALL else -1.0
    side_label = "Call" if contract.side is OptionSide.CALL else "Put"
    return MetricValue(
        available=True,
        value=sign * unsigned,
        label=f"Estimated {side_label} GEX",
        evidence={
            "dealer_proxy_sign": DEALER_PROXY_SIGN,
            "contract_multiplier": contract.multiplier,
            "risk_free_rate": rate,
            "dividend_yield": dividend_yield,
        },
    )


def estimate_gamma_flip(points: Iterable[tuple[float, float]]) -> MetricValue:
    finite_points = sorted(
        {
            (float(x), float(y))
            for x, y in points
            if math.isfinite(float(x)) and math.isfinite(float(y))
        }
    )
    for x, y in finite_points:
        if y == 0:
            return MetricValue(
                available=True,
                value=x,
                label="Estimated Gamma Flip",
                evidence={"method": "linear_interpolation"},
            )
    for (left_x, left_y), (right_x, right_y) in zip(
        finite_points, finite_points[1:]
    ):
        if left_y * right_y < 0:
            crossing = left_x + (-left_y) * (right_x - left_x) / (right_y - left_y)
            return MetricValue(
                available=True,
                value=crossing,
                label="Estimated Gamma Flip",
                evidence={"method": "linear_interpolation"},
            )
    return MetricValue(
        available=False,
        reason_codes=("gamma_crossing_unavailable",),
        label="Estimated Gamma Flip",
    )


def _wall(
    contracts: Iterable[NormalizedOptionContract],
    side: OptionSide,
) -> MetricValue:
    usable = [
        contract
        for contract in contracts
        if contract.side is side
        and contract.open_interest is not None
        and math.isfinite(float(contract.open_interest))
        and contract.open_interest >= 0
    ]
    label = f"Estimated {'Call' if side is OptionSide.CALL else 'Put'} Wall"
    if not usable:
        return MetricValue(
            available=False,
            reason_codes=("open_interest_unavailable",),
            label=label,
        )
    selected = min(usable, key=lambda row: (-float(row.open_interest), row.strike))
    return MetricValue(
        available=True,
        value=float(selected.strike),
        label=label,
        evidence={"method": "maximum_open_interest"},
    )


def estimate_open_interest_walls(
    contracts: Iterable[NormalizedOptionContract],
) -> tuple[MetricValue, MetricValue]:
    rows = tuple(contracts)
    return _wall(rows, OptionSide.CALL), _wall(rows, OptionSide.PUT)

