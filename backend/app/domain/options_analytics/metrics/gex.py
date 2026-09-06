"""Assumption-labeled gamma exposure estimates."""

from __future__ import annotations

import math
from collections.abc import Iterable
from itertools import pairwise

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
    return (
        math.exp(-dividend_yield * time_years)
        * density
        / (spot * volatility * sqrt_time)
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


def _interpolate_gamma_crossing(
    points: Iterable[tuple[float, float]],
) -> float | None:
    finite_points = sorted(
        {
            (float(x), float(y))
            for x, y in points
            if math.isfinite(float(x)) and math.isfinite(float(y))
        }
    )
    for index, (x, y) in enumerate(finite_points):
        if y != 0:
            continue
        left = next(
            (value for _, value in reversed(finite_points[:index]) if value != 0),
            None,
        )
        right = next(
            (value for _, value in finite_points[index + 1 :] if value != 0),
            None,
        )
        if left is not None and right is not None and left * right < 0:
            return x
    for (left_x, left_y), (right_x, right_y) in pairwise(finite_points):
        if left_y * right_y < 0:
            return left_x + (-left_y) * (right_x - left_x) / (right_y - left_y)
    return None


def estimate_gamma_flip(
    contracts: Iterable[NormalizedOptionContract],
    *,
    pinned_spot: float,
    time_years: float,
    rate: float | None,
    dividend_yield: float,
) -> MetricValue:
    rows = tuple(contracts)
    if not _finite_positive(pinned_spot) or rate is None or not math.isfinite(rate):
        return MetricValue(
            available=False,
            reason_codes=("gex_inputs_unavailable",),
            label="Estimated Gamma Flip",
        )
    usable_strikes = sorted(
        {
            float(row.strike)
            for row in rows
            if _finite_positive(row.strike)
            and _finite_positive(row.implied_volatility)
            and row.open_interest is not None
            and row.open_interest >= 0
            and row.contract_size == "REGULAR"
            and row.multiplier == 100
        }
    )
    if not usable_strikes:
        return MetricValue(
            available=False,
            reason_codes=("gamma_crossing_unavailable",),
            label="Estimated Gamma Flip",
        )
    lower = max(usable_strikes[0], pinned_spot * 0.80)
    upper = min(usable_strikes[-1], pinned_spot * 1.20)
    if lower > upper:
        return MetricValue(
            available=False,
            reason_codes=("gamma_crossing_unavailable",),
            label="Estimated Gamma Flip",
        )
    step = pinned_spot * 0.01
    grid = {lower, upper}
    cursor = lower
    while cursor <= upper:
        grid.add(cursor)
        cursor += step
    grid.update(strike for strike in usable_strikes if lower <= strike <= upper)
    profile = []
    for hypothetical_spot in sorted(grid):
        total = 0.0
        available = False
        for contract in rows:
            metric = estimate_contract_gex(
                contract,
                spot=hypothetical_spot,
                time_years=time_years,
                rate=rate,
                dividend_yield=dividend_yield,
            )
            if metric.available:
                total += float(metric.value)
                available = True
        if available:
            profile.append((hypothetical_spot, total))
    crossing = _interpolate_gamma_crossing(profile)
    if crossing is not None:
        return MetricValue(
            available=True,
            value=crossing,
            label="Estimated Gamma Flip",
            evidence={
                "method": "chain_repricing_linear_interpolation",
                "grid_step": step,
                "range_low": lower,
                "range_high": upper,
            },
        )
    return MetricValue(
        available=False,
        reason_codes=("gamma_crossing_unavailable",),
        label="Estimated Gamma Flip",
    )


def _gex_wall(
    contracts: Iterable[NormalizedOptionContract],
    side: OptionSide,
    *,
    spot: float,
    time_years: float,
    rate: float | None,
    dividend_yield: float,
) -> MetricValue:
    label = f"Estimated {'Call' if side is OptionSide.CALL else 'Put'} Wall"
    if rate is None or not math.isfinite(rate):
        return MetricValue(
            available=False,
            reason_codes=("gex_inputs_unavailable",),
            label=label,
        )
    gex_by_strike: dict[float, float] = {}
    for contract in contracts:
        if contract.side is not side:
            continue
        metric = estimate_contract_gex(
            contract,
            spot=spot,
            time_years=time_years,
            rate=rate,
            dividend_yield=dividend_yield,
        )
        if metric.available:
            strike = float(contract.strike)
            gex_by_strike[strike] = gex_by_strike.get(strike, 0.0) + float(
                metric.value
            )
    if not gex_by_strike:
        return MetricValue(
            available=False,
            reason_codes=("open_interest_unavailable",),
            label=label,
        )
    selected_strike, aggregate_gex = min(
        gex_by_strike.items(),
        key=lambda row: (-abs(row[1]), row[0]),
    )
    return MetricValue(
        available=True,
        value=selected_strike,
        label=label,
        evidence={
            "method": "maximum_absolute_estimated_side_gex",
            "aggregation": "sum_by_strike",
            "absolute_estimated_gex": abs(aggregate_gex),
        },
    )


def estimate_gex_walls(
    contracts: Iterable[NormalizedOptionContract],
    *,
    spot: float,
    time_years: float,
    rate: float | None,
    dividend_yield: float,
) -> tuple[MetricValue, MetricValue]:
    rows = tuple(contracts)
    inputs = {
        "spot": spot,
        "time_years": time_years,
        "rate": rate,
        "dividend_yield": dividend_yield,
    }
    return (
        _gex_wall(rows, OptionSide.CALL, **inputs),
        _gex_wall(rows, OptionSide.PUT, **inputs),
    )
