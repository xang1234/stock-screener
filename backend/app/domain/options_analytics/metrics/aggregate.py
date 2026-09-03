"""Compose pure calculators into one chain-level result."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields, is_dataclass
from datetime import date
from typing import Any

from ..models import ChainObservation, MetricValue
from .activity import ActivityMetrics, calculate_activity_metrics
from .gex import (
    DEALER_PROXY_SIGN,
    estimate_contract_gex,
    estimate_gamma_flip,
    estimate_open_interest_walls,
)
from .max_pain import calculate_max_pain
from .volatility import (
    calculate_25_delta_skew,
    calculate_atm_iv,
    calculate_realized_volatility,
    calculate_volatility_risk_premium,
)


@dataclass(frozen=True)
class ChainMetrics:
    max_pain: MetricValue
    net_gex: MetricValue
    gamma_flip: MetricValue
    call_wall: MetricValue
    put_wall: MetricValue
    atm_iv: MetricValue
    skew_25_delta: MetricValue
    realized_volatility: MetricValue
    vrp: MetricValue
    activity: ActivityMetrics
    gex_by_strike: tuple[tuple[float, float], ...]

    def assert_json_finite(self) -> None:
        def visit(value: Any) -> None:
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("Options metric result contains a non-finite number")
            if is_dataclass(value):
                for item in fields(value):
                    visit(getattr(value, item.name))
            elif isinstance(value, dict):
                for item in value.values():
                    visit(item)
            elif isinstance(value, (tuple, list)):
                for item in value:
                    visit(item)

        visit(self)


def calculate_chain_metrics(
    observation: ChainObservation,
    *,
    as_of_date: date,
    risk_free_rate: float,
    dividend_yield: float,
    closes: tuple[float | None, ...],
) -> ChainMetrics:
    contracts = observation.contracts
    time_years = (observation.expiration - as_of_date).days / 365
    gex_values = [
        (contract, estimate_contract_gex(
            contract,
            spot=observation.source_spot_price,
            time_years=time_years,
            rate=risk_free_rate,
            dividend_yield=dividend_yield,
        ))
        for contract in contracts
    ]
    usable_gex = [(contract, metric) for contract, metric in gex_values if metric.available]
    if usable_gex:
        net_gex = MetricValue(
            available=True,
            value=sum(float(metric.value) for _, metric in usable_gex),
            label="Estimated Net GEX",
            evidence={"dealer_proxy_sign": DEALER_PROXY_SIGN},
        )
    else:
        net_gex = MetricValue(
            available=False,
            reason_codes=("gex_inputs_unavailable",),
            label="Estimated Net GEX",
        )
    by_strike: dict[float, float] = {}
    for contract, metric in usable_gex:
        by_strike[contract.strike] = by_strike.get(contract.strike, 0.0) + float(
            metric.value
        )
    gex_by_strike = tuple(sorted(by_strike.items()))
    call_wall, put_wall = estimate_open_interest_walls(contracts)
    atm_iv = calculate_atm_iv(contracts, spot=observation.source_spot_price)
    realized = calculate_realized_volatility(closes)
    result = ChainMetrics(
        max_pain=calculate_max_pain(contracts),
        net_gex=net_gex,
        gamma_flip=estimate_gamma_flip(gex_by_strike),
        call_wall=call_wall,
        put_wall=put_wall,
        atm_iv=atm_iv,
        skew_25_delta=calculate_25_delta_skew(contracts),
        realized_volatility=realized,
        vrp=calculate_volatility_risk_premium(
            atm_iv=atm_iv.value if atm_iv.available else None,
            realized_volatility=realized.value if realized.available else None,
        ),
        activity=calculate_activity_metrics(
            contracts, spot=observation.source_spot_price
        ),
        gex_by_strike=gex_by_strike,
    )
    result.assert_json_finite()
    return result

