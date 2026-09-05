"""Open-interest and volume activity metrics without directional claims."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from ..models import MetricValue, NormalizedOptionContract, OptionSide

ACTIVITY_VOLUME_FLOOR = 100


@dataclass(frozen=True)
class ActivityMetrics:
    call_put_volume_ratio: MetricValue
    volume_oi_ratio: MetricValue
    near_spot_volume_concentration: MetricValue
    near_spot_open_interest_concentration: MetricValue
    highest_contract_activity_ratio: MetricValue
    activity_intensity: MetricValue


def _non_negative(value: int | None) -> int | None:
    if value is None or not math.isfinite(float(value)) or value < 0:
        return None
    return int(value)


def _complete_total(
    contracts: tuple[NormalizedOptionContract, ...],
    field: str,
) -> int | None:
    if not contracts:
        return None
    values = tuple(_non_negative(getattr(row, field)) for row in contracts)
    if any(value is None for value in values):
        return None
    return sum(value for value in values if value is not None)


def _near_spot_concentration(
    contracts: tuple[NormalizedOptionContract, ...],
    *,
    field: str,
    spot: float,
    incomplete_reason: str,
    unavailable_reason: str,
) -> MetricValue:
    total = _complete_total(contracts, field)
    if total is None:
        return MetricValue(available=False, reason_codes=(incomplete_reason,))
    if total == 0 or not math.isfinite(float(spot)) or spot <= 0:
        return MetricValue(available=False, reason_codes=(unavailable_reason,))
    near_total = sum(
        value
        for row in contracts
        if abs(row.strike - spot) / spot <= 0.05
        and (value := _non_negative(getattr(row, field))) is not None
    )
    return MetricValue(available=True, value=near_total / total)


def _highest_contract_activity_ratio(
    contracts: tuple[NormalizedOptionContract, ...],
) -> MetricValue:
    volumes = tuple(_non_negative(row.volume) for row in contracts)
    if any(volume is None for volume in volumes):
        return MetricValue(
            available=False,
            reason_codes=("contract_volume_incomplete",),
        )
    qualifying = tuple(
        (row, volume)
        for row, volume in zip(contracts, volumes, strict=True)
        if volume is not None and volume >= ACTIVITY_VOLUME_FLOOR
    )
    if not qualifying:
        return MetricValue(
            available=False,
            reason_codes=("contract_volume_floor_not_met",),
        )
    open_interest = tuple(_non_negative(row.open_interest) for row, _ in qualifying)
    if any(value is None for value in open_interest):
        return MetricValue(
            available=False,
            reason_codes=("contract_open_interest_incomplete",),
        )
    if any(value == 0 for value in open_interest):
        return MetricValue(
            available=False,
            reason_codes=("contract_open_interest_zero",),
        )
    ratios = (
        volume / value
        for (_, volume), value in zip(qualifying, open_interest, strict=True)
        if value is not None
    )
    return MetricValue(available=True, value=max(ratios))


def calculate_activity_metrics(
    contracts: tuple[NormalizedOptionContract, ...], *, spot: float
) -> ActivityMetrics:
    call_volume = _complete_total(
        tuple(row for row in contracts if row.side is OptionSide.CALL),
        "volume",
    )
    put_volume = _complete_total(
        tuple(row for row in contracts if row.side is OptionSide.PUT),
        "volume",
    )
    if call_volume is None or put_volume is None:
        call_put_ratio = MetricValue(
            available=False,
            reason_codes=("side_volume_incomplete",),
            label="Call / Put Volume",
        )
    elif put_volume == 0:
        call_put_ratio = MetricValue(
            available=False,
            reason_codes=("put_volume_zero",),
            label="Call / Put Volume",
        )
    else:
        call_put_ratio = MetricValue(
            available=True,
            value=call_volume / put_volume,
            label="Call / Put Volume",
        )

    total_volume = _complete_total(contracts, "volume")
    total_open_interest = _complete_total(contracts, "open_interest")
    if total_volume is None or total_open_interest is None:
        ratio = MetricValue(
            available=False,
            reason_codes=("activity_totals_incomplete",),
        )
    elif total_open_interest == 0:
        ratio = MetricValue(available=False, reason_codes=("open_interest_zero",))
    else:
        ratio = MetricValue(available=True, value=total_volume / total_open_interest)

    volume_concentration = _near_spot_concentration(
        contracts,
        field="volume",
        spot=spot,
        incomplete_reason="volume_total_incomplete",
        unavailable_reason="volume_concentration_unavailable",
    )
    open_interest_concentration = _near_spot_concentration(
        contracts,
        field="open_interest",
        spot=spot,
        incomplete_reason="open_interest_total_incomplete",
        unavailable_reason="open_interest_concentration_unavailable",
    )
    highest_contract_ratio = _highest_contract_activity_ratio(contracts)

    if total_volume is None:
        intensity = MetricValue(
            available=False,
            reason_codes=("activity_totals_incomplete",),
        )
    elif not ratio.available:
        intensity = MetricValue(available=False, reason_codes=ratio.reason_codes)
    else:
        intensity = MetricValue(
            available=True,
            value=float(ratio.value),
            label="Activity Intensity",
        )
    return ActivityMetrics(
        call_put_volume_ratio=call_put_ratio,
        volume_oi_ratio=ratio,
        near_spot_volume_concentration=volume_concentration,
        near_spot_open_interest_concentration=open_interest_concentration,
        highest_contract_activity_ratio=highest_contract_ratio,
        activity_intensity=intensity,
    )


def rank_activity(values: Mapping[str, MetricValue]) -> dict[str, int | None]:
    available = sorted(
        (
            (symbol.strip().upper(), float(metric.value))
            for symbol, metric in values.items()
            if metric.available and metric.value is not None
        ),
        key=lambda row: (-row[1], row[0]),
    )
    ranks = {symbol: rank for rank, (symbol, _) in enumerate(available, 1)}
    return {
        symbol.strip().upper(): ranks.get(symbol.strip().upper()) for symbol in values
    }
