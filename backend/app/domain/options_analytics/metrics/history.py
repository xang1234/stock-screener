"""Ticker-continuous historical options metrics."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, fields
from datetime import date

from ..history import (
    IV_HISTORY_REQUIRED,
    IV_HISTORY_WINDOW,
    SHORT_HISTORY_REQUIRED,
    SHORT_HISTORY_WINDOW,
    HistoricalObservation,
)
from ..models import MetricValue, ObservationState


@dataclass(frozen=True)
class HistoricalMetrics:
    iv_percentile: MetricValue
    iv_rank: MetricValue
    max_pain_change_5: MetricValue
    net_gex_change_5: MetricValue
    gamma_flip_change_5: MetricValue
    atm_iv_change_5: MetricValue
    skew_25_delta_change_5: MetricValue
    realized_volatility_change_5: MetricValue
    vrp_change_5: MetricValue
    activity_intensity_change_5: MetricValue


_CHANGE_FIELDS = (
    "max_pain",
    "net_gex",
    "gamma_flip",
    "atm_iv",
    "skew_25_delta",
    "realized_volatility",
    "vrp",
    "activity_intensity",
)

_LABELS = {
    "iv_percentile": "ATM IV Percentile",
    "iv_rank": "ATM IV Rank",
    "max_pain_change_5": "5-Observation Max Pain Change",
    "net_gex_change_5": "5-Observation Net GEX Change",
    "gamma_flip_change_5": "5-Observation Gamma Flip Change",
    "atm_iv_change_5": "5-Observation ATM IV Change",
    "skew_25_delta_change_5": "5-Observation 25-Delta Skew Change",
    "realized_volatility_change_5": "5-Observation Realized Volatility Change",
    "vrp_change_5": "5-Observation Volatility Risk Premium Change",
    "activity_intensity_change_5": "5-Observation Activity Intensity Change",
}


def _unavailable(
    name: str, reason_codes: tuple[str, ...] = ("building_history",)
) -> MetricValue:
    return MetricValue(
        available=False,
        reason_codes=reason_codes,
        label=_LABELS[name],
    )


def _available(name: str, value: float) -> MetricValue:
    return MetricValue(available=True, value=value, label=_LABELS[name])


def _all_unavailable() -> HistoricalMetrics:
    return HistoricalMetrics(
        **{field.name: _unavailable(field.name) for field in fields(HistoricalMetrics)}
    )


def _finite(value: float | None, *, positive: bool = False) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    if not math.isfinite(numeric) or (positive and numeric <= 0):
        return None
    return numeric


def calculate_historical_metrics(
    observations: Iterable[HistoricalObservation],
    trailing_sessions: Sequence[date],
    *,
    calculation_version: str,
) -> HistoricalMetrics:
    """Calculate history metrics without requiring consecutive cohort membership."""

    ordered_sessions = tuple(dict.fromkeys(trailing_sessions))
    allowed_sessions = set(ordered_sessions[-IV_HISTORY_WINDOW:])
    by_session: dict[date, HistoricalObservation] = {}
    for row in observations:
        if (
            row.calculation_version == calculation_version
            and row.state is ObservationState.AVAILABLE
            and row.session in allowed_sessions
        ):
            by_session[row.session] = row
    if not ordered_sessions or ordered_sessions[-1] not in by_session:
        return _all_unavailable()
    current_row = by_session[ordered_sessions[-1]]
    ordered = [
        by_session[session] for session in ordered_sessions if session in by_session
    ]

    values: dict[str, MetricValue] = {}
    short_sessions = set(ordered_sessions[-SHORT_HISTORY_WINDOW:])
    short_rows = [row for row in ordered if row.session in short_sessions]
    for source_name in _CHANGE_FIELDS:
        usable = [
            value
            for row in short_rows
            if (value := _finite(getattr(row, source_name))) is not None
        ]
        target_name = f"{source_name}_change_5"
        current_value = _finite(getattr(current_row, source_name))
        if current_value is None:
            values[target_name] = _unavailable(target_name, ("metric_unavailable",))
        elif len(usable) >= SHORT_HISTORY_REQUIRED:
            values[target_name] = _available(
                target_name,
                current_value - usable[-SHORT_HISTORY_REQUIRED],
            )
        else:
            values[target_name] = _unavailable(target_name)

    iv_values = [
        value
        for row in ordered
        if (value := _finite(row.atm_iv, positive=True)) is not None
    ]
    current_iv = _finite(current_row.atm_iv, positive=True)
    if current_iv is None:
        values["iv_percentile"] = _unavailable("iv_percentile", ("metric_unavailable",))
        values["iv_rank"] = _unavailable("iv_rank", ("metric_unavailable",))
    elif len(iv_values) >= IV_HISTORY_REQUIRED:
        values["iv_percentile"] = _available(
            "iv_percentile",
            sum(value <= current_iv for value in iv_values) / len(iv_values),
        )
        minimum = min(iv_values)
        maximum = max(iv_values)
        values["iv_rank"] = (
            _available("iv_rank", (current_iv - minimum) / (maximum - minimum))
            if maximum > minimum
            else MetricValue(
                available=False,
                reason_codes=("iv_range_zero",),
                label=_LABELS["iv_rank"],
            )
        )
    else:
        values["iv_percentile"] = _unavailable("iv_percentile")
        values["iv_rank"] = _unavailable("iv_rank")

    return HistoricalMetrics(
        **{field.name: values[field.name] for field in fields(HistoricalMetrics)}
    )


__all__ = ["HistoricalMetrics", "calculate_historical_metrics"]
