"""Ticker-continuous history readiness policy."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date

from .models import HistoryReadiness, ObservationState

SHORT_HISTORY_REQUIRED = 5
SHORT_HISTORY_WINDOW = 7
IV_HISTORY_REQUIRED = 20
IV_HISTORY_WINDOW = 30


@dataclass(frozen=True)
class HistoricalObservation:
    session: date
    calculation_version: str
    state: ObservationState
    max_pain: float | None = None
    net_gex: float | None = None
    gamma_flip: float | None = None
    atm_iv: float | None = None
    skew_25_delta: float | None = None
    realized_volatility: float | None = None
    vrp: float | None = None
    activity_intensity: float | None = None


def history_readiness(
    observations: Iterable[HistoricalObservation],
    trailing_sessions: Sequence[date],
    *,
    calculation_version: str,
) -> HistoryReadiness:
    rows = tuple(observations)
    compatible_sessions = {
        row.session
        for row in rows
        if row.calculation_version == calculation_version
        and row.state is ObservationState.AVAILABLE
    }
    ordered_sessions = tuple(dict.fromkeys(trailing_sessions))
    short_window = set(ordered_sessions[-SHORT_HISTORY_WINDOW:])
    iv_window = set(ordered_sessions[-IV_HISTORY_WINDOW:])
    short_count = len(compatible_sessions & short_window)
    iv_count = len(
        {
            row.session
            for row in rows
            if row.calculation_version == calculation_version
            and row.state is ObservationState.AVAILABLE
            and row.session in iv_window
            and row.atm_iv is not None
            and math.isfinite(float(row.atm_iv))
            and row.atm_iv > 0
        }
    )
    short_ready = short_count >= SHORT_HISTORY_REQUIRED
    iv_ready = iv_count >= IV_HISTORY_REQUIRED
    return HistoryReadiness(
        short_history_available=short_ready,
        iv_history_available=iv_ready,
        short_observation_count=short_count,
        iv_observation_count=iv_count,
        lifetime_observation_count=len(compatible_sessions),
        reason_codes=() if short_ready and iv_ready else ("building_history",),
    )
