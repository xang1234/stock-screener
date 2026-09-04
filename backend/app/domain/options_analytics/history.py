"""Ticker-continuous history readiness policy."""

from __future__ import annotations

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


def history_readiness(
    observations: Iterable[HistoricalObservation],
    trailing_sessions: Sequence[date],
    *,
    calculation_version: str,
) -> HistoryReadiness:
    compatible_sessions = {
        row.session
        for row in observations
        if row.calculation_version == calculation_version
        and row.state is ObservationState.AVAILABLE
    }
    ordered_sessions = tuple(dict.fromkeys(trailing_sessions))
    short_window = set(ordered_sessions[-SHORT_HISTORY_WINDOW:])
    iv_window = set(ordered_sessions[-IV_HISTORY_WINDOW:])
    short_count = len(compatible_sessions & short_window)
    iv_count = len(compatible_sessions & iv_window)
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
