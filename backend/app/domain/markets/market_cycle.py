"""Market-cycle state machine over a benchmark's daily bars.

A pure reducer: the full frame goes in and one ``MarketCycle`` per bar comes
out, so a daily run and a backfill over the same bars always agree.

- UPTREND -> CORRECTION when a close is ``correction_pct`` below the highest
  close since the start, the last follow-through day or the last recovery, or
  when a close breaks the confirming FTD's rally low (a failed FTD).
- CORRECTION -> RALLY_ATTEMPT on the first close above the prior close (day 1).
- An intraday low under the rally low moves it and resets an attempt.
- RALLY_ATTEMPT -> UPTREND on day ``ftd_min_rally_day`` or later with a gain
  of ``ftd_gain_pct`` on volume above the prior session's (an FTD).
- An attempt that reaches ``attempt_expiry_sessions`` without an FTD falls
  back to CORRECTION with the same rally low.
- CORRECTION or RALLY_ATTEMPT -> UPTREND, without an FTD, on a close above the
  high the correction started from.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from typing import NamedTuple

# Absorbs float error so a gain of exactly ftd_gain_pct confirms (101.5 / 100 - 1
# is 0.014999999999999902).
_GAIN_TOLERANCE = 1e-9


class CycleState(StrEnum):
    UPTREND = "uptrend"
    CORRECTION = "correction"
    RALLY_ATTEMPT = "rally_attempt"


@dataclass(frozen=True)
class CycleParams:
    correction_pct: float = 0.08
    ftd_gain_pct: float = 0.015
    ftd_min_rally_day: int = 4
    attempt_expiry_sessions: int = 25


DEFAULT_CYCLE_PARAMS = CycleParams()


@dataclass(frozen=True)
class MarketCycle:
    state: CycleState
    rally_low: float | None = None  # intraday low being defended
    rally_low_date: date | None = None
    rally_day: int | None = None  # RALLY_ATTEMPT only
    ftd_date: date | None = None  # the FTD confirming the current uptrend
    failed_ftd_date: date | None = None  # only on the session the FTD failed


class CycleBar(NamedTuple):
    date: date
    close: float
    low: float
    volume: float


def market_cycle(bars: Sequence[CycleBar], params: CycleParams = DEFAULT_CYCLE_PARAMS) -> list[MarketCycle]:
    """One ``MarketCycle`` per bar.

    Expects one bar per session in date order, with positive finite close and
    low; ``compute_exposure`` enforces this before calling. A missing, zero or
    NaN volume on the session or the one before never confirms an FTD. The
    first bar starts in UPTREND, so the result depends on where the bars
    start until the first correction.
    """
    closes = [bar.close for bar in bars]
    volumes = [bar.volume for bar in bars]

    state = CycleState.UPTREND
    peak = float("-inf")
    low = low_date = day = ftd_date = None
    cycles: list[MarketCycle] = []
    for i, (bar_date, close, bar_low, _) in enumerate(bars):
        failed_ftd_date = None
        if state is CycleState.UPTREND:
            peak = max(peak, close)
            broke = low is not None and close < low
            if broke or close <= peak * (1 - params.correction_pct):
                failed_ftd_date = bar_date if broke else None
                state, low, low_date, ftd_date = CycleState.CORRECTION, bar_low, bar_date, None
        elif close > peak:
            state, peak, low, low_date, day = CycleState.UPTREND, close, None, None, None
        elif bar_low < low:
            state, low, low_date, day = CycleState.CORRECTION, bar_low, bar_date, None
        elif state is CycleState.CORRECTION:
            if close > closes[i - 1]:
                state, day = CycleState.RALLY_ATTEMPT, 1
        else:
            day += 1
            if (
                day >= params.ftd_min_rally_day
                and close / closes[i - 1] - 1 >= params.ftd_gain_pct - _GAIN_TOLERANCE
                and volumes[i] > volumes[i - 1] > 0
            ):
                state, peak, ftd_date, day = CycleState.UPTREND, close, bar_date, None
            elif day >= params.attempt_expiry_sessions:
                state, day = CycleState.CORRECTION, None
        cycles.append(MarketCycle(state, low, low_date, day, ftd_date, failed_ftd_date))
    return cycles
