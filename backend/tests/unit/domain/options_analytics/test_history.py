from __future__ import annotations

from datetime import date, timedelta

from app.domain.options_analytics.history import HistoricalObservation, history_readiness
from app.domain.options_analytics.models import ObservationState


def _sessions(count: int) -> tuple[date, ...]:
    start = date(2026, 1, 5)
    sessions: list[date] = []
    cursor = start
    while len(sessions) < count:
        if cursor.weekday() < 5:
            sessions.append(cursor)
        cursor += timedelta(days=1)
    return tuple(sessions)


def _observation(session: date, version: str = "v1") -> HistoricalObservation:
    return HistoricalObservation(
        session=session,
        calculation_version=version,
        state=ObservationState.AVAILABLE,
    )


def test_five_compatible_observations_in_last_seven_enable_short_history() -> None:
    sessions = _sessions(10)
    observations = [_observation(session) for session in sessions[-7::] if session != sessions[-3]][:5]

    readiness = history_readiness(observations, sessions, calculation_version="v1")

    assert readiness.short_observation_count == 5
    assert readiness.short_history_available is True


def test_twenty_compatible_observations_in_last_thirty_enable_iv_history() -> None:
    sessions = _sessions(30)
    observations = [
        _observation(session)
        for index, session in enumerate(sessions)
        if index % 3 != 0
    ][:20]

    readiness = history_readiness(observations, sessions, calculation_version="v1")

    assert readiness.iv_observation_count == 20
    assert readiness.iv_history_available is True


def test_gaps_do_not_reset_lifetime_history_and_incompatible_versions_are_ignored() -> None:
    sessions = _sessions(30)
    observations = [
        _observation(sessions[0]),
        _observation(sessions[-1]),
        _observation(sessions[-2], version="v0"),
    ]

    readiness = history_readiness(observations, sessions, calculation_version="v1")

    assert readiness.lifetime_observation_count == 2
    assert readiness.short_observation_count == 1
    assert readiness.iv_observation_count == 2
    assert readiness.short_history_available is False
    assert readiness.iv_history_available is False
    assert readiness.reason_codes == ("building_history",)
