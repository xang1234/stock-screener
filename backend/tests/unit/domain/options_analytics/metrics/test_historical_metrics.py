from __future__ import annotations

from datetime import date, timedelta

import pytest

from app.domain.options_analytics.history import HistoricalObservation
from app.domain.options_analytics.metrics.history import calculate_historical_metrics
from app.domain.options_analytics.models import ObservationState


def _sessions(count: int) -> tuple[date, ...]:
    end = date(2026, 9, 4)
    return tuple(end - timedelta(days=count - index - 1) for index in range(count))


def _row(session: date, value: float) -> HistoricalObservation:
    return HistoricalObservation(
        session=session,
        calculation_version="v1",
        state=ObservationState.AVAILABLE,
        max_pain=value,
        net_gex=value * 10,
        gamma_flip=value + 1,
        atm_iv=value / 100,
        skew_25_delta=-value / 1000,
        realized_volatility=value / 200,
        vrp=value / 300,
        activity_intensity=value / 10,
    )


def test_five_observation_metrics_are_absolute_changes_from_oldest_to_current() -> None:
    sessions = _sessions(7)
    observations = tuple(
        _row(session, value)
        for session, value in zip(sessions[-5:], (1, 2, 4, 8, 10), strict=True)
    )

    metrics = calculate_historical_metrics(
        observations,
        sessions,
        calculation_version="v1",
    )

    assert metrics.max_pain_change_5.value == 9
    assert metrics.net_gex_change_5.value == 90
    assert metrics.gamma_flip_change_5.value == 9
    assert metrics.atm_iv_change_5.value == pytest.approx(0.09)
    assert metrics.skew_25_delta_change_5.value == pytest.approx(-0.009)
    assert metrics.realized_volatility_change_5.value == pytest.approx(0.045)
    assert metrics.vrp_change_5.value == pytest.approx(0.03)
    assert metrics.activity_intensity_change_5.value == pytest.approx(0.9)


def test_iv_percentile_and_rank_use_twenty_compatible_atm_iv_observations() -> None:
    sessions = _sessions(30)
    observations = tuple(
        _row(session, value) for value, session in enumerate(sessions[-20:], 1)
    )

    metrics = calculate_historical_metrics(
        observations,
        sessions,
        calculation_version="v1",
    )

    assert metrics.iv_percentile.value == 1.0
    assert metrics.iv_rank.value == 1.0


def test_historical_metrics_stay_unavailable_until_each_window_is_ready() -> None:
    sessions = _sessions(30)
    metrics = calculate_historical_metrics(
        tuple(_row(session, value) for value, session in enumerate(sessions[-4:], 1)),
        sessions,
        calculation_version="v1",
    )

    assert metrics.max_pain_change_5.available is False
    assert metrics.iv_percentile.available is False
    assert metrics.iv_rank.available is False
    assert metrics.max_pain_change_5.reason_codes == ("building_history",)


def test_historical_metrics_require_a_valid_current_observation() -> None:
    sessions = _sessions(30)
    prior = tuple(
        _row(session, value) for value, session in enumerate(sessions[-21:-1], 1)
    )
    invalid_current = HistoricalObservation(
        session=sessions[-1],
        calculation_version="v1",
        state=ObservationState.INSUFFICIENT_QUALITY,
        atm_iv=0.50,
        max_pain=120,
    )

    metrics = calculate_historical_metrics(
        (*prior, invalid_current),
        sessions,
        calculation_version="v1",
    )

    assert metrics.iv_percentile.available is False
    assert metrics.max_pain_change_5.available is False


def test_each_historical_metric_requires_its_current_value() -> None:
    sessions = _sessions(30)
    prior = tuple(
        _row(session, value) for value, session in enumerate(sessions[-21:-1], 1)
    )
    current_without_values = HistoricalObservation(
        session=sessions[-1],
        calculation_version="v1",
        state=ObservationState.AVAILABLE,
    )

    metrics = calculate_historical_metrics(
        (*prior, current_without_values),
        sessions,
        calculation_version="v1",
    )

    assert metrics.iv_percentile.reason_codes == ("metric_unavailable",)
    assert metrics.iv_rank.reason_codes == ("metric_unavailable",)
    assert metrics.max_pain_change_5.reason_codes == ("metric_unavailable",)
