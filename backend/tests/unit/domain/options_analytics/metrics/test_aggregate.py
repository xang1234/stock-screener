from __future__ import annotations

from datetime import date, datetime, timezone

import pytest
from app.domain.options_analytics.metrics.aggregate import calculate_chain_metrics
from app.domain.options_analytics.models import (
    ChainObservation,
    NormalizedOptionContract,
    OptionSide,
)


def _contract(side: OptionSide, strike: float, *, iv: float, volume: int, oi: int, delta: float):
    return NormalizedOptionContract(
        side=side,
        strike=strike,
        bid=1,
        ask=2,
        last_price=1.5,
        volume=volume,
        open_interest=oi,
        implied_volatility=iv,
        last_trade_at=datetime(2026, 3, 1, tzinfo=timezone.utc),
        contract_size="REGULAR",
        multiplier=100,
        delta=delta,
    )


def test_aggregate_calculates_from_full_chain_and_is_json_finite() -> None:
    observation = ChainObservation(
        symbol="AAPL",
        expiration=date(2026, 3, 20),
        source_spot_price=100,
        fetched_at=datetime(2026, 3, 1, tzinfo=timezone.utc),
        contracts=(
            _contract(OptionSide.CALL, 96, iv=0.24, volume=120, oi=300, delta=0.70),
            _contract(OptionSide.PUT, 96, iv=0.31, volume=130, oi=400, delta=-0.25),
            _contract(OptionSide.CALL, 100, iv=0.20, volume=200, oi=500, delta=0.50),
            _contract(OptionSide.PUT, 100, iv=0.30, volume=200, oi=600, delta=-0.50),
            _contract(OptionSide.CALL, 104, iv=0.22, volume=140, oi=700, delta=0.25),
            _contract(OptionSide.PUT, 104, iv=0.32, volume=110, oi=200, delta=-0.70),
        ),
    )

    result = calculate_chain_metrics(
        observation,
        as_of_date=date(2026, 3, 1),
        risk_free_rate=0.04,
        dividend_yield=0.01,
        closes=tuple(float(close) for close in range(100, 121)),
    )

    assert result.max_pain.available is True
    assert result.net_gex.available is True
    assert result.atm_iv.value == pytest.approx(0.25)
    assert result.skew_25_delta.value == pytest.approx(0.09)
    assert result.realized_volatility.available is True
    assert result.vrp.available is True
    assert result.activity.activity_intensity.available is True
    result.assert_json_finite()


def test_aggregate_preserves_partial_metric_availability() -> None:
    observation = ChainObservation(
        symbol="AAPL",
        expiration=date(2026, 3, 20),
        source_spot_price=100,
        fetched_at=datetime(2026, 3, 1, tzinfo=timezone.utc),
        contracts=(
            _contract(OptionSide.CALL, 100, iv=0.20, volume=120, oi=0, delta=0.50),
            _contract(OptionSide.PUT, 100, iv=float("inf"), volume=120, oi=0, delta=-0.50),
        ),
    )

    result = calculate_chain_metrics(
        observation,
        as_of_date=date(2026, 3, 1),
        risk_free_rate=0.04,
        dividend_yield=0.01,
        closes=(),
    )

    assert result.atm_iv.available is False
    assert result.realized_volatility.available is False
    assert result.activity.volume_oi_ratio.available is False
    result.assert_json_finite()
