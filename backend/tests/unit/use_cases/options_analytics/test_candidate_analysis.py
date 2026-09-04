from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from app.domain.options_analytics.metrics.aggregate import calculate_chain_metrics
from app.domain.options_analytics.models import (
    CandidateKind,
    ChainObservation,
    NormalizedOptionContract,
    OptionCandidate,
    OptionSide,
)
from app.use_cases.options_analytics.analysis_models import (
    OptionsMetricValues,
    OptionsStrikePoint,
)
from app.use_cases.options_analytics.analysis_projection import metric_values
from app.use_cases.options_analytics.candidate_analysis import (
    AnalysisContext,
    OptionsCandidateAnalyzer,
    UnavailableCandidateAnalysis,
)


class Calendar:
    def is_session(self, value: date) -> bool:
        return value.weekday() < 5

    def sessions_ending_on(self, value: date, count: int) -> tuple[date, ...]:
        return tuple(
            value - timedelta(days=offset) for offset in reversed(range(count))
        )


class Provider:
    def __init__(self) -> None:
        self.calls = 0

    def list_expirations(self, symbol: str):
        self.calls += 1
        raise AssertionError(symbol)


def test_missing_spot_returns_a_distinct_unavailable_result_without_provider_io() -> (
    None
):
    provider = Provider()
    analyzer = OptionsCandidateAnalyzer(
        provider=provider,
        calendar=Calendar(),
        calculation_version="v1",
    )
    candidate = OptionCandidate(
        symbol="AAPL",
        kind=CandidateKind.CURRENT,
        composite_score=99,
        daily_dollar_volume=200_000_001,
        spot_price=None,
    )

    result = analyzer.analyze(
        candidate,
        AnalysisContext(
            as_of_date=date(2026, 9, 4),
            market="US",
            risk_free_rate=0.04,
        ),
    )

    assert isinstance(result, UnavailableCandidateAnalysis)
    assert result.reason_codes == ("source_spot_unavailable",)
    assert result.assumptions == {
        "dividend_yield": 0.0,
        "dividend_source": "zero_assumption",
    }
    assert provider.calls == 0


def test_persistence_projection_has_closed_metric_and_strike_fields() -> None:
    metrics = OptionsMetricValues(atm_iv=0.25, call_volume=120)
    point = OptionsStrikePoint(strike=100, call_open_interest=200)

    assert metrics.atm_iv == 0.25
    assert metrics.call_volume == 120
    assert point.call_open_interest == 200
    assert point.strike == 100


def test_persistence_projection_preserves_incomplete_side_totals_as_missing() -> None:
    def contract(side, strike, *, volume, open_interest):
        return NormalizedOptionContract(
            side=side,
            strike=strike,
            bid=1,
            ask=2,
            last_price=1.5,
            volume=volume,
            open_interest=open_interest,
            implied_volatility=0.25,
            last_trade_at=None,
            contract_size="REGULAR",
            multiplier=100,
        )

    observation = ChainObservation(
        symbol="AAPL",
        expiration=date(2026, 9, 18),
        source_spot_price=100,
        fetched_at=datetime(2026, 9, 4, tzinfo=timezone.utc),
        contracts=(
            contract(OptionSide.CALL, 100, volume=50, open_interest=200),
            contract(OptionSide.CALL, 105, volume=None, open_interest=300),
            contract(OptionSide.PUT, 100, volume=0, open_interest=None),
            contract(OptionSide.PUT, 95, volume=0, open_interest=None),
        ),
    )
    metrics = calculate_chain_metrics(
        observation,
        as_of_date=date(2026, 9, 4),
        risk_free_rate=0.04,
        dividend_yield=0.0,
        closes=(),
    )

    values = metric_values(metrics, observation)

    assert values.call_volume is None
    assert values.call_open_interest == 500
    assert values.put_volume == 0
    assert values.put_open_interest is None
