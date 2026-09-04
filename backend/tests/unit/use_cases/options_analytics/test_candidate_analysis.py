from __future__ import annotations

from datetime import date, timedelta

from app.domain.options_analytics.models import CandidateKind, OptionCandidate
from app.use_cases.options_analytics.candidate_analysis import (
    AnalysisContext,
    OptionsCandidateAnalyzer,
    UnavailableCandidateAnalysis,
)


class Calendar:
    def is_session(self, value: date) -> bool:
        return value.weekday() < 5

    def sessions_ending_on(self, value: date, count: int) -> tuple[date, ...]:
        return tuple(value - timedelta(days=offset) for offset in reversed(range(count)))


class Provider:
    def __init__(self) -> None:
        self.calls = 0

    def list_expirations(self, symbol: str):
        self.calls += 1
        raise AssertionError(symbol)


class History:
    def symbol_history(self, symbol: str, *, market: str, calculation_version: str):
        raise AssertionError((symbol, market, calculation_version))


def test_missing_spot_returns_a_distinct_unavailable_result_without_provider_io() -> None:
    provider = Provider()
    analyzer = OptionsCandidateAnalyzer(
        provider=provider,
        history_reader=History(),
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
