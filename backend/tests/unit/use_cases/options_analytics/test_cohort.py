from __future__ import annotations

from datetime import date, timedelta

from app.domain.options_analytics.models import CandidateKind, OptionCandidateInput
from app.domain.options_analytics.ports import (
    CandidateSourceSnapshot,
    LastCurrentMembership,
)
from app.use_cases.options_analytics.cohort import OptionsCandidateCohortBuilder


class Calendar:
    def sessions_ending_on(self, value: date, count: int) -> tuple[date, ...]:
        sessions = []
        cursor = value
        while len(sessions) < count:
            if cursor.weekday() < 5:
                sessions.append(cursor)
            cursor -= timedelta(days=1)
        return tuple(reversed(sessions))


class Source:
    def __init__(self, current: OptionCandidateInput) -> None:
        self.current = current

    def read(self, source_feature_run_id: int) -> CandidateSourceSnapshot:
        assert source_feature_run_id == 7
        return CandidateSourceSnapshot(
            source_feature_run_id=7,
            as_of_date=date(2026, 9, 4),
            top_candidate_inputs=(self.current,),
            leader_inputs=(),
            current_candidates=(),
        )

    def read_continuity_inputs(self, symbols, as_of_date):
        assert as_of_date == date(2026, 9, 4)
        return {
            symbol: OptionCandidateInput(symbol, 80, None, 100)
            for symbol in symbols
        }


class Memberships:
    def last_current_memberships(self, market: str, calculation_version: str):
        assert (market, calculation_version) == ("US", "v1")
        return {
            "AAPL": LastCurrentMembership(
                symbol="AAPL",
                as_of_date=date(2026, 9, 3),
                prior_best_rank=2,
                dividend_yield=0.01,
                dividend_source="pinned_feature_run",
            ),
            "MSFT": LastCurrentMembership(
                symbol="MSFT",
                as_of_date=date(2026, 9, 3),
                prior_best_rank=3,
                dividend_yield=0.0,
                dividend_source="zero_assumption",
            ),
        }


def test_cohort_builder_restores_current_status_and_keeps_only_dropouts_as_continuity() -> None:
    current = OptionCandidateInput("AAPL", 99, 200_000_001, 101, 0.01)
    cohort = OptionsCandidateCohortBuilder(
        candidate_source=Source(current),
        membership_reader=Memberships(),
        calendar=Calendar(),
        calculation_version="v1",
    ).build(7)

    assert cohort.by_symbol("AAPL").kind is CandidateKind.CURRENT
    assert cohort.by_symbol("MSFT").kind is CandidateKind.CONTINUITY
    assert cohort.by_symbol("MSFT").dividend_source == "zero_assumption"
    assert cohort.current == (cohort.by_symbol("AAPL"),)
