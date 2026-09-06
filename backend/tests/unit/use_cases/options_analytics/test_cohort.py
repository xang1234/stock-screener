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
        self.continuity_symbols: tuple[str, ...] = ()

    def read(self, source_feature_run_id: int) -> CandidateSourceSnapshot:
        assert source_feature_run_id == 7
        return CandidateSourceSnapshot(
            source_feature_run_id=7,
            as_of_date=date(2026, 9, 4),
            top_candidate_inputs=(self.current,),
            leader_inputs=(),
        )

    def read_continuity_inputs(self, symbols, as_of_date):
        assert as_of_date == date(2026, 9, 4)
        self.continuity_symbols = tuple(symbols)
        return {
            symbol: OptionCandidateInput(symbol, 80, None, 100) for symbol in symbols
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


def test_cohort_builder_restores_current_status_and_keeps_only_dropouts_as_continuity() -> (
    None
):
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


def test_cohort_builder_limits_history_reads_to_ranked_recent_dropouts() -> None:
    current = OptionCandidateInput("CURRENT", 99, 200_000_001, 101, 0.01)
    source = Source(current)
    memberships = {
        "CURRENT": LastCurrentMembership(
            symbol="CURRENT",
            as_of_date=date(2026, 9, 3),
            prior_best_rank=1,
            dividend_yield=None,
            dividend_source=None,
        ),
        **{
            f"RECENT{rank:02}": LastCurrentMembership(
                symbol=f"RECENT{rank:02}",
                as_of_date=date(2026, 9, 3),
                prior_best_rank=rank,
                dividend_yield=None,
                dividend_source=None,
            )
            for rank in range(25, 0, -1)
        },
        "OLD": LastCurrentMembership(
            symbol="OLD",
            as_of_date=date(2026, 8, 27),
            prior_best_rank=1,
            dividend_yield=None,
            dividend_source=None,
        ),
    }

    class ManyMemberships:
        def last_current_memberships(self, market: str, calculation_version: str):
            assert (market, calculation_version) == ("US", "v1")
            return memberships

    cohort = OptionsCandidateCohortBuilder(
        candidate_source=source,
        membership_reader=ManyMemberships(),
        calendar=Calendar(),
        calculation_version="v1",
    ).build(7)

    expected = tuple(f"RECENT{rank:02}" for rank in range(1, 21))
    assert source.continuity_symbols == expected
    assert tuple(candidate.symbol for candidate in cohort.candidates[1:]) == expected
