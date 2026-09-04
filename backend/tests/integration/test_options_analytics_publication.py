from __future__ import annotations

from datetime import date, timedelta

from app.domain.options_analytics.history import HistoricalObservation, history_readiness
from app.domain.options_analytics.models import (
    CandidateKind,
    ObservationState,
    OptionCandidateInput,
)
from app.domain.options_analytics.quality import evaluate_publication
from app.domain.options_analytics.selection import (
    CandidateHistoryInput,
    build_candidate_cohort,
    select_current_candidates,
)


def _input(symbol: str, score: float, volume: float = 200_000_000):
    return OptionCandidateInput(symbol, score, volume, 100.0)


def test_liquid_current_publication_and_returning_ticker_history_are_continuous():
    candidates = [_input(f"C{index:02d}", 100 - index) for index in range(42)]
    candidates.append(_input("EXACT", 999, 100_000_000))
    leaders = [_input("C00", 100), _input("LEADER", 98)]

    current = select_current_candidates(candidates, leaders)

    assert len([row for row in current if row.candidate_rank is not None]) == 40
    assert "EXACT" not in {row.symbol for row in current}
    overlap = next(row for row in current if row.symbol == "C00")
    assert (overlap.candidate_rank, overlap.leader_rank) == (1, 1)
    assert len(current) == 41

    publication = evaluate_publication(
        current,
        core_valid_symbols={row.symbol for row in current[:37]},
    )
    assert publication.publish is True
    assert publication.coverage >= 0.90

    dropped = CandidateHistoryInput(_input("RETURN", 80), 3, 5)
    continuity = build_candidate_cohort([], [], continuity=[dropped])
    assert continuity[0].kind is CandidateKind.CONTINUITY
    returned = build_candidate_cohort([_input("RETURN", 95)], [], continuity=[dropped])
    assert returned[0].kind is CandidateKind.CURRENT

    sessions = tuple(date(2026, 8, 3) + timedelta(days=index) for index in range(30))
    observations = [
        HistoricalObservation(sessions[index], "options-analytics-v1", ObservationState.AVAILABLE)
        for index in (0, 4, 9, 15, 22, 29)
    ]
    readiness = history_readiness(
        observations,
        sessions,
        calculation_version="options-analytics-v1",
    )
    assert readiness.lifetime_observation_count == 6
    assert readiness.lifetime_observation_count > 1


def test_below_gate_does_not_authorize_pointer_movement():
    current = select_current_candidates(
        [_input(f"S{index}", 100 - index) for index in range(10)],
        [],
    )
    decision = evaluate_publication(
        current,
        core_valid_symbols={row.symbol for row in current[:8]},
    )
    published_pointer = 17
    if decision.publish:
        published_pointer = 18

    assert decision.publish is False
    assert decision.reason_codes == ("insufficient_core_coverage",)
    assert published_pointer == 17
