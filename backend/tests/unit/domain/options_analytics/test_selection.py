from __future__ import annotations

from app.domain.options_analytics.models import CandidateKind, OptionCandidateInput
from app.domain.options_analytics.selection import (
    CandidateHistoryInput,
    build_candidate_cohort,
    select_current_candidates,
)


def _input(symbol: str, score: float, dollar_volume: float = 150_000_000) -> OptionCandidateInput:
    return OptionCandidateInput(
        symbol=symbol,
        composite_score=score,
        daily_dollar_volume=dollar_volume,
        spot_price=100.0,
    )


def test_each_source_uses_score_descending_then_canonical_symbol() -> None:
    selected = select_current_candidates(
        top_candidates=[_input("ccc", 90), _input("BBB", 95), _input("aaa", 90)],
        leaders=[_input("zzz", 80), _input("YYY", 80)],
    )

    assert [row.symbol for row in selected] == ["BBB", "AAA", "CCC", "YYY", "ZZZ"]
    assert [(row.symbol, row.candidate_rank) for row in selected[:3]] == [
        ("BBB", 1),
        ("AAA", 2),
        ("CCC", 3),
    ]
    assert [(row.symbol, row.leader_rank) for row in selected[3:]] == [
        ("YYY", 1),
        ("ZZZ", 2),
    ]


def test_liquidity_threshold_is_strictly_greater_than_100_million_usd() -> None:
    selected = select_current_candidates(
        top_candidates=[
            _input("EXACT", 99, 100_000_000),
            _input("ABOVE", 98, 100_000_000.01),
        ],
        leaders=[],
    )

    assert [row.symbol for row in selected] == ["ABOVE"]


def test_sources_are_capped_independently_and_do_not_borrow_unused_slots() -> None:
    one_candidate = [_input("C000", 100)]
    forty_five_leaders = [_input(f"L{index:03}", 100 - index) for index in range(45)]

    selected = select_current_candidates(one_candidate, forty_five_leaders)

    assert len(selected) == 41
    assert sum(row.candidate_rank is not None for row in selected) == 1
    assert sum(row.leader_rank is not None for row in selected) == 40
    assert max(row.leader_rank or 0 for row in selected) == 40


def test_duplicate_symbol_preserves_both_sources_and_ranks() -> None:
    selected = select_current_candidates(
        [_input("AAPL", 99), _input("MSFT", 98)],
        [_input("NVDA", 100), _input("AAPL", 97)],
    )

    apple = next(row for row in selected if row.symbol == "AAPL")
    assert apple.kind is CandidateKind.CURRENT
    assert apple.candidate_rank == 1
    assert apple.leader_rank == 2
    assert apple.is_candidate is True
    assert apple.is_leader is True


def test_continuity_lasts_five_sessions_and_is_gone_before_sixth() -> None:
    cohort = build_candidate_cohort(
        top_candidates=[],
        leaders=[],
        continuity=[
            CandidateHistoryInput(
                _input("FIFTH", 80), sessions_since_current=5, prior_best_rank=2
            ),
            CandidateHistoryInput(
                _input("SIXTH", 90), sessions_since_current=6, prior_best_rank=1
            ),
        ],
    )

    assert [(row.symbol, row.kind) for row in cohort] == [
        ("FIFTH", CandidateKind.CONTINUITY)
    ]


def test_continuity_cap_prefers_recent_membership_then_prior_rank_then_symbol() -> None:
    continuity = [
        CandidateHistoryInput(
            _input(f"S{index:02}", 50),
            sessions_since_current=2 if index < 5 else 1,
            prior_best_rank=30 - index,
        )
        for index in range(25)
    ]

    cohort = build_candidate_cohort([], [], continuity=continuity)

    assert len(cohort) == 20
    assert [row.symbol for row in cohort[:3]] == ["S24", "S23", "S22"]
    assert all(row.kind is CandidateKind.CONTINUITY for row in cohort)


def test_current_membership_overrides_continuity_and_total_is_bounded() -> None:
    candidates = [_input(f"C{index:03}", 200 - index) for index in range(40)]
    leaders = [_input(f"L{index:03}", 200 - index) for index in range(40)]
    continuity = [
        CandidateHistoryInput(
            _input("C000" if index == 0 else f"H{index:03}", 50),
            sessions_since_current=1,
            prior_best_rank=index + 1,
        )
        for index in range(31)
    ]

    cohort = build_candidate_cohort(candidates, leaders, continuity=continuity)

    assert len(cohort) == 100
    assert sum(row.kind is CandidateKind.CURRENT for row in cohort) == 80
    assert sum(row.kind is CandidateKind.CONTINUITY for row in cohort) == 20
    assert next(row for row in cohort if row.symbol == "C000").kind is CandidateKind.CURRENT
