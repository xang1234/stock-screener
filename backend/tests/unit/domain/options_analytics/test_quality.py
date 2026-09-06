from __future__ import annotations

from datetime import date, datetime, timezone

from app.domain.options_analytics.models import (
    CandidateKind,
    ChainObservation,
    NormalizedOptionContract,
    OptionCandidateInput,
    OptionSide,
)
from app.domain.options_analytics.quality import (
    evaluate_publication,
    has_core_chain_coverage,
)
from app.domain.options_analytics.selection import (
    CandidateHistoryInput,
    build_candidate_cohort,
)


def _input(symbol: str) -> OptionCandidateInput:
    return OptionCandidateInput(
        symbol=symbol,
        composite_score=90,
        daily_dollar_volume=150_000_000,
        spot_price=100,
    )


def test_empty_current_cohort_blocks_publication() -> None:
    decision = evaluate_publication([], core_valid_symbols=set())

    assert decision.publish is False
    assert decision.coverage == 0.0
    assert decision.reason_codes == ("empty_current_cohort",)


def test_exactly_90_percent_current_coverage_publishes() -> None:
    cohort = build_candidate_cohort([_input(f"S{index}") for index in range(10)], [])

    decision = evaluate_publication(
        cohort,
        core_valid_symbols={f"S{index}" for index in range(9)},
    )

    assert decision.publish is True
    assert decision.current_count == 10
    assert decision.core_valid_current_count == 9
    assert decision.coverage == 0.9
    assert decision.reason_codes == ()


def test_coverage_below_90_percent_blocks_publication() -> None:
    cohort = build_candidate_cohort([_input(f"S{index}") for index in range(9)], [])

    decision = evaluate_publication(
        cohort,
        core_valid_symbols={f"S{index}" for index in range(8)},
    )

    assert decision.publish is False
    assert decision.coverage == 8 / 9
    assert decision.reason_codes == ("insufficient_core_coverage",)


def test_continuity_never_enters_publication_denominator() -> None:
    cohort = build_candidate_cohort(
        [_input("CURRENT")],
        [],
        continuity=[CandidateHistoryInput(_input("HISTORY"), 1, 1)],
    )

    decision = evaluate_publication(cohort, core_valid_symbols={"CURRENT"})

    assert [row.kind for row in cohort] == [CandidateKind.CURRENT, CandidateKind.CONTINUITY]
    assert decision.current_count == 1
    assert decision.core_valid_current_count == 1
    assert decision.coverage == 1.0
    assert decision.publish is True


def test_core_chain_requires_five_contracts_per_side_positive_oi_and_three_strikes() -> None:
    def observation(calls=5, puts=5, strikes=(95, 100, 105), put_oi=10):
        contracts = []
        for side, count, oi in (
            (OptionSide.CALL, calls, 10),
            (OptionSide.PUT, puts, put_oi),
        ):
            for index in range(count):
                contracts.append(
                    NormalizedOptionContract(
                        side=side,
                        strike=strikes[index % len(strikes)],
                        bid=1,
                        ask=2,
                        last_price=1.5,
                        volume=10,
                        open_interest=oi,
                        implied_volatility=0.25,
                        last_trade_at=None,
                        contract_size="REGULAR",
                        multiplier=100,
                    )
                )
        return ChainObservation(
            symbol="AAPL",
            expiration=date(2026, 9, 18),
            source_spot_price=100,
            fetched_at=datetime(2026, 9, 4, tzinfo=timezone.utc),
            contracts=tuple(contracts),
        )

    assert has_core_chain_coverage(observation()) is True
    assert has_core_chain_coverage(observation(calls=4)) is False
    assert has_core_chain_coverage(observation(put_oi=0)) is False
    assert has_core_chain_coverage(observation(strikes=(100,))) is False
