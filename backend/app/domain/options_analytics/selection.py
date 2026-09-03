"""Deterministic Current and Continuity Candidate selection."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from app.domain.scanning.leadership_policy import leadership_order_key

from .models import CandidateKind, OptionCandidate, OptionCandidateInput

MIN_DAILY_DOLLAR_VOLUME_USD = 100_000_000
SOURCE_CANDIDATE_LIMIT = 40
CONTINUITY_SESSION_LIMIT = 5
CONTINUITY_CANDIDATE_LIMIT = 20
TOTAL_COHORT_LIMIT = 100


@dataclass(frozen=True)
class CandidateHistoryInput:
    candidate: OptionCandidateInput
    sessions_since_current: int
    prior_best_rank: int


def _is_liquid(candidate: OptionCandidateInput) -> bool:
    value = candidate.daily_dollar_volume
    return value is not None and math.isfinite(value) and value > MIN_DAILY_DOLLAR_VOLUME_USD


def _rank_source(rows: Iterable[OptionCandidateInput]) -> list[OptionCandidateInput]:
    eligible_by_symbol: dict[str, OptionCandidateInput] = {}
    for row in rows:
        if not _is_liquid(row):
            continue
        existing = eligible_by_symbol.get(row.symbol)
        if existing is None or leadership_order_key(row) < leadership_order_key(existing):
            eligible_by_symbol[row.symbol] = row
    return sorted(eligible_by_symbol.values(), key=leadership_order_key)[:SOURCE_CANDIDATE_LIMIT]


def _as_current(
    source: OptionCandidateInput,
    *,
    candidate_rank: int | None,
    leader_rank: int | None,
) -> OptionCandidate:
    return OptionCandidate(
        symbol=source.symbol,
        kind=CandidateKind.CURRENT,
        composite_score=source.composite_score,
        daily_dollar_volume=source.daily_dollar_volume,
        spot_price=source.spot_price,
        dividend_yield=source.dividend_yield,
        price_closes=source.price_closes,
        candidate_rank=candidate_rank,
        leader_rank=leader_rank,
    )


def select_current_candidates(
    top_candidates: Iterable[OptionCandidateInput],
    leaders: Iterable[OptionCandidateInput],
) -> list[OptionCandidate]:
    """Select each source independently, then merge without losing provenance."""
    ranked_candidates = _rank_source(top_candidates)
    ranked_leaders = _rank_source(leaders)
    candidate_ranks = {row.symbol: rank for rank, row in enumerate(ranked_candidates, 1)}
    leader_ranks = {row.symbol: rank for rank, row in enumerate(ranked_leaders, 1)}

    merged: list[OptionCandidate] = []
    seen: set[str] = set()
    for row in (*ranked_candidates, *ranked_leaders):
        if row.symbol in seen:
            continue
        seen.add(row.symbol)
        merged.append(
            _as_current(
                row,
                candidate_rank=candidate_ranks.get(row.symbol),
                leader_rank=leader_ranks.get(row.symbol),
            )
        )
    return merged


def _continuity_order_key(row: CandidateHistoryInput) -> tuple[int, int, str]:
    return (row.sessions_since_current, row.prior_best_rank, row.candidate.symbol)


def build_candidate_cohort(
    top_candidates: Iterable[OptionCandidateInput],
    leaders: Iterable[OptionCandidateInput],
    *,
    continuity: Iterable[CandidateHistoryInput] = (),
) -> list[OptionCandidate]:
    current = select_current_candidates(top_candidates, leaders)
    current_symbols = {row.symbol for row in current}
    continuity_by_symbol: dict[str, CandidateHistoryInput] = {}
    for row in continuity:
        if not 1 <= row.sessions_since_current <= CONTINUITY_SESSION_LIMIT:
            continue
        if row.candidate.symbol in current_symbols:
            continue
        existing = continuity_by_symbol.get(row.candidate.symbol)
        if existing is None or _continuity_order_key(row) < _continuity_order_key(existing):
            continuity_by_symbol[row.candidate.symbol] = row

    selected_history = sorted(
        continuity_by_symbol.values(), key=_continuity_order_key
    )[:CONTINUITY_CANDIDATE_LIMIT]
    continuity_candidates = [
        OptionCandidate(
            symbol=row.candidate.symbol,
            kind=CandidateKind.CONTINUITY,
            composite_score=row.candidate.composite_score,
            daily_dollar_volume=row.candidate.daily_dollar_volume,
            spot_price=row.candidate.spot_price,
            dividend_yield=row.candidate.dividend_yield,
            price_closes=row.candidate.price_closes,
            sessions_since_current=row.sessions_since_current,
            prior_best_rank=row.prior_best_rank,
        )
        for row in selected_history
    ]
    return (current + continuity_candidates)[:TOTAL_COHORT_LIMIT]

