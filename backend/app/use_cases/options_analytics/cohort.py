"""Assemble the pinned Current and Continuity options cohort."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import date
from typing import Protocol

from app.domain.options_analytics.models import (
    CandidateKind,
    OptionCandidate,
)
from app.domain.options_analytics.ports import (
    LastCurrentMembership,
    OptionsCandidateSource,
    SessionCalendar,
)
from app.domain.options_analytics.selection import (
    CandidateHistoryInput,
    build_candidate_cohort,
    select_current_candidates,
)


class MembershipReader(Protocol):
    def last_current_memberships(
        self,
        market: str,
        calculation_version: str,
    ) -> Mapping[str, LastCurrentMembership]: ...


@dataclass(frozen=True)
class OptionsCohortSnapshot:
    source_feature_run_id: int
    as_of_date: date
    candidates: tuple[OptionCandidate, ...]

    @property
    def current(self) -> tuple[OptionCandidate, ...]:
        return tuple(
            candidate
            for candidate in self.candidates
            if candidate.kind is CandidateKind.CURRENT
        )

    def by_symbol(self, symbol: str) -> OptionCandidate:
        canonical = symbol.strip().upper()
        return next(
            candidate for candidate in self.candidates if candidate.symbol == canonical
        )


class OptionsCandidateCohortBuilder:
    def __init__(
        self,
        *,
        candidate_source: OptionsCandidateSource,
        membership_reader: MembershipReader,
        calendar: SessionCalendar,
        calculation_version: str,
    ) -> None:
        self._candidate_source = candidate_source
        self._membership_reader = membership_reader
        self._calendar = calendar
        self._calculation_version = calculation_version

    def build(self, source_run_id: int, *, market: str = "US") -> OptionsCohortSnapshot:
        source = self._candidate_source.read(source_run_id)
        current = tuple(
            select_current_candidates(
                source.top_candidate_inputs,
                source.leader_inputs,
            )
        )
        current_symbols = {candidate.symbol for candidate in current}
        memberships = self._membership_reader.last_current_memberships(
            market,
            self._calculation_version,
        )
        recent_sessions = tuple(self._calendar.sessions_ending_on(source.as_of_date, 6))
        inputs = self._candidate_source.read_continuity_inputs(
            tuple(memberships),
            source.as_of_date,
        )
        continuity: list[CandidateHistoryInput] = []
        for symbol, membership in memberships.items():
            if (
                symbol in current_symbols
                or membership.as_of_date not in recent_sessions
            ):
                continue
            candidate_input = inputs.get(symbol)
            if candidate_input is None:
                continue
            continuity.append(
                CandidateHistoryInput(
                    candidate=replace(
                        candidate_input,
                        dividend_yield=membership.dividend_yield,
                        dividend_source=membership.dividend_source,
                    ),
                    sessions_since_current=sum(
                        session > membership.as_of_date for session in recent_sessions
                    ),
                    prior_best_rank=membership.prior_best_rank,
                )
            )
        candidates = tuple(
            build_candidate_cohort(
                source.top_candidate_inputs,
                source.leader_inputs,
                continuity=continuity,
            )
        )
        return OptionsCohortSnapshot(
            source_feature_run_id=source.source_feature_run_id,
            as_of_date=source.as_of_date,
            candidates=candidates,
        )


__all__ = ["OptionsCandidateCohortBuilder", "OptionsCohortSnapshot"]
