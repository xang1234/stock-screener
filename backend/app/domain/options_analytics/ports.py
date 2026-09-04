"""Dependency-inversion ports for the options analytics bounded context."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Protocol, runtime_checkable

from .models import ChainObservation, OptionCandidate, OptionCandidateInput


class OptionsProviderError(RuntimeError):
    """A non-retryable provider or payload failure for one symbol."""


class TransientOptionsProviderError(OptionsProviderError):
    """A provider failure that may be retried within the symbol budget."""


class OptionsProvider(Protocol):
    def list_expirations(self, symbol: str) -> Sequence[date]: ...

    def fetch_chain(
        self, symbol: str, expiration: date, *, source_spot_price: float
    ) -> ChainObservation: ...

    def risk_free_rate(self, on_or_before: date) -> float | None: ...


@dataclass(frozen=True)
class CandidateSourceSnapshot:
    source_feature_run_id: int
    as_of_date: date
    top_candidate_inputs: tuple[OptionCandidateInput, ...]
    leader_inputs: tuple[OptionCandidateInput, ...]
    current_candidates: tuple[OptionCandidate, ...]


@dataclass(frozen=True)
class LastCurrentMembership:
    symbol: str
    as_of_date: date
    prior_best_rank: int
    dividend_yield: float | None
    dividend_source: str | None


@runtime_checkable
class OptionsCandidateSource(Protocol):
    def read(self, source_feature_run_id: int) -> CandidateSourceSnapshot: ...

    def read_continuity_inputs(
        self,
        symbols: Sequence[str],
        as_of_date: date,
    ) -> Mapping[str, OptionCandidateInput]: ...


class SessionCalendar(Protocol):
    def is_session(self, value: date) -> bool: ...

    def sessions_ending_on(self, value: date, count: int) -> Sequence[date]: ...


class CancellationToken(Protocol):
    def is_cancelled(self) -> bool: ...
