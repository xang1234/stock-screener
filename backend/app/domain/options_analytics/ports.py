"""Dependency-inversion ports for the options analytics bounded context."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime
from typing import Any, Protocol

from .models import ChainObservation, OptionCandidate, OptionsRunSummary


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


class CandidateSourceReader(Protocol):
    def read_candidate_cohort_inputs(self, source_run_id: int) -> Any: ...


class OptionsAnalyticsRepository(Protocol):
    def start_or_reuse_run(self, **values: Any) -> Any: ...

    def stage_candidates(self, run_id: int, candidates: Sequence[OptionCandidate]) -> None: ...

    def save_observation(self, run_id: int, observation: ChainObservation) -> None: ...

    def publish(self, run_id: int, summary: OptionsRunSummary) -> None: ...


class SessionCalendar(Protocol):
    def is_session(self, value: date) -> bool: ...

    def sessions_ending_on(self, value: date, count: int) -> Sequence[date]: ...


class Clock(Protocol):
    def now(self) -> datetime: ...


class ProgressReporter(Protocol):
    def report(self, **values: Any) -> None: ...


class CancellationToken(Protocol):
    def is_cancelled(self) -> bool: ...
