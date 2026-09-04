"""Typed persistence ports used by options analytics application services."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Protocol

from app.domain.options_analytics.history import HistoricalObservation
from app.domain.options_analytics.models import (
    OptionCandidate,
    OptionsRunSummary,
)
from app.domain.options_analytics.ports import LastCurrentMembership

from .analysis_models import CandidateAnalysis


@dataclass(frozen=True)
class OptionsHistoryRecord:
    run_id: int
    as_of_date: date
    calculation_version: str
    observation_state: str
    core_valid: bool
    max_pain: float | None = None
    net_gex: float | None = None
    gamma_flip: float | None = None
    atm_iv: float | None = None
    skew_25_delta: float | None = None
    realized_volatility: float | None = None
    vrp: float | None = None
    activity_intensity: float | None = None
    iv_percentile: float | None = None
    iv_rank: float | None = None
    max_pain_change_5: float | None = None
    net_gex_change_5: float | None = None
    gamma_flip_change_5: float | None = None
    atm_iv_change_5: float | None = None
    skew_25_delta_change_5: float | None = None
    realized_volatility_change_5: float | None = None
    vrp_change_5: float | None = None
    activity_intensity_change_5: float | None = None


class OptionsRunRecord(Protocol):
    id: int
    source_feature_run_id: int | None
    status: str
    expected_count: int
    completed_count: int
    core_valid_current_count: int
    failed_count: int
    retried_count: int
    coverage: float
    warnings_json: Sequence[str] | None
    risk_free_rate: float | None
    assumptions_json: Mapping[str, object] | None


class OptionsRunItemRecord(Protocol):
    security_symbol: str
    candidate_kind: str
    observation_state: str
    core_valid: bool
    activity_intensity: float | None
    retry_count: int


class OptionsRunWriter(Protocol):
    def start_or_reuse(
        self,
        *,
        market: str,
        source_feature_run_id: int,
        calculation_version: str,
        schema_version: str,
        provider: str,
        input_signature: str,
        as_of_date: date,
        force: bool = False,
    ) -> OptionsRunRecord: ...

    def stage_candidates(
        self,
        run_id: int,
        candidates: Sequence[OptionCandidate],
    ) -> Sequence[OptionsRunItemRecord]: ...

    def save_run_assumptions(
        self,
        run_id: int,
        *,
        risk_free_rate: float | None,
        assumptions: Mapping[str, object],
    ) -> None: ...

    def incomplete_symbols(self, run_id: int) -> tuple[str, ...]: ...

    def save_analysis(self, run_id: int, analysis: CandidateAnalysis) -> None: ...

    def items_for_run(self, run_id: int) -> Sequence[OptionsRunItemRecord]: ...

    def save_activity_ranks(
        self,
        run_id: int,
        ranks: Mapping[str, int | None],
    ) -> None: ...

    def publish(self, run_id: int, summary: OptionsRunSummary) -> OptionsRunRecord: ...

    def mark_failed_quality(
        self,
        run_id: int,
        *,
        reason_codes: Sequence[str],
    ) -> OptionsRunRecord: ...

    def cancel(self, run_id: int) -> OptionsRunRecord: ...


class PublishedOptionsReader(Protocol):
    def get_published_run(
        self,
        market: str,
        calculation_version: str,
    ) -> OptionsRunRecord | None: ...

    def get_published_symbol_detail(
        self,
        symbol: str,
        market: str,
        calculation_version: str,
    ) -> OptionsRunItemRecord | None: ...

    def get_run_diagnostics(self, run_id: int) -> OptionsRunRecord | None: ...

    def latest_source_feature_run_id(self, market: str) -> int | None: ...

    def last_current_memberships(
        self,
        market: str,
        calculation_version: str,
    ) -> Mapping[str, LastCurrentMembership]: ...

    def analysis_history(
        self,
        symbol: str,
        *,
        market: str,
        calculation_version: str,
    ) -> Sequence[HistoricalObservation]: ...

    def symbol_history(
        self,
        symbol: str,
        *,
        market: str,
        calculation_version: str,
    ) -> Sequence[OptionsHistoryRecord]: ...


class OptionsRetention(Protocol):
    def prune(
        self,
        *,
        aggregate_before: date,
        strike_history_run_limit: int = 30,
    ) -> None: ...


__all__ = [
    "OptionsHistoryRecord",
    "OptionsRetention",
    "OptionsRunItemRecord",
    "OptionsRunRecord",
    "OptionsRunWriter",
    "PublishedOptionsReader",
]
