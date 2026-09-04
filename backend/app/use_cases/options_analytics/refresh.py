"""Bounded orchestration for one quality-gated options analytics run."""

from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

from app.domain.options_analytics.metrics.activity import rank_activity
from app.domain.options_analytics.models import (
    CandidateKind,
    MetricValue,
    ObservationState,
    OptionCandidate,
    OptionsRunStatus,
    OptionsRunSummary,
)
from app.domain.options_analytics.ports import (
    CancellationToken,
    OptionsCandidateSource,
    OptionsProvider,
    OptionsProviderError,
    SessionCalendar,
)
from app.domain.options_analytics.quality import evaluate_publication

from .analysis_models import (
    AnalysisContext,
    CandidateAnalysis,
)
from .candidate_analysis import OptionsCandidateAnalyzer
from .cohort import OptionsCandidateCohortBuilder, OptionsCohortSnapshot
from .ports import OptionsRetention, OptionsRunItemRecord, OptionsRunWriter, PublishedOptionsReader


@dataclass(frozen=True)
class RefreshOptionsAnalyticsCommand:
    source_run_id: int
    market: str = "US"
    enabled: bool = True
    force: bool = False


@dataclass(frozen=True)
class _RunCounts:
    completed: int
    failed: int
    retried: int
    core_valid_symbols: frozenset[str]


class RefreshOptionsAnalyticsUseCase:
    def __init__(
        self,
        *,
        candidate_source: OptionsCandidateSource,
        run_writer: OptionsRunWriter,
        published_reader: PublishedOptionsReader,
        retention: OptionsRetention,
        provider: OptionsProvider,
        calendar: SessionCalendar,
        cancellation: CancellationToken,
        calculation_version: str,
        schema_version: str,
        max_workers: int = 2,
    ) -> None:
        self._run_writer = run_writer
        self._retention = retention
        self._provider = provider
        self._calendar = calendar
        self._cancellation = cancellation
        self._calculation_version = calculation_version
        self._schema_version = schema_version
        self._max_workers = min(max(int(max_workers), 1), 2)
        self._cohort_builder = OptionsCandidateCohortBuilder(
            candidate_source=candidate_source,
            membership_reader=published_reader,
            calendar=calendar,
            calculation_version=calculation_version,
        )
        self._analyzer = OptionsCandidateAnalyzer(
            provider=provider,
            history_reader=published_reader,
            calendar=calendar,
            calculation_version=calculation_version,
        )

    def execute(self, command: RefreshOptionsAnalyticsCommand) -> dict[str, Any]:
        market = command.market.strip().upper()
        if not command.enabled:
            return {"status": "skipped", "reason_codes": ["options_analytics_disabled"]}
        if market != "US":
            return {"status": "skipped", "reason_codes": ["market_unsupported"]}

        cohort = self._cohort_builder.build(command.source_run_id, market=market)
        run = self._run_writer.start_or_reuse(
            market=market,
            source_feature_run_id=cohort.source_feature_run_id,
            calculation_version=self._calculation_version,
            schema_version=self._schema_version,
            provider="yahoo",
            input_signature=self._input_signature(
                cohort.source_feature_run_id,
                cohort.candidates,
                calculation_version=self._calculation_version,
                schema_version=self._schema_version,
            ),
            as_of_date=cohort.as_of_date,
            force=command.force,
        )
        if run.status == OptionsRunStatus.PUBLISHED.value and not command.force:
            return self._existing_run_result(run)

        self._run_writer.stage_candidates(run.id, cohort.candidates)
        if self._cancellation.is_cancelled():
            self._run_writer.cancel(run.id)
            return {"run_id": run.id, "status": "cancelled", "coverage": 0.0}

        risk_free_rate, risk_free_source, run_warnings = self._resolve_risk_free(
            cohort
        )
        self._run_writer.save_run_assumptions(
            run.id,
            risk_free_rate=risk_free_rate,
            assumptions={"risk_free_source": risk_free_source},
        )
        results = self._collect_analyses(
            run.id,
            cohort,
            market=market,
            risk_free_rate=risk_free_rate,
            run_warnings=run_warnings,
        )
        for analysis in results:
            self._persist_analysis(run.id, analysis)
        return self._finish_run(run.id, cohort)

    def _resolve_risk_free(
        self,
        cohort: OptionsCohortSnapshot,
    ) -> tuple[float | None, str, tuple[str, ...]]:
        try:
            return (
                self._provider.risk_free_rate(cohort.as_of_date),
                "Yahoo ^IRX close on or before source date",
                (),
            )
        except OptionsProviderError:
            return None, "unavailable", ("risk_free_rate_unavailable",)

    def _collect_analyses(
        self,
        run_id: int,
        cohort: OptionsCohortSnapshot,
        *,
        market: str,
        risk_free_rate: float | None,
        run_warnings: tuple[str, ...],
    ) -> tuple[CandidateAnalysis, ...]:
        by_symbol = {candidate.symbol: candidate for candidate in cohort.candidates}
        context = AnalysisContext(
            as_of_date=cohort.as_of_date,
            market=market,
            risk_free_rate=risk_free_rate,
            run_warnings=run_warnings,
        )
        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            futures = {
                executor.submit(self._analyzer.analyze, by_symbol[symbol], context)
                for symbol in self._run_writer.incomplete_symbols(run_id)
            }
            return tuple(future.result() for future in as_completed(futures))

    def _persist_analysis(self, run_id: int, analysis: CandidateAnalysis) -> None:
        self._run_writer.save_analysis(run_id, analysis)

    def _finish_run(
        self,
        run_id: int,
        cohort: OptionsCohortSnapshot,
    ) -> dict[str, Any]:
        items = self._run_writer.items_for_run(run_id)
        counts = self._counts(items)
        activity_values = {
            item.security_symbol: (
                MetricValue(available=True, value=item.activity_intensity)
                if item.activity_intensity is not None
                else MetricValue(
                    available=False,
                    reason_codes=("activity_unavailable",),
                )
            )
            for item in items
            if item.candidate_kind == CandidateKind.CURRENT.value
        }
        self._run_writer.save_activity_ranks(run_id, rank_activity(activity_values))

        decision = evaluate_publication(
            cohort.current,
            core_valid_symbols=counts.core_valid_symbols,
        )
        status = (
            OptionsRunStatus.PUBLISHED
            if decision.publish
            else OptionsRunStatus.FAILED_QUALITY
        )
        summary = OptionsRunSummary(
            status=status,
            expected_count=len(cohort.candidates),
            completed_count=counts.completed,
            core_valid_current_count=decision.core_valid_current_count,
            failed_count=counts.failed,
            retried_count=counts.retried,
            coverage=decision.coverage,
            reason_codes=decision.reason_codes,
        )
        if decision.publish:
            self._run_writer.publish(run_id, summary)
            sessions = self._calendar.sessions_ending_on(cohort.as_of_date, 252)
            self._retention.prune(aggregate_before=sessions[0])
        else:
            self._run_writer.mark_failed_quality(
                run_id,
                reason_codes=decision.reason_codes,
            )
        return {
            "run_id": run_id,
            "source_run_id": cohort.source_feature_run_id,
            "status": status.value,
            "expected_count": len(cohort.candidates),
            "completed_count": counts.completed,
            "core_valid_current_count": decision.core_valid_current_count,
            "failed_count": counts.failed,
            "retried_count": counts.retried,
            "coverage": decision.coverage,
            "reason_codes": list(decision.reason_codes),
        }

    @staticmethod
    def _counts(items: Sequence[OptionsRunItemRecord]) -> _RunCounts:
        terminal = {
            ObservationState.AVAILABLE.value,
            ObservationState.UNAVAILABLE.value,
            ObservationState.INSUFFICIENT_QUALITY.value,
        }
        failed = {
            ObservationState.UNAVAILABLE.value,
            ObservationState.INSUFFICIENT_QUALITY.value,
        }
        return _RunCounts(
            completed=sum(item.observation_state in terminal for item in items),
            failed=sum(item.observation_state in failed for item in items),
            retried=sum(item.retry_count for item in items),
            core_valid_symbols=frozenset(
                item.security_symbol
                for item in items
                if item.candidate_kind == CandidateKind.CURRENT.value
                and item.observation_state == ObservationState.AVAILABLE.value
                and item.core_valid
            ),
        )

    @staticmethod
    def _input_signature(
        source_run_id: int,
        cohort: tuple[OptionCandidate, ...],
        *,
        calculation_version: str,
        schema_version: str,
    ) -> str:
        material = f"{calculation_version}:{schema_version}:{source_run_id}:" + ",".join(
            f"{candidate.symbol}:{candidate.kind.value}" for candidate in cohort
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    @staticmethod
    def _existing_run_result(run: object) -> dict[str, Any]:
        return {
            "run_id": run.id,
            "source_run_id": run.source_feature_run_id,
            "status": run.status,
            "expected_count": run.expected_count,
            "completed_count": run.completed_count,
            "core_valid_current_count": run.core_valid_current_count,
            "failed_count": run.failed_count,
            "retried_count": run.retried_count,
            "coverage": run.coverage,
            "reason_codes": list(run.warnings_json or []),
        }
