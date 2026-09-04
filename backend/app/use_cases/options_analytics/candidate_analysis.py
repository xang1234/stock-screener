"""Analyze one cohort member within the bounded provider request budget."""

from __future__ import annotations

import math
from collections.abc import Sequence

from app.domain.options_analytics.expiration import select_monthly_expiration
from app.domain.options_analytics.history import HistoricalObservation, history_readiness
from app.domain.options_analytics.metrics.aggregate import calculate_chain_metrics
from app.domain.options_analytics.models import ObservationState, OptionCandidate
from app.domain.options_analytics.ports import (
    OptionsProvider,
    OptionsProviderError,
    SessionCalendar,
    TransientOptionsProviderError,
)
from app.domain.options_analytics.quality import has_core_chain_coverage

from .analysis_models import (
    AnalysisContext,
    AvailableCandidateAnalysis,
    CandidateAnalysis,
    UnavailableCandidateAnalysis,
)
from .analysis_projection import (
    dividend_assumption,
    metric_evidence,
    metric_values,
    quality_evidence,
    quality_warnings,
    strike_points,
    unavailable_quality_evidence,
)


class SymbolHistoryReader:
    def symbol_history(
        self,
        symbol: str,
        *,
        market: str,
        calculation_version: str,
    ) -> Sequence[object]:
        raise NotImplementedError


class OptionsCandidateAnalyzer:
    def __init__(
        self,
        *,
        provider: OptionsProvider,
        history_reader: SymbolHistoryReader,
        calendar: SessionCalendar,
        calculation_version: str,
    ) -> None:
        self._provider = provider
        self._history_reader = history_reader
        self._calendar = calendar
        self._calculation_version = calculation_version

    def analyze(
        self,
        candidate: OptionCandidate,
        context: AnalysisContext,
    ) -> CandidateAnalysis:
        dividend_yield, dividend_source, dividend_warning = dividend_assumption(
            candidate
        )
        assumptions = {
            "dividend_yield": dividend_yield,
            "dividend_source": dividend_source,
        }
        warnings = tuple(
            warning
            for warning in (*context.run_warnings, dividend_warning)
            if warning is not None
        )
        if (
            candidate.spot_price is None
            or not math.isfinite(float(candidate.spot_price))
            or candidate.spot_price <= 0
        ):
            return UnavailableCandidateAnalysis(
                candidate=candidate,
                reason_codes=("source_spot_unavailable",),
                evidence={"quality": unavailable_quality_evidence(candidate)},
                assumptions=assumptions,
                warnings=warnings,
                retry_count=0,
            )

        for attempt in range(1, 4):
            try:
                expiration = select_monthly_expiration(
                    as_of_date=context.as_of_date,
                    listed_expirations=self._provider.list_expirations(candidate.symbol),
                    calendar=self._calendar,
                )
                if expiration is None:
                    return self._unavailable(
                        candidate,
                        ("expiration_unavailable",),
                        assumptions,
                        warnings,
                        attempt - 1,
                    )
                observation = self._provider.fetch_chain(
                    candidate.symbol,
                    expiration,
                    source_spot_price=float(candidate.spot_price),
                )
                metrics = calculate_chain_metrics(
                    observation,
                    as_of_date=context.as_of_date,
                    risk_free_rate=context.risk_free_rate,
                    dividend_yield=dividend_yield,
                    closes=candidate.price_closes,
                )
                return self._available(
                    candidate,
                    observation,
                    metrics,
                    context,
                    assumptions,
                    warnings,
                    attempt - 1,
                    dividend_yield,
                )
            except TransientOptionsProviderError:
                if attempt == 3:
                    return self._unavailable(
                        candidate,
                        ("provider_unavailable",),
                        assumptions,
                        warnings,
                        2,
                    )
            except OptionsProviderError:
                return self._unavailable(
                    candidate,
                    ("provider_unavailable",),
                    assumptions,
                    warnings,
                    attempt - 1,
                )
        raise AssertionError("unreachable")

    def _available(
        self,
        candidate,
        observation,
        metrics,
        context,
        assumptions,
        run_warnings,
        retry_count,
        dividend_yield,
    ) -> AvailableCandidateAnalysis:
        core_valid = has_core_chain_coverage(observation)
        historical = (*self._historical_observations(candidate, context),)
        historical = (
            *historical,
            HistoricalObservation(
                session=context.as_of_date,
                calculation_version=self._calculation_version,
                state=(
                    ObservationState.AVAILABLE
                    if core_valid
                    else ObservationState.INSUFFICIENT_QUALITY
                ),
            ),
        )
        readiness = history_readiness(
            historical,
            self._calendar.sessions_ending_on(context.as_of_date, 30),
            calculation_version=self._calculation_version,
        )
        reasons = list(readiness.reason_codes)
        if not core_valid:
            reasons.append("insufficient_core_quality")
        quality = quality_evidence(observation, as_of_date=context.as_of_date)
        return AvailableCandidateAnalysis(
            candidate=candidate,
            observation=observation,
            metrics=metrics,
            core_valid=core_valid,
            metric_values=metric_values(metrics, observation),
            strike_points=strike_points(
                observation,
                as_of_date=context.as_of_date,
                risk_free_rate=context.risk_free_rate,
                dividend_yield=dividend_yield,
            ),
            evidence=metric_evidence(metrics, quality=quality),
            assumptions={
                "risk_free_rate": context.risk_free_rate,
                **assumptions,
            },
            reason_codes=tuple(reasons),
            warnings=quality_warnings(
                observation,
                as_of_date=context.as_of_date,
                run_warnings=run_warnings,
                recent_sessions=self._calendar.sessions_ending_on(
                    context.as_of_date,
                    2,
                ),
            ),
            retry_count=retry_count,
            history_readiness=readiness,
        )

    def _historical_observations(
        self,
        candidate: OptionCandidate,
        context: AnalysisContext,
    ) -> tuple[HistoricalObservation, ...]:
        rows = self._history_reader.symbol_history(
            candidate.symbol,
            market=context.market,
            calculation_version=self._calculation_version,
        )
        observations = []
        for row in rows:
            if isinstance(row, HistoricalObservation):
                observations.append(row)
            else:
                observations.append(
                    HistoricalObservation(
                        session=row.run.as_of_date,
                        calculation_version=row.run.calculation_version,
                        state=ObservationState(row.observation_state),
                    )
                )
        return tuple(observations)

    @staticmethod
    def _unavailable(
        candidate: OptionCandidate,
        reasons: tuple[str, ...],
        assumptions: dict[str, object],
        warnings: tuple[str, ...],
        retry_count: int,
    ) -> UnavailableCandidateAnalysis:
        return UnavailableCandidateAnalysis(
            candidate=candidate,
            reason_codes=reasons,
            evidence={"quality": unavailable_quality_evidence(candidate)},
            assumptions=assumptions,
            warnings=warnings,
            retry_count=retry_count,
        )


__all__ = [
    "AnalysisContext",
    "AvailableCandidateAnalysis",
    "OptionsCandidateAnalyzer",
    "UnavailableCandidateAnalysis",
]
