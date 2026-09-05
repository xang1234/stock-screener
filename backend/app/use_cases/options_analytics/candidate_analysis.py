"""Analyze one cohort member within the bounded provider request budget."""

from __future__ import annotations

import math
from collections.abc import Callable

from app.domain.options_analytics.expiration import select_monthly_expiration
from app.domain.options_analytics.history import (
    HistoricalObservation,
    history_readiness,
)
from app.domain.options_analytics.metrics.aggregate import (
    ChainMetrics,
    calculate_chain_metrics,
)
from app.domain.options_analytics.metrics.history import calculate_historical_metrics
from app.domain.options_analytics.models import (
    ChainObservation,
    ObservationState,
    OptionCandidate,
)
from app.domain.options_analytics.ports import (
    OptionsProvider,
    OptionsProviderError,
    SessionCalendar,
    ThrottledOptionsProviderError,
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
    historical_metric_evidence,
    metric_evidence,
    metric_values,
    quality_evidence,
    quality_warnings,
    strike_points,
    unavailable_quality_evidence,
)


class OptionsCandidateAnalyzer:
    def __init__(
        self,
        *,
        provider: OptionsProvider,
        calendar: SessionCalendar,
        calculation_version: str,
        throttle_backoff: Callable[[int], None] = lambda _attempt: None,
    ) -> None:
        self._provider = provider
        self._calendar = calendar
        self._calculation_version = calculation_version
        self._throttle_backoff = throttle_backoff

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
                    listed_expirations=self._provider.list_expirations(
                        candidate.symbol
                    ),
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
            except ThrottledOptionsProviderError:
                if attempt == 3:
                    return self._unavailable(
                        candidate,
                        ("provider_unavailable",),
                        assumptions,
                        warnings,
                        2,
                    )
                self._throttle_backoff(attempt)
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
        candidate: OptionCandidate,
        observation: ChainObservation,
        metrics: ChainMetrics,
        context: AnalysisContext,
        assumptions: dict[str, object],
        run_warnings: tuple[str, ...],
        retry_count: int,
        dividend_yield: float,
    ) -> AvailableCandidateAnalysis:
        core_valid = has_core_chain_coverage(observation)
        values = metric_values(metrics, observation)
        historical = context.historical_observations
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
                max_pain=values.max_pain,
                net_gex=values.net_gex,
                gamma_flip=values.gamma_flip,
                atm_iv=values.atm_iv,
                skew_25_delta=values.skew_25_delta,
                realized_volatility=values.realized_volatility,
                vrp=values.vrp,
                activity_intensity=values.activity_intensity,
            ),
        )
        trailing_sessions = self._calendar.sessions_ending_on(context.as_of_date, 30)
        readiness = history_readiness(
            historical,
            trailing_sessions,
            calculation_version=self._calculation_version,
        )
        historical_metrics = calculate_historical_metrics(
            historical,
            trailing_sessions,
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
            metric_values=values,
            historical_metrics=historical_metrics,
            strike_points=strike_points(
                observation,
                as_of_date=context.as_of_date,
                risk_free_rate=context.risk_free_rate,
                dividend_yield=dividend_yield,
            ),
            evidence={
                **metric_evidence(metrics, quality=quality),
                **historical_metric_evidence(historical_metrics),
            },
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
