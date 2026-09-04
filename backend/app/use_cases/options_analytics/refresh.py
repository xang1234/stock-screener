"""Bounded refresh orchestration with atomic quality-gated publication."""

from __future__ import annotations

import hashlib
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from datetime import date
from typing import Any

from app.domain.options_analytics.expiration import (
    retain_contracts_for_persistence,
    select_monthly_expiration,
)
from app.domain.options_analytics.history import (
    HistoricalObservation,
    history_readiness,
)
from app.domain.options_analytics.metrics.activity import rank_activity
from app.domain.options_analytics.metrics.aggregate import (
    ChainMetrics,
    calculate_chain_metrics,
)
from app.domain.options_analytics.metrics.gex import estimate_contract_gex
from app.domain.options_analytics.models import (
    CandidateKind,
    ChainObservation,
    MetricValue,
    ObservationState,
    OptionCandidate,
    OptionSide,
    OptionsRunStatus,
    OptionsRunSummary,
)
from app.domain.options_analytics.ports import (
    OptionsProviderError,
    TransientOptionsProviderError,
)
from app.domain.options_analytics.quality import (
    evaluate_publication,
    has_core_chain_coverage,
)
from app.domain.options_analytics.selection import (
    CandidateHistoryInput,
    build_candidate_cohort,
)


@dataclass(frozen=True)
class RefreshOptionsAnalyticsCommand:
    source_run_id: int
    market: str = "US"
    enabled: bool = True
    force: bool = False


@dataclass(frozen=True)
class _FetchResult:
    candidate: OptionCandidate
    observation: ChainObservation | None
    metrics: ChainMetrics | None
    retry_count: int
    reason_codes: tuple[str, ...] = ()

    @property
    def core_valid(self) -> bool:
        return bool(
            self.metrics
            and self.observation
            and has_core_chain_coverage(self.observation)
        )


class RefreshOptionsAnalyticsUseCase:
    def __init__(
        self,
        *,
        candidate_source: Any,
        repository: Any,
        provider: Any,
        calendar: Any,
        clock: Any,
        cancellation: Any,
        calculation_version: str,
        schema_version: str,
        max_workers: int = 2,
    ) -> None:
        self._candidate_source = candidate_source
        self._repository = repository
        self._provider = provider
        self._calendar = calendar
        self._clock = clock
        self._cancellation = cancellation
        self._calculation_version = calculation_version
        self._schema_version = schema_version
        self._max_workers = min(max(int(max_workers), 1), 2)

    def execute(self, command: RefreshOptionsAnalyticsCommand) -> dict[str, Any]:
        market = command.market.strip().upper()
        if not command.enabled:
            return {"status": "skipped", "reason_codes": ["options_analytics_disabled"]}
        if market != "US":
            return {"status": "skipped", "reason_codes": ["market_unsupported"]}
        source = self._candidate_source.read(command.source_run_id)
        current = tuple(source.current_candidates)
        current_symbols = {candidate.symbol for candidate in current}
        memberships = self._repository.last_current_memberships(
            market, self._calculation_version
        )
        recent_sessions = tuple(
            self._calendar.sessions_ending_on(source.as_of_date, 6)
        )
        continuity_inputs = self._candidate_source.read_continuity_inputs(
            tuple(memberships), source.as_of_date
        )
        continuity = []
        for symbol, membership in memberships.items():
            if symbol in current_symbols or membership.as_of_date not in recent_sessions:
                continue
            sessions_since_current = sum(
                session > membership.as_of_date for session in recent_sessions
            )
            candidate_input = continuity_inputs.get(symbol)
            if candidate_input is None:
                continue
            candidate_input = replace(
                candidate_input,
                dividend_yield=getattr(membership, "dividend_yield", None),
            )
            continuity.append(
                CandidateHistoryInput(
                    candidate=candidate_input,
                    sessions_since_current=sessions_since_current,
                    prior_best_rank=membership.prior_best_rank,
                )
            )
        cohort = tuple(
            build_candidate_cohort(
                source.top_candidate_inputs,
                source.leader_inputs,
                continuity=continuity,
            )
        )
        current = tuple(
            candidate for candidate in cohort if candidate.kind is CandidateKind.CURRENT
        )
        signature = self._input_signature(
            source.source_feature_run_id,
            cohort,
            calculation_version=self._calculation_version,
            schema_version=self._schema_version,
        )
        run = self._repository.start_or_reuse(
            market=market,
            source_feature_run_id=source.source_feature_run_id,
            calculation_version=self._calculation_version,
            schema_version=self._schema_version,
            provider="yahoo",
            input_signature=signature,
            as_of_date=source.as_of_date,
            force=command.force,
        )
        if (
            run.status == OptionsRunStatus.PUBLISHED.value
            and not command.force
        ):
            return self._existing_run_result(run)
        self._repository.stage_candidates(run.id, cohort)
        self._repository.commit()
        if self._cancellation.is_cancelled():
            return {"run_id": run.id, "status": "cancelled", "coverage": 0.0}

        risk_free_warning: str | None = None
        try:
            risk_free_rate = self._provider.risk_free_rate(source.as_of_date)
            risk_free_source = "Yahoo ^IRX close on or before source date"
        except OptionsProviderError:
            risk_free_rate = None
            risk_free_source = "unavailable"
            risk_free_warning = "risk_free_rate_unavailable"
        self._repository.save_run_assumptions(
            run.id,
            risk_free_rate=risk_free_rate,
            assumptions={"risk_free_source": risk_free_source},
        )
        self._repository.commit()
        candidates_by_symbol = {candidate.symbol: candidate for candidate in cohort}
        incomplete = self._repository.incomplete_symbols(run.id)
        results: list[_FetchResult] = []
        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            futures = {
                executor.submit(
                    self._fetch_candidate,
                    candidates_by_symbol[symbol],
                    source.as_of_date,
                    risk_free_rate,
                ): symbol
                for symbol in incomplete
            }
            for future in as_completed(futures):
                results.append(future.result())

        for result in results:
            candidate = result.candidate
            if result.observation is None or result.metrics is None:
                dividend_yield, dividend_source, _ = self._dividend_assumption(
                    candidate
                )
                self._repository.save_unavailable(
                    run.id,
                    candidate.symbol,
                    reason_codes=result.reason_codes,
                    evidence={
                        "quality": self._unavailable_quality_evidence(candidate)
                    },
                    assumptions={
                        "dividend_yield": dividend_yield,
                        "dividend_source": dividend_source,
                    },
                    retry_count=result.retry_count,
                )
                self._repository.commit()
                continue
            dividend_yield, dividend_source, dividend_warning = (
                self._dividend_assumption(candidate)
            )
            historical = self._historical_observations(
                self._repository.symbol_history(
                    candidate.symbol,
                    market=market,
                    calculation_version=self._calculation_version,
                )
            )
            historical = (
                *historical,
                HistoricalObservation(
                    session=source.as_of_date,
                    calculation_version=self._calculation_version,
                    state=(
                        ObservationState.AVAILABLE
                        if result.core_valid
                        else ObservationState.INSUFFICIENT_QUALITY
                    ),
                ),
            )
            readiness = history_readiness(
                historical,
                self._calendar.sessions_ending_on(source.as_of_date, 30),
                calculation_version=self._calculation_version,
            )
            item_reason_codes = list(readiness.reason_codes)
            if not result.core_valid:
                item_reason_codes.append("insufficient_core_quality")
            self._repository.save_item_result(
                run.id,
                candidate.symbol,
                observation=result.observation,
                core_valid=result.core_valid,
                metric_values=self._metric_values(
                    result.metrics, result.observation
                ),
                strike_points=self._strike_points(
                    result.observation,
                    as_of_date=source.as_of_date,
                    risk_free_rate=risk_free_rate,
                    dividend_yield=dividend_yield,
                ),
                evidence=self._metric_evidence(
                    result.metrics,
                    quality=self._quality_evidence(
                        result.observation,
                        as_of_date=source.as_of_date,
                    ),
                ),
                assumptions={
                    "risk_free_rate": risk_free_rate,
                    "dividend_yield": dividend_yield,
                    "dividend_source": dividend_source,
                },
                reason_codes=tuple(item_reason_codes),
                warnings=self._quality_warnings(
                    result.observation,
                    as_of_date=source.as_of_date,
                    run_warnings=tuple(
                        warning
                        for warning in (risk_free_warning, dividend_warning)
                        if warning is not None
                    ),
                ),
                retry_count=result.retry_count,
                history_readiness=readiness,
            )
            self._repository.commit()
        persisted_items = self._repository.items_for_run(run.id)
        core_valid_symbols = {
            item.security_symbol
            for item in persisted_items
            if item.candidate_kind == CandidateKind.CURRENT.value
            and item.observation_state == ObservationState.AVAILABLE.value
            and item.core_valid
        }
        activity_values = {
            item.security_symbol: (
                MetricValue(available=True, value=item.activity_intensity)
                if item.activity_intensity is not None
                else MetricValue(
                    available=False, reason_codes=("activity_unavailable",)
                )
            )
            for item in persisted_items
            if item.candidate_kind == CandidateKind.CURRENT.value
        }
        self._repository.save_activity_ranks(run.id, rank_activity(activity_values))
        self._repository.commit()

        decision = evaluate_publication(current, core_valid_symbols=core_valid_symbols)
        terminal_states = {
            ObservationState.AVAILABLE.value,
            ObservationState.UNAVAILABLE.value,
            ObservationState.INSUFFICIENT_QUALITY.value,
        }
        completed = sum(
            item.observation_state in terminal_states for item in persisted_items
        )
        failed = sum(
            item.observation_state
            in {
                ObservationState.UNAVAILABLE.value,
                ObservationState.INSUFFICIENT_QUALITY.value,
            }
            for item in persisted_items
        )
        retried = sum(item.retry_count for item in persisted_items)
        summary = OptionsRunSummary(
            status=(
                OptionsRunStatus.PUBLISHED
                if decision.publish
                else OptionsRunStatus.FAILED_QUALITY
            ),
            expected_count=len(cohort),
            completed_count=completed,
            core_valid_current_count=decision.core_valid_current_count,
            failed_count=failed,
            retried_count=retried,
            coverage=decision.coverage,
            reason_codes=decision.reason_codes,
        )
        if decision.publish:
            self._repository.publish(run.id, summary)
            self._repository.commit()
            sessions = self._calendar.sessions_ending_on(source.as_of_date, 252)
            self._repository.prune(aggregate_before=sessions[0])
            self._repository.commit()
            status = OptionsRunStatus.PUBLISHED.value
        else:
            self._repository.mark_failed_quality(
                run.id, reason_codes=decision.reason_codes
            )
            self._repository.commit()
            status = OptionsRunStatus.FAILED_QUALITY.value
        return {
            "run_id": run.id,
            "source_run_id": source.source_feature_run_id,
            "status": status,
            "expected_count": len(cohort),
            "completed_count": completed,
            "core_valid_current_count": decision.core_valid_current_count,
            "failed_count": failed,
            "retried_count": retried,
            "coverage": decision.coverage,
            "reason_codes": list(decision.reason_codes),
        }

    def _fetch_candidate(
        self,
        candidate: OptionCandidate,
        as_of_date: date,
        risk_free_rate: float | None,
    ) -> _FetchResult:
        if (
            candidate.spot_price is None
            or not math.isfinite(float(candidate.spot_price))
            or candidate.spot_price <= 0
        ):
            return _FetchResult(
                candidate,
                None,
                None,
                0,
                ("source_spot_unavailable",),
            )
        for attempt in range(1, 4):
            try:
                expirations = self._provider.list_expirations(candidate.symbol)
                expiration = select_monthly_expiration(
                    as_of_date=as_of_date,
                    listed_expirations=expirations,
                    calendar=self._calendar,
                )
                if expiration is None:
                    return _FetchResult(
                        candidate,
                        None,
                        None,
                        attempt - 1,
                        ("expiration_unavailable",),
                    )
                observation = self._provider.fetch_chain(
                    candidate.symbol,
                    expiration,
                    source_spot_price=float(candidate.spot_price),
                )
                metrics = calculate_chain_metrics(
                    observation,
                    as_of_date=as_of_date,
                    risk_free_rate=risk_free_rate,
                    dividend_yield=self._dividend_assumption(candidate)[0],
                    closes=candidate.price_closes,
                )
                return _FetchResult(candidate, observation, metrics, attempt - 1)
            except TransientOptionsProviderError:
                if attempt == 3:
                    return _FetchResult(
                        candidate,
                        None,
                        None,
                        2,
                        ("provider_unavailable",),
                    )
            except OptionsProviderError:
                return _FetchResult(
                    candidate,
                    None,
                    None,
                    attempt - 1,
                    ("provider_unavailable",),
                )
        raise AssertionError("unreachable")

    def _historical_observations(self, rows: Any) -> tuple[HistoricalObservation, ...]:
        observations = []
        for row in rows:
            if isinstance(row, HistoricalObservation):
                observations.append(row)
                continue
            observations.append(
                HistoricalObservation(
                    session=row.run.as_of_date,
                    calculation_version=row.run.calculation_version,
                    state=ObservationState(row.observation_state),
                )
            )
        return tuple(observations)

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
    def _existing_run_result(run: Any) -> dict[str, Any]:
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

    @staticmethod
    def _dividend_assumption(
        candidate: OptionCandidate,
    ) -> tuple[float, str, str | None]:
        value = candidate.dividend_yield
        if value is None or not math.isfinite(float(value)) or float(value) < 0:
            return 0.0, "zero_assumption", "zero_dividend_assumption"
        return float(value), "pinned_feature_run", None

    @staticmethod
    def _metric_values(
        metrics: ChainMetrics, observation: ChainObservation
    ) -> dict[str, float | int | None]:
        def total(side: OptionSide, field: str) -> int:
            return sum(
                int(value)
                for contract in observation.contracts
                if contract.side is side
                and (value := getattr(contract, field)) is not None
                and value >= 0
            )

        return {
            "max_pain": metrics.max_pain.value,
            "net_gex": metrics.net_gex.value,
            "gamma_flip": metrics.gamma_flip.value,
            "call_wall": metrics.call_wall.value,
            "put_wall": metrics.put_wall.value,
            "atm_iv": metrics.atm_iv.value,
            "skew_25_delta": metrics.skew_25_delta.value,
            "realized_volatility": metrics.realized_volatility.value,
            "vrp": metrics.vrp.value,
            "activity_intensity": metrics.activity.activity_intensity.value,
            "call_open_interest": total(OptionSide.CALL, "open_interest"),
            "put_open_interest": total(OptionSide.PUT, "open_interest"),
            "call_volume": total(OptionSide.CALL, "volume"),
            "put_volume": total(OptionSide.PUT, "volume"),
            "volume_oi_ratio": metrics.activity.volume_oi_ratio.value,
            "near_spot_volume_concentration": (
                metrics.activity.near_spot_volume_concentration.value
            ),
        }

    @staticmethod
    def _strike_points(
        observation: ChainObservation,
        *,
        as_of_date: date,
        risk_free_rate: float | None,
        dividend_yield: float,
    ) -> list[dict[str, Any]]:
        retained = retain_contracts_for_persistence(
            observation.contracts,
            spot_price=observation.source_spot_price,
        )
        time_years = (observation.expiration - as_of_date).days / 365
        points: dict[float, dict[str, Any]] = {}
        for contract in retained:
            point = points.setdefault(contract.strike, {"strike": contract.strike})
            prefix = "call" if contract.side.value == "call" else "put"
            point[f"{prefix}_open_interest"] = contract.open_interest
            point[f"{prefix}_volume"] = contract.volume
            point[f"{prefix}_iv"] = contract.implied_volatility
            estimated_gex = (
                estimate_contract_gex(
                    contract,
                    spot=observation.source_spot_price,
                    time_years=time_years,
                    rate=risk_free_rate,
                    dividend_yield=dividend_yield,
                )
                if risk_free_rate is not None
                else MetricValue(
                    available=False,
                    reason_codes=("risk_free_rate_unavailable",),
                )
            )
            point[f"estimated_{prefix}_gex"] = (
                estimated_gex.value if estimated_gex.available else None
            )
        return [points[strike] for strike in sorted(points)]

    @staticmethod
    def _metric_evidence(
        metrics: ChainMetrics, *, quality: dict[str, Any]
    ) -> dict[str, Any]:
        values = {
            "max_pain": metrics.max_pain,
            "net_gex": metrics.net_gex,
            "gamma_flip": metrics.gamma_flip,
            "call_wall": metrics.call_wall,
            "put_wall": metrics.put_wall,
            "atm_iv": metrics.atm_iv,
            "skew_25_delta": metrics.skew_25_delta,
            "realized_volatility": metrics.realized_volatility,
            "vrp": metrics.vrp,
            "activity_intensity": metrics.activity.activity_intensity,
            "volume_oi_ratio": metrics.activity.volume_oi_ratio,
            "near_spot_volume_concentration": (
                metrics.activity.near_spot_volume_concentration
            ),
        }
        evidence = {
            name: {
                "available": metric.available,
                "label": metric.label,
                "reason_codes": list(metric.reason_codes),
                "evidence": dict(metric.evidence),
            }
            for name, metric in values.items()
        }
        evidence["quality"] = quality
        return evidence

    @staticmethod
    def _unavailable_quality_evidence(candidate: OptionCandidate) -> dict[str, Any]:
        source_spot = candidate.spot_price
        if (
            source_spot is None
            or not math.isfinite(float(source_spot))
            or source_spot <= 0
        ):
            source_spot = None
        return {
            "source_spot_price": source_spot,
            "provider_spot_price": None,
            "spot_disagreement_ratio": None,
            "latest_contract_trade_at": None,
            "days_to_expiration": None,
            "normalized_call_count": 0,
            "normalized_put_count": 0,
            "distinct_strike_count": 0,
            "open_interest_coverage": 0.0,
            "iv_coverage": 0.0,
            "volume_coverage": 0.0,
            "two_sided_quote_coverage": 0.0,
        }

    @staticmethod
    def _quality_evidence(
        observation: ChainObservation, *, as_of_date: date
    ) -> dict[str, Any]:
        contracts = tuple(observation.contracts)
        retained = retain_contracts_for_persistence(
            contracts,
            spot_price=observation.source_spot_price,
        )
        trade_times = [
            contract.last_trade_at
            for contract in retained
            if contract.last_trade_at is not None
        ]
        latest_trade = max(trade_times) if trade_times else None
        total = len(contracts)

        def coverage(predicate: Any) -> float:
            if total == 0:
                return 0.0
            return sum(bool(predicate(contract)) for contract in contracts) / total

        evidence: dict[str, Any] = {
            "source_spot_price": observation.source_spot_price,
            "provider_spot_price": observation.provider_spot_price,
            "latest_contract_trade_at": (
                latest_trade.isoformat() if latest_trade is not None else None
            ),
            "days_to_expiration": (observation.expiration - as_of_date).days,
            "normalized_call_count": sum(
                contract.side is OptionSide.CALL for contract in contracts
            ),
            "normalized_put_count": sum(
                contract.side is OptionSide.PUT for contract in contracts
            ),
            "distinct_strike_count": len(
                {
                    contract.strike
                    for contract in contracts
                    if math.isfinite(float(contract.strike)) and contract.strike > 0
                }
            ),
            "open_interest_coverage": coverage(
                lambda contract: contract.open_interest is not None
                and contract.open_interest >= 0
            ),
            "iv_coverage": coverage(
                lambda contract: contract.implied_volatility is not None
                and math.isfinite(float(contract.implied_volatility))
                and contract.implied_volatility > 0
            ),
            "volume_coverage": coverage(
                lambda contract: contract.volume is not None and contract.volume >= 0
            ),
            "two_sided_quote_coverage": coverage(
                lambda contract: contract.bid is not None
                and math.isfinite(float(contract.bid))
                and contract.bid >= 0
                and contract.ask is not None
                and math.isfinite(float(contract.ask))
                and contract.ask >= 0
            ),
        }
        if (
            observation.provider_spot_price is not None
            and math.isfinite(float(observation.provider_spot_price))
            and observation.source_spot_price > 0
        ):
            evidence["spot_disagreement_ratio"] = abs(
                float(observation.provider_spot_price)
                - observation.source_spot_price
            ) / observation.source_spot_price
        return evidence

    def _quality_warnings(
        self,
        observation: ChainObservation,
        *,
        as_of_date: date,
        run_warnings: tuple[str, ...],
    ) -> tuple[str, ...]:
        evidence = self._quality_evidence(observation, as_of_date=as_of_date)
        warnings = list(run_warnings)
        disagreement = evidence.get("spot_disagreement_ratio")
        if disagreement is not None and disagreement > 0.02:
            warnings.append("provider_spot_disagreement")
        latest_trade = evidence.get("latest_contract_trade_at")
        recent_sessions = set(self._calendar.sessions_ending_on(as_of_date, 2))
        if latest_trade is None or date.fromisoformat(latest_trade[:10]) not in recent_sessions:
            warnings.append("stale_contract_trades")
        return tuple(warnings)
