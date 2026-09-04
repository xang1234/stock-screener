"""Transactional command repository for options analytics runs."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import date, datetime, timezone

from sqlalchemy.orm import Session

from app.domain.options_analytics.models import (
    CandidateKind,
    ObservationState,
    OptionCandidate,
    OptionsRunStatus,
    OptionsRunSummary,
)
from app.infra.db.models.options_analytics import (
    OptionsAnalyticsPointer,
    OptionsAnalyticsRun,
    OptionsAnalyticsRunItem,
    OptionsAnalyticsStrikePoint,
)
from app.use_cases.options_analytics.analysis_models import (
    AvailableCandidateAnalysis,
    CandidateAnalysis,
    OptionsStrikePoint,
    UnavailableCandidateAnalysis,
)

_OBSERVATION_VALUE_FIELDS = (
    "expiration",
    "observation_at",
    "max_pain",
    "net_gex",
    "gamma_flip",
    "call_wall",
    "put_wall",
    "atm_iv",
    "skew_25_delta",
    "realized_volatility",
    "vrp",
    "activity_intensity",
    "iv_percentile",
    "iv_rank",
    "max_pain_change_5",
    "net_gex_change_5",
    "gamma_flip_change_5",
    "atm_iv_change_5",
    "skew_25_delta_change_5",
    "realized_volatility_change_5",
    "vrp_change_5",
    "activity_intensity_change_5",
    "activity_rank",
    "call_open_interest",
    "put_open_interest",
    "call_volume",
    "put_volume",
    "call_put_volume_ratio",
    "volume_oi_ratio",
    "near_spot_volume_concentration",
)


class SqlOptionsRunWriter:
    def __init__(self, session: Session) -> None:
        self._session = session

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
    ) -> OptionsAnalyticsRun:
        existing = (
            self._session.query(OptionsAnalyticsRun)
            .filter(OptionsAnalyticsRun.input_signature == input_signature)
            .order_by(OptionsAnalyticsRun.attempt_number.desc())
            .first()
        )
        if existing is not None and not force:
            return existing
        run = OptionsAnalyticsRun(
            market=market.strip().upper(),
            origin="local",
            source_feature_run_id=source_feature_run_id,
            calculation_version=calculation_version,
            schema_version=schema_version,
            provider=provider,
            input_signature=input_signature,
            attempt_number=1 if existing is None else existing.attempt_number + 1,
            status=OptionsRunStatus.STAGED.value,
            as_of_date=as_of_date,
            expected_count=0,
            current_count=0,
            continuity_count=0,
            completed_count=0,
            core_valid_current_count=0,
            failed_count=0,
            retried_count=0,
            coverage=0.0,
        )
        self._session.add(run)
        self._session.flush()
        return run

    def stage_candidates(
        self,
        run_id: int,
        candidates: Iterable[OptionCandidate],
    ) -> tuple[OptionsAnalyticsRunItem, ...]:
        run = self._get_run(run_id)
        existing = {
            item.security_symbol: item
            for item in self._session.query(OptionsAnalyticsRunItem)
            .filter(OptionsAnalyticsRunItem.run_id == run_id)
            .all()
        }
        for candidate in candidates:
            if candidate.symbol in existing:
                continue
            item = OptionsAnalyticsRunItem(
                run_id=run_id,
                security_symbol=candidate.symbol,
                candidate_kind=candidate.kind.value,
                candidate_rank=candidate.candidate_rank,
                leader_rank=candidate.leader_rank,
                spot_price=candidate.spot_price,
                observation_state="pending",
                core_valid=False,
                short_history_observation_count=0,
                iv_history_observation_count=0,
                lifetime_observation_count=0,
                retry_count=0,
            )
            self._session.add(item)
            existing[candidate.symbol] = item
        run.expected_count = len(existing)
        run.current_count = sum(
            item.candidate_kind == CandidateKind.CURRENT.value
            for item in existing.values()
        )
        run.continuity_count = run.expected_count - run.current_count
        self._session.commit()
        return tuple(existing[symbol] for symbol in sorted(existing))

    def save_run_assumptions(
        self,
        run_id: int,
        *,
        risk_free_rate: float | None,
        assumptions: Mapping[str, object],
    ) -> None:
        run = self._get_run(run_id)
        run.risk_free_rate = risk_free_rate
        run.assumptions_json = dict(assumptions)
        self._session.commit()

    def incomplete_symbols(self, run_id: int) -> tuple[str, ...]:
        run = self._get_run(run_id)
        completed_states = (
            (
                ObservationState.AVAILABLE.value,
                ObservationState.UNAVAILABLE.value,
                ObservationState.INSUFFICIENT_QUALITY.value,
            )
            if run.status == OptionsRunStatus.PUBLISHED.value
            else (ObservationState.AVAILABLE.value,)
        )
        rows = (
            self._session.query(OptionsAnalyticsRunItem.security_symbol)
            .filter(
                OptionsAnalyticsRunItem.run_id == run_id,
                ~OptionsAnalyticsRunItem.observation_state.in_(completed_states),
            )
            .order_by(OptionsAnalyticsRunItem.security_symbol)
            .all()
        )
        return tuple(row[0] for row in rows)

    def save_analysis(self, run_id: int, analysis: CandidateAnalysis) -> None:
        item = self._get_item(run_id, analysis.candidate.symbol)
        if isinstance(analysis, UnavailableCandidateAnalysis):
            self._apply_unavailable(item, analysis)
        else:
            self._apply_available(item, analysis)
        self._session.commit()

    def items_for_run(self, run_id: int) -> tuple[OptionsAnalyticsRunItem, ...]:
        return tuple(
            self._session.query(OptionsAnalyticsRunItem)
            .filter(OptionsAnalyticsRunItem.run_id == run_id)
            .order_by(OptionsAnalyticsRunItem.security_symbol)
            .all()
        )

    def save_activity_ranks(
        self,
        run_id: int,
        ranks: Mapping[str, int | None],
    ) -> None:
        items = {
            item.security_symbol: item
            for item in self._session.query(OptionsAnalyticsRunItem)
            .filter(OptionsAnalyticsRunItem.run_id == run_id)
            .all()
        }
        for symbol, rank in ranks.items():
            item = items.get(symbol.strip().upper())
            if item is not None:
                item.activity_rank = rank
        self._session.commit()

    def publish(self, run_id: int, summary: OptionsRunSummary) -> OptionsAnalyticsRun:
        run = self._get_run(run_id)
        self._apply_summary(run, summary)
        run.status = OptionsRunStatus.PUBLISHED.value
        run.completed_at = datetime.now(timezone.utc)
        run.published_at = run.completed_at
        key = (run.market, run.calculation_version)
        pointer = self._session.get(OptionsAnalyticsPointer, key)
        if pointer is None:
            self._session.add(
                OptionsAnalyticsPointer(
                    market=run.market,
                    calculation_version=run.calculation_version,
                    run_id=run.id,
                )
            )
        else:
            pointer.run_id = run.id
        self._session.commit()
        return run

    def mark_failed_quality(
        self,
        run_id: int,
        *,
        reason_codes: Sequence[str],
    ) -> OptionsAnalyticsRun:
        run = self._get_run(run_id)
        run.status = OptionsRunStatus.FAILED_QUALITY.value
        run.completed_at = datetime.now(timezone.utc)
        run.warnings_json = list(reason_codes)
        self._session.commit()
        return run

    def cancel(self, run_id: int) -> OptionsAnalyticsRun:
        run = self._get_run(run_id)
        run.status = OptionsRunStatus.CANCELLED.value
        run.completed_at = datetime.now(timezone.utc)
        self._session.commit()
        return run

    def _apply_available(
        self,
        item: OptionsAnalyticsRunItem,
        analysis: AvailableCandidateAnalysis,
    ) -> None:
        observation = analysis.observation
        item.spot_price = observation.source_spot_price
        item.expiration = observation.expiration
        item.observation_at = observation.fetched_at
        item.observation_state = (
            ObservationState.AVAILABLE.value
            if analysis.core_valid
            else ObservationState.INSUFFICIENT_QUALITY.value
        )
        item.core_valid = analysis.core_valid
        item.retry_count = analysis.retry_count
        item.evidence_json = dict(analysis.evidence)
        item.assumptions_json = dict(analysis.assumptions)
        item.warnings_json = list(analysis.warnings)
        item.reasons_json = list(analysis.reason_codes)
        readiness = analysis.history_readiness
        item.short_history_observation_count = readiness.short_observation_count
        item.iv_history_observation_count = readiness.iv_observation_count
        item.lifetime_observation_count = readiness.lifetime_observation_count
        values = analysis.metric_values
        item.max_pain = values.max_pain
        item.net_gex = values.net_gex
        item.gamma_flip = values.gamma_flip
        item.call_wall = values.call_wall
        item.put_wall = values.put_wall
        item.atm_iv = values.atm_iv
        item.skew_25_delta = values.skew_25_delta
        item.realized_volatility = values.realized_volatility
        item.vrp = values.vrp
        item.activity_intensity = values.activity_intensity
        historical = analysis.historical_metrics
        item.iv_percentile = historical.iv_percentile.value
        item.iv_rank = historical.iv_rank.value
        item.max_pain_change_5 = historical.max_pain_change_5.value
        item.net_gex_change_5 = historical.net_gex_change_5.value
        item.gamma_flip_change_5 = historical.gamma_flip_change_5.value
        item.atm_iv_change_5 = historical.atm_iv_change_5.value
        item.skew_25_delta_change_5 = historical.skew_25_delta_change_5.value
        item.realized_volatility_change_5 = (
            historical.realized_volatility_change_5.value
        )
        item.vrp_change_5 = historical.vrp_change_5.value
        item.activity_intensity_change_5 = historical.activity_intensity_change_5.value
        item.call_open_interest = values.call_open_interest
        item.put_open_interest = values.put_open_interest
        item.call_volume = values.call_volume
        item.put_volume = values.put_volume
        item.call_put_volume_ratio = values.call_put_volume_ratio
        item.volume_oi_ratio = values.volume_oi_ratio
        item.near_spot_volume_concentration = values.near_spot_volume_concentration
        self._replace_strike_points(item, analysis.strike_points)

    def _apply_unavailable(
        self,
        item: OptionsAnalyticsRunItem,
        analysis: UnavailableCandidateAnalysis,
    ) -> None:
        for field in _OBSERVATION_VALUE_FIELDS:
            setattr(item, field, None)
        item.short_history_observation_count = 0
        item.iv_history_observation_count = 0
        item.lifetime_observation_count = 0
        self._replace_strike_points(item, ())
        item.observation_state = ObservationState.UNAVAILABLE.value
        item.core_valid = False
        item.reasons_json = list(analysis.reason_codes)
        item.evidence_json = dict(analysis.evidence)
        item.assumptions_json = dict(analysis.assumptions)
        item.warnings_json = list(analysis.warnings)
        item.retry_count = analysis.retry_count

    def _replace_strike_points(
        self,
        item: OptionsAnalyticsRunItem,
        values: Sequence[OptionsStrikePoint],
    ) -> None:
        item.strike_points.clear()
        self._session.flush()
        item.strike_points.extend(
            OptionsAnalyticsStrikePoint(
                strike=float(row.strike),
                call_open_interest=row.call_open_interest,
                put_open_interest=row.put_open_interest,
                call_volume=row.call_volume,
                put_volume=row.put_volume,
                call_iv=row.call_iv,
                put_iv=row.put_iv,
                estimated_call_gex=row.estimated_call_gex,
                estimated_put_gex=row.estimated_put_gex,
            )
            for row in values
        )

    def _get_run(self, run_id: int) -> OptionsAnalyticsRun:
        run = self._session.get(OptionsAnalyticsRun, run_id)
        if run is None:
            raise LookupError(f"Options Analytics Run {run_id} does not exist")
        return run

    def _get_item(self, run_id: int, symbol: str) -> OptionsAnalyticsRunItem:
        item = (
            self._session.query(OptionsAnalyticsRunItem)
            .filter(
                OptionsAnalyticsRunItem.run_id == run_id,
                OptionsAnalyticsRunItem.security_symbol == symbol.strip().upper(),
            )
            .one_or_none()
        )
        if item is None:
            raise LookupError(f"Options item {symbol!r} is not staged in run {run_id}")
        return item

    @staticmethod
    def _apply_summary(run: OptionsAnalyticsRun, summary: OptionsRunSummary) -> None:
        run.expected_count = summary.expected_count
        run.completed_count = summary.completed_count
        run.core_valid_current_count = summary.core_valid_current_count
        run.failed_count = summary.failed_count
        run.retried_count = summary.retried_count
        run.coverage = summary.coverage
        run.warnings_json = list(summary.reason_codes)


__all__ = ["SqlOptionsRunWriter"]
