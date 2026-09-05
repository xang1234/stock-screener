"""Persistence gateway for typed, aggregate-only options history."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from datetime import date, timezone

from sqlalchemy.orm import Session, selectinload

from app.domain.options_analytics.models import (
    CandidateKind,
    ObservationState,
    OptionsRunStatus,
)
from app.infra.db.models.options_analytics import (
    OptionsAnalyticsRun,
    OptionsAnalyticsRunItem,
)
from app.schemas.options_history_transfer import OptionsHistoryObservation


class SqlOptionsHistoryRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def export_history_observations(
        self,
        market: str,
        calculation_version: str,
    ) -> tuple[OptionsHistoryObservation, ...]:
        runs = (
            self._session.query(OptionsAnalyticsRun)
            .options(selectinload(OptionsAnalyticsRun.items))
            .filter(
                OptionsAnalyticsRun.market == market.strip().upper(),
                OptionsAnalyticsRun.calculation_version == calculation_version,
                OptionsAnalyticsRun.status == OptionsRunStatus.PUBLISHED.value,
            )
            .order_by(
                OptionsAnalyticsRun.as_of_date.asc(),
                OptionsAnalyticsRun.attempt_number.desc(),
                OptionsAnalyticsRun.id.desc(),
            )
            .all()
        )
        observations: list[OptionsHistoryObservation] = []
        seen_sessions: set[date] = set()
        for run in runs:
            if run.as_of_date in seen_sessions:
                continue
            seen_sessions.add(run.as_of_date)
            external_key = run.external_source_feature_run_key or (
                f"{run.market}:{run.as_of_date.isoformat()}:{run.input_signature}"
            )
            for item in sorted(run.items, key=lambda row: row.security_symbol):
                observations.append(self._to_observation(run, item, external_key))
        return tuple(observations)

    def import_history_transfer(
        self,
        observations: Sequence[OptionsHistoryObservation],
        *,
        market: str,
        calculation_version: str,
        schema_version: str,
    ) -> dict[str, int | str]:
        del schema_version
        grouped: dict[str, list[OptionsHistoryObservation]] = {}
        for row in observations:
            grouped.setdefault(row.external_source_feature_run_key, []).append(row)

        imported_runs = 0
        imported_observations = 0
        for external_key, rows in sorted(grouped.items()):
            if self._history_run_exists(market, calculation_version, external_key):
                continue
            run = self._import_run(
                rows,
                external_key=external_key,
                market=market,
                calculation_version=calculation_version,
            )
            self._session.add(run)
            self._session.flush()
            self._session.add_all([self._import_item(run.id, row) for row in rows])
            imported_runs += 1
            imported_observations += len(rows)
        self._session.commit()
        return {
            "status": "imported",
            "imported_runs": imported_runs,
            "imported_observations": imported_observations,
        }

    def _history_run_exists(
        self,
        market: str,
        calculation_version: str,
        external_key: str,
    ) -> bool:
        return (
            self._session.query(OptionsAnalyticsRun.id)
            .filter(
                OptionsAnalyticsRun.market == market.strip().upper(),
                OptionsAnalyticsRun.calculation_version == calculation_version,
                OptionsAnalyticsRun.origin == "history_transfer",
                OptionsAnalyticsRun.external_source_feature_run_key == external_key,
            )
            .first()
            is not None
        )

    @staticmethod
    def _to_observation(
        run: OptionsAnalyticsRun,
        item: OptionsAnalyticsRunItem,
        external_key: str,
    ) -> OptionsHistoryObservation:
        return OptionsHistoryObservation(
            external_source_feature_run_key=external_key,
            as_of_date=run.as_of_date,
            schema_version=run.schema_version,
            provider=run.provider,
            published_at=run.published_at,
            risk_free_rate=run.risk_free_rate,
            run_assumptions=dict(run.assumptions_json or {}),
            symbol=item.security_symbol,
            candidate_kind=item.candidate_kind,
            candidate_rank=item.candidate_rank,
            leader_rank=item.leader_rank,
            spot_price=item.spot_price,
            expiration=item.expiration,
            observation_state=item.observation_state,
            core_valid=item.core_valid,
            observation_at=item.observation_at,
            max_pain=item.max_pain,
            net_gex=item.net_gex,
            gamma_flip=item.gamma_flip,
            call_wall=item.call_wall,
            put_wall=item.put_wall,
            atm_iv=item.atm_iv,
            skew_25_delta=item.skew_25_delta,
            realized_volatility=item.realized_volatility,
            vrp=item.vrp,
            activity_intensity=item.activity_intensity,
            iv_percentile=item.iv_percentile,
            iv_rank=item.iv_rank,
            max_pain_change_5=item.max_pain_change_5,
            net_gex_change_5=item.net_gex_change_5,
            gamma_flip_change_5=item.gamma_flip_change_5,
            atm_iv_change_5=item.atm_iv_change_5,
            skew_25_delta_change_5=item.skew_25_delta_change_5,
            realized_volatility_change_5=item.realized_volatility_change_5,
            vrp_change_5=item.vrp_change_5,
            activity_intensity_change_5=item.activity_intensity_change_5,
            activity_rank=item.activity_rank,
            call_open_interest=item.call_open_interest,
            put_open_interest=item.put_open_interest,
            call_volume=item.call_volume,
            put_volume=item.put_volume,
            call_put_volume_ratio=item.call_put_volume_ratio,
            volume_oi_ratio=item.volume_oi_ratio,
            near_spot_volume_concentration=item.near_spot_volume_concentration,
            near_spot_open_interest_concentration=(
                item.near_spot_open_interest_concentration
            ),
            short_history_observation_count=item.short_history_observation_count,
            iv_history_observation_count=item.iv_history_observation_count,
            lifetime_observation_count=item.lifetime_observation_count,
            retry_count=item.retry_count,
            evidence=dict(item.evidence_json or {}),
            assumptions=dict(item.assumptions_json or {}),
            warnings=list(item.warnings_json or []),
            reason_codes=list(item.reasons_json or []),
        )

    @staticmethod
    def _import_run(
        rows: Sequence[OptionsHistoryObservation],
        *,
        external_key: str,
        market: str,
        calculation_version: str,
    ) -> OptionsAnalyticsRun:
        first = rows[0]
        current = [row for row in rows if row.candidate_kind is CandidateKind.CURRENT]
        completed_states = {
            ObservationState.AVAILABLE,
            ObservationState.UNAVAILABLE,
            ObservationState.INSUFFICIENT_QUALITY,
        }
        core_valid_count = sum(row.core_valid for row in current)
        digest = hashlib.sha256(
            f"history-transfer:{market}:{calculation_version}:{external_key}".encode()
        ).hexdigest()
        published_at = first.published_at
        if published_at is not None and published_at.tzinfo is None:
            published_at = published_at.replace(tzinfo=timezone.utc)
        return OptionsAnalyticsRun(
            market=market.strip().upper(),
            origin="history_transfer",
            source_feature_run_id=None,
            external_source_feature_run_key=external_key,
            calculation_version=calculation_version,
            schema_version=first.schema_version,
            provider=first.provider,
            input_signature=digest,
            attempt_number=1,
            status=OptionsRunStatus.PUBLISHED.value,
            as_of_date=first.as_of_date,
            risk_free_rate=first.risk_free_rate,
            expected_count=len(rows),
            current_count=len(current),
            continuity_count=len(rows) - len(current),
            completed_count=sum(
                row.observation_state in completed_states for row in rows
            ),
            core_valid_current_count=core_valid_count,
            failed_count=sum(
                row.observation_state
                in {
                    ObservationState.UNAVAILABLE,
                    ObservationState.INSUFFICIENT_QUALITY,
                }
                for row in rows
            ),
            retried_count=sum(row.retry_count for row in rows),
            coverage=core_valid_count / len(current) if current else 0.0,
            assumptions_json=dict(first.run_assumptions),
            warnings_json=[],
            diagnostics_json={"history_transfer": True},
            completed_at=published_at,
            published_at=published_at,
        )

    @staticmethod
    def _import_item(
        run_id: int, row: OptionsHistoryObservation
    ) -> OptionsAnalyticsRunItem:
        return OptionsAnalyticsRunItem(
            run_id=run_id,
            security_symbol=row.symbol,
            candidate_kind=row.candidate_kind.value,
            candidate_rank=row.candidate_rank,
            leader_rank=row.leader_rank,
            spot_price=row.spot_price,
            expiration=row.expiration,
            observation_state=row.observation_state.value,
            core_valid=row.core_valid,
            observation_at=row.observation_at,
            max_pain=row.max_pain,
            net_gex=row.net_gex,
            gamma_flip=row.gamma_flip,
            call_wall=row.call_wall,
            put_wall=row.put_wall,
            atm_iv=row.atm_iv,
            skew_25_delta=row.skew_25_delta,
            realized_volatility=row.realized_volatility,
            vrp=row.vrp,
            activity_intensity=row.activity_intensity,
            iv_percentile=row.iv_percentile,
            iv_rank=row.iv_rank,
            max_pain_change_5=row.max_pain_change_5,
            net_gex_change_5=row.net_gex_change_5,
            gamma_flip_change_5=row.gamma_flip_change_5,
            atm_iv_change_5=row.atm_iv_change_5,
            skew_25_delta_change_5=row.skew_25_delta_change_5,
            realized_volatility_change_5=row.realized_volatility_change_5,
            vrp_change_5=row.vrp_change_5,
            activity_intensity_change_5=row.activity_intensity_change_5,
            activity_rank=row.activity_rank,
            call_open_interest=row.call_open_interest,
            put_open_interest=row.put_open_interest,
            call_volume=row.call_volume,
            put_volume=row.put_volume,
            call_put_volume_ratio=row.call_put_volume_ratio,
            volume_oi_ratio=row.volume_oi_ratio,
            near_spot_volume_concentration=row.near_spot_volume_concentration,
            near_spot_open_interest_concentration=(
                row.near_spot_open_interest_concentration
            ),
            short_history_observation_count=row.short_history_observation_count,
            iv_history_observation_count=row.iv_history_observation_count,
            lifetime_observation_count=row.lifetime_observation_count,
            retry_count=row.retry_count,
            evidence_json=dict(row.evidence),
            assumptions_json=dict(row.assumptions),
            warnings_json=list(row.warnings),
            reasons_json=list(row.reason_codes),
        )


__all__ = ["SqlOptionsHistoryRepository"]
