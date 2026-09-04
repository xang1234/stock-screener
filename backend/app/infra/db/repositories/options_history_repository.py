"""Persistence gateway for portable options aggregate history."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from datetime import date, datetime, timezone
from typing import Any

from sqlalchemy.orm import Session, selectinload

from app.domain.options_analytics.models import CandidateKind, ObservationState, OptionsRunStatus
from app.infra.db.models.options_analytics import OptionsAnalyticsRun, OptionsAnalyticsRunItem

TRANSFER_ITEM_COLUMNS = (
    "spot_price",
    "observation_state",
    "core_valid",
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
    "activity_rank",
    "call_open_interest",
    "put_open_interest",
    "call_volume",
    "put_volume",
    "volume_oi_ratio",
    "near_spot_volume_concentration",
    "short_history_observation_count",
    "iv_history_observation_count",
    "lifetime_observation_count",
    "retry_count",
)


class SqlOptionsHistoryRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def export_history_observations(
        self,
        market: str,
        calculation_version: str,
    ) -> tuple[dict[str, Any], ...]:
        runs = (
            self._session.query(OptionsAnalyticsRun)
            .options(selectinload(OptionsAnalyticsRun.items))
            .filter(
                OptionsAnalyticsRun.market == market.strip().upper(),
                OptionsAnalyticsRun.calculation_version == calculation_version,
                OptionsAnalyticsRun.status == OptionsRunStatus.PUBLISHED.value,
            )
            .order_by(OptionsAnalyticsRun.as_of_date, OptionsAnalyticsRun.id)
            .all()
        )
        observations: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for run in runs:
            external_key = run.external_source_feature_run_key or (
                f"{run.market}:{run.as_of_date.isoformat()}:{run.input_signature}"
            )
            for item in sorted(run.items, key=lambda row: row.security_symbol):
                identity = (external_key, item.security_symbol)
                if identity in seen:
                    continue
                seen.add(identity)
                row = {
                    "external_source_feature_run_key": external_key,
                    "as_of_date": run.as_of_date.isoformat(),
                    "schema_version": run.schema_version,
                    "provider": run.provider,
                    "published_at": self._iso(run.published_at),
                    "risk_free_rate": run.risk_free_rate,
                    "run_assumptions": dict(run.assumptions_json or {}),
                    "symbol": item.security_symbol,
                    "candidate_kind": item.candidate_kind,
                    "candidate_rank": item.candidate_rank,
                    "leader_rank": item.leader_rank,
                    "expiration": item.expiration.isoformat() if item.expiration else None,
                    "observation_at": self._iso(item.observation_at),
                    "evidence": dict(item.evidence_json or {}),
                    "assumptions": dict(item.assumptions_json or {}),
                    "warnings": list(item.warnings_json or []),
                    "reason_codes": list(item.reasons_json or []),
                }
                row.update({name: getattr(item, name) for name in TRANSFER_ITEM_COLUMNS})
                observations.append(row)
        return tuple(observations)

    def import_history_transfer(
        self,
        observations: Sequence[Mapping[str, Any]],
        *,
        market: str,
        calculation_version: str,
        schema_version: str,
    ) -> dict[str, int | str]:
        del schema_version
        grouped: dict[str, list[Mapping[str, Any]]] = {}
        for row in observations:
            grouped.setdefault(str(row["external_source_feature_run_key"]), []).append(row)

        imported_runs = 0
        imported_observations = 0
        for external_key, rows in sorted(grouped.items()):
            existing = (
                self._session.query(OptionsAnalyticsRun.id)
                .filter(
                    OptionsAnalyticsRun.market == market.strip().upper(),
                    OptionsAnalyticsRun.calculation_version == calculation_version,
                    OptionsAnalyticsRun.origin == "history_transfer",
                    OptionsAnalyticsRun.external_source_feature_run_key == external_key,
                )
                .first()
            )
            if existing is not None:
                continue
            first = rows[0]
            current_count = sum(
                row["candidate_kind"] == CandidateKind.CURRENT.value for row in rows
            )
            completed_count = sum(
                row["observation_state"]
                in {
                    ObservationState.AVAILABLE.value,
                    ObservationState.UNAVAILABLE.value,
                    ObservationState.INSUFFICIENT_QUALITY.value,
                }
                for row in rows
            )
            core_valid_count = sum(
                row["candidate_kind"] == CandidateKind.CURRENT.value
                and bool(row["core_valid"])
                for row in rows
            )
            published_at = self._parse_datetime(first.get("published_at"))
            digest = hashlib.sha256(
                f"history-transfer:{market}:{calculation_version}:{external_key}".encode()
            ).hexdigest()
            run = OptionsAnalyticsRun(
                market=market.strip().upper(),
                origin="history_transfer",
                source_feature_run_id=None,
                external_source_feature_run_key=external_key,
                calculation_version=calculation_version,
                schema_version=str(first["schema_version"]),
                provider=str(first["provider"]),
                input_signature=digest,
                attempt_number=1,
                status=OptionsRunStatus.PUBLISHED.value,
                as_of_date=date.fromisoformat(str(first["as_of_date"])),
                risk_free_rate=first.get("risk_free_rate"),
                expected_count=len(rows),
                current_count=current_count,
                continuity_count=len(rows) - current_count,
                completed_count=completed_count,
                core_valid_current_count=core_valid_count,
                failed_count=sum(
                    row["observation_state"]
                    in {
                        ObservationState.UNAVAILABLE.value,
                        ObservationState.INSUFFICIENT_QUALITY.value,
                    }
                    for row in rows
                ),
                retried_count=sum(int(row.get("retry_count") or 0) for row in rows),
                coverage=core_valid_count / current_count if current_count else 0.0,
                assumptions_json=dict(first.get("run_assumptions") or {}),
                warnings_json=[],
                diagnostics_json={"history_transfer": True},
                completed_at=published_at,
                published_at=published_at,
            )
            self._session.add(run)
            self._session.flush()
            for values in rows:
                item_values = {name: values.get(name) for name in TRANSFER_ITEM_COLUMNS}
                self._session.add(
                    OptionsAnalyticsRunItem(
                        run_id=run.id,
                        security_symbol=str(values["symbol"]).strip().upper(),
                        candidate_kind=str(values["candidate_kind"]),
                        candidate_rank=values.get("candidate_rank"),
                        leader_rank=values.get("leader_rank"),
                        expiration=(
                            date.fromisoformat(str(values["expiration"]))
                            if values.get("expiration")
                            else None
                        ),
                        observation_at=self._parse_datetime(values.get("observation_at")),
                        evidence_json=dict(values.get("evidence") or {}),
                        assumptions_json=dict(values.get("assumptions") or {}),
                        warnings_json=list(values.get("warnings") or []),
                        reasons_json=list(values.get("reason_codes") or []),
                        **item_values,
                    )
                )
            imported_runs += 1
            imported_observations += len(rows)
        self._session.commit()
        return {
            "status": "imported",
            "imported_runs": imported_runs,
            "imported_observations": imported_observations,
        }

    @staticmethod
    def _iso(value: datetime | None) -> str | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

    @staticmethod
    def _parse_datetime(value: Any) -> datetime | None:
        if value is None:
            return None
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


__all__ = ["SqlOptionsHistoryRepository"]
