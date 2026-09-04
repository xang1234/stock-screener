"""SQLAlchemy repository for Options Analytics Runs and published history."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

from sqlalchemy.orm import Session, selectinload

from app.domain.options_analytics.models import (
    CandidateKind,
    ChainObservation,
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

_METRIC_COLUMNS = {
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
}


@dataclass(frozen=True)
class LastCurrentMembership:
    symbol: str
    as_of_date: date
    prior_best_rank: int


class SqlOptionsAnalyticsRepository:
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
        attempt = 1 if existing is None else existing.attempt_number + 1
        run = OptionsAnalyticsRun(
            market=market.strip().upper(),
            origin="local",
            source_feature_run_id=source_feature_run_id,
            calculation_version=calculation_version,
            schema_version=schema_version,
            provider=provider,
            input_signature=input_signature,
            attempt_number=attempt,
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
        self, run_id: int, candidates: Iterable[OptionCandidate]
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
        self._session.flush()
        return tuple(existing[symbol] for symbol in sorted(existing))

    def save_item_result(
        self,
        run_id: int,
        symbol: str,
        *,
        observation: ChainObservation,
        metric_values: Mapping[str, float | None] | None = None,
        strike_points: Sequence[Mapping[str, Any]] = (),
        evidence: Mapping[str, Any] | None = None,
        assumptions: Mapping[str, Any] | None = None,
        warnings: Sequence[str] = (),
        reason_codes: Sequence[str] = (),
        retry_count: int = 0,
    ) -> OptionsAnalyticsRunItem:
        item = self._get_item(run_id, symbol)
        item.spot_price = observation.source_spot_price
        item.expiration = observation.expiration
        item.observation_at = observation.fetched_at
        item.observation_state = ObservationState.AVAILABLE.value
        item.retry_count = retry_count
        item.evidence_json = dict(evidence or {})
        item.assumptions_json = dict(assumptions or {})
        item.warnings_json = list(warnings)
        item.reasons_json = list(reason_codes)
        for name, value in (metric_values or {}).items():
            if name not in _METRIC_COLUMNS:
                raise ValueError(f"Unsupported options metric column: {name}")
            setattr(item, name, value)
        for values in strike_points:
            strike = float(values["strike"])
            point = (
                self._session.query(OptionsAnalyticsStrikePoint)
                .filter(
                    OptionsAnalyticsStrikePoint.item_id == item.id,
                    OptionsAnalyticsStrikePoint.strike == strike,
                )
                .first()
            )
            if point is None:
                point = OptionsAnalyticsStrikePoint(item_id=item.id, strike=strike)
                self._session.add(point)
            for name, value in values.items():
                if name != "strike" and hasattr(point, name):
                    setattr(point, name, value)
        self._session.flush()
        return item

    def save_unavailable(
        self,
        run_id: int,
        symbol: str,
        *,
        reason_codes: Sequence[str],
        retry_count: int = 0,
    ) -> OptionsAnalyticsRunItem:
        item = self._get_item(run_id, symbol)
        item.observation_state = ObservationState.UNAVAILABLE.value
        item.reasons_json = list(reason_codes)
        item.retry_count = retry_count
        self._session.flush()
        return item

    def incomplete_symbols(self, run_id: int) -> tuple[str, ...]:
        rows = (
            self._session.query(OptionsAnalyticsRunItem.security_symbol)
            .filter(
                OptionsAnalyticsRunItem.run_id == run_id,
                ~OptionsAnalyticsRunItem.observation_state.in_(
                    (ObservationState.AVAILABLE.value, ObservationState.UNAVAILABLE.value)
                ),
            )
            .order_by(OptionsAnalyticsRunItem.security_symbol)
            .all()
        )
        return tuple(row[0] for row in rows)

    def publish(
        self, run_id: int, summary: OptionsRunSummary
    ) -> OptionsAnalyticsRun:
        run = self._get_run(run_id)
        self._apply_summary(run, summary)
        run.status = OptionsRunStatus.PUBLISHED.value
        run.completed_at = datetime.now(timezone.utc)
        run.published_at = run.completed_at
        key = (run.market, run.calculation_version)
        pointer = self._session.get(OptionsAnalyticsPointer, key)
        if pointer is None:
            pointer = OptionsAnalyticsPointer(
                market=run.market,
                calculation_version=run.calculation_version,
                run_id=run.id,
            )
            self._session.add(pointer)
        else:
            pointer.run_id = run.id
        self._session.flush()
        return run

    def mark_failed_quality(
        self, run_id: int, *, reason_codes: Sequence[str]
    ) -> OptionsAnalyticsRun:
        run = self._get_run(run_id)
        run.status = OptionsRunStatus.FAILED_QUALITY.value
        run.completed_at = datetime.now(timezone.utc)
        run.warnings_json = list(reason_codes)
        self._session.flush()
        return run

    def get_published_run(
        self, market: str, calculation_version: str
    ) -> OptionsAnalyticsRun | None:
        pointer = self._session.get(
            OptionsAnalyticsPointer,
            (market.strip().upper(), calculation_version),
        )
        if pointer is None:
            return None
        return (
            self._session.query(OptionsAnalyticsRun)
            .options(
                selectinload(OptionsAnalyticsRun.items).selectinload(
                    OptionsAnalyticsRunItem.strike_points
                )
            )
            .filter(
                OptionsAnalyticsRun.id == pointer.run_id,
                OptionsAnalyticsRun.status == OptionsRunStatus.PUBLISHED.value,
            )
            .one_or_none()
        )

    def symbol_history(
        self, symbol: str, *, market: str, calculation_version: str
    ) -> tuple[OptionsAnalyticsRunItem, ...]:
        rows = (
            self._session.query(OptionsAnalyticsRunItem)
            .join(OptionsAnalyticsRun)
            .options(selectinload(OptionsAnalyticsRunItem.run))
            .filter(
                OptionsAnalyticsRunItem.security_symbol == symbol.strip().upper(),
                OptionsAnalyticsRun.market == market.strip().upper(),
                OptionsAnalyticsRun.calculation_version == calculation_version,
                OptionsAnalyticsRun.status == OptionsRunStatus.PUBLISHED.value,
                OptionsAnalyticsRunItem.observation_state
                == ObservationState.AVAILABLE.value,
            )
            .order_by(OptionsAnalyticsRun.as_of_date, OptionsAnalyticsRun.id)
            .all()
        )
        return tuple(rows)

    def last_current_memberships(
        self, market: str, calculation_version: str
    ) -> dict[str, LastCurrentMembership]:
        rows = (
            self._session.query(OptionsAnalyticsRunItem, OptionsAnalyticsRun.as_of_date)
            .join(OptionsAnalyticsRun)
            .filter(
                OptionsAnalyticsRun.market == market.strip().upper(),
                OptionsAnalyticsRun.calculation_version == calculation_version,
                OptionsAnalyticsRun.status == OptionsRunStatus.PUBLISHED.value,
                OptionsAnalyticsRunItem.candidate_kind == CandidateKind.CURRENT.value,
            )
            .order_by(OptionsAnalyticsRun.as_of_date.desc(), OptionsAnalyticsRun.id.desc())
            .all()
        )
        memberships: dict[str, LastCurrentMembership] = {}
        for item, as_of_date in rows:
            if item.security_symbol in memberships:
                continue
            ranks = [rank for rank in (item.candidate_rank, item.leader_rank) if rank]
            memberships[item.security_symbol] = LastCurrentMembership(
                symbol=item.security_symbol,
                as_of_date=as_of_date,
                prior_best_rank=min(ranks) if ranks else 10_000,
            )
        return memberships

    def prune(
        self,
        *,
        aggregate_before: date,
        strike_history_run_limit: int = 30,
    ) -> None:
        pointed_run_ids = {
            run_id
            for (run_id,) in self._session.query(OptionsAnalyticsPointer.run_id)
        }
        old_run_ids = {
            run_id
            for (run_id,) in self._session.query(OptionsAnalyticsRun.id).filter(
                OptionsAnalyticsRun.as_of_date < aggregate_before,
                ~OptionsAnalyticsRun.id.in_(pointed_run_ids or {-1}),
            )
        }
        if old_run_ids:
            self._session.query(OptionsAnalyticsRunItem).filter(
                OptionsAnalyticsRunItem.run_id.in_(old_run_ids)
            ).delete(synchronize_session=False)

        published_ids = [
            run_id
            for (run_id,) in self._session.query(OptionsAnalyticsRun.id)
            .filter(OptionsAnalyticsRun.status == OptionsRunStatus.PUBLISHED.value)
            .order_by(
                OptionsAnalyticsRun.as_of_date.desc(),
                OptionsAnalyticsRun.id.desc(),
            )
            .all()
        ]
        retained_run_ids = (
            set(published_ids[:strike_history_run_limit]) | pointed_run_ids
        )
        stale_run_ids = set(published_ids) - retained_run_ids
        if stale_run_ids:
            stale_item_ids = self._session.query(
                OptionsAnalyticsRunItem.id
            ).filter(OptionsAnalyticsRunItem.run_id.in_(stale_run_ids))
            self._session.query(OptionsAnalyticsStrikePoint).filter(
                OptionsAnalyticsStrikePoint.item_id.in_(stale_item_ids)
            ).delete(synchronize_session=False)
        self._session.flush()

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
