"""Subscription allowance and provider-attempt accounting.

Every provider call is: reserve a bounded amount → durably mark dispatched
(commit) → perform I/O with no transaction held → append the immutable
result → reconcile. Reservations are counted in requests and reported
tokens; no dollar cost is ever inferred. A ``pre_dispatch`` failure releases
its reservation; an uncertain one stays charged until reported usage or the
end of its allocation period.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.domain.company_exposure.contracts import (
    DispatchPhase,
    ReservationState,
    ResourceUnit,
    utc_now,
)
from app.infra.db.repositories.company_exposure_work_repo import (
    ROOT_PROVIDER_ATTEMPTS,
    ReservationLedger,
)
from app.models.company_exposure import (
    ResearchProviderAttempt,
    ResearchProviderResult,
    ResearchReservation,
    ResearchResourcePool,
)
from app.services.company_exposure.config import (
    SUBSCRIPTION_MODEL,
    SUBSCRIPTION_PROVIDER,
    ExposureRuntimeConfig,
)

REQUEST_POOL = f"llm:{SUBSCRIPTION_PROVIDER}"


@dataclass(frozen=True, slots=True)
class DispatchRequest:
    logical_operation_key: str
    operation: str
    capability: str
    input_hash: str
    policy_hash: str
    max_output_tokens: int
    estimated_input_tokens: int
    root_request_id: UUID | None = None
    request_id: UUID | None = None
    route: str = SUBSCRIPTION_PROVIDER
    model: str = SUBSCRIPTION_MODEL
    parameters: dict = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ReservationTicket:
    allowed: bool
    state: str
    reason: str | None = None
    id: UUID | None = None
    token_reservation_id: UUID | None = None
    attempt_id: UUID | None = None
    attempt_number: int | None = None
    allocation_id: str | None = None
    period: str | None = None
    period_end: datetime | None = None
    route: str = SUBSCRIPTION_PROVIDER
    model: str = SUBSCRIPTION_MODEL


@dataclass(frozen=True, slots=True)
class DispatchOutcome:
    success: bool
    dispatch_phase: DispatchPhase
    retryable: bool = False
    failure_code: str | None = None
    provider_request_id: str | None = None
    reported_usage: dict | None = None
    response_hash: str | None = None
    retry_after_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class ReservationUsage:
    state: str
    dispatch_phase: str | None
    reported_tokens: int | None
    actual_known: bool
    actual_dollar_cost: None = None


@dataclass(frozen=True, slots=True)
class ResourceAmounts:
    requests: int | None
    tokens: int | None


@dataclass(frozen=True, slots=True)
class PeriodCloseReport:
    period: str
    expired_reservation_ids: tuple[UUID, ...]


class ResearchResources:
    def __init__(
        self,
        session: Session,
        config: ExposureRuntimeConfig,
        *,
        clock: Callable[[], datetime] = utc_now,
    ):
        self.session = session
        self.config = config
        self.clock = clock
        self.ledger = ReservationLedger(session, clock=clock)

    # -- pools -------------------------------------------------------------

    def _pool(self, unit: ResourceUnit, period: str, period_end: datetime):
        capacity = (
            self.config.daily_request_limit
            if unit == ResourceUnit.REQUESTS
            else self.config.daily_token_limit
        )
        return self.ledger.ensure_pool(
            pool_key=REQUEST_POOL,
            unit=unit,
            period=period,
            period_end=period_end,
            capacity=capacity,
        )

    def limit(self, allocation_id: str) -> ResourceAmounts:
        del allocation_id
        return ResourceAmounts(
            requests=self.config.daily_request_limit,
            tokens=self.config.daily_token_limit,
        )

    def available(self, allocation_id: str, *, period: str) -> ResourceAmounts:
        del allocation_id
        values = {}
        for unit in (ResourceUnit.REQUESTS, ResourceUnit.REPORTED_TOKENS):
            pool = self.session.execute(
                select(ResearchResourcePool).where(
                    ResearchResourcePool.pool_key == REQUEST_POOL,
                    ResearchResourcePool.unit == unit.value,
                    ResearchResourcePool.period == period,
                )
            ).scalar_one_or_none()
            capacity = (
                self.config.daily_request_limit
                if unit == ResourceUnit.REQUESTS
                else self.config.daily_token_limit
            )
            used = 0 if pool is None else int(pool.reserved_amount)
            values[unit] = None if capacity is None else max(0, capacity - used)
        return ResourceAmounts(
            requests=values[ResourceUnit.REQUESTS],
            tokens=values[ResourceUnit.REPORTED_TOKENS],
        )

    # -- reserve / dispatch / finish --------------------------------------

    def reserve(self, dispatch: DispatchRequest) -> ReservationTicket:
        if (dispatch.route, dispatch.model) != (SUBSCRIPTION_PROVIDER, SUBSCRIPTION_MODEL):
            return ReservationTicket(False, "unavailable_capability", "route_not_approved")
        if not self.config.route_approved(dispatch.capability):
            return ReservationTicket(False, "unavailable_capability", "route_not_approved")
        if not self.config.subscription_key_present:
            return ReservationTicket(
                False, "unavailable_capability", "subscription_credentials_missing"
            )
        if self.config.daily_request_limit is None:
            return ReservationTicket(False, "paused_allowance", "allocation_not_configured")
        if dispatch.max_output_tokens <= 0 or dispatch.estimated_input_tokens < 0:
            raise ValueError("invalid_token_bounds")

        period, period_end = self.config.allocation_period(self.clock())
        request_pool = self._pool(ResourceUnit.REQUESTS, period, period_end)
        token_pool = (
            self._pool(ResourceUnit.REPORTED_TOKENS, period, period_end)
            if self.config.daily_token_limit is not None
            else None
        )
        # Canonical lock order: request pool, token pool, then root budget.
        outcome = self.ledger.reserve(
            pool_id=request_pool.id,
            amount=1,
            purpose=dispatch.operation,
            logical_operation_key=dispatch.logical_operation_key,
            root_request_id=dispatch.root_request_id,
            root_budget_key=(
                ROOT_PROVIDER_ATTEMPTS if dispatch.root_request_id is not None else None
            ),
        )
        if not outcome.allowed:
            return ReservationTicket(False, "paused_allowance", outcome.reason)
        token_reservation_id = None
        if token_pool is not None:
            token_outcome = self.ledger.reserve(
                pool_id=token_pool.id,
                amount=dispatch.max_output_tokens + dispatch.estimated_input_tokens,
                purpose=dispatch.operation,
                logical_operation_key=dispatch.logical_operation_key,
            )
            if not token_outcome.allowed:
                self.ledger.transition(
                    outcome.reservation_id,
                    ReservationState.RELEASED,
                    dispatch_phase=DispatchPhase.PRE_DISPATCH,
                    detail={"reason": token_outcome.reason},
                )
                return ReservationTicket(False, "paused_allowance", token_outcome.reason)
            token_reservation_id = token_outcome.reservation_id

        number = int(
            self.session.execute(
                select(func.max(ResearchProviderAttempt.attempt_number)).where(
                    ResearchProviderAttempt.logical_operation_key
                    == dispatch.logical_operation_key
                )
            ).scalar_one()
            or 0
        ) + 1
        attempt = ResearchProviderAttempt(
            logical_operation_key=dispatch.logical_operation_key,
            attempt_number=number,
            request_id=dispatch.request_id,
            root_request_id=dispatch.root_request_id,
            operation=dispatch.operation,
            route=dispatch.route,
            model=dispatch.model,
            parameters={
                **dispatch.parameters,
                "capability": dispatch.capability,
                "max_output_tokens": dispatch.max_output_tokens,
                "fallbacks": False,
                "retries": 0,
            },
            input_hash=dispatch.input_hash,
            policy_hash=dispatch.policy_hash,
            reservation_id=outcome.reservation_id,
        )
        self.session.add(attempt)
        self.session.flush()
        return ReservationTicket(
            True,
            ReservationState.RESERVED.value,
            id=outcome.reservation_id,
            token_reservation_id=token_reservation_id,
            attempt_id=attempt.id,
            attempt_number=number,
            allocation_id=REQUEST_POOL,
            period=period,
            period_end=period_end,
        )

    def _ids(self, ticket: ReservationTicket) -> list[UUID]:
        return [i for i in (ticket.id, ticket.token_reservation_id) if i is not None]

    def mark_dispatched(self, ticket: ReservationTicket) -> None:
        for reservation_id in self._ids(ticket):
            self.ledger.transition(reservation_id, ReservationState.DISPATCHED)

    def finish(self, ticket: ReservationTicket, outcome: DispatchOutcome) -> None:
        phase = DispatchPhase(outcome.dispatch_phase)
        usage = outcome.reported_usage or {}
        total = usage.get("total_tokens")
        if total is None and {"prompt_tokens", "completion_tokens"} <= set(usage):
            total = int(usage["prompt_tokens"]) + int(usage["completion_tokens"])
        self.session.add(
            ResearchProviderResult(
                attempt_id=ticket.attempt_id,
                outcome=(
                    "success"
                    if outcome.success
                    else "uncertain"
                    if phase == DispatchPhase.UNCERTAIN
                    else "retryable_failure"
                    if outcome.retryable
                    else "terminal_failure"
                ),
                dispatch_phase=phase.value,
                failure_code=outcome.failure_code,
                provider_request_id=outcome.provider_request_id,
                reported_usage=usage,
                usage_known=total is not None,
                response_hash=outcome.response_hash,
                retry_after_seconds=(
                    None
                    if outcome.retry_after_seconds is None
                    else int(round(outcome.retry_after_seconds))
                ),
            )
        )
        self.session.flush()
        if phase == DispatchPhase.PRE_DISPATCH:
            for reservation_id in self._ids(ticket):
                self.ledger.transition(
                    reservation_id,
                    ReservationState.RELEASED,
                    dispatch_phase=phase,
                    detail={"failure_code": outcome.failure_code},
                )
            return
        if phase == DispatchPhase.UNCERTAIN:
            for reservation_id in self._ids(ticket):
                self.ledger.transition(
                    reservation_id,
                    ReservationState.UNCERTAIN,
                    dispatch_phase=phase,
                    detail={"failure_code": outcome.failure_code},
                )
            return
        # A response was received: the request counts; tokens only if reported.
        self.ledger.transition(
            ticket.id, ReservationState.RECONCILED, dispatch_phase=phase, actual_amount=1
        )
        if ticket.token_reservation_id is not None:
            self.ledger.transition(
                ticket.token_reservation_id,
                ReservationState.RECONCILED,
                dispatch_phase=phase,
                actual_amount=None if total is None else int(total),
            )

    def read(self, reservation_id: UUID) -> ReservationUsage:
        events = self.ledger.events(reservation_id)
        latest = events[-1]
        phases = [e.dispatch_phase for e in events if e.dispatch_phase]
        reservation = self.session.get(ResearchReservation, reservation_id)
        tokens = None
        if reservation.unit == ResourceUnit.REPORTED_TOKENS.value and latest.actual_known:
            tokens = latest.actual_amount
        return ReservationUsage(
            state=latest.state,
            dispatch_phase=phases[-1] if phases else None,
            reported_tokens=tokens,
            actual_known=bool(latest.actual_known),
        )

    def close_period(self, allocation_id: str, period: str) -> PeriodCloseReport:
        expired: list[UUID] = []
        for unit in (ResourceUnit.REQUESTS, ResourceUnit.REPORTED_TOKENS):
            expired.extend(
                self.ledger.close_period(pool_key=allocation_id, unit=unit, period=period)
            )
        return PeriodCloseReport(period=period, expired_reservation_ids=tuple(expired))

    def close_ended_periods(self) -> list[PeriodCloseReport]:
        """Close every pool whose period has ended (provider-free maintenance)."""

        now = self.clock()
        pools = self.session.execute(
            select(ResearchResourcePool).where(
                ResearchResourcePool.closed_at.is_(None),
                ResearchResourcePool.period_end.is_not(None),
            )
        ).scalars()
        reports = []
        seen = set()
        for pool in list(pools):
            end = pool.period_end
            if end.tzinfo is None:
                end = end.replace(tzinfo=timezone.utc)
            if end <= now and (pool.pool_key, pool.period) not in seen:
                seen.add((pool.pool_key, pool.period))
                reports.append(self.close_period(pool.pool_key, pool.period))
        return reports
