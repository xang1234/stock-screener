"""Leased research work and the typed reservation ledger.

Work items are operational (leases, retry timing). Everything that explains
what happened -- request envelopes, events, reservations and their state
transitions -- is append-only.

Reservation locking order is canonical: resource-pool rows (sorted by id)
and then root-budget rows. These short transactions never hold a taxonomy
fence and never span network I/O.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

from sqlalchemy import and_, func, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.domain.company_exposure.contracts import (
    DispatchPhase,
    ResearchJobState,
    ResearchLimits,
    ReservationState,
    ResourceUnit,
)
from app.models.company_exposure_work import (
    ExposureResearchRequest,
    ResearchEvent,
    ResearchReservation,
    ResearchReservationEvent,
    ResearchResourcePool,
    ResearchRootBudget,
    ResearchWorkItem,
)

LEASE_DURATION = timedelta(minutes=5)
ROOT_PROVIDER_ATTEMPTS = "provider_attempts"
ROOT_SEARCH_QUERIES = "search_queries"
ROOT_DOCUMENTS = "documents"
ROOT_BROWSER_REQUESTS = "browser_requests"

_TERMINAL_RESERVATION_STATES = frozenset(
    {
        ReservationState.RELEASED,
        ReservationState.RECONCILED,
        ReservationState.EXPIRED_UNCERTAIN,
    }
)
_ALLOWED_TRANSITIONS = {
    ReservationState.RESERVED: {ReservationState.DISPATCHED, ReservationState.RELEASED},
    ReservationState.DISPATCHED: {
        ReservationState.RECONCILED,
        ReservationState.UNCERTAIN,
        ReservationState.RELEASED,
    },
    ReservationState.UNCERTAIN: {
        ReservationState.RECONCILED,
        ReservationState.EXPIRED_UNCERTAIN,
    },
}


class WorkLeaseError(RuntimeError):
    """The caller does not hold the active lease for this work item."""


class ReservationTransitionError(RuntimeError):
    """A reservation state change is not permitted."""


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _aware(value: datetime | None) -> datetime | None:
    # SQLite returns naive values for timezone-aware columns.
    if value is not None and value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def default_root_budgets(limits: ResearchLimits) -> dict[str, int]:
    return {
        ROOT_PROVIDER_ATTEMPTS: limits.provider_attempts_per_root_job,
        ROOT_SEARCH_QUERIES: limits.search_queries_per_root_job,
        ROOT_DOCUMENTS: limits.documents_per_issuer,
        ROOT_BROWSER_REQUESTS: limits.browser_navigations_per_issuer * 128,
    }


class CompanyExposureWorkRepository:
    def __init__(self, session: Session, *, clock: Callable[[], datetime] = _utcnow):
        self.session = session
        self.clock = clock

    # -- request envelopes -------------------------------------------------

    def create_request(
        self,
        *,
        kind: str,
        requester_principal: str,
        idempotency_namespace: str,
        idempotency_key: str,
        economic_theme_id: UUID,
        security_id: int | None = None,
        issuer_id: UUID | None = None,
        market: str | None = None,
        supplied_links: list | None = None,
        limits: ResearchLimits | None = None,
        policy_revision_ids: list | None = None,
        trigger_origin: str = "requested",
        parent: ExposureResearchRequest | None = None,
    ) -> tuple[ExposureResearchRequest, bool]:
        """Return ``(request, created)``; the same idempotency key reuses it."""

        key = (
            ExposureResearchRequest.idempotency_namespace == idempotency_namespace,
            ExposureResearchRequest.idempotency_key == idempotency_key,
        )
        existing = self.session.execute(
            select(ExposureResearchRequest).where(*key)
        ).scalar_one_or_none()
        if existing is not None:
            return existing, False
        limits = limits or ResearchLimits()
        root_id = None if parent is None else parent.effective_root_id
        try:
            with self.session.begin_nested():
                request = ExposureResearchRequest(
                    id=uuid4(),
                    root_request_id=root_id,
                    parent_request_id=None if parent is None else parent.id,
                    kind=kind,
                    requester_principal=requester_principal,
                    idempotency_namespace=idempotency_namespace,
                    idempotency_key=idempotency_key,
                    security_id=security_id,
                    issuer_id=issuer_id,
                    economic_theme_id=economic_theme_id,
                    market=market,
                    supplied_links=list(supplied_links or []),
                    requested_limits=default_root_budgets(limits),
                    policy_revision_ids=[str(v) for v in (policy_revision_ids or [])],
                    trigger_origin=trigger_origin,
                )
                self.session.add(request)
                self.session.flush()
                if parent is None:
                    for budget_key, limit in default_root_budgets(limits).items():
                        self.session.add(
                            ResearchRootBudget(
                                root_request_id=request.id,
                                budget_key=budget_key,
                                limit_amount=limit,
                                used_amount=0,
                            )
                        )
                self.append_event(request.id, ResearchJobState.QUEUED, {})
                self.session.flush()
                return request, True
        except IntegrityError:
            return (
                self.session.execute(
                    select(ExposureResearchRequest).where(*key)
                ).scalar_one(),
                False,
            )

    def append_event(self, request_id: UUID, state, detail: dict) -> ResearchEvent:
        current = self.session.execute(
            select(func.max(ResearchEvent.sequence)).where(
                ResearchEvent.request_id == request_id
            )
        ).scalar_one()
        event = ResearchEvent(
            request_id=request_id,
            sequence=int(current or 0) + 1,
            state=str(getattr(state, "value", state)),
            detail=detail,
        )
        self.session.add(event)
        self.session.flush()
        return event

    def events(self, request_id: UUID) -> list[ResearchEvent]:
        return list(
            self.session.execute(
                select(ResearchEvent)
                .where(ResearchEvent.request_id == request_id)
                .order_by(ResearchEvent.sequence)
            ).scalars()
        )

    def latest_state(self, request_id: UUID) -> str | None:
        events = self.events(request_id)
        return events[-1].state if events else None

    # -- leased work -------------------------------------------------------

    def enqueue(
        self,
        *,
        request: ExposureResearchRequest,
        stage: str,
        input_hash: str,
        policy_bundle_version: str,
        priority: int = 0,
        available_at: datetime | None = None,
    ) -> ResearchWorkItem:
        key = (
            ResearchWorkItem.request_id == request.id,
            ResearchWorkItem.stage == stage,
            ResearchWorkItem.input_hash == input_hash,
            ResearchWorkItem.policy_bundle_version == policy_bundle_version,
        )
        existing = self.session.execute(
            select(ResearchWorkItem).where(*key)
        ).scalar_one_or_none()
        if existing is not None:
            return existing
        try:
            with self.session.begin_nested():
                item = ResearchWorkItem(
                    request_id=request.id,
                    root_request_id=request.effective_root_id,
                    stage=stage,
                    input_hash=input_hash,
                    policy_bundle_version=policy_bundle_version,
                    priority=priority,
                    status="pending",
                    available_at=available_at or self.clock(),
                    claim_count=0,
                )
                self.session.add(item)
                self.session.flush()
                return item
        except IntegrityError:
            return self.session.execute(select(ResearchWorkItem).where(*key)).scalar_one()

    def claim_next(self, *, worker_id: str) -> ResearchWorkItem | None:
        now = self.clock()
        item = self.session.execute(
            select(ResearchWorkItem)
            .where(
                ResearchWorkItem.available_at <= now,
                or_(
                    ResearchWorkItem.status.in_(("pending", "retryable")),
                    and_(
                        ResearchWorkItem.status == "leased",
                        ResearchWorkItem.lease_expires_at <= now,
                    ),
                ),
            )
            .order_by(
                ResearchWorkItem.priority,
                ResearchWorkItem.available_at,
                ResearchWorkItem.created_at,
            )
            .with_for_update(skip_locked=True)
            .limit(1)
        ).scalar_one_or_none()
        if item is None:
            return None
        item.status = "leased"
        item.lease_token = uuid4()
        item.lease_owner = worker_id
        item.lease_expires_at = now + LEASE_DURATION
        item.claim_count = int(item.claim_count or 0) + 1
        self.session.flush()
        return item

    def _require_lease(self, work_id: UUID, lease_token: UUID) -> ResearchWorkItem:
        item = self.session.execute(
            select(ResearchWorkItem)
            .where(ResearchWorkItem.id == work_id)
            .with_for_update()
        ).scalar_one_or_none()
        if (
            item is None
            or item.status != "leased"
            or item.lease_token != lease_token
            or _aware(item.lease_expires_at) <= self.clock()
        ):
            raise WorkLeaseError("lease_not_held")
        return item

    def heartbeat(
        self, work_id: UUID, lease_token: UUID, *, extend: timedelta = LEASE_DURATION
    ) -> ResearchWorkItem:
        item = self._require_lease(work_id, lease_token)
        item.lease_expires_at = self.clock() + extend
        self.session.flush()
        return item

    def complete_step(
        self,
        work_id: UUID,
        lease_token: UUID,
        *,
        status: str,
        retry_at: datetime | None = None,
        pause_reason: str | None = None,
    ) -> ResearchWorkItem:
        if status not in {"completed", "retryable", "failed", "paused"}:
            raise ValueError("invalid_work_completion_status")
        item = self._require_lease(work_id, lease_token)
        item.status = status
        item.pause_reason = pause_reason
        item.lease_token = None
        item.lease_owner = None
        item.lease_expires_at = None
        if status == "retryable":
            item.available_at = retry_at or self.clock()
        self.session.flush()
        return item

    def _items_for(self, request_id: UUID) -> list[ResearchWorkItem]:
        return list(
            self.session.execute(
                select(ResearchWorkItem)
                .where(
                    or_(
                        ResearchWorkItem.request_id == request_id,
                        ResearchWorkItem.root_request_id == request_id,
                    )
                )
                .with_for_update()
            ).scalars()
        )

    def pause(self, request_id: UUID, *, reason: str) -> None:
        for item in self._items_for(request_id):
            if item.status in {"pending", "retryable", "leased"}:
                item.status = "paused"
                item.pause_reason = reason
                item.lease_token = None
                item.lease_owner = None
                item.lease_expires_at = None
        self.append_event(request_id, reason, {"paused": True})

    def resume(self, request_id: UUID) -> None:
        for item in self._items_for(request_id):
            if item.status == "paused":
                item.status = "pending"
                item.pause_reason = None
                item.available_at = self.clock()
        self.append_event(request_id, ResearchJobState.QUEUED, {"resumed": True})

    def cancel(self, request_id: UUID, *, reason: str) -> None:
        """Stop further work. Only never-dispatched reservations are released;
        a dispatched request may have executed, so it stays charged as
        uncertain until a result reconciles it or its period expires."""

        for item in self._items_for(request_id):
            if item.status not in {"completed", "failed", "cancelled"}:
                item.status = "cancelled"
                item.lease_token = None
                item.lease_owner = None
                item.lease_expires_at = None
        ledger = ReservationLedger(self.session, clock=self.clock)
        reservations = self.session.execute(
            select(ResearchReservation).where(
                ResearchReservation.root_request_id == request_id
            )
        ).scalars()
        for reservation in reservations:
            state = ledger.state(reservation.id)
            if state == ReservationState.RESERVED:
                ledger.transition(
                    reservation.id,
                    ReservationState.RELEASED,
                    dispatch_phase=DispatchPhase.PRE_DISPATCH,
                    detail={"reason": reason},
                )
            elif state == ReservationState.DISPATCHED:
                ledger.transition(
                    reservation.id,
                    ReservationState.UNCERTAIN,
                    dispatch_phase=DispatchPhase.UNCERTAIN,
                    detail={"reason": reason},
                )
        self.append_event(request_id, ResearchJobState.CANCELLED, {"reason": reason})


@dataclass(frozen=True, slots=True)
class ReserveOutcome:
    allowed: bool
    reservation_id: UUID | None = None
    reason: str | None = None
    requested: int = 0
    available: int | None = None


class ReservationLedger:
    """Concurrency-safe reserve/transition over pool and root-budget rows."""

    def __init__(self, session: Session, *, clock: Callable[[], datetime] = _utcnow):
        self.session = session
        self.clock = clock

    def ensure_pool(
        self,
        *,
        pool_key: str,
        unit: ResourceUnit | str,
        period: str,
        period_end: datetime | None,
        capacity: int | None,
    ) -> ResearchResourcePool:
        unit = ResourceUnit(unit).value
        key = (
            ResearchResourcePool.pool_key == pool_key,
            ResearchResourcePool.unit == unit,
            ResearchResourcePool.period == period,
        )
        pool = self.session.execute(
            select(ResearchResourcePool).where(*key)
        ).scalar_one_or_none()
        if pool is None:
            try:
                with self.session.begin_nested():
                    pool = ResearchResourcePool(
                        pool_key=pool_key,
                        unit=unit,
                        period=period,
                        period_end=period_end,
                        capacity=capacity,
                        reserved_amount=0,
                    )
                    self.session.add(pool)
                    self.session.flush()
            except IntegrityError:
                pool = self.session.execute(
                    select(ResearchResourcePool).where(*key)
                ).scalar_one()
        if pool.capacity != capacity and pool.closed_at is None:
            # Capacity is configuration; the reserved counter is history.
            locked = self.session.execute(
                select(ResearchResourcePool)
                .where(ResearchResourcePool.id == pool.id)
                .with_for_update()
            ).scalar_one()
            locked.capacity = capacity
            self.session.flush()
            pool = locked
        return pool

    def reserve(
        self,
        *,
        pool_id: UUID,
        amount: int,
        purpose: str,
        logical_operation_key: str,
        root_request_id: UUID | None = None,
        root_budget_key: str | None = None,
        root_amount: int = 1,
        currency: str | None = None,
        policy_revision_id: UUID | None = None,
    ) -> ReserveOutcome:
        if amount < 0 or root_amount < 0:
            raise ValueError("negative_reservation")
        pool = self.session.execute(
            select(ResearchResourcePool)
            .where(ResearchResourcePool.id == pool_id)
            .with_for_update()
        ).scalar_one()
        if pool.closed_at is not None:
            return ReserveOutcome(False, reason="period_closed", requested=amount)
        if pool.capacity is None:
            return ReserveOutcome(
                False, reason="allocation_not_configured", requested=amount
            )
        available = int(pool.capacity) - int(pool.reserved_amount)
        if amount > available:
            return ReserveOutcome(
                False, reason="capacity_exhausted", requested=amount, available=available
            )
        budget = None
        if root_request_id is not None and root_budget_key is not None:
            budget = self.session.execute(
                select(ResearchRootBudget)
                .where(
                    ResearchRootBudget.root_request_id == root_request_id,
                    ResearchRootBudget.budget_key == root_budget_key,
                )
                .with_for_update()
            ).scalar_one_or_none()
            if budget is None:
                return ReserveOutcome(False, reason="root_budget_missing")
            if int(budget.used_amount) + root_amount > int(budget.limit_amount):
                return ReserveOutcome(
                    False,
                    reason="root_budget_exhausted",
                    requested=root_amount,
                    available=int(budget.limit_amount) - int(budget.used_amount),
                )
            budget.used_amount = int(budget.used_amount) + root_amount
        pool.reserved_amount = int(pool.reserved_amount) + amount
        reservation = ResearchReservation(
            id=uuid4(),
            pool_id=pool.id,
            root_request_id=root_request_id,
            root_budget_key=root_budget_key if budget is not None else None,
            unit=pool.unit,
            amount=amount,
            currency=currency,
            period=pool.period,
            period_end=pool.period_end,
            purpose=purpose,
            logical_operation_key=logical_operation_key,
            policy_revision_id=policy_revision_id,
        )
        self.session.add(reservation)
        self.session.flush()
        self._append(
            reservation.id,
            ReservationState.RESERVED,
            detail={"root_amount": root_amount if budget is not None else 0},
        )
        return ReserveOutcome(
            True,
            reservation_id=reservation.id,
            requested=amount,
            available=available - amount,
        )

    def events(self, reservation_id: UUID) -> list[ResearchReservationEvent]:
        return list(
            self.session.execute(
                select(ResearchReservationEvent)
                .where(ResearchReservationEvent.reservation_id == reservation_id)
                .order_by(ResearchReservationEvent.sequence)
            ).scalars()
        )

    def state(self, reservation_id: UUID) -> ReservationState:
        events = self.events(reservation_id)
        if not events:
            raise ReservationTransitionError("reservation_without_events")
        return ReservationState(events[-1].state)

    def _append(
        self,
        reservation_id: UUID,
        state: ReservationState,
        *,
        dispatch_phase: DispatchPhase | None = None,
        actual_amount: int | None = None,
        actual_known: bool = False,
        detail: dict | None = None,
    ) -> ResearchReservationEvent:
        current = self.session.execute(
            select(func.max(ResearchReservationEvent.sequence)).where(
                ResearchReservationEvent.reservation_id == reservation_id
            )
        ).scalar_one()
        event = ResearchReservationEvent(
            reservation_id=reservation_id,
            sequence=int(current or 0) + 1,
            state=state.value,
            dispatch_phase=None if dispatch_phase is None else dispatch_phase.value,
            actual_amount=actual_amount,
            actual_known=actual_known,
            detail=dict(detail or {}),
        )
        self.session.add(event)
        self.session.flush()
        return event

    def transition(
        self,
        reservation_id: UUID,
        new_state: ReservationState | str,
        *,
        dispatch_phase: DispatchPhase | str | None = None,
        actual_amount: int | None = None,
        detail: dict | None = None,
    ) -> ResearchReservationEvent:
        new_state = ReservationState(new_state)
        phase = None if dispatch_phase is None else DispatchPhase(dispatch_phase)
        reservation = self.session.get(ResearchReservation, reservation_id)
        if reservation is None:
            raise ReservationTransitionError("reservation_missing")
        pool = self.session.execute(
            select(ResearchResourcePool)
            .where(ResearchResourcePool.id == reservation.pool_id)
            .with_for_update()
        ).scalar_one()
        budget = None
        if reservation.root_request_id is not None and reservation.root_budget_key:
            budget = self.session.execute(
                select(ResearchRootBudget)
                .where(
                    ResearchRootBudget.root_request_id == reservation.root_request_id,
                    ResearchRootBudget.budget_key == reservation.root_budget_key,
                )
                .with_for_update()
            ).scalar_one()
        current = self.state(reservation_id)
        if new_state not in _ALLOWED_TRANSITIONS.get(current, set()):
            raise ReservationTransitionError(
                f"illegal_transition:{current.value}->{new_state.value}"
            )
        if (
            current == ReservationState.DISPATCHED
            and new_state == ReservationState.RELEASED
            and phase != DispatchPhase.PRE_DISPATCH
        ):
            raise ReservationTransitionError("dispatched_release_requires_pre_dispatch")

        if new_state == ReservationState.RELEASED:
            pool.reserved_amount = max(0, int(pool.reserved_amount) - int(reservation.amount))
            if budget is not None:
                root_amount = int(self.events(reservation_id)[0].detail.get("root_amount", 0))
                budget.used_amount = max(0, int(budget.used_amount) - root_amount)
        elif new_state == ReservationState.RECONCILED and actual_amount is not None:
            if actual_amount < 0:
                raise ValueError("negative_actual_amount")
            delta = int(actual_amount) - int(reservation.amount)
            pool.reserved_amount = max(0, int(pool.reserved_amount) + delta)
        return self._append(
            reservation_id,
            new_state,
            dispatch_phase=phase,
            actual_amount=actual_amount,
            actual_known=actual_amount is not None,
            detail=detail,
        )

    def adjust_pool(self, pool_id: UUID, delta: int) -> ResearchResourcePool:
        """Operational counter correction explained by an append-only record
        elsewhere (e.g. an evidence tombstone for reclaimed blob bytes)."""

        pool = self.session.execute(
            select(ResearchResourcePool)
            .where(ResearchResourcePool.id == pool_id)
            .with_for_update()
        ).scalar_one()
        pool.reserved_amount = max(0, int(pool.reserved_amount) + int(delta))
        self.session.flush()
        return pool

    def consume_root(
        self, root_request_id: UUID, budget_key: str, amount: int = 1
    ) -> ReserveOutcome:
        """Charge a cumulative root budget that has no shared pool."""

        budget = self.session.execute(
            select(ResearchRootBudget)
            .where(
                ResearchRootBudget.root_request_id == root_request_id,
                ResearchRootBudget.budget_key == budget_key,
            )
            .with_for_update()
        ).scalar_one_or_none()
        if budget is None:
            return ReserveOutcome(False, reason="root_budget_missing")
        if int(budget.used_amount) + amount > int(budget.limit_amount):
            return ReserveOutcome(
                False,
                reason="root_budget_exhausted",
                requested=amount,
                available=int(budget.limit_amount) - int(budget.used_amount),
            )
        budget.used_amount = int(budget.used_amount) + amount
        self.session.flush()
        return ReserveOutcome(True, requested=amount)

    def close_period(self, *, pool_key: str, unit: ResourceUnit | str, period: str) -> list[UUID]:
        """Expire every still-uncertain reservation of a closed period.

        Uncertain reservations stay charged; they never carry forward.
        """

        unit = ResourceUnit(unit).value
        pool = self.session.execute(
            select(ResearchResourcePool)
            .where(
                ResearchResourcePool.pool_key == pool_key,
                ResearchResourcePool.unit == unit,
                ResearchResourcePool.period == period,
            )
            .with_for_update()
        ).scalar_one_or_none()
        if pool is None:
            return []
        expired = []
        reservations = self.session.execute(
            select(ResearchReservation.id).where(ResearchReservation.pool_id == pool.id)
        ).scalars()
        for reservation_id in list(reservations):
            state = self.state(reservation_id)
            if state == ReservationState.DISPATCHED:
                self._append(
                    reservation_id,
                    ReservationState.UNCERTAIN,
                    dispatch_phase=DispatchPhase.UNCERTAIN,
                    detail={"reason": "period_closed_without_result"},
                )
                state = ReservationState.UNCERTAIN
            if state == ReservationState.UNCERTAIN:
                self._append(
                    reservation_id,
                    ReservationState.EXPIRED_UNCERTAIN,
                    detail={"reason": "allocation_period_closed"},
                )
                expired.append(reservation_id)
        if pool.closed_at is None:
            pool.closed_at = self.clock()
        self.session.flush()
        return expired
