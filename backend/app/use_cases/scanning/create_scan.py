"""CreateScanUseCase — orchestrates scan creation with idempotency.

This use case owns the business rules for creating a new scan:
  1. Check idempotency key (return existing scan if duplicate)
  2. Resolve universe symbols via the UniverseRepository port
  3. Persist the Scan record via ScanRepository
  4. Dispatch the background scan task via TaskDispatcher

The use case depends ONLY on domain ports (abstract interfaces) —
never on SQLAlchemy, Celery, or any other infrastructure.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime

from app.domain.common.errors import ValidationError
from app.domain.common.uow import UnitOfWork
from app.domain.scanning.signature import (
    build_scan_signature_payload,
    hash_scan_signature,
    hash_universe_symbols,
)
from app.schemas.universe import UniverseType
from app.domain.scanning.ports import TaskDispatcher

logger = logging.getLogger(__name__)


# ── Command (input) ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class CreateScanCommand:
    """Immutable value object describing what the caller wants to create."""

    # Opaque universe definition — passed through to UniverseRepository
    universe_def: object

    # Pre-computed universe metadata (router extracts from UniverseDefinition)
    universe_label: str
    universe_key: str
    universe_type: str
    universe_market: str | None = None
    universe_exchange: str | None = None
    universe_index: str | None = None
    universe_symbols: list[str] | None = None

    # Screener configuration
    screeners: list[str] = field(default_factory=lambda: ["minervini"])
    composite_method: str = "weighted_average"
    criteria: dict | None = None

    # Idempotency
    idempotency_key: str | None = None


# ── Result (output) ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class CreateScanResult:
    """What the use case returns to the caller."""

    scan_id: str
    status: str
    total_stocks: int
    is_duplicate: bool
    feature_run_id: int | None = None


@dataclass(frozen=True)
class ActiveScanConflict:
    """Structured metadata describing the active scan blocking a new request."""

    scan_id: str
    status: str
    trigger_source: str
    total_stocks: int
    started_at: datetime | None


class ActiveScanConflictError(RuntimeError):
    """Raised when a user tries to queue a second scan while one is active."""

    def __init__(self, active_scan: ActiveScanConflict) -> None:
        super().__init__(f"Scan {active_scan.scan_id} is already {active_scan.status}")
        self.active_scan = active_scan

    def to_dict(self) -> dict[str, object]:
        return {
            "code": "scan_already_active",
            "message": "Another scan is already queued or running.",
            "active_scan": {
                "scan_id": self.active_scan.scan_id,
                "status": self.active_scan.status,
                "trigger_source": self.active_scan.trigger_source,
                "total_stocks": self.active_scan.total_stocks,
                "started_at": (
                    self.active_scan.started_at.isoformat()
                    if self.active_scan.started_at is not None
                    else None
                ),
            },
        }


# ── Use Case ─────────────────────────────────────────────────────────────


class CreateScanUseCase:
    """Create a scan record, resolve its universe, and dispatch execution."""

    def __init__(self, dispatcher: TaskDispatcher) -> None:
        self._dispatcher = dispatcher

    def execute(self, uow: UnitOfWork, cmd: CreateScanCommand) -> CreateScanResult:
        """Run the use case.

        Opens a UoW context, persists the Scan, commits (so the Celery
        worker can read it), dispatches the task, then stores the task ID.
        """
        with uow:
            # ── Idempotency check ────────────────────────────────────
            if cmd.idempotency_key is not None:
                existing = uow.scans.get_by_idempotency_key(cmd.idempotency_key)
                if existing is not None:
                    return CreateScanResult(
                        scan_id=existing.scan_id,
                        status=existing.status,
                        total_stocks=existing.total_stocks or 0,
                        is_duplicate=True,
                        feature_run_id=getattr(existing, "feature_run_id", None),
                    )

            active_scan = uow.scans.get_active_scan()
            if active_scan is not None:
                raise ActiveScanConflictError(
                    ActiveScanConflict(
                        scan_id=active_scan.scan_id,
                        status=active_scan.status,
                        trigger_source=getattr(active_scan, "trigger_source", "manual") or "manual",
                        total_stocks=active_scan.total_stocks or 0,
                        started_at=getattr(active_scan, "started_at", None),
                    )
                )

            # ── Resolve universe symbols ─────────────────────────────
            symbols = uow.universe.resolve_symbols(cmd.universe_def)
            if not symbols:
                raise ValidationError(
                    f"No symbols found for universe '{cmd.universe_label}'. "
                    "Try refreshing the universe."
                )

            feature_run_id = None
            instant_match = None
            should_attempt_instant = cmd.universe_type == UniverseType.ALL.value
            signature_payload = build_scan_signature_payload(
                universe_type=cmd.universe_type,
                screeners=cmd.screeners,
                composite_method=cmd.composite_method,
                criteria=cmd.criteria,
            )
            input_hash = hash_scan_signature(signature_payload)
            universe_hash = hash_universe_symbols(symbols)

            if should_attempt_instant:
                try:
                    instant_match = uow.feature_runs.find_latest_published_exact(
                        input_hash=input_hash,
                        universe_hash=universe_hash,
                    )
                    if instant_match is not None:
                        feature_run_id = instant_match.id
                        logger.info(
                            "Creating instant snapshot-backed scan from feature run %d",
                            instant_match.id,
                        )
                    else:
                        logger.info("No exact published feature run match — scan uses async path")
                except Exception:
                    logger.warning(
                        "Exact feature run lookup failed — proceeding with async scan",
                        exc_info=True,
                    )

            # ── Create scan record ───────────────────────────────────
            scan_id = str(uuid.uuid4())
            scan = uow.scans.create(
                scan_id=scan_id,
                criteria=cmd.criteria or {},
                universe=cmd.universe_label,
                universe_key=cmd.universe_key,
                universe_type=cmd.universe_type,
                universe_market=cmd.universe_market,
                universe_exchange=cmd.universe_exchange,
                universe_index=cmd.universe_index,
                universe_symbols=cmd.universe_symbols,
                screener_types=cmd.screeners,
                composite_method=cmd.composite_method,
                total_stocks=len(symbols),
                passed_stocks=0,
                status="queued",
                trigger_source="manual",
                task_id=None,
                idempotency_key=cmd.idempotency_key,
                feature_run_id=feature_run_id,
            )

            if instant_match is not None:
                uow.scans.update_status(
                    scan_id,
                    "completed",
                    total_stocks=len(symbols),
                    passed_stocks=instant_match.stats.passed_symbols
                    if instant_match.stats and instant_match.stats.passed_symbols is not None
                    else 0,
                )
                uow.commit()
                return CreateScanResult(
                    scan_id=scan_id,
                    status="completed",
                    total_stocks=len(symbols),
                    is_duplicate=False,
                    feature_run_id=feature_run_id,
                )

            # Commit so the scan row is visible to the Celery worker.
            uow.commit()

            # ── Dispatch background task ─────────────────────────────
            try:
                task_id = self._dispatcher.dispatch_scan(
                    scan_id, symbols, cmd.criteria or {}
                )
            except Exception:
                scan.status = "failed"
                uow.commit()
                raise

            scan.task_id = task_id
            uow.commit()

        return CreateScanResult(
            scan_id=scan_id,
            status="queued",
            total_stocks=len(symbols),
            is_duplicate=False,
            feature_run_id=feature_run_id,
        )
