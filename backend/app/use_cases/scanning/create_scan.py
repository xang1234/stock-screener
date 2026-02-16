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

import uuid
from dataclasses import dataclass, field

from app.domain.common.errors import ValidationError
from app.domain.common.uow import UnitOfWork
from app.domain.scanning.ports import TaskDispatcher


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
            if cmd.idempotency_key:
                existing = uow.scans.get_by_idempotency_key(cmd.idempotency_key)
                if existing is not None:
                    return CreateScanResult(
                        scan_id=existing.scan_id,
                        status=existing.status,
                        total_stocks=existing.total_stocks or 0,
                        is_duplicate=True,
                    )

            # ── Resolve universe symbols ─────────────────────────────
            symbols = uow.universe.resolve_symbols(cmd.universe_def)
            if not symbols:
                raise ValidationError(
                    f"No symbols found for universe '{cmd.universe_label}'. "
                    "Try refreshing the universe."
                )

            # ── Create scan record ───────────────────────────────────
            scan_id = str(uuid.uuid4())
            scan = uow.scans.create(
                scan_id=scan_id,
                criteria=cmd.criteria or {},
                universe=cmd.universe_label,
                universe_key=cmd.universe_key,
                universe_type=cmd.universe_type,
                universe_exchange=cmd.universe_exchange,
                universe_index=cmd.universe_index,
                universe_symbols=cmd.universe_symbols,
                screener_types=cmd.screeners,
                composite_method=cmd.composite_method,
                total_stocks=len(symbols),
                passed_stocks=0,
                status="queued",
                task_id=None,
                idempotency_key=cmd.idempotency_key,
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
        )
