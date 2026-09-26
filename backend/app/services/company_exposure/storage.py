"""Private content-addressed store for retained original documents.

Layout: ``<root>/sha256/<aa>/<hash>`` for blobs and ``<root>/tmp/`` for
writes in progress. Writes are temp-file → fsync → rename, so a committed
revision only ever references a complete blob; a crash leaves at most an
unreferenced blob or a stale temp file for garbage collection.

Capacity (spec §9.7): a shared byte reservation pool (default 5 GiB) plus a
free-space floor (default 1 GiB). Bytes are reserved before download; a
duplicate hash settles at zero so identical content is never charged twice.
When full, callers get ``paused_storage`` before any I/O.

Garbage collection removes only unreferenced blobs older than 30 days and
abandoned temp files older than 24 hours, rechecking references under an
exclusive storage lock that acquisition also takes (shared) when it binds a
revision to an existing blob. Every removal leaves a tombstone. Published
history is never evicted to make room.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID, uuid4

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from app.domain.company_exposure.contracts import (
    ReservationState,
    ResourceUnit,
    bytes_hash,
    utc_now,
)
from app.infra.db.repositories.company_exposure_work_repo import ReservationLedger
from app.models.company_exposure import EvidenceTombstoneEvent, ExposureDocumentRevision

STORAGE_POOL = "storage:exposure-evidence"
STORAGE_PERIOD = "all"
UNREFERENCED_GRACE = timedelta(days=30)
ABANDONED_TEMP_GRACE = timedelta(hours=24)
_STORAGE_LOCK_KEY = 78_124_031


class StorageUnavailable(RuntimeError):
    def __init__(self, code: str, *, required: int = 0, available: int | None = None):
        super().__init__(code)
        self.code = code
        self.required = required
        self.available = available


@dataclass(frozen=True, slots=True)
class StorageTicket:
    allowed: bool
    reservation_id: UUID | None = None
    bytes_bound: int = 0
    reason: str | None = None
    available: int | None = None


@dataclass(frozen=True, slots=True)
class BlobRef:
    content_hash: str
    media_type: str
    byte_length: int
    key: str
    deduplicated: bool = False


@dataclass(frozen=True, slots=True)
class StorageGCReport:
    dry_run: bool
    unreferenced: tuple[str, ...] = ()
    abandoned_temp: tuple[str, ...] = ()
    reclaimed_bytes: int = 0
    skipped_referenced: tuple[str, ...] = ()
    tombstones: tuple[UUID, ...] = field(default_factory=tuple)


def blob_key(content_hash: str) -> str:
    return f"sha256/{content_hash[:2]}/{content_hash}"


def storage_lock(session: Session, *, exclusive: bool) -> None:
    """Transaction-scoped lock separating GC deletion from new references."""

    if session.get_bind().dialect.name != "postgresql":
        return
    function = "pg_advisory_xact_lock" if exclusive else "pg_advisory_xact_lock_shared"
    session.execute(text(f"SELECT {function}(:key)"), {"key": _STORAGE_LOCK_KEY})


class OriginalStore:
    def __init__(
        self,
        session: Session,
        root: str | Path,
        *,
        max_bytes: int,
        min_free_bytes: int,
        clock: Callable[[], datetime] = utc_now,
        disk_free: Callable[[Path], int] | None = None,
    ):
        self.session = session
        self.root = Path(root)
        self.max_bytes = max_bytes
        self.min_free_bytes = min_free_bytes
        self.clock = clock
        self.ledger = ReservationLedger(session, clock=clock)
        self._disk_free = disk_free or (lambda path: shutil.disk_usage(path).free)

    # -- capacity -------------------------------------------------------------

    def _pool(self):
        return self.ledger.ensure_pool(
            pool_key=STORAGE_POOL,
            unit=ResourceUnit.BLOB_BYTES,
            period=STORAGE_PERIOD,
            period_end=None,
            capacity=self.max_bytes,
        )

    def usage(self) -> dict:
        pool = self._pool()
        self.root.mkdir(parents=True, exist_ok=True)
        return {
            "capacity_bytes": self.max_bytes,
            "reserved_or_used_bytes": int(pool.reserved_amount),
            "filesystem_free_bytes": self._disk_free(self.root),
            "min_free_bytes": self.min_free_bytes,
        }

    def reserve(self, bytes_bound: int, *, purpose: str, operation_key: str) -> StorageTicket:
        """Reserve the worst-case size against the shared store before I/O."""

        if bytes_bound <= 0:
            raise ValueError("bytes_bound_must_be_positive")
        self.root.mkdir(parents=True, exist_ok=True)
        free = self._disk_free(self.root)
        if free - bytes_bound < self.min_free_bytes:
            return StorageTicket(
                False,
                bytes_bound=bytes_bound,
                reason="paused_storage",
                available=max(0, free - self.min_free_bytes),
            )
        outcome = self.ledger.reserve(
            pool_id=self._pool().id,
            amount=bytes_bound,
            purpose=purpose,
            logical_operation_key=operation_key,
        )
        if not outcome.allowed:
            return StorageTicket(
                False, bytes_bound=bytes_bound, reason="paused_storage",
                available=outcome.available,
            )
        return StorageTicket(True, outcome.reservation_id, bytes_bound)

    def release(self, ticket: StorageTicket, *, reason: str) -> None:
        if ticket.allowed and ticket.reservation_id is not None:
            self.ledger.transition(
                ticket.reservation_id,
                ReservationState.RELEASED,
                dispatch_phase="pre_dispatch",
                detail={"reason": reason},
            )

    # -- blobs ------------------------------------------------------------------

    def path_for(self, key: str) -> Path:
        path = (self.root / key).resolve()
        if self.root.resolve() not in path.parents:
            raise ValueError("blob_key_outside_store")
        return path

    def exists(self, content_hash: str) -> bool:
        return self.path_for(blob_key(content_hash)).is_file()

    def put(self, data: bytes, media_type: str, ticket: StorageTicket) -> BlobRef:
        if not ticket.allowed or ticket.reservation_id is None:
            raise StorageUnavailable("storage_not_reserved")
        if len(data) > ticket.bytes_bound:
            self.release(ticket, reason="blob_exceeds_reservation")
            raise StorageUnavailable("blob_exceeds_reservation", required=len(data))
        digest = bytes_hash(data)
        key = blob_key(digest)
        target = self.path_for(key)
        self.ledger.transition(ticket.reservation_id, ReservationState.DISPATCHED)
        if target.is_file():
            # Identical content is stored once and never charged twice.
            self.ledger.transition(
                ticket.reservation_id,
                ReservationState.RECONCILED,
                dispatch_phase="dispatched",
                actual_amount=0,
            )
            return BlobRef(digest, media_type, len(data), key, deduplicated=True)
        temp_dir = self.root / "tmp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = temp_dir / f"{uuid4().hex}.part"
        try:
            with open(temp, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp, target)
        except OSError:
            temp.unlink(missing_ok=True)
            self.ledger.transition(
                ticket.reservation_id,
                ReservationState.RECONCILED,
                dispatch_phase="dispatched",
                actual_amount=0,
            )
            raise StorageUnavailable("storage_write_failed", required=len(data)) from None
        self.ledger.transition(
            ticket.reservation_id,
            ReservationState.RECONCILED,
            dispatch_phase="dispatched",
            actual_amount=len(data),
        )
        return BlobRef(digest, media_type, len(data), key)

    def read(self, key: str) -> bytes:
        tombstone = self.session.execute(
            select(EvidenceTombstoneEvent).where(EvidenceTombstoneEvent.blob_key == key)
        ).scalar_one_or_none()
        if tombstone is not None:
            raise StorageUnavailable(f"evidence_removed:{tombstone.reason}")
        path = self.path_for(key)
        if not path.is_file():
            raise StorageUnavailable("evidence_blob_missing")
        return path.read_bytes()

    def open_authorized(self, blob: BlobRef | str, principal) -> bytes:
        """Authorized evidence read; never a public mirror."""

        roles = getattr(principal, "roles", None)
        if not getattr(principal, "subject", None) or roles is None:
            raise PermissionError("authorized_principal_required")
        key = blob if isinstance(blob, str) else blob.key
        return self.read(key)

    # -- garbage collection -----------------------------------------------------

    def _referenced(self, key: str) -> bool:
        return (
            self.session.execute(
                select(ExposureDocumentRevision.id)
                .where(ExposureDocumentRevision.blob_key == key)
                .limit(1)
            ).scalar_one_or_none()
            is not None
        )

    def collect_unreferenced_blobs(
        self, as_of: datetime | None = None, *, dry_run: bool = True
    ) -> StorageGCReport:
        as_of = as_of or self.clock()
        if as_of.tzinfo is None:
            raise ValueError("as_of must be timezone-aware")
        if not self.root.exists():
            return StorageGCReport(dry_run=dry_run)
        storage_lock(self.session, exclusive=True)
        unreferenced, abandoned, skipped, tombstones = [], [], [], []
        reclaimed = 0
        blob_root = self.root / "sha256"
        for path in sorted(blob_root.glob("*/*")) if blob_root.exists() else []:
            key = str(path.relative_to(self.root))
            modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
            if self._referenced(key):
                skipped.append(key)
                continue
            if as_of - modified < UNREFERENCED_GRACE:
                continue
            unreferenced.append(key)
            if not dry_run:
                size = path.stat().st_size
                path.unlink()
                reclaimed += size
                tombstone = EvidenceTombstoneEvent(
                    blob_key=key,
                    content_hash=path.name,
                    byte_length=size,
                    reason="unreferenced_gc",
                    authority="system:company-exposure-storage-gc",
                )
                self.session.add(tombstone)
                self.session.flush()
                tombstones.append(tombstone.id)
        temp_root = self.root / "tmp"
        for path in sorted(temp_root.glob("*.part")) if temp_root.exists() else []:
            modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
            if as_of - modified < ABANDONED_TEMP_GRACE:
                continue
            abandoned.append(path.name)
            if not dry_run:
                path.unlink(missing_ok=True)
        if reclaimed and not dry_run:
            self.ledger.adjust_pool(self._pool().id, -reclaimed)
        return StorageGCReport(
            dry_run=dry_run,
            unreferenced=tuple(unreferenced),
            abandoned_temp=tuple(abandoned),
            reclaimed_bytes=reclaimed,
            skipped_referenced=tuple(skipped),
            tombstones=tuple(tombstones),
        )
