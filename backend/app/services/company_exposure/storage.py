"""Private content-addressed store for retained original documents.

Layout: ``<root>/sha256/<aa>/<hash>`` for blobs and ``<root>/tmp/`` for
writes in progress. Writes are temp-file → fsync → rename, so a committed
revision only ever references a complete blob; a crash leaves at most an
unreferenced blob or a stale temp file.

Capacity (spec §9.7): a shared byte reservation pool (default 5 GiB) plus a
free-space floor (default 1 GiB). Bytes are reserved before download; a
duplicate hash settles at zero so identical content is never charged twice.
When full, callers get ``paused_storage`` before any I/O.

Nothing is evicted to make room; a removed original leaves a tombstone and
reads of it fail with a typed reason.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.domain.company_exposure.contracts import (
    ReservationState,
    ResourceUnit,
    bytes_hash,
    utc_now,
)
from app.infra.db.repositories.company_exposure_work_repo import ReservationLedger
from app.models.company_exposure import EvidenceTombstoneEvent

STORAGE_POOL = "storage:exposure-evidence"
STORAGE_PERIOD = "all"


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


def blob_key(content_hash: str) -> str:
    return f"sha256/{content_hash[:2]}/{content_hash}"


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

    def reserve(
        self, bytes_bound: int, *, purpose: str, operation_key: str
    ) -> StorageTicket:
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
                False,
                bytes_bound=bytes_bound,
                reason="paused_storage",
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
            raise StorageUnavailable(
                "storage_write_failed", required=len(data)
            ) from None
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
