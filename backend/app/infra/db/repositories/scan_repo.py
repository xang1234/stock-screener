"""SQLAlchemy implementation of ScanRepository."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy.orm import Session

from app.domain.scanning.ports import ScanRepository
from app.models.scan_result import Scan


class SqlScanRepository(ScanRepository):
    """Persist and retrieve Scan rows via SQLAlchemy."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def create(self, *, scan_id: str, **fields) -> Scan:
        scan = Scan(scan_id=scan_id, **fields)
        self._session.add(scan)
        self._session.flush()  # assigns PK without committing
        return scan

    def get_by_scan_id(self, scan_id: str) -> Scan | None:
        return (
            self._session.query(Scan)
            .filter(Scan.scan_id == scan_id)
            .first()
        )

    def get_by_idempotency_key(self, key: str) -> Scan | None:
        return (
            self._session.query(Scan)
            .filter(Scan.idempotency_key == key)
            .first()
        )

    def update_status(self, scan_id: str, status: str, **fields) -> None:
        scan = self.get_by_scan_id(scan_id)
        if scan is None:
            return

        scan.status = status

        if "total_stocks" in fields:
            scan.total_stocks = fields["total_stocks"]
        if "passed_stocks" in fields:
            scan.passed_stocks = fields["passed_stocks"]
        if status in ("completed", "cancelled"):
            scan.completed_at = datetime.utcnow()

        self._session.flush()

    def list_recent(self, limit: int = 20) -> list[Scan]:
        return (
            self._session.query(Scan)
            .order_by(Scan.started_at.desc())
            .limit(limit)
            .all()
        )

    def delete(self, scan_id: str) -> bool:
        scan = self.get_by_scan_id(scan_id)
        if scan is None:
            return False
        self._session.delete(scan)
        self._session.flush()
        return True
