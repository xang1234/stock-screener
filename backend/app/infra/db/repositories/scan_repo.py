"""SQLAlchemy implementation of ScanRepository."""

from __future__ import annotations

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
