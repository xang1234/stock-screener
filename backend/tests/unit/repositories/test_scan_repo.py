"""Tests for SqlScanRepository using in-memory SQLite.

Covers create, lookup (by scan_id / idempotency_key), update_status,
and uniqueness constraints.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.domain.scanning.errors import SingleActiveScanViolation
from app.models.scan_result import Scan
from app.infra.db.repositories.scan_repo import SqlScanRepository


@pytest.fixture
def repo(session: Session) -> SqlScanRepository:
    return SqlScanRepository(session)


def _make_scan_id() -> str:
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# TestCreate
# ---------------------------------------------------------------------------


class TestCreate:
    def test_creates_scan_and_assigns_pk(self, repo: SqlScanRepository, session: Session):
        scan = repo.create(scan_id=_make_scan_id(), status="running")

        assert scan.id is not None
        # Verify actually persisted
        row = session.get(Scan, scan.id)
        assert row is not None

    def test_stores_all_fields(self, repo: SqlScanRepository, session: Session):
        sid = _make_scan_id()
        criteria = {"min_score": 70, "screeners": ["minervini"]}
        screener_types = ["minervini", "canslim"]

        scan = repo.create(
            scan_id=sid,
            status="running",
            criteria=criteria,
            screener_types=screener_types,
            idempotency_key="idem-001",
            universe="all",
        )

        row = session.get(Scan, scan.id)
        assert row.scan_id == sid
        assert row.criteria == criteria
        assert row.screener_types == screener_types
        assert row.idempotency_key == "idem-001"
        assert row.universe == "all"

    def test_unique_scan_id_constraint(self, repo: SqlScanRepository, session: Session):
        sid = _make_scan_id()
        repo.create(scan_id=sid, status="completed")

        with pytest.raises(IntegrityError):
            repo.create(scan_id=sid, status="completed")

    def test_unique_idempotency_key_constraint(self, repo: SqlScanRepository, session: Session):
        repo.create(scan_id=_make_scan_id(), status="completed", idempotency_key="key-dup")

        with pytest.raises(IntegrityError):
            repo.create(scan_id=_make_scan_id(), status="completed", idempotency_key="key-dup")

    def test_partial_unique_active_status_constraint(self, repo: SqlScanRepository):
        repo.create(scan_id=_make_scan_id(), status="queued")

        with pytest.raises(SingleActiveScanViolation):
            repo.create(scan_id=_make_scan_id(), status="running")

    def test_completed_scans_do_not_hit_active_status_constraint(self, repo: SqlScanRepository):
        repo.create(scan_id=_make_scan_id(), status="completed")
        repo.create(scan_id=_make_scan_id(), status="completed")


# ---------------------------------------------------------------------------
# TestGetByScanId
# ---------------------------------------------------------------------------


class TestGetByScanId:
    def test_returns_existing_scan(self, repo: SqlScanRepository):
        sid = _make_scan_id()
        repo.create(scan_id=sid, status="running")

        result = repo.get_by_scan_id(sid)

        assert result is not None
        assert result.scan_id == sid

    def test_returns_none_for_missing(self, repo: SqlScanRepository):
        assert repo.get_by_scan_id("nonexistent-id") is None


# ---------------------------------------------------------------------------
# TestGetByIdempotencyKey
# ---------------------------------------------------------------------------


class TestGetByIdempotencyKey:
    def test_returns_matching_scan(self, repo: SqlScanRepository):
        sid = _make_scan_id()
        repo.create(scan_id=sid, status="running", idempotency_key="key-abc")

        result = repo.get_by_idempotency_key("key-abc")

        assert result is not None
        assert result.scan_id == sid

    def test_returns_none_when_no_match(self, repo: SqlScanRepository):
        assert repo.get_by_idempotency_key("nonexistent-key") is None


# ---------------------------------------------------------------------------
# TestUpdateStatus
# ---------------------------------------------------------------------------


class TestUpdateStatus:
    def test_updates_status_and_fields(self, repo: SqlScanRepository, session: Session):
        sid = _make_scan_id()
        repo.create(scan_id=sid, status="running")

        repo.update_status(sid, "completed", total_stocks=500, passed_stocks=42)

        row = repo.get_by_scan_id(sid)
        assert row.status == "completed"
        assert row.total_stocks == 500
        assert row.passed_stocks == 42
        assert row.completed_at is not None

    def test_sets_completed_at_only_for_completed_status(self, repo: SqlScanRepository):
        sid = _make_scan_id()
        repo.create(scan_id=sid, status="running")

        repo.update_status(sid, "failed")

        row = repo.get_by_scan_id(sid)
        assert row.status == "failed"
        assert row.completed_at is None

    def test_partial_field_update(self, repo: SqlScanRepository):
        sid = _make_scan_id()
        repo.create(scan_id=sid, status="running", total_stocks=100)

        # Update only status — total_stocks should remain untouched
        repo.update_status(sid, "failed")

        row = repo.get_by_scan_id(sid)
        assert row.status == "failed"
        assert row.total_stocks == 100

    def test_no_op_for_missing_scan(self, repo: SqlScanRepository):
        # Should not raise — silent no-op
        repo.update_status("nonexistent-id", "completed")
