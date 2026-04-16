"""Tests for synchronous orphaned scan cleanup."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models.scan_result import Scan, ScanResult
from app.tasks.cache_tasks import run_orphaned_scan_cleanup


@pytest.fixture
def cleanup_session_factory():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine, tables=[Scan.__table__, ScanResult.__table__])
    factory = sessionmaker(bind=engine)
    try:
        yield factory
    finally:
        engine.dispose()


def _add_scan(session, *, scan_id: str, status: str, started_at: datetime) -> None:
    session.add(
        Scan(
            scan_id=scan_id,
            status=status,
            started_at=started_at,
            screener_types=["minervini"],
        )
    )
    session.add(ScanResult(scan_id=scan_id, symbol=f"{scan_id}-SYM"))


def test_run_orphaned_scan_cleanup_deletes_cancelled_and_stale_scans(cleanup_session_factory):
    now_utc = datetime(2026, 4, 16, 12, 0, 0)
    session = cleanup_session_factory()
    try:
        _add_scan(
            session,
            scan_id="cancelled-scan",
            status="cancelled",
            started_at=now_utc - timedelta(hours=3),
        )
        _add_scan(
            session,
            scan_id="stale-running-scan",
            status="running",
            started_at=now_utc - timedelta(hours=2),
        )
        _add_scan(
            session,
            scan_id="fresh-running-scan",
            status="running",
            started_at=now_utc - timedelta(minutes=30),
        )
        _add_scan(
            session,
            scan_id="completed-scan",
            status="completed",
            started_at=now_utc - timedelta(hours=5),
        )
        session.commit()
    finally:
        session.close()

    result = run_orphaned_scan_cleanup(
        session_factory=cleanup_session_factory,
        now_utc=now_utc,
    )

    assert result["deleted_scans"] == 2
    assert result["deleted_results"] == 2

    verification = cleanup_session_factory()
    try:
        remaining_scan_ids = {scan.scan_id for scan in verification.query(Scan).all()}
        remaining_result_scan_ids = {row.scan_id for row in verification.query(ScanResult).all()}
    finally:
        verification.close()

    assert remaining_scan_ids == {"fresh-running-scan", "completed-scan"}
    assert remaining_result_scan_ids == {"fresh-running-scan", "completed-scan"}


def test_run_orphaned_scan_cleanup_returns_zero_when_nothing_matches(cleanup_session_factory):
    now_utc = datetime(2026, 4, 16, 12, 0, 0)
    session = cleanup_session_factory()
    try:
        _add_scan(
            session,
            scan_id="fresh-running-scan",
            status="running",
            started_at=now_utc - timedelta(minutes=20),
        )
        session.commit()
    finally:
        session.close()

    result = run_orphaned_scan_cleanup(
        session_factory=cleanup_session_factory,
        now_utc=now_utc,
    )

    assert result["deleted_scans"] == 0
    assert result["deleted_results"] == 0
