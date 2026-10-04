"""Market-scoped scan bootstrap variants, build-once, and the latest ordering guard (#492)."""

from __future__ import annotations

from datetime import datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models.scan_result import Scan
from app.services.ui_snapshot_service import UISnapshotService


@pytest.fixture
def service_and_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    return UISnapshotService(session_factory), session_factory


def _add_scan(session_factory, scan_id, market, completed_hour, status="completed"):
    with session_factory() as db:
        db.add(
            Scan(
                scan_id=scan_id,
                status=status,
                universe=f"market:{market}" if market else "all",
                universe_type="market" if market else "all",
                universe_key=f"market:{market}" if market else "all",
                universe_market=market,
                total_stocks=10,
                passed_stocks=1,
                started_at=datetime(2026, 9, 30, completed_hour, 0),
                completed_at=datetime(2026, 9, 30, completed_hour, 5) if status != "running" else None,
            )
        )
        db.commit()


def test_market_variant_lists_and_selects_only_that_market(service_and_session):
    service, session_factory = service_and_session
    _add_scan(session_factory, "us-1", "US", 9)
    _add_scan(session_factory, "hk-1", "HK", 10)
    _add_scan(session_factory, "us-2", "US", 11)
    _add_scan(session_factory, "hk-2", "HK", 12)

    snapshot = service.publish_scan_bootstrap(market="HK")

    payload = snapshot.payload
    assert payload["market"] == "HK"
    assert [s["scan_id"] for s in payload["recent_scans"]["scans"]] == ["hk-2", "hk-1"]
    assert payload["selected_scan"]["scan_id"] == "hk-2"
    assert snapshot.source_revision == "hk-2"
    assert service.get_scan_bootstrap(market="HK").is_stale is False


def test_market_variant_goes_stale_only_for_its_own_market(service_and_session):
    service, session_factory = service_and_session
    _add_scan(session_factory, "us-1", "US", 9)
    _add_scan(session_factory, "hk-1", "HK", 10)
    service.publish_scan_bootstrap(market="US")

    _add_scan(session_factory, "hk-2", "HK", 11)  # newer scan, other market
    assert service.get_scan_bootstrap(market="US").is_stale is False

    _add_scan(session_factory, "us-2", "US", 12)
    assert service.get_scan_bootstrap(market="US").is_stale is True


def test_publish_for_scan_builds_once_for_scan_and_market_latest(service_and_session, monkeypatch):
    service, session_factory = service_and_session
    _add_scan(session_factory, "us-1", "US", 9)
    _add_scan(session_factory, "us-2", "US", 10)
    builds = []
    original = service._build_scan_payload
    monkeypatch.setattr(
        service,
        "_build_scan_payload",
        lambda scan_id, market=None: builds.append((scan_id, market)) or original(scan_id, market),
    )

    service.publish_scan_bootstraps_for("us-2")

    assert builds == [("us-2", "US")]
    explicit = service.get_scan_bootstrap("us-2")
    latest = service.get_scan_bootstrap(market="US")
    assert explicit.is_stale is False and latest.is_stale is False
    assert explicit.payload == latest.payload
    assert latest.source_revision == "us-2"


def test_publish_for_older_scan_rebuilds_market_latest_separately(service_and_session, monkeypatch):
    service, session_factory = service_and_session
    _add_scan(session_factory, "us-1", "US", 9)
    _add_scan(session_factory, "us-2", "US", 10)
    builds = []
    original = service._build_scan_payload
    monkeypatch.setattr(
        service,
        "_build_scan_payload",
        lambda scan_id, market=None: builds.append((scan_id, market)) or original(scan_id, market),
    )

    service.publish_scan_bootstraps_for("us-1")  # e.g. an old scan republished after cancel

    # The market latest is rebuilt for the resolved latest scan, not re-resolved.
    assert builds == [("us-1", "US"), ("us-2", "US")]
    assert service.get_scan_bootstrap(market="US").source_revision == "us-2"


def test_failed_market_latest_is_logged_as_that_variant_and_keeps_the_scan_variant(
    service_and_session, monkeypatch, caplog
):
    service, session_factory = service_and_session
    _add_scan(session_factory, "us-1", "US", 9)

    def broken_latest(*args, **kwargs):
        raise RuntimeError("latest publish failed")

    monkeypatch.setattr(service, "_publish_scan_latest", broken_latest)

    with caplog.at_level("ERROR"):
        result = service.publish_scan_bootstraps_for("us-1")

    assert result.source_revision == "us-1"
    assert service.get_scan_bootstrap("us-1").is_stale is False
    failures = [r for r in caplog.records if r.getMessage() == "UI snapshot publish failed"]
    assert [r.variant_key for r in failures] == ["latest:US"]


def test_scan_without_market_publishes_the_global_latest(service_and_session):
    service, session_factory = service_and_session
    _add_scan(session_factory, "all-1", None, 9)

    service.publish_scan_bootstraps_for("all-1")

    latest = service.get_scan_bootstrap()
    assert latest is not None and latest.source_revision == "all-1"
    assert latest.payload["market"] is None


def test_latest_matches_the_history_list_order(service_and_session):
    """'Latest' is the first finished scan in GET /scans order (started_at), which
    the scan page also auto-loads; resolution and payload must agree on it."""
    service, session_factory = service_and_session
    with session_factory() as db:
        for scan_id, started, completed in (("us-a", 9, 12), ("us-b", 10, 11)):
            db.add(Scan(
                scan_id=scan_id, status="completed", universe="market:US", universe_type="market",
                universe_key="market:US", universe_market="US", total_stocks=1, passed_stocks=1,
                started_at=datetime(2026, 9, 30, started), completed_at=datetime(2026, 9, 30, completed),
            ))
        db.commit()

    snapshot = service.publish_scan_bootstrap(market="US")

    assert [s["scan_id"] for s in snapshot.payload["recent_scans"]["scans"]] == ["us-b", "us-a"]
    assert snapshot.source_revision == "us-b"
    assert snapshot.payload["selected_scan"]["scan_id"] == "us-b"
    assert service.get_scan_bootstrap(market="US").is_stale is False


def test_slow_outdated_latest_build_does_not_repoint(service_and_session, monkeypatch):
    """Task A builds latest for us-1; meanwhile us-2 completes and task B publishes it.
    A must not move the pointer back to us-1 when it finally writes."""
    service, session_factory = service_and_session
    _add_scan(session_factory, "us-1", "US", 9)
    original = service._build_scan_payload
    raced = []

    def slow_build(scan_id, market=None):
        payload = original(scan_id, market)
        if not raced:
            raced.append(True)
            _add_scan(session_factory, "us-2", "US", 10)
            service.publish_scan_bootstrap(market="US")  # the faster, newer task
        return payload

    monkeypatch.setattr(service, "_build_scan_payload", slow_build)

    result = service.publish_scan_bootstrap(market="US")

    current = service.get_scan_bootstrap(market="US")
    assert current.source_revision == "us-2"
    assert current.is_stale is False
    assert result.source_revision == "us-2"  # the outdated build reports the current snapshot
