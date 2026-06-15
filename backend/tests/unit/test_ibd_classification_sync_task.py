"""Unit tests for the live-site IBD classification GitHub sync.

Covers the service function ``sync_ibd_classification_from_github`` (network
boundary injected) and the Celery task that delegates to it.
"""

from __future__ import annotations

import pytest

from app.services import ibd_classification_bundle as bundle


class _FakeSyncService:
    """Stand-in for GitHubReleaseSyncService returning a canned fetch result."""

    def __init__(self, result: dict):
        self._result = result

    def fetch_latest_bundle(self, **kwargs):
        return self._result


def test_sync_returns_non_fatal_for_live_only():
    """live_only is a clean no-op: no import, no error, no reason."""
    fake = _FakeSyncService({"status": "live_only"})

    out = bundle.sync_ibd_classification_from_github(
        db=None, market="sg", github_sync_service=fake
    )

    assert out == {"market": "SG", "status": "live_only", "imported": None, "reason": None}


def test_sync_imports_bundle_on_success(monkeypatch):
    """On success the bundle is read and imported, and stats are returned."""
    fake = _FakeSyncService({"status": "success", "bundle_path": "/tmp/bundle.json.gz"})
    monkeypatch.setattr(bundle, "read_bundle", lambda _path: {"classifications": []})
    monkeypatch.setattr(
        bundle, "import_classifications", lambda _db, _payload: {"inserted": 3, "updated": 1}
    )

    out = bundle.sync_ibd_classification_from_github(
        db=object(), market="US", github_sync_service=fake
    )

    assert out["status"] == "success"
    assert out["imported"] == {"inserted": 3, "updated": 1}


class _DummySession:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_task_delegates_to_service_and_stamps_timestamp(monkeypatch):
    """The Celery task opens a session, calls the service, and adds a timestamp."""
    from app.tasks import industry_tasks
    from app.tasks.workload_coordination import disable_serialized_market_workload

    def _fake_service(db, *, market):
        return {"market": market, "status": "success", "imported": {"inserted": 1}, "reason": None}

    monkeypatch.setattr(industry_tasks, "sync_ibd_classification_from_github", _fake_service)
    monkeypatch.setattr(industry_tasks, "SessionLocal", lambda: _DummySession())

    with disable_serialized_market_workload():
        out = industry_tasks.sync_ibd_classification.run(market="hk")

    assert out["market"] == "HK"
    assert out["status"] == "success"
    assert "timestamp" in out


def test_task_is_registered_routed_and_scheduled():
    """The sync task must be registered, routed to a market jobs queue, and have
    one weekly beat entry per supported market."""
    import app.tasks.industry_tasks  # noqa: F401 - force task registration
    from app.celery_app import celery_app
    from app.tasks.market_queues import SUPPORTED_MARKETS

    name = "app.tasks.industry_tasks.sync_ibd_classification"
    assert name in celery_app.tasks
    assert celery_app.conf.task_routes.get(name) is not None

    beat = celery_app.conf.beat_schedule or {}
    entries = {k for k in beat if k.startswith("weekly-ibd-classification-sync-")}
    # cache_warmup_enabled gates the schedule; only assert when entries exist.
    if entries:
        assert len(entries) == len(SUPPORTED_MARKETS)
        sample = beat[next(iter(entries))]
        assert sample["task"] == name
        assert "market" in sample["kwargs"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
