import types

import pytest
from fastapi import HTTPException

from app.api.v1 import scans as scans_module
from app.api.v1.scans import ScanCreateRequest


class FakeSession:
    def __init__(self):
        self.added = []
        self.commits = 0
        self.refreshes = 0

    def add(self, obj):
        self.added.append(obj)

    def commit(self):
        self.commits += 1

    def refresh(self, _obj):
        self.refreshes += 1

    def query(self, *_args, **_kwargs):
        raise AssertionError("query should not be called for custom universe")


@pytest.mark.asyncio
async def test_create_scan_commits_before_dispatch(monkeypatch):
    fake_db = FakeSession()
    called = {"delay": False}

    def fake_delay(scan_id, symbols, criteria):
        assert scan_id
        assert symbols == ["AAPL"]
        assert fake_db.commits >= 1
        assert fake_db.added
        assert fake_db.added[-1].task_id is None
        called["delay"] = True
        return types.SimpleNamespace(id="task-123")

    monkeypatch.setattr(scans_module, "run_bulk_scan", types.SimpleNamespace(delay=fake_delay))

    request = ScanCreateRequest(
        universe="custom",
        symbols=["AAPL"],
        criteria=None,
        screeners=["minervini"],
        composite_method="weighted_average",
    )

    response = await scans_module.create_scan(request, db=fake_db)

    assert called["delay"] is True
    assert response.status == "queued"
    assert fake_db.added[-1].task_id == "task-123"
    assert fake_db.commits >= 2


@pytest.mark.asyncio
async def test_create_scan_marks_failed_on_dispatch_error(monkeypatch):
    fake_db = FakeSession()

    def fake_delay(_scan_id, _symbols, _criteria):
        raise RuntimeError("dispatch failed")

    monkeypatch.setattr(scans_module, "run_bulk_scan", types.SimpleNamespace(delay=fake_delay))

    request = ScanCreateRequest(
        universe="custom",
        symbols=["AAPL"],
        criteria=None,
        screeners=["minervini"],
        composite_method="weighted_average",
    )

    with pytest.raises(HTTPException) as exc_info:
        await scans_module.create_scan(request, db=fake_db)

    assert exc_info.value.status_code == 500
    assert fake_db.added[-1].status == "failed"
    assert fake_db.commits >= 2
