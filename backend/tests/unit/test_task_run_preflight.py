"""Pre-flight for manual Runs: refuse up front when the run would only skip."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import httpx
import pytest
import pytest_asyncio

import app.tasks.daily_market_pipeline_tasks as pipeline
from app.database import get_db
from app.main import app
from app.services.task_registry_service import TaskRegistryService, TaskRunBlockedError, _run_blockers


class _Calendar:
    def __init__(self, trading=True, error=None):
        self.trading, self.error = trading, error

    def market_now(self, _market):
        if self.error:
            raise self.error
        return datetime(2026, 10, 17, 9, 0)  # a Saturday

    def is_trading_day(self, _market, _today):
        return self.trading


class _Claims:
    def __init__(self, starts=None, error=None):
        self.starts, self.error, self.calls = starts or {}, error, []

    def get_pipeline_chain_starts(self, markets):
        self.calls.append(list(markets))
        if self.error:
            raise self.error
        return {m: self.starts.get(m) for m in markets}


def _blocker(monkeypatch, *, calendar=None, claims=None):
    monkeypatch.setattr(pipeline, "_preflight_calendar_cache", [])
    monkeypatch.setattr(pipeline, "MarketCalendarService", lambda: calendar or _Calendar())
    monkeypatch.setattr(pipeline, "_pipeline_chain_claims", lambda: claims or _Claims())
    return pipeline.pipeline_run_blocker("us")


def test_a_trading_day_with_no_pipeline_running_is_not_blocked(monkeypatch):
    assert _blocker(monkeypatch) is None


def test_a_non_trading_day_blocks_with_the_date(monkeypatch):
    blocker = _blocker(monkeypatch, calendar=_Calendar(trading=False))

    assert blocker["reason"] == "not_trading_day"
    assert blocker["as_of_date"] == "2026-10-17"
    assert "Not a trading day in US" in blocker["message"]


def test_a_held_chain_claim_blocks_with_when_it_started(monkeypatch):
    blocker = _blocker(monkeypatch, claims=_Claims(starts={"US": "2026-10-16T20:30:00+00:00"}))

    assert blocker["reason"] == "market_pipeline_already_active"
    assert blocker["active_since"] == "2026-10-16T20:30:00+00:00"


def test_unreadable_checks_never_block(monkeypatch):
    # The task re-checks everything itself; a pre-flight read failure must
    # not lock operators out of Run.
    assert _blocker(
        monkeypatch,
        calendar=_Calendar(error=RuntimeError("calendar down")),
        claims=_Claims(error=ConnectionError("redis down")),
    ) is None


def test_the_polled_list_reuses_one_calendar_across_markets(monkeypatch):
    # Each MarketCalendarService() re-reads every market's coverage manifests;
    # the list asks once per market on every 5-30s poll.
    built = []
    monkeypatch.setattr(pipeline, "_preflight_calendar_cache", [])
    monkeypatch.setattr(pipeline, "MarketCalendarService", lambda: built.append(1) or _Calendar())
    monkeypatch.setattr(pipeline, "_pipeline_chain_claims", lambda: _Claims())

    for market in ("US", "HK", "JP"):
        pipeline.pipeline_run_blocker(market)

    assert len(built) == 1


def test_every_market_shares_one_claim_read(monkeypatch):
    # One MGET for all markets: a hung Redis costs one socket timeout, not
    # one per market, on a list the UI polls.
    claims = _Claims(starts={"HK": "2026-10-16T08:00:00+00:00"})
    monkeypatch.setattr(pipeline, "_preflight_calendar_cache", [])
    monkeypatch.setattr(pipeline, "MarketCalendarService", lambda: _Calendar())
    monkeypatch.setattr(pipeline, "_pipeline_chain_claims", lambda: claims)

    blockers = pipeline.pipeline_run_blockers(["US", "HK", "JP"])

    assert claims.calls == [["US", "HK", "JP"]]
    assert blockers["HK"]["reason"] == "market_pipeline_already_active"
    assert blockers["US"] is None and blockers["JP"] is None


def test_a_held_claim_with_an_unreadable_start_still_blocks(monkeypatch):
    blocker = _blocker(monkeypatch, claims=_Claims(starts={"US": ""}))

    assert blocker["reason"] == "market_pipeline_already_active"
    assert blocker["active_since"] is None


def test_registry_entries_without_a_pre_flight_are_never_blocked():
    assert _run_blockers({"x": {"task_function": "x"}}) == {}


def test_a_raising_pre_flight_never_blocks():
    with patch.object(pipeline, "pipeline_run_blockers", side_effect=RuntimeError("boom")):
        assert _run_blockers({"daily-market-pipeline-us": {
            "run_blockers": "app.tasks.daily_market_pipeline_tasks.pipeline_run_blockers",
            "manual_dispatch_kwargs": {"market": "US"},
        }}) == {"daily-market-pipeline-us": None}


def test_a_blocked_trigger_queues_nothing_and_writes_no_run_row():
    blocker = {"reason": "not_trading_day", "message": "Not a trading day in US (2026-10-17)"}
    db = MagicMock()
    service = TaskRegistryService()
    task = MagicMock()
    service._task_imports["daily-market-pipeline-us"] = task

    with patch.object(pipeline, "pipeline_run_blockers", return_value={"US": blocker}), pytest.raises(
        TaskRunBlockedError, match="Not a trading day in US"
    ):
        service.trigger_task("daily-market-pipeline-us", db)

    task.apply_async.assert_not_called()
    db.add.assert_not_called()


def test_the_task_list_carries_each_pipelines_blocker(db_session):
    calls = []

    def blockers_for(markets):
        calls.append(markets)
        return {m: {"reason": "market_pipeline_already_active", "message": "busy"} if m == "HK" else None
                for m in markets}

    with patch.object(pipeline, "pipeline_run_blockers", side_effect=blockers_for):
        tasks = {
            t["name"]: t
            for t in TaskRegistryService().get_all_scheduled_tasks(db_session, include_run_blockers=True)
        }
        plain = TaskRegistryService().get_all_scheduled_tasks(db_session)

    assert tasks["daily-market-pipeline-hk"]["run_blocker"]["reason"] == "market_pipeline_already_active"
    assert tasks["daily-market-pipeline-us"]["run_blocker"] is None
    assert tasks["monthly-price-data-cleanup"]["run_blocker"] is None
    # Other callers (the MCP copilot) don't pay for live checks or see the key.
    assert all("run_blocker" not in t for t in plain)
    # Every pipeline in one batched pre-flight call.
    assert len(calls) == 1 and "US" in calls[0] and "HK" in calls[0]


def test_the_task_list_names_each_jobs_market(db_session):
    # The Operations page groups jobs of inactive markets by this field.
    from app.schemas.task import ScheduledTaskResponse

    tasks = {
        t["name"]: ScheduledTaskResponse(**t)
        for t in TaskRegistryService().get_all_scheduled_tasks(db_session)
    }

    assert tasks["daily-market-pipeline-hk"].market == "HK"
    assert tasks["weekly-universe-refresh-de"].market == "DE"
    assert tasks["daily-us-options-analytics"].market == "US"
    assert tasks["weekly-global-maintenance"].market is None


@pytest_asyncio.fixture
async def client():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.asyncio
async def test_the_run_endpoint_answers_409_with_the_reason(client, monkeypatch):
    from app.services import server_auth

    monkeypatch.setattr(server_auth.settings, "server_auth_enabled", False)
    app.dependency_overrides[get_db] = lambda: MagicMock()
    blocker = {"reason": "market_pipeline_already_active", "message": "The US pipeline is already in progress"}
    try:
        with patch.object(pipeline, "pipeline_run_blockers", return_value={"US": blocker}):
            response = await client.post("/api/v1/tasks/daily-market-pipeline-us/run")
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert response.status_code == 409
    assert response.json()["detail"] == "The US pipeline is already in progress"
