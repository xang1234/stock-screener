# Issue 261 Runtime Activity Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix GitHub issue #261 by making price refresh worker loss publish a terminal runtime activity state, allowing truly stale runtime owners to be superseded, surfacing stale rows as warnings, and adding diagnostics around the SIGKILL point.

**Architecture:** Keep task execution, persisted runtime activity reduction, API presentation, and diagnostics as separate concerns. A Celery parent-process request hook handles hard worker loss that child-process code cannot catch. The reducer remains conservative by default, with stale-owner override allowed only when the persistence layer proves the previous owner is stale and not live.

**Tech Stack:** Python 3.11, Celery prefork, SQLAlchemy, Redis-backed locks and heartbeats, FastAPI service modules, pytest, React, TanStack Query, Vitest.

---

## File Structure

- Create `backend/app/services/runtime_activity_staleness.py`
  - Pure helpers for runtime activity age parsing and stale/stuck presentation.
- Create `backend/tests/unit/test_runtime_activity_staleness.py`
  - Unit tests for stale threshold behavior and stale payload conversion.
- Create `backend/app/tasks/runtime_activity_failure_hooks.py`
  - Celery `Request` hook plus testable helper for hard worker failure publication.
- Create `backend/tests/unit/test_runtime_activity_failure_hooks.py`
  - Unit tests for publishing failed activity and releasing matching leases on worker loss.
- Modify `backend/app/tasks/cache_tasks.py`
  - Attach the tracked task base to `smart_refresh_cache`.
- Modify `backend/app/services/runtime_activity_reducer.py`
  - Add an explicit opt-in override flag for stale running owners.
- Modify `backend/tests/unit/test_runtime_activity_reducer.py`
  - Cover default preserve behavior and explicit override behavior.
- Modify `backend/app/services/market_activity_service.py`
  - Decide when the persisted stale running owner can be overridden; present stale rows on read.
- Modify `backend/tests/unit/test_market_activity_service.py`
  - Cover stale owner override, live owner preservation, and stale read-side warnings.
- Modify `backend/app/services/runtime_activity_presenter.py`
  - Treat stale/stuck market rows as warning summary states.
- Modify `frontend/src/components/Layout/RuntimeActivityStatusButton.jsx`
  - Show stale/stuck warnings in the header summary.
- Modify `frontend/src/components/Layout/Layout.test.jsx`
  - Cover stale runtime warning copy.
- Create `backend/app/services/runtime_diagnostics.py`
  - Dependency-free elapsed time and max RSS logging context manager.
- Create `backend/tests/unit/test_runtime_diagnostics.py`
  - Unit tests for diagnostic log emission.
- Modify `backend/app/services/price_refresh_plan_builder.py`
  - Instrument universe load, GitHub seed sync, and coverage classification.
- Modify `backend/tests/unit/test_price_refresh_planning.py` or create `backend/tests/unit/test_price_refresh_plan_builder_diagnostics.py`
  - Cover that plan-building emits diagnostic stage logs.

## Task 1: Add Runtime Activity Staleness Helpers

**Files:**
- Create: `backend/app/services/runtime_activity_staleness.py`
- Test: `backend/tests/unit/test_runtime_activity_staleness.py`

- [ ] **Step 1: Write the failing tests**

Create `backend/tests/unit/test_runtime_activity_staleness.py` with:

```python
from __future__ import annotations

from datetime import datetime, timedelta, timezone


def _record(**overrides):
    from app.services.runtime_activity_contract import RuntimeActivityRecord

    values = {
        "market": "US",
        "lifecycle": "daily_refresh",
        "stage_key": "prices",
        "status": "running",
        "task_name": "app.tasks.cache_tasks.smart_refresh_cache",
        "task_id": "task-old",
        "message": "Refreshing market prices",
        "updated_at": "2026-06-23T05:39:46+00:00",
    }
    values.update(overrides)
    return RuntimeActivityRecord.create(**values)


def test_running_activity_is_stale_after_threshold():
    from app.services.runtime_activity_staleness import is_stale_running_activity

    record = _record(updated_at="2026-06-23T05:39:46+00:00")
    now = datetime(2026, 6, 23, 6, 10, 0, tzinfo=timezone.utc)

    assert is_stale_running_activity(record, now=now) is True


def test_running_activity_is_not_stale_before_threshold():
    from app.services.runtime_activity_staleness import is_stale_running_activity

    record = _record(updated_at="2026-06-23T05:39:46+00:00")
    now = datetime(2026, 6, 23, 5, 50, 0, tzinfo=timezone.utc)

    assert is_stale_running_activity(record, now=now) is False


def test_non_running_activity_is_not_stale():
    from app.services.runtime_activity_staleness import is_stale_running_activity

    record = _record(status="completed", updated_at="2026-06-23T05:00:00+00:00")
    now = datetime(2026, 6, 23, 7, 0, 0, tzinfo=timezone.utc)

    assert is_stale_running_activity(record, now=now) is False


def test_stale_payload_marks_running_record_stale_without_mutating_original():
    from app.services.runtime_activity_staleness import stale_runtime_activity_payload

    record = _record(updated_at="2026-06-23T05:39:46+00:00")
    payload = stale_runtime_activity_payload(record, reason="No live worker owns this task.")

    assert payload["status"] == "stale"
    assert payload["message"] == "Refreshing market prices - stale: No live worker owns this task."
    assert payload["task_id"] == "task-old"
    assert record.status == "running"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
cd backend
source venv/bin/activate
pytest tests/unit/test_runtime_activity_staleness.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'app.services.runtime_activity_staleness'`.

- [ ] **Step 3: Implement the helpers**

Create `backend/app/services/runtime_activity_staleness.py`:

```python
"""Runtime activity stale-state helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .runtime_activity_contract import RuntimeActivityRecord, progress_mode

RUNNING_ACTIVITY_STALE_AFTER_SECONDS = 30 * 60


def parse_activity_timestamp(value: Any) -> datetime | None:
    if not value or not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def activity_age_seconds(
    record: RuntimeActivityRecord,
    *,
    now: datetime | None = None,
) -> float | None:
    updated_at = parse_activity_timestamp(record.updated_at)
    if updated_at is None:
        return None
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    else:
        current = current.astimezone(timezone.utc)
    return max((current - updated_at).total_seconds(), 0.0)


def is_stale_running_activity(
    record: RuntimeActivityRecord,
    *,
    now: datetime | None = None,
    stale_after_seconds: int = RUNNING_ACTIVITY_STALE_AFTER_SECONDS,
) -> bool:
    if record.status != "running":
        return False
    age = activity_age_seconds(record, now=now)
    return age is not None and age >= stale_after_seconds


def stale_runtime_activity_payload(
    record: RuntimeActivityRecord,
    *,
    reason: str,
) -> dict[str, Any]:
    payload = record.to_payload()
    base_message = str(record.message or "Runtime activity").strip()
    payload["status"] = "stale"
    payload["message"] = f"{base_message} - stale: {reason}"
    payload["progress_mode"] = progress_mode(
        "stale",
        payload.get("percent"),
        payload.get("current"),
        payload.get("total"),
    )
    return payload
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
cd backend
source venv/bin/activate
pytest tests/unit/test_runtime_activity_staleness.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit Task 1**

```bash
git add backend/app/services/runtime_activity_staleness.py backend/tests/unit/test_runtime_activity_staleness.py
git commit -m "test: add runtime activity staleness helpers"
```

## Task 2: Capture Hard Worker Failures for Smart Refresh

**Files:**
- Create: `backend/app/tasks/runtime_activity_failure_hooks.py`
- Modify: `backend/app/tasks/cache_tasks.py`
- Test: `backend/tests/unit/test_runtime_activity_failure_hooks.py`
- Test: `backend/tests/unit/test_cache_refresh_unification.py`

- [ ] **Step 1: Write the failing hard-failure tests**

Create `backend/tests/unit/test_runtime_activity_failure_hooks.py`:

```python
from __future__ import annotations

from types import SimpleNamespace


class _FakeDb:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _FakeLock:
    def __init__(self) -> None:
        self.released = []

    def release(self, task_id, market=None):
        self.released.append({"task_id": task_id, "market": market})
        return True


class _FakeCoordination:
    def __init__(self) -> None:
        self.market_releases = []
        self.external_releases = []

    def release_market_workload(self, task_id, market=None):
        self.market_releases.append({"task_id": task_id, "market": market})
        return True

    def release_external_fetch(self, task_id):
        self.external_releases.append(task_id)
        return True


class _FakePriceCache:
    def __init__(self) -> None:
        self.heartbeats = []

    def complete_warmup_heartbeat(self, status="completed", market=None):
        self.heartbeats.append({"status": status, "market": market})


def test_publish_runtime_activity_failure_marks_smart_refresh_failed(monkeypatch):
    import app.tasks.runtime_activity_failure_hooks as module

    fake_db = _FakeDb()
    fake_lock = _FakeLock()
    fake_coordination = _FakeCoordination()
    fake_price_cache = _FakePriceCache()
    failures = []

    monkeypatch.setattr(module, "SessionLocal", lambda: fake_db)
    monkeypatch.setattr(module, "get_data_fetch_lock", lambda: fake_lock)
    monkeypatch.setattr(module, "get_workload_coordination", lambda: fake_coordination)
    monkeypatch.setattr(module, "get_price_cache", lambda: fake_price_cache)
    monkeypatch.setattr(
        module,
        "mark_market_activity_failed",
        lambda _db, **kwargs: failures.append(kwargs),
    )

    module.publish_runtime_activity_failure(
        task_name="app.tasks.cache_tasks.smart_refresh_cache",
        task_id="lost-task",
        kwargs={"market": "us", "activity_lifecycle": "bootstrap"},
        exception=RuntimeError("Worker exited prematurely: signal 9 (SIGKILL)"),
    )

    assert failures == [
        {
            "market": "US",
            "stage_key": "prices",
            "lifecycle": "bootstrap",
            "task_name": "app.tasks.cache_tasks.smart_refresh_cache",
            "task_id": "lost-task",
            "message": "Task worker exited before cleanup: Worker exited prematurely: signal 9 (SIGKILL)",
        }
    ]
    assert fake_lock.released == [{"task_id": "lost-task", "market": "US"}]
    assert fake_coordination.market_releases == [{"task_id": "lost-task", "market": "US"}]
    assert fake_coordination.external_releases == ["lost-task"]
    assert fake_price_cache.heartbeats == [{"status": "failed", "market": "US"}]
    assert fake_db.closed is True


def test_publish_runtime_activity_failure_ignores_untracked_tasks(monkeypatch):
    import app.tasks.runtime_activity_failure_hooks as module

    failures = []
    monkeypatch.setattr(
        module,
        "mark_market_activity_failed",
        lambda _db, **kwargs: failures.append(kwargs),
    )

    module.publish_runtime_activity_failure(
        task_name="app.tasks.cache_tasks.warm_spy_cache",
        task_id="other-task",
        kwargs={"market": "US"},
        exception=RuntimeError("boom"),
    )

    assert failures == []


def test_request_hook_delegates_failure_publication(monkeypatch):
    import app.tasks.runtime_activity_failure_hooks as module

    calls = []

    class _Request(module.RuntimeActivityFailureRequest):
        task = SimpleNamespace(name="app.tasks.cache_tasks.smart_refresh_cache")
        id = "lost-task"
        kwargs = {"market": "US"}

    monkeypatch.setattr(
        module.Request,
        "on_failure",
        lambda self, exc_info, send_failed_event=True, return_ok=False: None,
    )
    monkeypatch.setattr(
        module,
        "publish_runtime_activity_failure",
        lambda **kwargs: calls.append(kwargs),
    )

    request = object.__new__(_Request)
    request.on_failure(SimpleNamespace(exception=RuntimeError("signal 9")))

    assert calls[0]["task_name"] == "app.tasks.cache_tasks.smart_refresh_cache"
    assert calls[0]["task_id"] == "lost-task"
    assert calls[0]["kwargs"] == {"market": "US"}
    assert "signal 9" in str(calls[0]["exception"])
```

Add this assertion to `backend/tests/unit/test_cache_refresh_unification.py`:

```python
def test_smart_refresh_cache_uses_runtime_activity_tracked_task_base():
    import app.tasks.cache_tasks as module
    from app.tasks.runtime_activity_failure_hooks import RuntimeActivityTrackedTask

    assert isinstance(module.smart_refresh_cache, RuntimeActivityTrackedTask)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
cd backend
source venv/bin/activate
pytest tests/unit/test_runtime_activity_failure_hooks.py tests/unit/test_cache_refresh_unification.py::test_smart_refresh_cache_uses_runtime_activity_tracked_task_base -v
```

Expected: FAIL because `runtime_activity_failure_hooks.py` does not exist and `smart_refresh_cache` has no tracked base.

- [ ] **Step 3: Implement the Celery request hook**

Create `backend/app/tasks/runtime_activity_failure_hooks.py`:

```python
"""Celery request hooks for runtime activity failure publication."""

from __future__ import annotations

import logging
from typing import Any

from celery import Task
from celery.worker.request import Request

from ..database import SessionLocal
from ..services.market_activity_service import mark_market_activity_failed
from ..tasks.market_queues import normalize_market
from ..wiring.bootstrap import (
    get_data_fetch_lock,
    get_price_cache,
    get_workload_coordination,
)

logger = logging.getLogger(__name__)

_TRACKED_TASKS = {
    "app.tasks.cache_tasks.smart_refresh_cache": {
        "stage_key": "prices",
        "default_lifecycle": "daily_refresh",
    },
}


def _failure_exception(exc_info: Any) -> BaseException | Any:
    return getattr(exc_info, "exception", None) or exc_info


def _failure_message(exception: Any) -> str:
    raw = str(exception).strip() or exception.__class__.__name__
    return f"Task worker exited before cleanup: {raw}"


def publish_runtime_activity_failure(
    *,
    task_name: str,
    task_id: str | None,
    kwargs: dict[str, Any] | None,
    exception: Any,
) -> None:
    task_config = _TRACKED_TASKS.get(task_name)
    if task_config is None or not task_id:
        return

    kwargs = kwargs or {}
    market = normalize_market(kwargs.get("market") or "US")
    lifecycle = kwargs.get("activity_lifecycle") or task_config["default_lifecycle"]

    db = SessionLocal()
    try:
        mark_market_activity_failed(
            db,
            market=market,
            stage_key=task_config["stage_key"],
            lifecycle=lifecycle,
            task_name=task_name,
            task_id=task_id,
            message=_failure_message(exception),
        )
        try:
            get_price_cache().complete_warmup_heartbeat("failed", market=market)
        except Exception:
            logger.warning(
                "Failed to mark warmup heartbeat failed after task worker loss",
                extra={"task_name": task_name, "task_id": task_id, "market": market},
                exc_info=True,
            )
        try:
            get_data_fetch_lock().release(task_id, market=market)
        except Exception:
            logger.warning(
                "Failed to release data fetch lock after task worker loss",
                extra={"task_name": task_name, "task_id": task_id, "market": market},
                exc_info=True,
            )
        try:
            coordination = get_workload_coordination()
            coordination.release_market_workload(task_id, market=market)
            coordination.release_external_fetch(task_id)
        except Exception:
            logger.warning(
                "Failed to release workload leases after task worker loss",
                extra={"task_name": task_name, "task_id": task_id, "market": market},
                exc_info=True,
            )
    finally:
        db.close()


class RuntimeActivityFailureRequest(Request):
    """Parent-process failure hook for prefork worker losses."""

    def on_failure(self, exc_info, send_failed_event=True, return_ok=False):
        super().on_failure(
            exc_info,
            send_failed_event=send_failed_event,
            return_ok=return_ok,
        )
        publish_runtime_activity_failure(
            task_name=getattr(getattr(self, "task", None), "name", ""),
            task_id=getattr(self, "id", None),
            kwargs=getattr(self, "kwargs", None),
            exception=_failure_exception(exc_info),
        )


class RuntimeActivityTrackedTask(Task):
    Request = RuntimeActivityFailureRequest
```

Modify `backend/app/tasks/cache_tasks.py`:

```python
from .runtime_activity_failure_hooks import RuntimeActivityTrackedTask
```

Then change the task decorator for `smart_refresh_cache` to:

```python
@celery_app.task(
    bind=True,
    base=RuntimeActivityTrackedTask,
    name='app.tasks.cache_tasks.smart_refresh_cache',
    soft_time_limit=14400,
)
@serialized_data_fetch('smart_refresh_cache')
def smart_refresh_cache(
    self,
    mode: str = "auto",
    market: str | None = None,
    activity_lifecycle: str | None = None,
):
    """Run the smart price refresh workflow behind the Celery data-fetch lock."""
    return run_smart_price_refresh(
        task=self,
        mode=mode,
        market=market,
        activity_lifecycle=activity_lifecycle,
    )
```

- [ ] **Step 4: Run the focused tests**

Run:

```bash
cd backend
source venv/bin/activate
pytest tests/unit/test_runtime_activity_failure_hooks.py tests/unit/test_cache_refresh_unification.py::test_smart_refresh_cache_uses_runtime_activity_tracked_task_base -v
```

Expected: PASS.

- [ ] **Step 5: Commit Task 2**

```bash
git add backend/app/tasks/runtime_activity_failure_hooks.py backend/app/tasks/cache_tasks.py backend/tests/unit/test_runtime_activity_failure_hooks.py backend/tests/unit/test_cache_refresh_unification.py
git commit -m "fix: publish runtime activity on smart refresh worker loss"
```

## Task 3: Allow Proven Stale Running Owners to Be Superseded

**Files:**
- Modify: `backend/app/services/runtime_activity_reducer.py`
- Modify: `backend/app/services/market_activity_service.py`
- Test: `backend/tests/unit/test_runtime_activity_reducer.py`
- Test: `backend/tests/unit/test_market_activity_service.py`

- [ ] **Step 1: Write failing reducer tests**

Append to `backend/tests/unit/test_runtime_activity_reducer.py`:

```python
def test_reduce_market_activity_preserves_running_record_from_different_owner_by_default():
    from app.services.runtime_activity_reducer import reduce_market_activity

    existing = _record(status="running", stage_key="prices", task_id="old-task")
    incoming = _record(status="running", stage_key="prices", task_id="new-task")

    transition = reduce_market_activity(existing, incoming)

    assert transition.should_persist is False
    assert transition.record == existing


def test_reduce_market_activity_allows_explicit_running_owner_override():
    from app.services.runtime_activity_reducer import reduce_market_activity

    existing = _record(status="running", stage_key="prices", task_id="old-task")
    incoming = _record(status="running", stage_key="prices", task_id="new-task")

    transition = reduce_market_activity(
        existing,
        incoming,
        allow_running_owner_override=True,
    )

    assert transition.should_persist is True
    assert transition.record == incoming
```

- [ ] **Step 2: Write failing service tests**

Append to `backend/tests/unit/test_market_activity_service.py`:

```python
def _persist_activity_row(db_session, payload):
    from app.models.app_settings import AppSetting
    from app.services.market_activity_service import (
        MARKET_ACTIVITY_KEY_PREFIX,
        RUNTIME_ACTIVITY_CATEGORY,
    )

    db_session.add(
        AppSetting(
            key=f"{MARKET_ACTIVITY_KEY_PREFIX}{payload['market']}",
            value=json.dumps(payload),
            category=RUNTIME_ACTIVITY_CATEGORY,
            description=f"Latest runtime activity state for {payload['market']}",
        )
    )
    db_session.commit()


def test_stale_running_activity_allows_new_task_owner_to_start(db_session, monkeypatch):
    from app.services import market_activity_service as module
    from app.services.runtime_activity_contract import PersistedRuntimeActivity, RuntimeActivityRecord

    old_record = RuntimeActivityRecord.create(
        market="US",
        lifecycle="daily_refresh",
        stage_key="prices",
        status="running",
        task_name="app.tasks.cache_tasks.smart_refresh_cache",
        task_id="old-task",
        message="Refreshing market prices",
        updated_at="2026-06-23T05:00:00+00:00",
    )
    _persist_activity_row(db_session, PersistedRuntimeActivity.from_record(old_record).to_payload())
    monkeypatch.setattr(module, "get_data_fetch_lock", lambda: _FakeLock())
    monkeypatch.setattr(
        module,
        "_utcnow_iso",
        lambda: "2026-06-23T06:00:00+00:00",
    )

    result = module.mark_market_activity_started(
        db_session,
        market="US",
        stage_key="prices",
        lifecycle="daily_refresh",
        task_name="app.tasks.cache_tasks.smart_refresh_cache",
        task_id="new-task",
        message="Refreshing market prices",
    )

    assert result["status"] == "running"
    assert result["task_id"] == "new-task"
    assert result["updated_at"] == "2026-06-23T06:00:00+00:00"


def test_stale_running_activity_does_not_override_when_old_owner_is_live(db_session, monkeypatch):
    from app.services import market_activity_service as module
    from app.services.runtime_activity_contract import PersistedRuntimeActivity, RuntimeActivityRecord

    old_record = RuntimeActivityRecord.create(
        market="US",
        lifecycle="daily_refresh",
        stage_key="prices",
        status="running",
        task_name="app.tasks.cache_tasks.smart_refresh_cache",
        task_id="old-task",
        message="Refreshing market prices",
        updated_at="2026-06-23T05:00:00+00:00",
    )
    _persist_activity_row(db_session, PersistedRuntimeActivity.from_record(old_record).to_payload())
    monkeypatch.setattr(
        module,
        "get_data_fetch_lock",
        lambda: _FakeLock({"US": {"task_id": "old-task"}}),
    )
    monkeypatch.setattr(
        module,
        "_utcnow_iso",
        lambda: "2026-06-23T06:00:00+00:00",
    )

    result = module.mark_market_activity_started(
        db_session,
        market="US",
        stage_key="prices",
        lifecycle="daily_refresh",
        task_name="app.tasks.cache_tasks.smart_refresh_cache",
        task_id="new-task",
        message="Refreshing market prices",
    )

    assert result["status"] == "running"
    assert result["task_id"] == "old-task"
```

- [ ] **Step 3: Run the tests to verify they fail**

Run:

```bash
cd backend
source venv/bin/activate
pytest tests/unit/test_runtime_activity_reducer.py::test_reduce_market_activity_preserves_running_record_from_different_owner_by_default tests/unit/test_runtime_activity_reducer.py::test_reduce_market_activity_allows_explicit_running_owner_override tests/unit/test_market_activity_service.py::test_stale_running_activity_allows_new_task_owner_to_start tests/unit/test_market_activity_service.py::test_stale_running_activity_does_not_override_when_old_owner_is_live -v
```

Expected: reducer override test fails with unexpected keyword argument; service new-owner test preserves `old-task`.

- [ ] **Step 4: Add explicit reducer override**

Modify `backend/app/services/runtime_activity_reducer.py`:

```python
def reduce_market_activity(
    existing_payload: RuntimeActivityRecord | Mapping[str, Any] | None,
    incoming_payload: RuntimeActivityRecord | RuntimeActivityUpdate | Mapping[str, Any],
    *,
    allow_running_owner_override: bool = False,
) -> RuntimeActivityTransition:
    """Return the activity payload that should win this state transition."""
    existing = _coerce_activity_record(existing_payload)
    incoming = _coerce_activity_record(incoming_payload, existing=existing)
    if incoming is None:
        raise ValueError("incoming runtime activity payload is invalid")

    if existing is None:
        return RuntimeActivityTransition(should_persist=True, record=incoming)

    existing_status = existing.status
    if existing_status not in _PRESERVED_EXISTING_STATUSES:
        return RuntimeActivityTransition(should_persist=True, record=incoming)

    payload_status = incoming.status
    same_task = existing.task_id == incoming.task_id
    same_stage = existing.stage_key == incoming.stage_key
    same_owner = same_task and same_stage
    incoming_has_owner = incoming.task_id is not None

    if existing_status == "running":
        if payload_status == "queued":
            return RuntimeActivityTransition(should_persist=False, record=existing)
        if not same_owner and not allow_running_owner_override:
            return RuntimeActivityTransition(should_persist=False, record=existing)
    elif existing_status == "completed":
        if payload_status != "failed":
            incoming_new_cycle = (
                payload_status in {"queued", "running"}
                and incoming_has_owner
                and not same_owner
            )
            if not incoming_new_cycle:
                return RuntimeActivityTransition(should_persist=False, record=existing)
    elif existing_status == "failed":
        if _should_supersede_failed_activity(existing, incoming):
            return RuntimeActivityTransition(should_persist=True, record=incoming)
        if payload_status == "failed" and same_owner:
            if _should_preserve_existing_failed_message(existing, incoming):
                return RuntimeActivityTransition(should_persist=False, record=existing)
        elif not _is_new_cycle_after_failed(
            existing,
            incoming,
            same_owner=same_owner,
            incoming_has_owner=incoming_has_owner,
        ):
            return RuntimeActivityTransition(should_persist=False, record=existing)

    return RuntimeActivityTransition(should_persist=True, record=incoming)
```

- [ ] **Step 5: Add stale-owner decision in persistence**

Modify `backend/app/services/market_activity_service.py` imports:

```python
from ..services.runtime_activity_staleness import is_stale_running_activity
```

Add helpers near `_load_market_activity`:

```python
def _record_from_payload(payload: dict[str, Any] | None) -> RuntimeActivityRecord | None:
    if not isinstance(payload, dict):
        return None
    try:
        return PersistedRuntimeActivity.from_payload(payload).to_record()
    except ValueError:
        return None


def _incoming_owner(payload: RuntimeActivityUpdate | RuntimeActivityRecord | dict[str, Any]) -> tuple[str | None, str | None, str | None]:
    if isinstance(payload, RuntimeActivityUpdate):
        return payload.task_id, payload.stage_key, payload.status
    if isinstance(payload, RuntimeActivityRecord):
        return payload.task_id, payload.stage_key, payload.status
    if isinstance(payload, dict):
        return payload.get("task_id"), payload.get("stage_key"), payload.get("status")
    return None, None, None


def _running_activity_has_live_owner(record: RuntimeActivityRecord) -> bool:
    if not record.task_id:
        return False
    try:
        current_task = get_data_fetch_lock().get_current_task(market=record.market)
    except Exception:
        return True
    return bool(current_task and current_task.get("task_id") == record.task_id)


def _should_override_stale_running_owner(
    existing: RuntimeActivityRecord | None,
    incoming_payload: RuntimeActivityUpdate | RuntimeActivityRecord | dict[str, Any],
) -> bool:
    if existing is None or existing.status != "running":
        return False
    incoming_task_id, incoming_stage_key, incoming_status = _incoming_owner(incoming_payload)
    if incoming_status not in {"running", "completed", "failed"}:
        return False
    if not incoming_task_id:
        return False
    if existing.task_id == incoming_task_id and existing.stage_key == incoming_stage_key:
        return False
    if not is_stale_running_activity(existing):
        return False
    return not _running_activity_has_live_owner(existing)
```

Then update `_save_market_activity`:

```python
    existing_record = _record_from_payload(existing_payload)
    allow_running_owner_override = _should_override_stale_running_owner(
        existing_record,
        payload,
    )
    transition = reduce_market_activity(
        existing_payload if isinstance(existing_payload, dict) else None,
        payload,
        allow_running_owner_override=allow_running_owner_override,
    )
```

- [ ] **Step 6: Run focused tests**

Run:

```bash
cd backend
source venv/bin/activate
pytest tests/unit/test_runtime_activity_reducer.py tests/unit/test_market_activity_service.py::test_stale_running_activity_allows_new_task_owner_to_start tests/unit/test_market_activity_service.py::test_stale_running_activity_does_not_override_when_old_owner_is_live -v
```

Expected: PASS.

- [ ] **Step 7: Commit Task 3**

```bash
git add backend/app/services/runtime_activity_reducer.py backend/app/services/market_activity_service.py backend/tests/unit/test_runtime_activity_reducer.py backend/tests/unit/test_market_activity_service.py
git commit -m "fix: let stale runtime activity owners be superseded safely"
```

## Task 4: Present Stale Runtime Activity as a Warning

**Files:**
- Modify: `backend/app/services/market_activity_service.py`
- Modify: `backend/app/services/runtime_activity_presenter.py`
- Modify: `frontend/src/components/Layout/RuntimeActivityStatusButton.jsx`
- Test: `backend/tests/unit/test_market_activity_service.py`
- Test: `frontend/src/components/Layout/Layout.test.jsx`

- [ ] **Step 1: Write failing backend read-side test**

Append to `backend/tests/unit/test_market_activity_service.py`:

```python
def test_runtime_activity_status_marks_orphaned_running_row_stale(db_session, monkeypatch):
    from app.services import market_activity_service as module
    from app.services.runtime_activity_contract import PersistedRuntimeActivity, RuntimeActivityRecord

    old_record = RuntimeActivityRecord.create(
        market="US",
        lifecycle="daily_refresh",
        stage_key="prices",
        status="running",
        task_name="app.tasks.cache_tasks.smart_refresh_cache",
        task_id="old-task",
        message="Refreshing market prices",
        updated_at="2026-06-23T05:00:00+00:00",
    )
    _persist_activity_row(db_session, PersistedRuntimeActivity.from_record(old_record).to_payload())
    monkeypatch.setattr(module, "get_data_fetch_lock", lambda: _FakeLock())
    monkeypatch.setattr(
        module,
        "get_runtime_bootstrap_status",
        lambda _db: _bootstrap_status(required=False, enabled=["US"], state="ready"),
    )

    payload = module.get_runtime_activity_status(db_session)

    us_market = next(item for item in payload["markets"] if item["market"] == "US")
    assert us_market["status"] == "stale"
    assert "No live data-fetch lock owns task old-task" in us_market["message"]
    assert payload["summary"]["status"] == "warning"
    assert payload["summary"]["active_market_count"] == 0
```

- [ ] **Step 2: Write failing header test**

Append to `frontend/src/components/Layout/Layout.test.jsx`:

```jsx
  it('shows stale runtime activity as a header warning', () => {
    useRuntimeActivityMock.mockReturnValue({
      dataUpdatedAt: 1,
      data: {
        bootstrap: { state: 'ready' },
        summary: { active_market_count: 0, status: 'warning' },
        markets: [
          {
            market: 'US',
            status: 'stale',
            stage_label: 'Price Refresh',
            message: 'Refreshing market prices - stale: No live data-fetch lock owns task old-task.',
          },
        ],
      },
    });

    renderWithProviders(
      <MemoryRouter initialEntries={['/scan']}>
        <Layout>
          <div>content</div>
        </Layout>
      </MemoryRouter>
    );

    expect(screen.getByText('Refresh warning')).toBeInTheDocument();
    expect(screen.getByText('US · Price Refresh')).toBeInTheDocument();
  });
```

- [ ] **Step 3: Run tests to verify they fail**

Run:

```bash
cd backend
source venv/bin/activate
pytest tests/unit/test_market_activity_service.py::test_runtime_activity_status_marks_orphaned_running_row_stale -v
cd ../frontend
npm run test:run -- src/components/Layout/Layout.test.jsx
```

Expected: backend returns `running`; frontend header returns `Markets ready`.

- [ ] **Step 4: Add read-side stale overlay**

Modify `backend/app/services/market_activity_service.py` imports:

```python
from ..services.runtime_activity_staleness import (
    is_stale_running_activity,
    stale_runtime_activity_payload,
)
```

Add helper:

```python
def _overlay_stale_runtime_activity(record: dict[str, Any], market: str) -> dict[str, Any]:
    try:
        typed_record = RuntimeActivityRecord.from_payload(record)
    except ValueError:
        return record
    if not is_stale_running_activity(typed_record):
        return record
    if _running_activity_has_live_owner(typed_record):
        return record
    reason = f"No live data-fetch lock owns task {typed_record.task_id or 'unknown'}."
    return stale_runtime_activity_payload(typed_record, reason=reason)
```

Update `_market_payload`:

```python
    if record is None:
        return _idle_market_payload(market, None)
    live_record = _overlay_live_progress(record, market)
    stale_checked_record = _overlay_stale_runtime_activity(live_record, market)
    return _idle_market_payload(market, stale_checked_record)
```

- [ ] **Step 5: Treat stale/stuck as warning summary states**

Modify `backend/app/services/runtime_activity_presenter.py`:

```python
WARNING_ACTIVITY_STATUSES = frozenset({"failed", "stale", "stuck"})
```

Then update:

```python
    has_warning = any(record.status in WARNING_ACTIVITY_STATUSES for record in activity_records)
    summary_status = "warning" if has_warning else ("active" if active_markets else "idle")
```

- [ ] **Step 6: Update header warning selection**

Modify `frontend/src/components/Layout/RuntimeActivityStatusButton.jsx`:

```jsx
  const activeMarket = markets.find((market) => (
    market.status === 'running' || market.status === 'queued'
  ));
  const warningMarket = markets.find((market) => (
    market.status === 'failed' || market.status === 'stale' || market.status === 'stuck'
  ));
```

Replace the warning branch:

```jsx
  if (summary.status === 'warning' && warningMarket) {
    return {
      badge: 'Warn',
      badgeColor: 'warning',
      title: 'Refresh warning',
      detail: `${warningMarket.market} · ${warningMarket.stage_label || warningMarket.message || 'Task needs attention'}`,
    };
  }
```

- [ ] **Step 7: Run focused backend and frontend tests**

Run:

```bash
cd backend
source venv/bin/activate
pytest tests/unit/test_market_activity_service.py::test_runtime_activity_status_marks_orphaned_running_row_stale -v
cd ../frontend
npm run test:run -- src/components/Layout/Layout.test.jsx
```

Expected: PASS.

- [ ] **Step 8: Commit Task 4**

```bash
git add backend/app/services/market_activity_service.py backend/app/services/runtime_activity_presenter.py frontend/src/components/Layout/RuntimeActivityStatusButton.jsx backend/tests/unit/test_market_activity_service.py frontend/src/components/Layout/Layout.test.jsx
git commit -m "fix: surface stale runtime activity as warnings"
```

## Task 5: Add Dependency-Free Planner Diagnostics

**Files:**
- Create: `backend/app/services/runtime_diagnostics.py`
- Modify: `backend/app/services/price_refresh_plan_builder.py`
- Test: `backend/tests/unit/test_runtime_diagnostics.py`
- Test: `backend/tests/unit/test_price_refresh_plan_builder_diagnostics.py`

- [ ] **Step 1: Write failing diagnostics helper tests**

Create `backend/tests/unit/test_runtime_diagnostics.py`:

```python
from __future__ import annotations


class _FakeLogger:
    def __init__(self) -> None:
        self.messages = []

    def info(self, message, *args, **kwargs):
        self.messages.append({"message": message, "args": args, "kwargs": kwargs})


def test_log_runtime_stage_emits_start_and_finish(monkeypatch):
    import app.services.runtime_diagnostics as module

    logger = _FakeLogger()
    monkeypatch.setattr(module.time, "perf_counter", iter([10.0, 12.5]).__next__)
    monkeypatch.setattr(module, "_max_rss_mb", lambda: 128.0)

    with module.log_runtime_stage(logger, "price_refresh.load_universe", market="US"):
        pass

    assert logger.messages[0]["message"] == "Runtime stage started: %s"
    assert logger.messages[0]["args"] == ("price_refresh.load_universe",)
    assert logger.messages[0]["kwargs"]["extra"]["runtime_stage"] == "price_refresh.load_universe"
    assert logger.messages[1]["message"] == "Runtime stage finished: %s"
    assert logger.messages[1]["args"] == ("price_refresh.load_universe",)
    assert logger.messages[1]["kwargs"]["extra"]["elapsed_seconds"] == 2.5
    assert logger.messages[1]["kwargs"]["extra"]["max_rss_mb"] == 128.0
```

- [ ] **Step 2: Write failing plan-builder diagnostic test**

Create `backend/tests/unit/test_price_refresh_plan_builder_diagnostics.py`:

```python
from __future__ import annotations

from datetime import date
from types import SimpleNamespace


def test_build_price_refresh_planning_input_logs_diagnostic_stages(monkeypatch):
    import app.services.price_refresh_plan_builder as module

    stages = []

    class _Stage:
        def __init__(self, _logger, name, **_extra):
            self.name = name

        def __enter__(self):
            stages.append(("start", self.name))

        def __exit__(self, exc_type, exc, tb):
            stages.append(("finish", self.name))

    class _StockUniverse:
        symbol = SimpleNamespace(in_=lambda _symbols: True)
        market = "US"
        is_active = True
        market_cap = SimpleNamespace(desc=lambda: SimpleNamespace(nullslast=lambda: "market_cap_desc"))

    class _Query:
        def filter(self, *_args):
            return self

        def order_by(self, *_args):
            return self

        def all(self):
            return [SimpleNamespace(symbol="AAPL", market="US")]

    class _Db:
        def query(self, *_args):
            return _Query()

    monkeypatch.setattr(module, "log_runtime_stage", _Stage)
    monkeypatch.setattr("app.models.stock_universe.StockUniverse", _StockUniverse)
    monkeypatch.setattr(
        module,
        "classify_price_history",
        lambda *_args, **_kwargs: SimpleNamespace(fresh=("AAPL",), stale=(), no_history=()),
    )

    module.build_price_refresh_planning_input(
        _Db(),
        mode="delta",
        market="US",
        effective_market="US",
        normalize_market=lambda market: str(market).upper(),
        market_calendar_service=SimpleNamespace(last_completed_trading_day=lambda _market: date(2026, 6, 18)),
        sync_github_seed=lambda *_args, **_kwargs: {"status": "missing"},
    )

    assert ("start", "price_refresh.load_universe") in stages
    assert ("start", "price_refresh.sync_github_seed") in stages
    assert ("start", "price_refresh.classify_coverage") in stages
```

- [ ] **Step 3: Run tests to verify they fail**

Run:

```bash
cd backend
source venv/bin/activate
pytest tests/unit/test_runtime_diagnostics.py tests/unit/test_price_refresh_plan_builder_diagnostics.py -v
```

Expected: FAIL because `runtime_diagnostics.py` does not exist and plan builder does not import `log_runtime_stage`.

- [ ] **Step 4: Implement diagnostics helper**

Create `backend/app/services/runtime_diagnostics.py`:

```python
"""Dependency-free runtime diagnostics for long-running worker stages."""

from __future__ import annotations

from contextlib import contextmanager
import resource
import sys
import time
from typing import Iterator


def _max_rss_mb() -> float:
    raw = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return round(raw / (1024 * 1024), 2)
    return round(raw / 1024, 2)


@contextmanager
def log_runtime_stage(logger, name: str, **extra) -> Iterator[None]:
    started = time.perf_counter()
    logger.info(
        "Runtime stage started: %s",
        name,
        extra={
            "runtime_stage": name,
            **extra,
        },
    )
    try:
        yield
    finally:
        elapsed = round(time.perf_counter() - started, 3)
        logger.info(
            "Runtime stage finished: %s",
            name,
            extra={
                "runtime_stage": name,
                "elapsed_seconds": elapsed,
                "max_rss_mb": _max_rss_mb(),
                **extra,
            },
        )
```

- [ ] **Step 5: Instrument price refresh planning**

Modify `backend/app/services/price_refresh_plan_builder.py`:

```python
import logging
```

```python
from .runtime_diagnostics import log_runtime_stage
```

```python
logger = logging.getLogger(__name__)
```

Update `build_price_refresh_planning_input`:

```python
    with log_runtime_stage(
        logger,
        "price_refresh.load_universe",
        market=effective_market,
        mode=parsed_mode.value,
    ):
        universe = extend_universe_with_key_market_symbols(
            load_active_price_refresh_universe(
                db,
                market=market,
                effective_market=effective_market,
                normalize_market=normalize_market,
            ),
            market,
            normalize_market,
        )
```

Wrap GitHub seed sync:

```python
    if parsed_mode in LIVE_TOP_UP_MODES and all_symbols and market is not None:
        with log_runtime_stage(
            logger,
            "price_refresh.sync_github_seed",
            market=effective_market,
            mode=parsed_mode.value,
            symbol_count=len(all_symbols),
        ):
            github_seed = GitHubSeedOutcome.from_mapping(
                sync_github_seed(db, market=effective_market, allow_stale=True)
            )
```

Wrap coverage classification:

```python
    if parsed_mode in LIVE_TOP_UP_MODES and all_symbols:
        target_as_of = market_calendar_service.last_completed_trading_day(effective_market)
        with log_runtime_stage(
            logger,
            "price_refresh.classify_coverage",
            market=effective_market,
            mode=parsed_mode.value,
            symbol_count=len(all_symbols),
            target_as_of=target_as_of.isoformat() if target_as_of else None,
        ):
            coverage = classify_price_history(db, symbols=all_symbols, as_of_date=target_as_of)
```

- [ ] **Step 6: Run focused diagnostics tests**

Run:

```bash
cd backend
source venv/bin/activate
pytest tests/unit/test_runtime_diagnostics.py tests/unit/test_price_refresh_plan_builder_diagnostics.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit Task 5**

```bash
git add backend/app/services/runtime_diagnostics.py backend/app/services/price_refresh_plan_builder.py backend/tests/unit/test_runtime_diagnostics.py backend/tests/unit/test_price_refresh_plan_builder_diagnostics.py
git commit -m "chore: add diagnostics around price refresh planning"
```

## Task 6: Final Verification and Issue Handoff

**Files:**
- No new files.
- Verify all files changed in Tasks 1 through 5.

- [ ] **Step 1: Run focused backend tests**

Run:

```bash
cd backend
source venv/bin/activate
pytest \
  tests/unit/test_runtime_activity_staleness.py \
  tests/unit/test_runtime_activity_failure_hooks.py \
  tests/unit/test_runtime_activity_reducer.py \
  tests/unit/test_market_activity_service.py \
  tests/unit/test_runtime_diagnostics.py \
  tests/unit/test_price_refresh_plan_builder_diagnostics.py \
  tests/unit/test_cache_refresh_unification.py::test_smart_refresh_cache_uses_runtime_activity_tracked_task_base \
  -v
```

Expected: PASS.

- [ ] **Step 2: Run focused frontend tests**

Run:

```bash
cd frontend
npm run test:run -- src/components/Layout/Layout.test.jsx src/pages/OperationsPage.test.jsx
```

Expected: PASS.

- [ ] **Step 3: Run quality gates**

Run:

```bash
cd backend
source venv/bin/activate
pytest tests/unit/test_runtime_activity_reducer.py tests/unit/test_market_activity_service.py tests/unit/test_cache_refresh_unification.py -v
cd ../frontend
npm run lint
npm run test:run -- src/components/Layout/Layout.test.jsx src/pages/OperationsPage.test.jsx
```

Expected: all commands PASS.

- [ ] **Step 4: Manual behavior check in local compose**

Run:

```bash
docker compose up -d backend redis postgres celery-datafetch
docker compose logs -f celery-datafetch
```

Trigger a US daily market pipeline or smart refresh from the UI. If a process is killed or a stale `runtime.activity.market.US` row is present, `/api/v1/app/runtime/activity` must return a market row with `status: "failed"` after hard worker failure publication or `status: "stale"` on read-side fallback.

Expected API shape:

```json
{
  "summary": {
    "active_market_count": 0,
    "status": "warning"
  },
  "markets": [
    {
      "market": "US",
      "stage_key": "prices",
      "status": "stale"
    }
  ]
}
```

- [ ] **Step 5: Create a follow-up bead if SIGKILL diagnostics identify OOM**

If logs show `price_refresh.load_universe` or `price_refresh.classify_coverage` max RSS rising sharply before worker loss, create:

```bash
bd create --title="Reduce memory pressure in price refresh planning" --type=bug --priority=1
```

Expected: a new bead id is printed. Include the observed stage name, elapsed seconds, and max RSS in the bead body with `bd update <id> --description`.

- [ ] **Step 6: Session completion**

Run:

```bash
git status
bd sync
git pull --rebase
bd sync
git push
git status
```

Expected: final `git status` reports the branch is up to date with origin and has no unstaged implementation changes.

## Self-Review

Spec coverage:
- Fix 1, hard worker failure capture, is covered by Task 2.
- Fix 2, stale-owner reconciliation, is covered by Task 3.
- Fix 3, read-side stale presentation, is covered by Task 4.
- Fix 4, SIGKILL investigation, is covered by Task 5.

Placeholder scan:
- No `TBD`, `TODO`, or unspecified implementation placeholders remain.
- Every code-changing step includes concrete snippets or complete file contents.

Type consistency:
- Runtime activity records consistently use `RuntimeActivityRecord`, `RuntimeActivityUpdate`, and persisted payload dictionaries.
- Market codes are normalized with `normalize_market`.
- Stale states are read-side presentation states and do not expand `ACTIVE_ACTIVITY_STATUSES`.

