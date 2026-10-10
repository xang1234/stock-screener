"""#472: unfenced legacy Theme writers skip, 409 or refuse under economic authority."""

from __future__ import annotations

import sys

import httpx
import pytest
from sqlalchemy import func, select, text
from sqlalchemy.exc import DatabaseError

from app.database import get_db
from app.main import app
from app.models.economic_taxonomy_runtime import TaxonomyAuthority
from app.models.theme import ThemeCluster
from app.services import theme_content_recovery_service as recovery
from app.services.legacy_theme_write_guard import (
    ECONOMIC_AUTHORITY_SKIP_REASON,
    LEGACY_WRITE_MODES,
    LegacyThemeWritesBlocked,
    legacy_theme_writes_blocked,
    mark_legacy_theme_writer,
    skip_in_economic_authority,
)
from app.tasks import (
    live_attachment_tasks,
    theme_discovery_tasks,
    theme_intelligence_tasks,
)


def _authority(db_session, mode, *, writes_fenced=False):
    db_session.add(TaxonomyAuthority(
        id=1, mode=mode, processing_head_revision=1, authority_epoch=1,
        writes_fenced=writes_fenced, semantic_invalidation_revision=0,
        cutover_catch_up_cursor=[], rollback_state="ready",
    ))
    db_session.add(ThemeCluster(
        canonical_key="ai_memory", display_name="AI Memory", name="AI Memory",
        pipeline="technical",
    ))
    db_session.commit()


def _clusters(db_session):
    return db_session.scalar(select(func.count()).select_from(ThemeCluster))


GUARDED_TASKS = [
    (theme_discovery_tasks.extract_themes, ()),
    (theme_discovery_tasks.reprocess_failed_themes, ()),
    (theme_discovery_tasks.calculate_theme_metrics, ()),
    (theme_discovery_tasks.recompute_stale_theme_embeddings, ()),
    (theme_discovery_tasks.promote_candidate_themes, ()),
    (theme_discovery_tasks.apply_lifecycle_policies, ()),
    (theme_discovery_tasks.infer_theme_relationships, ()),
    (theme_discovery_tasks.validate_themes, ()),
    (theme_discovery_tasks.check_alerts, ()),
    (theme_discovery_tasks.run_full_pipeline, ()),
    (theme_discovery_tasks.consolidate_themes, ()),
    (theme_discovery_tasks.compute_l1_metrics, ()),
    (theme_discovery_tasks.run_taxonomy_assignment, ()),
    (theme_discovery_tasks.recompute_l1_centroid_embeddings, ()),
    (live_attachment_tasks.refresh_attachment_themes, ([],)),
    (theme_intelligence_tasks.refresh_groups, ()),
]


@pytest.mark.parametrize("task,args", GUARDED_TASKS, ids=[t.name.rsplit(".", 1)[-1] for t, _ in GUARDED_TASKS])
def test_task_skips_with_a_reason_under_economic_authority(db_session, task, args):
    _authority(db_session, "economic")
    before = _clusters(db_session)

    result = task(*args)

    assert result["status"] == "skipped"
    assert result["reason"] == ECONOMIC_AUTHORITY_SKIP_REASON
    assert _clusters(db_session) == before


def _switch_to_economic():
    from app.database import SessionLocal

    with SessionLocal() as other:
        other.get(TaxonomyAuthority, 1).mode = "economic"
        other.commit()


def test_task_write_after_a_mid_run_cutover_fails_closed(db_session):
    # The entry check passes in legacy mode; cutover happens before the body
    # commits, so its first write re-checks under the fence and is refused,
    # and the task is skipped as if the entry check had caught it.
    _authority(db_session, "legacy")
    from app.database import SessionLocal

    skipped = []

    @skip_in_economic_authority(on_skip=lambda run_id: skipped.append(run_id))
    def body(run_id):
        _switch_to_economic()
        with SessionLocal() as session:
            session.add(ThemeCluster(canonical_key="late", display_name="Late",
                                     name="Late", pipeline="technical"))
            session.commit()

    result = body("run-1")

    assert result["reason"] == ECONOMIC_AUTHORITY_SKIP_REASON
    assert skipped == ["run-1"]
    db_session.expire_all()
    assert _clusters(db_session) == 1


def test_task_that_swallows_the_fence_is_still_skipped(db_session):
    # Many task bodies catch every exception and return an error payload.
    _authority(db_session, "legacy")
    from app.database import SessionLocal

    skipped = []

    @skip_in_economic_authority(on_skip=lambda: skipped.append(True))
    def body():
        _switch_to_economic()
        try:
            with SessionLocal() as session:
                session.add(ThemeCluster(canonical_key="late", display_name="Late",
                                         name="Late", pipeline="technical"))
                session.commit()
        except Exception as exc:  # noqa: BLE001 - mimics task bodies
            return {"status": "error", "error": str(exc)}

    result = body()

    assert result["reason"] == ECONOMIC_AUTHORITY_SKIP_REASON
    assert skipped == [True]


def test_task_errors_unrelated_to_the_fence_still_raise(db_session):
    _authority(db_session, "legacy")

    @skip_in_economic_authority
    def body():
        raise ValueError("boom")

    with pytest.raises(ValueError):
        body()


@pytest.mark.asyncio
async def test_route_write_fenced_mid_request_returns_409(db_session, monkeypatch):
    # The entry check passes in legacy mode; cutover lands before the first
    # write, so the fence refuses it with the same 409 as the entry check.
    import uuid
    from types import SimpleNamespace

    from app.api.v1 import themes_content_pipeline
    from app.services import server_auth

    monkeypatch.setattr(server_auth.settings, "server_auth_enabled", False)
    _authority(db_session, "legacy")

    def cutover_then_uuid4():
        _switch_to_economic()
        return uuid.uuid4()

    monkeypatch.setattr(themes_content_pipeline, "uuid", SimpleNamespace(uuid4=cutover_then_uuid4))
    app.dependency_overrides[get_db] = lambda: db_session
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post("/api/v1/themes/pipeline/run")
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert response.status_code == 409, response.text
    assert response.json()["detail"]["code"] == "economic_generation_endpoint_required"


def test_pipeline_run_fenced_mid_run_is_recorded_as_skipped(db_session, monkeypatch):
    # Its own `failed` write is refused by the same fence, so the skip hook
    # must give the run its terminal state.
    from app.models.theme import ThemePipelineRun

    _authority(db_session, "legacy")
    db_session.add(ThemePipelineRun(run_id="run-1", pipeline="technical", status="queued"))
    db_session.commit()

    def cutover_then_ingest(*args, **kwargs):
        _switch_to_economic()
        return {}

    monkeypatch.setattr(theme_discovery_tasks, "ingest_content", cutover_then_ingest)
    monkeypatch.setattr(theme_discovery_tasks.run_full_pipeline, "update_state", lambda **kw: None)

    result = theme_discovery_tasks.run_full_pipeline(run_id="run-1", pipeline="technical")

    db_session.expire_all()
    run = db_session.query(ThemePipelineRun).filter_by(run_id="run-1").one()
    assert result["reason"] == ECONOMIC_AUTHORITY_SKIP_REASON
    assert run.status == "skipped"


def test_marked_request_session_write_fails_closed_after_cutover(db_session):
    _authority(db_session, "legacy")
    mark_legacy_theme_writer(db_session)
    _switch_to_economic()

    db_session.add(ThemeCluster(canonical_key="late", display_name="Late",
                                name="Late", pipeline="technical"))
    with pytest.raises(LegacyThemeWritesBlocked):
        db_session.flush()


def test_fence_rechecks_after_a_rolled_back_savepoint(db_session):
    # PostgreSQL drops a lock taken inside a savepoint when it rolls back, so
    # the next write in the outer transaction must take the fence again.
    _authority(db_session, "legacy")
    mark_legacy_theme_writer(db_session)
    savepoint = db_session.begin_nested()
    db_session.add(ThemeCluster(canonical_key="first", display_name="First",
                                name="First", pipeline="technical"))
    db_session.flush()
    savepoint.rollback()
    # Raw connection write: bypasses the ORM hooks, same transaction.
    db_session.connection().execute(
        text("UPDATE taxonomy_authority SET mode = 'economic' WHERE id = 1")
    )

    db_session.add(ThemeCluster(canonical_key="second", display_name="Second",
                                name="Second", pipeline="technical"))
    with pytest.raises(LegacyThemeWritesBlocked):
        db_session.flush()


def test_unmarked_sessions_are_not_fenced(db_session):
    # Economic producers and ingestion keep writing shared tables.
    _authority(db_session, "economic")

    db_session.add(ThemeCluster(canonical_key="other", display_name="Other",
                                name="Other", pipeline="technical"))
    db_session.flush()


@pytest.mark.parametrize("mode", sorted(LEGACY_WRITE_MODES))
def test_task_guard_runs_the_body_in_legacy_write_modes(db_session, mode):
    _authority(db_session, mode)

    @skip_in_economic_authority
    def body(value):
        return {"status": "ran", "value": value}

    assert body(3) == {"status": "ran", "value": 3}


GUARDED_ROUTES = [
    ("POST", "/pipeline/run"),
    ("POST", "/extract"),
    ("POST", "/calculate-metrics"),
    ("POST", "/validate-all"),
    ("POST", "/equivalence"),
    ("POST", "/equivalence/op-1/undo"),
    ("POST", "/alerts/1/dismiss"),
    ("POST", "/alerts/1/read"),
    ("POST", "/alerts/check"),
    ("POST", "/merge-suggestions/1/approve"),
    ("POST", "/merge-suggestions/1/reject"),
    ("POST", "/consolidate"),
    ("POST", "/consolidate/async"),
    ("POST", "/embeddings/refresh-campaign"),
    ("POST", "/merge-wave/strict-auto"),
    ("POST", "/merge-wave/manual-review"),
    ("POST", "/candidates/review"),
    ("POST", "/create-from-cluster"),
    ("POST", "/1/add-constituents"),
    ("DELETE", "/1"),
    ("POST", "/taxonomy/assign"),
    ("POST", "/taxonomy/assign/async"),
    ("PUT", "/taxonomy/1/reassign"),
    ("GET", "/1/validate"),
    ("GET", "/1/similar"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("method,path", GUARDED_ROUTES, ids=[f"{m} {p}" for m, p in GUARDED_ROUTES])
async def test_route_returns_409_under_economic_authority(db_session, monkeypatch, method, path):
    from app.api.v1.config import settings as config_settings
    from app.services import server_auth

    monkeypatch.setattr(server_auth.settings, "server_auth_enabled", False)
    # Admin-only routes authenticate before the authority check.
    monkeypatch.setattr(config_settings, "admin_api_key", "admin-secret")
    _authority(db_session, "economic")
    before = _clusters(db_session)
    app.dependency_overrides[get_db] = lambda: db_session
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.request(
                method, f"/api/v1/themes{path}", json={},
                headers={"X-Admin-Key": "admin-secret"},
            )
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert response.status_code == 409, response.text
    assert response.json()["detail"]["code"] == "economic_generation_endpoint_required"
    assert _clusters(db_session) == before


@pytest.mark.asyncio
async def test_rankings_read_skips_metric_recalculation_while_writes_are_fenced(db_session, monkeypatch):
    # A read endpoint: it still serves stored metrics instead of returning 409.
    from app.services import server_auth
    from app.services.theme_discovery_service import ThemeDiscoveryService

    monkeypatch.setattr(server_auth.settings, "server_auth_enabled", False)
    _authority(db_session, "legacy", writes_fenced=True)
    recalculated = []
    monkeypatch.setattr(ThemeDiscoveryService, "update_all_theme_metrics",
                        lambda self: recalculated.append(True))
    app.dependency_overrides[get_db] = lambda: db_session
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/api/v1/themes/rankings?recalculate=true")
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert response.status_code == 200, response.text
    assert recalculated == []


def test_skipped_pipeline_run_is_recorded_as_skipped(db_session):
    from app.models.theme import ThemePipelineRun

    _authority(db_session, "economic")
    db_session.add(ThemePipelineRun(run_id="run-1", pipeline="technical", status="queued"))
    db_session.commit()

    result = theme_discovery_tasks.run_full_pipeline(run_id="run-1", pipeline="technical")

    db_session.expire_all()
    run = db_session.query(ThemePipelineRun).filter_by(run_id="run-1").one()
    assert result["reason"] == ECONOMIC_AUTHORITY_SKIP_REASON
    assert run.status == "skipped"
    assert run.completed_at is not None
    assert ECONOMIC_AUTHORITY_SKIP_REASON in run.error_message


@pytest.mark.asyncio
async def test_pipeline_run_is_recorded_as_queued_before_dispatch(db_session, monkeypatch):
    # No fenced write follows the dispatch, so a cutover after it cannot leave
    # a queued task with an untracked run row.
    from app.database import SessionLocal
    from app.models.theme import ThemePipelineRun
    from app.services import server_auth

    monkeypatch.setattr(server_auth.settings, "server_auth_enabled", False)
    _authority(db_session, "legacy")
    seen = {}

    def fake_apply_async(*, kwargs, task_id):
        with SessionLocal() as other:
            row = other.query(ThemePipelineRun).filter_by(run_id=kwargs["run_id"]).one()
            seen.update(status=row.status, task_id=row.task_id, dispatched=task_id)

    monkeypatch.setattr(theme_discovery_tasks.run_full_pipeline, "apply_async", fake_apply_async)
    app.dependency_overrides[get_db] = lambda: db_session
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post("/api/v1/themes/pipeline/run")
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert response.status_code == 200, response.text
    assert seen["status"] == "queued"
    assert seen["task_id"] == seen["dispatched"] == response.json()["task_id"]


@pytest.mark.asyncio
async def test_failed_dispatch_after_cutover_is_recorded_as_failed(db_session, monkeypatch):
    # The request session is fenced; the failed status must still land, or the
    # run stays queued for a task that was never sent.
    from app.models.theme import ThemePipelineRun
    from app.services import server_auth

    monkeypatch.setattr(server_auth.settings, "server_auth_enabled", False)
    _authority(db_session, "legacy")

    def cutover_then_fail(*, kwargs, task_id):
        _switch_to_economic()
        raise RuntimeError("broker down")

    monkeypatch.setattr(theme_discovery_tasks.run_full_pipeline, "apply_async", cutover_then_fail)
    app.dependency_overrides[get_db] = lambda: db_session
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post("/api/v1/themes/pipeline/run")
    finally:
        app.dependency_overrides.pop(get_db, None)

    assert response.status_code == 503, response.text
    db_session.expire_all()
    run = db_session.query(ThemePipelineRun).one()
    assert run.status == "failed"
    assert "broker down" in run.error_message


def _load(path):
    import importlib
    import importlib.util
    from pathlib import Path

    if path.startswith("app/"):  # a package module (e.g. one defining dataclasses)
        return importlib.import_module(path[:-3].replace("/", "."))
    full = Path(__file__).resolve().parents[2] / path
    spec = importlib.util.spec_from_file_location(full.stem, full)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GUARDED_CLIS = [
    ("scripts/backfill_l1_taxonomy.py", []),
    ("scripts/backfill_silent_failures.py", []),
    ("scripts/backfill_theme_aliases.py", ["--yes"]),
    ("app/scripts/repair_jp_alpha_universe_symbols.py", ["--apply"]),
]


@pytest.mark.parametrize("path,argv", GUARDED_CLIS, ids=[p.rsplit("/", 1)[-1] for p, _ in GUARDED_CLIS])
def test_cli_refuses_to_write_under_economic_authority(db_session, monkeypatch, capsys, path, argv):
    _authority(db_session, "economic")
    module = _load(path)
    for name in ("initialize_process_runtime_services",):
        if hasattr(module, name):
            monkeypatch.setattr(module, name, lambda: None)
    if hasattr(module, "get_session_factory"):
        monkeypatch.setattr(module, "get_session_factory", lambda: lambda: db_session)
    monkeypatch.setattr(sys, "argv", [path, *argv])

    with pytest.raises(SystemExit) as exit_info:
        module.main()

    assert exit_info.value.code == 1
    assert "STOP:" in capsys.readouterr().err
    assert _clusters(db_session) == 1


def test_content_storage_reset_is_refused_under_economic_authority(db_session, monkeypatch):
    _authority(db_session, "economic")
    monkeypatch.setattr(recovery, "_reset_blocked_by_authority", lambda conn: True)
    dropped = []
    monkeypatch.setattr(recovery, "drop_theme_content_tables", lambda conn: dropped.append(conn))

    # Its own type: corruption is a 5xx incident, not the API's cutover 409.
    with pytest.raises(recovery.ThemeContentResetRefused) as refused:
        recovery.reset_corrupt_theme_content_storage(
            DatabaseError("SELECT 1", {}, Exception("database disk image is malformed"))
        )

    assert not isinstance(refused.value, LegacyThemeWritesBlocked)
    assert dropped == []


def test_reset_check_reads_the_authority_mode(db_session):
    _authority(db_session, "economic")

    assert recovery._reset_blocked_by_authority(db_session.connection()) is True


# Legacy-only readers with no economic counterpart: refused, not routed (#557).
REFUSED_READS = [
    "/merge-suggestions",
    "/merge-history",
    "/merge-plan/dry-run",
    "/candidates/queue",
    "/relationship-graph",
    "/equivalence/preview?source_id=1&target_id=2",
    "/equivalence/history",
    "/equivalence/search?q=ai",
    "/matching/telemetry",
    "/pipeline/state-health",
    "/pipeline/observability",
]


async def _get(db_session, path):
    app.dependency_overrides[get_db] = lambda: db_session
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            return await client.get(f"/api/v1/themes{path}")
    finally:
        app.dependency_overrides.pop(get_db, None)


@pytest.mark.asyncio
@pytest.mark.parametrize("path", REFUSED_READS)
async def test_legacy_only_read_returns_409_under_economic_authority(db_session, monkeypatch, path):
    from app.services import server_auth

    monkeypatch.setattr(server_auth.settings, "server_auth_enabled", False)
    _authority(db_session, "economic")

    response = await _get(db_session, path)

    assert response.status_code == 409, response.text
    assert response.json()["detail"]["code"] == "economic_generation_endpoint_required"


@pytest.mark.asyncio
async def test_legacy_only_read_still_serves_while_rollback_recovery_fences_writes(
    db_session, monkeypatch
):
    # Reading legacy rows is harmless while writes are fenced; only economic
    # authority, where nothing maintains them, refuses the read.
    from app.services import server_auth

    monkeypatch.setattr(server_auth.settings, "server_auth_enabled", False)
    _authority(db_session, "legacy", writes_fenced=True)

    response = await _get(db_session, "/merge-history")

    assert response.status_code == 200, response.text


@pytest.mark.parametrize("mode", sorted(LEGACY_WRITE_MODES))
def test_writes_fenced_blocks_legacy_writers_in_every_mode(db_session, mode):
    # Rollback recovery fences writes without leaving a legacy write mode.
    _authority(db_session, mode, writes_fenced=True)

    assert legacy_theme_writes_blocked(db_session) is True
