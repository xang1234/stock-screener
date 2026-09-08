from datetime import datetime, timezone
import importlib.util
import os

import pytest


def test_source_service_exists():
    assert importlib.util.find_spec("app.services.social_source_admin_service"), "registry service missing"


def test_pending_test_enable_and_stale_version(db_session):
    from app.services.social_source_admin_service import SocialSourceAdminService, SocialSourceStateError, SocialSourceVersionError
    from app.domain.social_signals.records import SourceTestOutcome
    service = SocialSourceAdminService(db_session)
    service.ensure_seed_sources()
    runtime = service.read_runtime()
    service.apply_runtime("validation", "official", runtime.version, "admin")
    source = service.create_source(" Japan ", "3001", "admin")
    assert (source.lifecycle, source.name, source.canonical_url) == ("pending", "Japan", "https://x.com/i/lists/3001")
    with pytest.raises(SocialSourceStateError, match="current_provider_test_required"):
        service.transition_source(source.source_id, "enabled", source.version, "admin")
    request = service.request_test(source.source_id, source.version, "admin")
    tested = service.record_test_result(source.source_id, "official", SourceTestOutcome("official", "passed", 5, datetime.now(timezone.utc)), "admin", request_id=request.request_id, expected_version=request.version)
    with pytest.raises(SocialSourceVersionError):
        service.record_test_result(source.source_id, "official", SourceTestOutcome("official", "passed", 5, datetime.now(timezone.utc)), "admin", request_id=request.request_id, expected_version=request.version)
    enabled = service.transition_source(tested.source_id, "enabled", tested.version, "admin")
    assert enabled.lifecycle == "enabled"
    with pytest.raises(SocialSourceVersionError):
        service.rename_source(enabled.source_id, "New", tested.version, "admin")
    assert [event.action for event in service.audit_events(source.source_id)] == ["created", "test_requested", "test_completed", "enabled"]


def test_provider_diagnostic_can_run_while_social_publishing_is_off(db_session):
    from app.services.social_source_admin_service import SocialSourceAdminService

    service = SocialSourceAdminService(db_session)
    service.ensure_seed_sources()
    runtime = service.read_runtime()
    service.apply_runtime("off", "official", runtime.version, "admin")
    source = service.create_source("Diagnostic list", "3002", "admin")

    requested = service.request_test(source.source_id, source.version, "admin")
    claimed = service.claim_test(source.source_id, "worker")

    assert claimed.request_id == requested.request_id
    assert claimed.provider == "official"


def prepare_service(db):
    from app.services.social_source_admin_service import SocialSourceAdminService
    service = SocialSourceAdminService(db)
    service.ensure_seed_sources()
    service.apply_runtime("validation", "official", service.read_runtime().version, "admin")
    return service


def enable_third(service):
    from app.domain.social_signals.records import SourceTestOutcome
    source = service.create_source("Third", "3001", "admin")
    request = service.request_test(source.source_id, source.version, "admin")
    tested = service.record_test_result(source.source_id, "official", SourceTestOutcome("official", "passed", 1, datetime.now(timezone.utc)), "admin", request_id=request.request_id, expected_version=request.version)
    return service.transition_source(source.source_id, "enabled", tested.version, "admin")


@pytest.fixture
def registry_engine(tmp_path):
    from sqlalchemy import create_engine
    from sqlalchemy.engine import make_url
    from app.database import Base
    from app.models.theme import ContentSource
    from app.infra.db.models.social_signals import SocialSourceRegistry, SocialSourceConfiguration, SocialSourceAuditEvent
    url = os.environ.get("SOCIAL_REGISTRY_DISPOSABLE_POSTGRES_URL")
    if url:
        parsed = make_url(url)
        assert parsed.host == "127.0.0.1" and parsed.database == "social_registry_test"
        engine = create_engine(url)
    else:
        engine = create_engine(f"sqlite:///{tmp_path / 'concurrency.db'}", connect_args={"timeout": 10})
    tables = [ContentSource.__table__, SocialSourceRegistry.__table__, SocialSourceConfiguration.__table__, SocialSourceAuditEvent.__table__]
    Base.metadata.create_all(engine, tables=tables)
    try:
        yield engine
    finally:
        Base.metadata.drop_all(engine, tables=tables)
        engine.dispose()


@pytest.mark.parametrize("other_target", ["disabled", "archived"])
def test_concurrent_transactions_preserve_minimum_two(registry_engine, other_target):
    """Without the shared lock, both transactions can observe three and remove one."""
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    from sqlalchemy.orm import sessionmaker
    from app.services.social_source_admin_service import SocialSourceAdminService, SocialSourceStateError
    factory = sessionmaker(bind=registry_engine)
    with factory() as db:
        service = prepare_service(db)
        third = enable_third(service)
        first = service.list_sources()[0]
    barrier = Barrier(2)

    def mutate(source, target):
        with factory() as db:
            barrier.wait(timeout=5)
            try:
                SocialSourceAdminService(db).transition_source(source.source_id, target, source.version, "admin")
                return "accepted"
            except SocialSourceStateError as exc:
                return str(exc)

    with ThreadPoolExecutor(max_workers=2) as workers:
        futures = [workers.submit(mutate, first, "disabled"), workers.submit(mutate, third, other_target)]
        assert sorted(future.result(timeout=15) for future in futures) == ["accepted", "minimum_two_enabled"]
    with factory() as db:
        service = SocialSourceAdminService(db)
        assert sum(source.lifecycle == "enabled" for source in service.list_sources(True)) == 2
        actions = [event.action for source in (first, third) for event in service.audit_events(source.source_id)]
        assert sum(action in {"disabled", "archived"} for action in actions) == 1


def test_concurrent_stale_versions_are_checked_after_lock(registry_engine):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    from sqlalchemy.orm import sessionmaker
    from app.services.social_source_admin_service import SocialSourceAdminService, SocialSourceVersionError
    factory = sessionmaker(bind=registry_engine)
    with factory() as db:
        service = prepare_service(db)
        source = service.list_sources()[0]
    barrier = Barrier(2)

    def rename(name):
        with factory() as db:
            barrier.wait(timeout=5)
            try:
                SocialSourceAdminService(db).rename_source(source.source_id, name, source.version, "admin")
                return "accepted"
            except SocialSourceVersionError:
                return "stale"

    with ThreadPoolExecutor(max_workers=2) as workers:
        futures = [workers.submit(rename, name) for name in ("One", "Two")]
        assert sorted(f.result(timeout=15) for f in futures) == ["accepted", "stale"]
    with factory() as db:
        assert [e.action for e in SocialSourceAdminService(db).audit_events(source.source_id)] == ["created", "renamed"]


def test_rollback_releases_registry_lock_for_another_transaction(registry_engine):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    from sqlalchemy.orm import sessionmaker
    from app.services.social_source_admin_service import SocialSourceAdminService
    factory = sessionmaker(bind=registry_engine)
    with factory() as db:
        source = prepare_service(db).list_sources()[0]
    entered = Event()

    def rename():
        with factory() as db:
            entered.set()
            return SocialSourceAdminService(db).rename_source(source.source_id, "After rollback", source.version, "admin")

    with factory() as db, ThreadPoolExecutor(max_workers=1) as workers:
        with pytest.raises(RuntimeError, match="rollback"):
            with SocialSourceAdminService(db)._transaction(lock=True):
                future = workers.submit(rename)
                assert entered.wait(timeout=5)
                assert not future.done()
                raise RuntimeError("rollback")
        assert future.result(timeout=15).name == "After rollback"


@pytest.mark.parametrize("intervening", ["rename", "disable", "archive", "new_test", "runtime"])
def test_late_test_result_cannot_overwrite_admin_changes(db_session, intervening):
    from app.services.social_source_admin_service import SocialSourceStateError
    from app.domain.social_signals.records import SourceTestOutcome
    service = prepare_service(db_session)
    source = service.create_source("Japan", "3001", "admin")
    request = service.request_test(source.source_id, source.version, "admin")
    if intervening == "rename":
        service.rename_source(source.source_id, "New", request.version, "admin")
    elif intervening in {"disable", "archive"}:
        service.transition_source(source.source_id, {"disable": "disabled", "archive": "archived"}[intervening], request.version, "admin")
    elif intervening == "new_test":
        service.request_test(source.source_id, request.version, "admin")
    else:
        service.apply_runtime("live", "xui", service.read_runtime().version, "admin")
    with pytest.raises(SocialSourceStateError):
        service.record_test_result(source.source_id, "official", SourceTestOutcome("official", "passed", 5, datetime.now(timezone.utc)), "admin", request_id=request.request_id, expected_version=request.version)
    assert "test_completed" not in [event.action for event in service.audit_events(source.source_id)]


def test_failed_mutation_rolls_back_and_seed_is_idempotent(db_session):
    from app.services.social_source_admin_service import SocialSourceStateError
    service = prepare_service(db_session)
    first = service.list_sources()[0]
    with pytest.raises(SocialSourceStateError, match="invalid_actor"):
        service.rename_source(first.source_id, "Lost edit", first.version, "")
    renamed = service.rename_source(first.source_id, "Kept edit", first.version, "admin")
    assert service.ensure_seed_sources()[0].name == "Kept edit"
    assert service.ensure_seed_sources()[0].version == renamed.version
    assert len(service.audit_events(first.source_id)) == 2


def test_explicit_test_can_run_for_enabled_sources(db_session):
    service = prepare_service(db_session)
    enabled = service.list_sources()[0]

    requested = service.request_test(enabled.source_id, enabled.version, "admin")
    claimed = service.claim_test(enabled.source_id, "worker")

    assert claimed.request_id == requested.request_id
    assert service.list_sources()[0].lifecycle == "enabled"


def test_caller_flushed_transaction_is_never_committed(db_session):
    from app.models.theme import ContentSource
    from app.services.social_source_admin_service import SocialSourceStateError, SocialSourceAdminService
    source = ContentSource(name="Unrelated", source_type="news")
    db_session.add(source)
    db_session.flush()
    with pytest.raises(SocialSourceStateError, match="caller_transaction_active"):
        SocialSourceAdminService(db_session).ensure_seed_sources()
    db_session.rollback()
    assert db_session.query(ContentSource).count() == 0


def test_official_capacity_is_durable_bounded_and_does_not_change_policy_version(registry_engine):
    from datetime import date
    from sqlalchemy.orm import sessionmaker
    from app.services.social_source_admin_service import SocialSourceAdminService
    factory = sessionmaker(bind=registry_engine)
    with factory() as db:
        service = SocialSourceAdminService(db)
        service.ensure_seed_sources()
        version = service.read_runtime().version
        assert service.reserve_official_capacity(date(2026, 9, 7), 6, 10) == 6
    with factory() as restarted:
        service = SocialSourceAdminService(restarted)
        assert service.reserve_official_capacity(date(2026, 9, 7), 6, 10) == 0
        assert service.reserve_official_capacity(date(2026, 9, 7), 1, 10) == 0
        assert service.read_runtime().version == version


def test_prior_day_cannot_reset_newer_official_capacity_day(registry_engine):
    from datetime import date
    from sqlalchemy.orm import sessionmaker
    from app.services.social_source_admin_service import SocialSourceAdminService
    factory = sessionmaker(bind=registry_engine)
    with factory() as db:
        service = SocialSourceAdminService(db); service.ensure_seed_sources()
        assert service.reserve_official_capacity(date(2026, 9, 8), 7, 10) == 7
        assert service.reserve_official_capacity(date(2026, 9, 7), 3, 10) == 0
        assert service.reserve_official_capacity(date(2026, 9, 8), 5, 10) == 0


def test_concurrent_official_capacity_reservations_never_exceed_daily_limit(registry_engine):
    from concurrent.futures import ThreadPoolExecutor
    from datetime import date
    from threading import Barrier
    from sqlalchemy.orm import sessionmaker
    from app.services.social_source_admin_service import SocialSourceAdminService
    factory = sessionmaker(bind=registry_engine)
    with factory() as db: SocialSourceAdminService(db).ensure_seed_sources()
    barrier = Barrier(2)
    def reserve():
        with factory() as db:
            barrier.wait(timeout=5)
            return SocialSourceAdminService(db).reserve_official_capacity(date(2026, 9, 7), 8, 10)
    with ThreadPoolExecutor(max_workers=2) as workers:
        grants = [future.result(timeout=15) for future in (workers.submit(reserve), workers.submit(reserve))]
    assert sorted(grants) == [0, 8]


def test_runtime_read_does_not_apply_environment(db_session, monkeypatch):
    from app.services.social_source_admin_service import SocialSourceAdminService
    monkeypatch.setenv("SOCIAL_SIGNALS_MODE", "live")
    service = SocialSourceAdminService(db_session)
    assert service.read_runtime().mode == "off"
    service.ensure_seed_sources()
    assert service.read_runtime().provider == "disabled"


@pytest.mark.parametrize("ref", ["", "1" * 33, "https://twitter.com/i/lists/3001", "https://x.com/i/lists/3001?x=1", "https://x.com/i/lists/3001/", "-1", "abc"])
def test_create_rejects_unapproved_list_refs(db_session, ref):
    from app.services.social_source_admin_service import SocialSourceStateError
    service = prepare_service(db_session)
    with pytest.raises(SocialSourceStateError, match="invalid_list_ref"):
        service.create_source("Valid name", ref, "admin")
    assert len(service.list_sources()) == 2


def test_normalized_list_identity_rejects_duplicate_decimal_id(db_session):
    from app.services.social_source_admin_service import SocialSourceStateError
    service = prepare_service(db_session)
    assert service.create_source("Japan", "03001", "admin").list_id == "3001"
    with pytest.raises(SocialSourceStateError, match="duplicate_list_id"):
        service.create_source("Duplicate", "https://x.com/i/lists/3001", "admin")


def test_admin_generation_changes_once_and_noops_preserve_it(db_session):
    service = prepare_service(db_session)
    initial = service.read_runtime()
    source = service.create_source("Japan", "3001", "admin")
    assert service.read_runtime().version == initial.version + 1
    source = service.rename_source(source.source_id, "Renamed", source.version, "admin")
    assert service.read_runtime().version == initial.version + 2
    service.rename_source(source.source_id, "Renamed", source.version, "admin")
    service.ensure_seed_sources()
    runtime = service.read_runtime()
    service.apply_runtime(runtime.mode, runtime.provider, runtime.version, "admin")
    assert service.read_runtime().version == initial.version + 2


def test_audits_are_immutable_and_outcomes_are_redacted(db_session):
    from sqlalchemy import update, delete
    from app.domain.social_signals.records import SourceTestOutcome
    from app.infra.db.models.social_signals import SocialSourceAuditEvent
    service = prepare_service(db_session)
    source = service.create_source("Japan", "3001", "admin")
    request = service.request_test(source.source_id, source.version, "admin")
    outcome = SourceTestOutcome("official", "failed", 0, datetime.now(timezone.utc), reason_code="BODY_AND_SECRET_MUST_NOT_APPEAR")
    service.record_test_result(source.source_id, "official", outcome, "admin", request_id=request.request_id, expected_version=request.version)
    assert "BODY_AND_SECRET" not in repr(service.audit_events(source.source_id))
    for statement in (update(SocialSourceAuditEvent).values(actor="tampered"), delete(SocialSourceAuditEvent)):
        with pytest.raises(ValueError, match="social_audit_append_only"):
            db_session.execute(statement)
        db_session.rollback()
    row = db_session.query(SocialSourceAuditEvent).first()
    row.actor = "tampered"
    with pytest.raises(ValueError, match="social_audit_append_only"):
        db_session.flush()
    db_session.rollback()
