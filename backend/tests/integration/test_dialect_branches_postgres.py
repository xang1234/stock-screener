"""PostgreSQL side of production dialect branches (issue #431).

Production runs PostgreSQL, but these branches were only ever exercised on
SQLite: the JSONB path delete in the feature store, the COT advisory locks and
upsert, ``ON CONFLICT DO NOTHING`` eligibility grants, and the Social registry
``FOR UPDATE`` locks. CI runs this file in the PostgreSQL integration sweep.
"""

from __future__ import annotations

import threading
import time
from datetime import date, datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine, event, func, select, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

from app.database import engine
from app.domain.common.query import SortOrder, SortSpec
from app.domain.feature_store.models import FeatureRowWrite
from app.domain.scanning.filter_expression_model import FilterExpression
from app.infra.db.models.feature_store import FeatureRun
from app.infra.db.models.social_signals import ContentPipelineEligibility, SocialSourceRegistry
from app.infra.db.repositories.cot_repository import (
    _COT_PUBLICATION_LOCK_ID,
    _COT_REFRESH_LOCK_ID,
    SqlCotRepository,
)
from app.infra.db.repositories.feature_store_repo import SqlFeatureStoreRepository
from app.models.theme import ContentItem, ContentSource

pytestmark = pytest.mark.skipif(
    engine.dialect.name != "postgresql", reason="exercises PostgreSQL-only SQL"
)


def test_feature_store_strips_setup_payload_with_jsonb_path_delete(db_session):
    run = FeatureRun(as_of_date=date(2026, 2, 17), run_type="daily_snapshot", status="completed")
    db_session.add(run)
    db_session.flush()
    repo = SqlFeatureStoreRepository(db_session)
    repo.upsert_snapshot_rows(run.id, [
        FeatureRowWrite(
            symbol="AAPL",
            as_of_date=date(2026, 2, 17),
            composite_score=90.0,
            overall_rating=5,
            passes_count=3,
            details={
                "rating": "Strong Buy",
                "setup_engine": {
                    "setup_score": 82.0,
                    "pattern_primary": "VCP",
                    "explain": {"summary": "large payload"},
                    "candidates": [{"pattern": "base"}],
                },
            },
        )
    ])

    statements: list[str] = []

    def capture(_conn, _cursor, statement, *_args):
        statements.append(statement.lower())

    event.listen(engine, "before_cursor_execute", capture)
    try:
        [item] = repo.query_all_as_scan_results(
            run.id,
            FilterExpression(),
            SortSpec(field="composite_score", order=SortOrder.DESC),
            include_setup_payload=False,
        )
    finally:
        event.remove(engine, "before_cursor_execute", capture)

    # The mapper also drops these fields, so check the SQL projection itself:
    # both JSONB path deletes must apply, and nothing else may be removed.
    feature_selects = [s for s in statements if "from stock_feature_daily" in s]
    assert any("#-" in s for s in feature_selects)
    from app.infra.db.repositories.feature_store_repo import (
        _feature_results_without_setup_payload_query,
    )

    [projected] = [
        row.details_json
        for row in _feature_results_without_setup_payload_query(db_session, run.id)
    ]
    assert projected["rating"] == "Strong Buy"
    assert projected["setup_engine"] == {"setup_score": 82.0, "pattern_primary": "VCP"}
    assert item.extended_fields["se_setup_score"] == 82.0
    assert item.extended_fields["se_pattern_primary"] == "VCP"
    assert "se_explain" not in item.extended_fields
    assert "se_candidates" not in item.extended_fields


def _advisory_lock_is_free(lock_id: int) -> bool:
    # Advisory locks are reentrant for their owning session, so probe from a
    # fresh, unpooled connection: a pooled one could be the leaking session.
    probe = create_engine(engine.url, poolclass=NullPool)
    try:
        with probe.connect() as connection:
            acquired = connection.scalar(select(func.pg_try_advisory_lock(lock_id)))
            if acquired:
                connection.execute(select(func.pg_advisory_unlock(lock_id)))
            return bool(acquired)
    finally:
        probe.dispose()


def _gold_managed_money(db_session):
    from app.domain.cot.models import Participant
    from app.infra.db.models.cot import CotInstrument, CotWeeklyPosition

    return db_session.scalars(
        select(CotWeeklyPosition)
        .join(CotInstrument, CotInstrument.id == CotWeeklyPosition.instrument_id)
        .where(
            CotInstrument.slug == "gold",
            CotWeeklyPosition.participant == Participant.MANAGED_MONEY.value,
        )
        .order_by(CotWeeklyPosition.report_date)
    ).all()


@pytest.fixture
def lock_timeout():
    """Fail fast instead of hanging if a leaked advisory lock blocks a refresh."""

    def on_checkout(dbapi_connection, _record, _proxy):
        with dbapi_connection.cursor() as cursor:
            cursor.execute("SET lock_timeout = '10s'")

    event.listen(engine, "checkout", on_checkout)
    try:
        yield
    finally:
        event.remove(engine, "checkout", on_checkout)
        engine.dispose()  # no pooled connection keeps the setting


def test_cot_refresh_publishes_and_upserts_under_advisory_locks(db_session, lock_timeout):
    from app.infra.db.models.cot import CotPublicationPointer
    from app.use_cases.cot.refresh import CotRefreshCommand, RefreshCotUseCase
    from tests.unit.test_cot_refresh import FakePriceHydrator, FakeSource

    source = FakeSource()
    use_case = RefreshCotUseCase(
        source=source,
        repository=SqlCotRepository(db_session),
        price_hydrator=FakePriceHydrator(),
    )

    first = use_case.execute(CotRefreshCommand(origin="test"))
    second = use_case.execute(CotRefreshCommand(origin="test"))
    original_long = _gold_managed_money(db_session)[155].long
    source.correct_gold_week(155, long_delta=50)  # rewrites stored rows via ON CONFLICT DO UPDATE
    third = use_case.execute(CotRefreshCommand(origin="test"))

    assert (first.status, second.status, third.status) == ("published", "no_change", "published")
    assert db_session.get(CotPublicationPointer, "latest_published").run_id == 3
    # Publishing alone would pass with DO NOTHING; the stored row itself must change.
    db_session.expire_all()
    corrected = _gold_managed_money(db_session)[155]
    assert corrected.long == original_long + 50
    assert corrected.import_run_id == 3
    # The session-level refresh lock is released even though it lives on its own connection.
    assert _advisory_lock_is_free(_COT_REFRESH_LOCK_ID)
    assert _advisory_lock_is_free(_COT_PUBLICATION_LOCK_ID)


def test_grant_eligibility_keeps_the_first_observation(db_session):
    from app.services.theme_evidence_eligibility_service import grant_eligibility

    source = ContentSource(
        name="Test Substack",
        source_type="substack",
        url="https://test.substack.com/feed",
        is_active=True,
        priority=50,
        pipelines=["technical"],
    )
    db_session.add(source)
    db_session.flush()
    item = ContentItem(
        source_id=source.id,
        source_type=source.source_type,
        source_name=source.name,
        title="Test Article",
        content="AI infrastructure is accelerating...",
        published_at=datetime.utcnow() - timedelta(days=1),
        is_processed=False,
    )
    db_session.add(item)
    db_session.flush()
    first_seen = datetime(2026, 9, 30, 12, 0, tzinfo=timezone.utc)

    grant_eligibility(db_session, item.id, "technical", "legacy", source.id, first_seen)
    grant_eligibility(db_session, item.id, "technical", "legacy", source.id, first_seen + timedelta(hours=1))
    db_session.commit()

    [observed_at] = db_session.scalars(
        select(ContentPipelineEligibility.observed_at).where(
            ContentPipelineEligibility.content_item_id == item.id
        )
    ).all()
    assert observed_at == first_seen


def _hold_social_analysis_transaction(factory, on_locked):
    from app.services.social_llm_budget_service import social_analysis_transaction

    with social_analysis_transaction(factory):
        on_locked()


def _hold_projection_registry_lock(factory, on_locked):
    from app.services.social_theme_projection_service import _lock_registry

    with factory() as db, db.begin():
        assert _lock_registry(db) is not None
        on_locked()


def _wait_for_lock_waiter(timeout: float = 15.0) -> None:
    probe = create_engine(engine.url, poolclass=NullPool)
    deadline = time.monotonic() + timeout
    try:
        with probe.connect() as connection:
            while time.monotonic() < deadline:
                waiting = connection.scalar(text(
                    "SELECT count(*) FROM pg_stat_activity "
                    "WHERE datname = current_database() AND wait_event_type = 'Lock'"
                ))
                if waiting:
                    return
                time.sleep(0.05)
    finally:
        probe.dispose()
    raise TimeoutError("no session ever waited on the lock")


def _hold_cot_refresh_lock(factory, on_locked):
    with factory() as db:
        with SqlCotRepository(db).serialized_refresh():
            on_locked()


@pytest.mark.parametrize(
    "hold_lock",
    [_hold_social_analysis_transaction, _hold_projection_registry_lock, _hold_cot_refresh_lock],
    ids=["social_analysis_transaction", "projection_lock_registry", "cot_refresh_advisory_lock"],
)
def test_lock_serializes_concurrent_sessions(db_session, hold_lock, lock_timeout):
    # lock_timeout: a leaked lock makes the contender error out in ~10s
    # instead of blocking forever and hanging pytest at shutdown.
    db_session.add(SocialSourceRegistry(id=1))
    db_session.commit()
    factory = sessionmaker(bind=engine)
    first_locked = threading.Event()
    events: list[str] = []
    errors: list[BaseException] = []

    def run(body):
        try:
            body()
        except BaseException as exc:  # surface thread failures in the test
            errors.append(exc)

    def first():
        def while_locked():
            first_locked.set()
            # Hold the lock until PostgreSQL reports the second session waiting
            # on it; without FOR UPDATE it never waits and this times out.
            _wait_for_lock_waiter()
            events.append("first released")
        hold_lock(factory, while_locked)

    def second():
        # Only contend once the first transaction really holds the lock.
        if not first_locked.wait(30):
            raise TimeoutError("first transaction never took the lock")
        hold_lock(factory, lambda: events.append("second locked"))

    threads = [
        threading.Thread(target=run, args=(body,), daemon=True) for body in (first, second)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(60)

    assert not any(thread.is_alive() for thread in threads), "a lock session never finished"
    assert errors == []
    assert events == ["first released", "second locked"]


@pytest.fixture
def isolated_scan_bootstrap():
    """Scans and bootstrap rows under a fake market, removed afterwards."""
    from app.models.scan_result import Scan
    from app.models.ui_view_snapshot import UIViewSnapshot, UIViewSnapshotPointer
    from app.services.ui_snapshot_service import UISnapshotService

    market = "ZT"
    variants = ("latest:ZT", "scan:zt-1", "scan:zt-2")
    factory = sessionmaker(bind=engine)

    def add_scan(scan_id, hour):
        with factory() as db:
            db.add(Scan(
                scan_id=scan_id, status="completed", universe="market:ZT", universe_type="market",
                universe_key="market:ZT", universe_market=market, total_stocks=1, passed_stocks=1,
                started_at=datetime(2026, 9, 30, hour), completed_at=datetime(2026, 9, 30, hour, 5),
            ))
            db.commit()

    def cleanup():
        with factory() as db:
            db.query(UIViewSnapshotPointer).filter(UIViewSnapshotPointer.variant_key.in_(variants)).delete(
                synchronize_session=False
            )
            db.query(UIViewSnapshot).filter(UIViewSnapshot.variant_key.in_(variants)).delete(
                synchronize_session=False
            )
            db.query(Scan).filter(Scan.universe_market == market).delete(synchronize_session=False)
            db.commit()

    service = UISnapshotService(factory)
    service._ensure_schema()
    cleanup()
    try:
        yield service, add_scan, factory
    finally:
        cleanup()


def test_scan_latest_publish_takes_the_pointer_lock(isolated_scan_bootstrap, lock_timeout):
    """The guarded latest publish must run on PostgreSQL (SQLite drops FOR UPDATE)
    and must wait for a concurrent holder of the pointer row lock."""
    from app.models.ui_view_snapshot import UIViewSnapshotPointer

    service, add_scan, factory = isolated_scan_bootstrap
    add_scan("zt-1", 9)
    assert service.publish_scan_bootstrap(market="ZT").source_revision == "zt-1"
    add_scan("zt-2", 10)
    service.publish_scan_bootstraps_for("zt-2")  # scan variant + guarded latest, one build
    current = service.get_scan_bootstrap(market="ZT")
    assert current.source_revision == "zt-2" and current.is_stale is False

    holder_locked = threading.Event()
    events: list[str] = []
    errors: list[BaseException] = []
    # Record each "which scan is latest" resolution: the guard's re-check must
    # run only after the pointer lock is free. (The later pointer UPDATE would
    # block on the holder anyway, so finishing order alone proves nothing.)
    resolve = service._resolve_scan_source_revision

    def recording_resolve(db, scan_id, market=None):
        events.append("resolve")
        return resolve(db, scan_id, market)

    service._resolve_scan_source_revision = recording_resolve

    def hold_pointer_lock():
        try:
            with factory() as db:
                db.query(UIViewSnapshotPointer).filter(
                    UIViewSnapshotPointer.variant_key == "latest:ZT"
                ).with_for_update(of=UIViewSnapshotPointer).one()
                holder_locked.set()
                _wait_for_lock_waiter()
                events.append("holder released")
        except BaseException as exc:
            errors.append(exc)

    def publish():
        try:
            if not holder_locked.wait(30):
                raise TimeoutError("holder never took the pointer lock")
            service.publish_scan_bootstrap(market="ZT")
            events.append("publish done")
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=fn, daemon=True) for fn in (hold_pointer_lock, publish)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(60)

    assert not any(thread.is_alive() for thread in threads)
    assert errors == []
    # Resolve the source, wait for the lock, then re-check under it.
    assert events == ["resolve", "holder released", "resolve", "publish done"]


def test_concurrent_first_publish_of_a_variant_retries_instead_of_failing(isolated_scan_bootstrap):
    """No pointer row exists yet, so nothing can be locked: both writers insert the
    same snapshot row. The one that loses the unique race must retry and succeed."""
    service, _add_scan, _factory = isolated_scan_bootstrap
    a_flushed = threading.Event()
    errors: list[BaseException] = []
    prune = service._prune_old_revisions

    def prune_then_pause_a(db, **kwargs):
        prune(db, **kwargs)
        if threading.current_thread().name == "A":
            a_flushed.set()
            _wait_for_lock_waiter()  # hold A's uncommitted rows until B blocks on them

    service._prune_old_revisions = prune_then_pause_a

    def publish(value):
        try:
            if threading.current_thread().name == "B" and not a_flushed.wait(30):
                raise TimeoutError("A never flushed its first publish")
            with _factory() as db:
                service._publish(
                    db=db, view_key="scan_bootstrap", variant_key="scan:zt-1",
                    source_revision="zt-1", payload={"writer": value},
                )
        except BaseException as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=publish, args=(name,), name=name, daemon=True) for name in ("A", "B")
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(60)

    assert not any(thread.is_alive() for thread in threads)
    assert errors == []
    assert service.get_scan_bootstrap("zt-1").payload == {"writer": "B"}  # B retried and updated


def test_social_analysis_transaction_requires_the_seeded_registry():
    from app.services.social_llm_budget_service import social_analysis_transaction

    with pytest.raises(ValueError, match="registry_not_initialized"):
        with social_analysis_transaction(sessionmaker(bind=engine)):
            pass


def test_engine_is_postgres_for_this_module():
    # Guard against the module silently running (and passing) on another dialect.
    with engine.connect() as connection:
        assert connection.scalar(text("SELECT version()")).startswith("PostgreSQL")
