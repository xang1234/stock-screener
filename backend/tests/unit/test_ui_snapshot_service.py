"""Unit tests for published UI snapshot persistence semantics."""

from __future__ import annotations

from datetime import date, datetime
import json
import sqlite3

from sqlalchemy import create_engine, func, text
from sqlalchemy.orm import sessionmaker

from app.db_migrations.universe_lifecycle_migration import migrate_universe_lifecycle
from app.db_migrations.ui_view_snapshot_migration import migrate_ui_view_snapshot_tables
from app.database import Base
from app.infra.db.models.feature_store import FeatureRun
from app.models.industry import IBDGroupRank
from app.models.scan_result import Scan
from app.models.stock_universe import StockUniverse
from app.models.theme import ThemeAlert, ThemeCluster, ThemeMergeSuggestion, ThemeMetrics, ThemePipelineRun
from app.models.ui_view_snapshot import UIViewSnapshot
from app.services.ui_snapshot_service import UISnapshotService, _force_forget_snapshot_tables


def test_ui_snapshot_service_marks_outdated_pointer_reads_as_stale_and_prunes_old_revisions():
    engine = create_engine("sqlite:///:memory:")
    Session = sessionmaker(bind=engine)
    migrate_ui_view_snapshot_tables(engine)

    service = UISnapshotService(Session)

    with Session() as db:
        first = service._publish(  # noqa: SLF001 - unit test exercises persistence semantics directly
            db=db,
            view_key="test_view",
            variant_key="default",
            source_revision="rev-1",
            payload={"value": 1},
        )
        fresh = service._get_snapshot(  # noqa: SLF001
            db=db,
            view_key="test_view",
            variant_key="default",
            source_revision="rev-1",
        )
        stale = service._get_snapshot(  # noqa: SLF001
            db=db,
            view_key="test_view",
            variant_key="default",
            source_revision="rev-2",
        )

        assert first.is_stale is False
        assert fresh is not None and fresh.is_stale is False
        assert stale is not None and stale.is_stale is True
        assert stale.source_revision == "rev-1"
        assert stale.payload == {"value": 1}

        second = service._publish(  # noqa: SLF001
            db=db,
            view_key="test_view",
            variant_key="default",
            source_revision="rev-2",
            payload={"value": 2},
        )
        third = service._publish(  # noqa: SLF001
            db=db,
            view_key="test_view",
            variant_key="default",
            source_revision="rev-3",
            payload={"value": 3},
        )

        kept_revisions = {
            row.source_revision
            for row in db.query(UIViewSnapshot)
            .filter(
                UIViewSnapshot.view_key == "test_view",
                UIViewSnapshot.variant_key == "default",
            )
            .all()
        }

        assert second.snapshot_revision != third.snapshot_revision
        assert kept_revisions == {"rev-2", "rev-3"}
        assert db.query(func.count(UIViewSnapshot.id)).scalar() == 2

def test_resolve_themes_source_revision_filters_pipeline_runs_to_pipeline_or_global():
    engine = create_engine("sqlite:///:memory:")

    Base.metadata.create_all(
        engine,
        tables=[
            ThemeMetrics.__table__,
            ThemeCluster.__table__,
            ThemePipelineRun.__table__,
            ThemeAlert.__table__,
            ThemeMergeSuggestion.__table__,
        ],
    )
    Session = sessionmaker(bind=engine)
    service = UISnapshotService(Session)

    with Session() as db:
        db.add_all(
            [
                ThemePipelineRun(run_id="global", pipeline=None, completed_at=datetime(2026, 3, 18, 10, 0, 0)),
                ThemePipelineRun(run_id="tech", pipeline="technical", completed_at=datetime(2026, 3, 18, 11, 0, 0)),
                ThemePipelineRun(run_id="fund", pipeline="fundamental", completed_at=datetime(2026, 3, 18, 12, 0, 0)),
            ]
        )
        db.commit()

        service._query_failed_items_count = lambda *_args, **_kwargs: 0  # noqa: SLF001

        technical_revision = service._resolve_themes_source_revision(db, "technical")  # noqa: SLF001
        fundamental_revision = service._resolve_themes_source_revision(db, "fundamental")  # noqa: SLF001

        assert technical_revision.split("|")[2] == "2026-03-18T11:00:00"
        assert fundamental_revision.split("|")[2] == "2026-03-18T12:00:00"


def test_publish_scan_bootstrap_serializes_universe_stats_counts():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(
        engine,
        tables=[
            FeatureRun.__table__,
            Scan.__table__,
            StockUniverse.__table__,
        ],
    )
    migrate_ui_view_snapshot_tables(engine)
    migrate_universe_lifecycle(engine)
    Session = sessionmaker(bind=engine)
    service = UISnapshotService(Session)

    with Session() as db:
        db.add_all(
            [
                StockUniverse(symbol="AAPL", exchange="NASDAQ", is_active=True, is_sp500=True),
                StockUniverse(symbol="MSFT", exchange="NASDAQ", is_active=True, is_sp500=True),
                StockUniverse(symbol="IBM", exchange="NYSE", is_active=True, is_sp500=False),
                StockUniverse(symbol="ZZZZ", exchange="AMEX", is_active=False, is_sp500=False),
            ]
        )
        db.commit()

    snapshot = service.publish_scan_bootstrap()

    assert snapshot.payload["universe_stats"] == {
        "total": 4,
        "active": 3,
        "by_exchange": {"NASDAQ": 2, "NYSE": 1},
        "sp500": 2,
        "by_status": {"active": 3, "inactive_manual": 1},
        "recent_deactivations": [],
    }
    assert json.loads(json.dumps(snapshot.payload)) == snapshot.payload


def test_publish_groups_bootstrap_returns_none_when_no_rankings_exist():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine, tables=[IBDGroupRank.__table__])
    migrate_ui_view_snapshot_tables(engine)
    Session = sessionmaker(bind=engine)
    service = UISnapshotService(Session)

    snapshot = service.publish_groups_bootstrap()

    assert snapshot is None
    assert service.get_groups_bootstrap() is None


def test_publish_all_skips_groups_when_rankings_are_missing(monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    migrate_ui_view_snapshot_tables(engine)
    Session = sessionmaker(bind=engine)
    service = UISnapshotService(Session)

    fixed_time = datetime(2026, 3, 29, 9, 30, 0)
    fake_snapshot = {
        "snapshot_revision": "1",
        "source_revision": "rev-1",
        "published_at": fixed_time,
        "is_stale": False,
        "payload": {"ok": True},
    }

    class _FakeSnapshot:
        def to_dict(self):
            return fake_snapshot

    monkeypatch.setattr(service, "publish_scan_bootstrap", lambda scan_id=None: _FakeSnapshot())
    monkeypatch.setattr(service, "publish_breadth_bootstrap", lambda: _FakeSnapshot())
    monkeypatch.setattr(service, "publish_groups_bootstrap", lambda: None)
    monkeypatch.setattr(service, "publish_themes_bootstrap", lambda pipeline, theme_view: _FakeSnapshot())
    monkeypatch.setattr("app.services.ui_snapshot_service.settings.feature_themes", False)

    published = service.publish_all()

    assert published["scan_latest"] == fake_snapshot
    assert published["breadth"] == fake_snapshot
    assert published["groups"] is None


def test_publish_groups_bootstrap_serializes_rankings_when_available():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine, tables=[IBDGroupRank.__table__])
    migrate_ui_view_snapshot_tables(engine)
    Session = sessionmaker(bind=engine)
    service = UISnapshotService(Session)

    with Session() as db:
        db.add(
            IBDGroupRank(
                industry_group="Software",
                date=date(2026, 3, 28),
                rank=1,
                avg_rs_rating=95.5,
                median_rs_rating=95.0,
                weighted_avg_rs_rating=95.2,
                rs_std_dev=1.0,
                num_stocks=12,
                num_stocks_rs_above_80=10,
                top_symbol="MSFT",
                top_rs_rating=99.0,
            )
        )
        db.commit()

    snapshot = service.publish_groups_bootstrap()

    assert snapshot is not None
    assert snapshot.payload["rankings"]["date"] == "2026-03-28"
    assert snapshot.payload["rankings"]["total_groups"] == 1
    assert snapshot.payload["rankings"]["rankings"][0]["industry_group"] == "Software"


def test_ui_snapshot_publish_coerces_nested_dates_to_json_safe_strings():
    engine = create_engine("sqlite:///:memory:")
    Session = sessionmaker(bind=engine)
    migrate_ui_view_snapshot_tables(engine)
    service = UISnapshotService(Session)

    with Session() as db:
        snapshot = service._publish(  # noqa: SLF001 - persistence semantics
            db=db,
            view_key="test_view",
            variant_key="default",
            source_revision="rev-date",
            payload={
                "current": {
                    "date": date(2026, 3, 19),
                    "published_at": datetime(2026, 3, 19, 12, 34, 56),
                },
                "history": [
                    {"date": date(2026, 3, 18)},
                    {"date": date(2026, 3, 17)},
                ],
            },
        )
        row = db.query(UIViewSnapshot).filter(
            UIViewSnapshot.view_key == "test_view",
            UIViewSnapshot.variant_key == "default",
            UIViewSnapshot.source_revision == "rev-date",
        ).one()

    assert snapshot.payload["current"]["date"] == "2026-03-19"
    assert snapshot.payload["current"]["published_at"] == "2026-03-19T12:34:56"
    assert snapshot.payload["history"][0]["date"] == "2026-03-18"
    assert row.payload_json["current"]["date"] == "2026-03-19"
    assert json.loads(json.dumps(snapshot.payload)) == snapshot.payload


def test_ui_snapshot_service_resets_corrupt_cache_tables_and_retries():
    engine = create_engine("sqlite:///:memory:")
    Session = sessionmaker(bind=engine)
    migrate_ui_view_snapshot_tables(engine)
    service = UISnapshotService(Session)

    with Session() as db:
        service._publish(  # noqa: SLF001
            db=db,
            view_key="test_view",
            variant_key="default",
            source_revision="rev-1",
            payload={"value": 1},
        )

    original = service._get_snapshot
    state = {"calls": 0}

    def flaky_get_snapshot(*, db, view_key, variant_key, source_revision):
        state["calls"] += 1
        if state["calls"] == 1:
            raise sqlite3.DatabaseError("database disk image is malformed")
        return original(
            db=db,
            view_key=view_key,
            variant_key=variant_key,
            source_revision=source_revision,
        )

    service._get_snapshot = flaky_get_snapshot  # noqa: SLF001

    snapshot = service._run_with_storage_recovery(  # noqa: SLF001
        lambda db: service._get_snapshot(  # noqa: SLF001
            db=db,
            view_key="test_view",
            variant_key="default",
            source_revision="rev-1",
        )
    )

    with Session() as db:
        assert db.query(func.count(UIViewSnapshot.id)).scalar() == 0

    assert state["calls"] == 2
    assert snapshot is None


def test_force_forget_snapshot_tables_removes_snapshot_schema_entries():
    engine = create_engine("sqlite:///:memory:")
    migrate_ui_view_snapshot_tables(engine)

    with engine.begin() as conn:
        _force_forget_snapshot_tables(conn)

    with engine.connect() as conn:
        names = {
            row[0]
            for row in conn.execute(
                text(
                    """
                    SELECT name
                    FROM sqlite_master
                    WHERE name LIKE 'ui_view_snapshot%'
                       OR tbl_name LIKE 'ui_view_snapshot%'
                    """
                )
            ).fetchall()
        }

    assert names == set()
