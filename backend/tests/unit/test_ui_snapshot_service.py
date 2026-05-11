"""Unit tests for published UI snapshot persistence semantics."""

from __future__ import annotations

from datetime import date, datetime
import json
import sqlite3

from sqlalchemy import create_engine, func, text
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.domain.markets import market_registry
from app.infra.db.models.feature_store import FeatureRun
from app.models.industry import IBDGroupRank
from app.models.market_breadth import MarketBreadth
from app.models.scan_result import Scan, ScanResult
from app.models.stock_universe import StockUniverse
from app.models.theme import ThemeAlert, ThemeCluster, ThemeMergeSuggestion, ThemeMetrics, ThemePipelineRun
from app.models.ui_view_snapshot import UIViewSnapshot
from app.services.ui_snapshot_service import UISnapshotService, _force_forget_snapshot_tables


def test_ui_snapshot_service_marks_outdated_pointer_reads_as_stale_and_prunes_old_revisions():
    engine = create_engine("sqlite:///:memory:")
    Session = sessionmaker(bind=engine)
    Base.metadata.create_all(engine)

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
            ScanResult.__table__,
            StockUniverse.__table__,
        ],
    )
    Base.metadata.create_all(engine)
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

    universe_stats = snapshot.payload["universe_stats"]
    core_universe_stats = {
        key: universe_stats[key]
        for key in ("total", "active", "by_exchange", "sp500", "by_status", "recent_deactivations")
    }
    assert core_universe_stats == {
        "total": 4,
        "active": 3,
        "by_exchange": {"NASDAQ": 2, "NYSE": 1},
        "sp500": 2,
        "by_status": {"active": 3, "inactive_manual": 1},
        "recent_deactivations": [],
    }
    assert universe_stats["by_market"]["US"]["counts"] == {
        "total": 4,
        "active": 3,
        "inactive": 1,
    }
    assert json.loads(json.dumps(snapshot.payload)) == snapshot.payload


def test_publish_scan_bootstrap_serializes_trigger_source_on_recent_scans():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(
        engine,
        tables=[
            FeatureRun.__table__,
            Scan.__table__,
            ScanResult.__table__,
            StockUniverse.__table__,
        ],
    )
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    service = UISnapshotService(Session)

    with Session() as db:
        db.add(
            Scan(
                scan_id="scan-auto",
                status="completed",
                trigger_source="auto",
                universe="all",
                universe_type="all",
                universe_key="all",
                total_stocks=100,
                passed_stocks=42,
                started_at=datetime(2026, 3, 29, 21, 45, 0),
                completed_at=datetime(2026, 3, 29, 21, 45, 0),
            )
        )
        db.commit()

    snapshot = service.publish_scan_bootstrap()

    assert snapshot.payload["recent_scans"]["scans"][0]["trigger_source"] == "auto"
    assert snapshot.payload["selected_scan"]["trigger_source"] == "auto"


def test_publish_groups_bootstrap_returns_none_when_no_rankings_exist():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine, tables=[IBDGroupRank.__table__])
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    service = UISnapshotService(Session)

    snapshot = service.publish_groups_bootstrap()

    assert snapshot is None
    assert service.get_groups_bootstrap() is None


def test_publish_all_skips_groups_when_rankings_are_missing(monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
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
    published_breadth_markets = []
    monkeypatch.setattr(
        service,
        "publish_breadth_bootstrap",
        lambda market="US": published_breadth_markets.append(market) or _FakeSnapshot(),
    )
    monkeypatch.setattr(service, "publish_groups_bootstrap", lambda: None)
    monkeypatch.setattr(service, "publish_themes_bootstrap", lambda pipeline, theme_view: _FakeSnapshot())
    monkeypatch.setattr("app.services.ui_snapshot_service.settings.feature_themes", False)

    published = service.publish_all()

    assert published["scan_latest"] == fake_snapshot
    assert published["breadth"] == fake_snapshot
    assert published["breadth_us"] == fake_snapshot
    assert published["breadth_hk"] == fake_snapshot
    assert published["breadth_in"] == fake_snapshot
    assert published["breadth_cn"] == fake_snapshot
    assert published_breadth_markets == list(market_registry.supported_market_codes())
    assert published["groups"] is None


def test_breadth_payload_uses_benchmark_fallback_when_primary_cache_is_empty(monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine, tables=[MarketBreadth.__table__])
    Session = sessionmaker(bind=engine)
    service = UISnapshotService(Session)

    with Session() as db:
        db.add(
            MarketBreadth(
                market="HK",
                date=date(2026, 4, 24),
                stocks_up_4pct=22,
                stocks_down_4pct=8,
                ratio_5day=2.75,
                ratio_10day=2.5,
                stocks_up_25pct_quarter=30,
                stocks_down_25pct_quarter=12,
                stocks_up_25pct_month=24,
                stocks_down_25pct_month=10,
                stocks_up_50pct_month=6,
                stocks_down_50pct_month=2,
                stocks_up_13pct_34days=18,
                stocks_down_13pct_34days=7,
                total_stocks_scanned=30,
                calculation_duration_seconds=1.25,
            )
        )
        db.commit()

    class _FakeBenchmarkCache:
        def get_benchmark_candidates(self, market):
            assert market == "HK"
            return ["^HSI", "2800.HK"]

        def get_benchmark_symbol(self, market):
            assert market == "HK"
            return "^HSI"

    monkeypatch.setattr("app.wiring.bootstrap.get_benchmark_cache", lambda: _FakeBenchmarkCache())

    history_calls = []

    def fake_cached_price_history(symbol, period):
        history_calls.append((symbol, period))
        if symbol == "2800.HK":
            return [{"date": "2026-04-24", "close": 18.4}]
        return []

    monkeypatch.setattr(service, "_get_cached_price_history", fake_cached_price_history)

    payload = service._build_breadth_payload("HK")  # noqa: SLF001 - intentional unit coverage

    assert payload["benchmark_symbol"] == "2800.HK"
    assert payload["benchmark_overlay"] == [{"date": "2026-04-24", "close": 18.4}]
    assert payload["spy_overlay"] == payload["benchmark_overlay"]
    assert history_calls == [("^HSI", "1mo"), ("2800.HK", "1mo")]


def test_publish_groups_bootstrap_serializes_rankings_when_available():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine, tables=[IBDGroupRank.__table__])
    Base.metadata.create_all(engine)
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
    Base.metadata.create_all(engine)
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


def test_ui_snapshot_publish_coerces_non_finite_numbers_to_null():
    engine = create_engine("sqlite:///:memory:")
    Session = sessionmaker(bind=engine)
    Base.metadata.create_all(engine)
    service = UISnapshotService(Session)

    with Session() as db:
        snapshot = service._publish(  # noqa: SLF001 - persistence semantics
            db=db,
            view_key="test_view",
            variant_key="default",
            source_revision="rev-non-finite",
            payload={
                "ratio_5day": float("nan"),
                "ratio_10day": float("inf"),
                "nested": {
                    "value": float("-inf"),
                    "ok": 1.25,
                },
                "series": [1.0, float("nan"), 2.0],
            },
        )
        row = db.query(UIViewSnapshot).filter(
            UIViewSnapshot.view_key == "test_view",
            UIViewSnapshot.variant_key == "default",
            UIViewSnapshot.source_revision == "rev-non-finite",
        ).one()

    assert snapshot.payload["ratio_5day"] is None
    assert snapshot.payload["ratio_10day"] is None
    assert snapshot.payload["nested"]["value"] is None
    assert snapshot.payload["nested"]["ok"] == 1.25
    assert snapshot.payload["series"] == [1.0, None, 2.0]
    assert row.payload_json["ratio_5day"] is None
    assert row.payload_json["nested"]["value"] is None
    assert json.loads(json.dumps(snapshot.payload)) == snapshot.payload


def test_ui_snapshot_service_resets_corrupt_cache_tables_and_retries():
    engine = create_engine("sqlite:///:memory:")
    Session = sessionmaker(bind=engine)
    Base.metadata.create_all(engine)
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
    Base.metadata.create_all(engine)

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
