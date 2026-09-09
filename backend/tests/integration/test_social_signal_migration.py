"""Execute the migration against a disposable database, including legacy backfill."""
import importlib.util
import os
from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations

TABLES = {
    "social_source_registry", "content_pipeline_eligibility",
    "social_source_configurations", "social_source_audit_events",
    "social_post_sources", "social_content_metrics", "social_post_tickers",
    "social_signal_runs", "social_signal_snapshots", "social_signal_run_pointers",
}


def test_upgrade_backfills_provenance_and_downgrade_preserves_content(tmp_path):
    path = Path(__file__).resolve().parents[2] / "alembic/versions/20260906_0035_add_social_signal_queue.py"
    assert path.exists(), "social migration is missing"
    spec = importlib.util.spec_from_file_location("social_migration", path)
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    assert migration.revision == "20260906_0035"
    assert migration.down_revision == "20260904_0034"
    url = os.environ.get("SOCIAL_REGISTRY_DISPOSABLE_POSTGRES_URL")
    if url:
        parsed = sa.engine.make_url(url)
        assert parsed.host == "127.0.0.1" and parsed.database == "social_registry_test"
    engine = sa.create_engine(url or f"sqlite:///{tmp_path / 'migration.db'}")
    with engine.begin() as connection:
        if engine.dialect.name == "sqlite":
            connection.exec_driver_sql("PRAGMA foreign_keys=ON")
        connection.exec_driver_sql("CREATE TABLE content_sources (id INTEGER PRIMARY KEY, pipelines JSON)")
        connection.exec_driver_sql("CREATE TABLE content_items (id INTEGER PRIMARY KEY, source_id INTEGER, fetched_at TIMESTAMP)")
        connection.exec_driver_sql("CREATE TABLE stock_universe (id INTEGER PRIMARY KEY)")
        connection.exec_driver_sql("INSERT INTO content_sources VALUES (1, '[\"technical\",\"fundamental\"]')")
        connection.exec_driver_sql("INSERT INTO content_items VALUES (7, 1, '2026-09-06 03:00:00')")
        migration.op = Operations(MigrationContext.configure(connection))
        migration.upgrade()
        inspector = sa.inspect(connection)
        assert TABLES.issubset(inspector.get_table_names())
        assert connection.exec_driver_sql(
            "SELECT mode, provider, version, official_budget_day, official_reserved_posts FROM social_source_registry"
        ).one() == ("off", "disabled", 1, None, 0)
        registry = migration.SocialSourceRegistry.__table__
        with pytest.raises(sa.exc.IntegrityError, match="ck_social_registry_official_reserved_posts"):
            with connection.begin_nested():
                connection.execute(registry.update().values(official_reserved_posts=-1))
        assert set(connection.exec_driver_sql("SELECT content_item_id,pipeline,channel,originating_source_id FROM content_pipeline_eligibility")) == {(7, "technical", "legacy", 1), (7, "fundamental", "legacy", 1)}
        assert inspector.get_pk_constraint("social_signal_run_pointers")["constrained_columns"] == ["key"]
        for table in TABLES - {"social_source_registry"}:
            assert inspector.get_foreign_keys(table), table
        source = migration.SocialSourceConfiguration.__table__
        connection.execute(source.insert().values(content_source_id=1, x_list_id="3001", lifecycle_state="pending", provenance="admin"))
        for values, constraint in [
            ({"lifecycle_state": "deleted"}, "ck_social_source_lifecycle"),
            ({"test_sample_count": 6}, "ck_social_source_sample_count"),
            ({"tested_provider": "fallback"}, "ck_social_source_test_provider"),
            ({"test_status": "success"}, "ck_social_source_test_status"),
            ({"lifecycle_state": "archived"}, "ck_social_source_archive_time"),
        ]:
            with pytest.raises(sa.exc.IntegrityError, match=constraint):
                with connection.begin_nested():
                    connection.execute(source.update().values(**values))
        run = migration.SocialSignalRun.__table__
        connection.execute(run.insert().values(id="run1", registry_id=1, registry_version=1, mode="validation", provider="official", status="staged", source_outcomes_json={}, application_progress_json={}, feature_run_ids_json={}, exposure_dates_json={}, coverage_json={}))
        snapshot = migration.SocialSignalSnapshot.__table__
        connection.execute(snapshot.insert().values(run_id="run1", window_days=7, candidate_key="unresolved:test", state="unresolved", social_score=50, confirmation_score=None, queue_score=None, explanation_json={}, coverage_json={}, resolution_policy_version="v1", formula_version="social-signal-v1", mention_count=1, observed_list_count=1, enabled_list_count=2, normalization_scope="global_fallback"))
        for values, constraint in [
            ({"queue_score": 30}, "ck_social_snapshot_nullable_scores"),
            ({"state": "fabricated"}, "ck_social_snapshot_state"),
            ({"window_days": 30}, "ck_social_snapshot_window"),
            ({"social_score": 101}, "ck_social_snapshot_social_score"),
        ]:
            with pytest.raises(sa.exc.IntegrityError, match=constraint):
                with connection.begin_nested():
                    connection.execute(snapshot.update().values(**values))
        pointer = migration.SocialSignalRunPointer.__table__
        with pytest.raises(sa.exc.IntegrityError, match="ck_social_pointer_key"):
            with connection.begin_nested():
                connection.execute(pointer.insert().values(key="unapproved", run_id="run1"))
        connection.execute(pointer.insert().values(key="latest_published", run_id="run1"))
        with pytest.raises(sa.exc.IntegrityError):
            with connection.begin_nested():
                connection.execute(pointer.update().values(run_id="missing"))
        expected_uniques = {
            "social_source_configurations": {"x_list_id"},
            "social_post_sources": {"content_item_id", "content_source_id"},
            "social_content_metrics": {"content_item_id"},
            "social_post_tickers": {"content_item_id", "candidate_key"},
            "social_signal_snapshots": {"run_id", "window_days", "candidate_key"},
        }
        for table, columns in expected_uniques.items():
            unique_keys = [set(row["column_names"]) for row in inspector.get_unique_constraints(table)]
            unique_keys.append(set(inspector.get_pk_constraint(table)["constrained_columns"]))
            assert columns in unique_keys
        migration.downgrade()
        assert TABLES.isdisjoint(sa.inspect(connection).get_table_names())
        assert connection.exec_driver_sql("SELECT COUNT(*) FROM content_items").scalar() == 1
        for name in ("content_items", "content_sources", "stock_universe"):
            connection.exec_driver_sql(f"DROP TABLE {name}")
    engine.dispose()
