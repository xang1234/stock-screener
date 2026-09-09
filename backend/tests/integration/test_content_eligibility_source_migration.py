"""Upgrade proof for retaining every content eligibility source membership."""

import importlib.util
from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations


MIGRATION_PATH = (
    Path(__file__).resolve().parents[2]
    / "alembic/versions/20260909_0039_retain_content_eligibility_sources.py"
)


def _migration():
    assert MIGRATION_PATH.exists(), "content eligibility source migration is missing"
    spec = importlib.util.spec_from_file_location(
        "content_eligibility_source_migration", MIGRATION_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_migration_retains_existing_row_and_allows_each_observing_source(tmp_path):
    engine = sa.create_engine(
        f"sqlite:///{tmp_path / 'content-eligibility-source.sqlite'}"
    )
    with engine.begin() as connection:
        connection.exec_driver_sql("PRAGMA foreign_keys=ON")
        connection.exec_driver_sql(
            "CREATE TABLE content_sources (id INTEGER PRIMARY KEY)"
        )
        connection.exec_driver_sql(
            "CREATE TABLE content_items (id INTEGER PRIMARY KEY)"
        )
        connection.exec_driver_sql(
            """
            CREATE TABLE content_pipeline_eligibility (
                content_item_id INTEGER NOT NULL,
                pipeline TEXT NOT NULL,
                channel TEXT NOT NULL,
                originating_source_id INTEGER,
                observed_at TIMESTAMP NOT NULL,
                PRIMARY KEY (content_item_id, pipeline, channel),
                FOREIGN KEY(content_item_id) REFERENCES content_items(id),
                FOREIGN KEY(originating_source_id) REFERENCES content_sources(id)
            )
            """
        )
        connection.exec_driver_sql("INSERT INTO content_sources VALUES (1), (2)")
        connection.exec_driver_sql("INSERT INTO content_items VALUES (7)")
        connection.exec_driver_sql(
            """
            INSERT INTO content_pipeline_eligibility
                (content_item_id, pipeline, channel, originating_source_id, observed_at)
            VALUES (7, 'technical', 'legacy', 1, '2026-09-09 00:00:00')
            """
        )

        migration = _migration()
        assert migration.revision == "20260909_0039"
        assert migration.down_revision == "20260908_0038"
        migration.op = Operations(MigrationContext.configure(connection))
        migration.upgrade()

        inspector = sa.inspect(connection)
        assert inspector.get_pk_constraint("content_pipeline_eligibility")[
            "constrained_columns"
        ] == ["id"]
        unique_columns = {
            tuple(row["column_names"])
            for row in inspector.get_unique_constraints(
                "content_pipeline_eligibility"
            )
        }
        assert (
            "content_item_id",
            "pipeline",
            "channel",
            "originating_source_id",
        ) in unique_columns
        connection.exec_driver_sql(
            """
            INSERT INTO content_pipeline_eligibility
                (content_item_id, pipeline, channel, originating_source_id, observed_at)
            VALUES (7, 'technical', 'legacy', 2, '2026-09-09 00:00:01')
            """
        )
        assert connection.exec_driver_sql(
            "SELECT originating_source_id FROM content_pipeline_eligibility ORDER BY originating_source_id"
        ).scalars().all() == [1, 2]
        with pytest.raises(sa.exc.IntegrityError):
            with connection.begin_nested():
                connection.exec_driver_sql(
                    """
                    INSERT INTO content_pipeline_eligibility
                        (content_item_id, pipeline, channel, originating_source_id, observed_at)
                    VALUES (7, 'technical', 'legacy', 2, '2026-09-09 00:00:02')
                    """
                )

        migration.downgrade()

        inspector = sa.inspect(connection)
        assert inspector.get_pk_constraint("content_pipeline_eligibility")[
            "constrained_columns"
        ] == ["content_item_id", "pipeline", "channel"]
        assert connection.exec_driver_sql(
            "SELECT COUNT(*) FROM content_pipeline_eligibility"
        ).scalar_one() == 1
    engine.dispose()
