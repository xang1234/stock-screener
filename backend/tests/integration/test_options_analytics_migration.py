from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations

BACKEND_ROOT = Path(__file__).resolve().parents[2]
MIGRATION_PATH = BACKEND_ROOT / "alembic" / "versions" / "20260904_0034_add_options_analytics.py"


def _load_migration():
    if not MIGRATION_PATH.is_file():
        pytest.fail(f"options analytics migration is missing: {MIGRATION_PATH}")
    spec = importlib.util.spec_from_file_location("options_analytics_migration", MIGRATION_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_revision(engine, operation: str) -> None:
    module = _load_migration()
    with engine.begin() as connection:
        operations = Operations(MigrationContext.configure(connection))
        original_op = module.op
        module.op = operations
        try:
            getattr(module, operation)()
        finally:
            module.op = original_op


def _create_feature_runs(engine) -> None:
    metadata = sa.MetaData()
    sa.Table("feature_runs", metadata, sa.Column("id", sa.Integer, primary_key=True))
    metadata.create_all(engine)


def test_options_migration_upgrades_checks_source_identity_and_downgrades(tmp_path) -> None:
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'options.sqlite'}")
    _create_feature_runs(engine)

    _run_revision(engine, "upgrade")
    inspector = sa.inspect(engine)
    expected = {
        "options_analytics_runs",
        "options_analytics_run_items",
        "options_analytics_strike_points",
        "options_analytics_pointers",
    }
    assert expected.issubset(inspector.get_table_names())

    with engine.begin() as connection:
        connection.execute(sa.text("INSERT INTO feature_runs (id) VALUES (1)"))
        connection.execute(
            sa.text(
                "INSERT INTO options_analytics_runs "
                "(market, origin, source_feature_run_id, calculation_version, schema_version, "
                "provider, input_signature, attempt_number, status, as_of_date, expected_count, "
                "current_count, continuity_count, completed_count, core_valid_current_count, "
                "failed_count, retried_count, coverage) VALUES "
                "('US', 'local', 1, 'v1', 'v1', 'yahoo', 'local-sig', 1, 'staged', "
                "'2026-09-04', 1, 1, 0, 0, 0, 0, 0, 0.0)"
            )
        )
        connection.execute(
            sa.text(
                "INSERT INTO options_analytics_runs "
                "(market, origin, external_source_feature_run_key, calculation_version, schema_version, "
                "provider, input_signature, attempt_number, status, as_of_date, expected_count, "
                "current_count, continuity_count, completed_count, core_valid_current_count, "
                "failed_count, retried_count, coverage) VALUES "
                "('US', 'history_transfer', 'external:42', 'v1', 'v1', 'yahoo', 'transfer-sig', "
                "1, 'published', '2026-09-03', 1, 1, 0, 1, 1, 0, 0, 1.0)"
            )
        )
        with pytest.raises(sa.exc.IntegrityError):
            connection.execute(
                sa.text(
                    "INSERT INTO options_analytics_runs "
                    "(market, origin, calculation_version, schema_version, provider, input_signature, "
                    "attempt_number, status, as_of_date, expected_count, current_count, continuity_count, "
                    "completed_count, core_valid_current_count, failed_count, retried_count, coverage) "
                    "VALUES ('US', 'local', 'v1', 'v1', 'yahoo', 'bad', 1, 'staged', "
                    "'2026-09-04', 0, 0, 0, 0, 0, 0, 0, 0.0)"
                )
            )

    _run_revision(engine, "downgrade")
    assert expected.isdisjoint(sa.inspect(engine).get_table_names())
    assert "feature_runs" in sa.inspect(engine).get_table_names()
    engine.dispose()

