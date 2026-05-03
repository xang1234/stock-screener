"""Tests for the predefined filter presets seed migration."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations


MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "alembic" / "versions"
MIGRATION_FILENAME = "20260503_0018_seed_predefined_filter_presets.py"


def _load_migration():
    path = MIGRATIONS_DIR / MIGRATION_FILENAME
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_filter_presets_table(engine: sa.Engine) -> None:
    metadata = sa.MetaData()
    sa.Table(
        "filter_presets",
        metadata,
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("name", sa.String(100), nullable=False, unique=True),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("filters", sa.Text, nullable=False),
        sa.Column("sort_by", sa.String(50), nullable=False),
        sa.Column("sort_order", sa.String(10), nullable=False),
        sa.Column("position", sa.Integer, nullable=False),
    )
    metadata.create_all(engine)


def _run_upgrade(module, connection) -> None:
    context = MigrationContext.configure(connection)
    module.op = Operations(context)
    module.upgrade()


def _run_downgrade(module, connection) -> None:
    context = MigrationContext.configure(connection)
    module.op = Operations(context)
    module.downgrade()


def test_seed_inserts_all_predefined_presets_into_empty_table():
    engine = sa.create_engine("sqlite:///:memory:")
    _make_filter_presets_table(engine)

    migration = _load_migration()
    expected_names = [name for name, *_ in migration.SEEDED_PRESETS]

    with engine.begin() as conn:
        _run_upgrade(migration, conn)

        rows = conn.execute(
            sa.text("SELECT name, position, sort_by, sort_order, filters FROM filter_presets ORDER BY position")
        ).fetchall()

    engine.dispose()

    assert [row[0] for row in rows] == expected_names
    assert [row[1] for row in rows] == list(range(len(expected_names)))

    # Every seeded row must be valid JSON containing the full default filter shape.
    expected_keys = set(migration._empty_filter_shape().keys())
    for _, _, _, _, filters_json in rows:
        parsed = json.loads(filters_json)
        assert set(parsed.keys()) == expected_keys


def test_seed_excludes_static_only_presets():
    """movers_9m and club_97 use semantics or fields the live site doesn't support."""
    migration = _load_migration()
    seeded_names = {name for name, *_ in migration.SEEDED_PRESETS}

    assert "9M Movers" not in seeded_names
    assert "97 Club" not in seeded_names


def test_minervini_preset_carries_canonical_overrides():
    migration = _load_migration()
    overrides_by_name = {
        name: overrides for name, _desc, overrides, *_ in migration.SEEDED_PRESETS
    }

    minervini = overrides_by_name["Minervini Trend Template"]
    assert minervini["minerviniScore"] == {"min": 70, "max": None}
    assert minervini["stage"] == 2
    assert minervini["maAlignment"] is True
    assert minervini["rsRating"] == {"min": 70, "max": None}

    canslim = overrides_by_name["CANSLIM"]
    assert canslim["canslimScore"] == {"min": 70, "max": None}
    assert canslim["epsGrowth"] == {"min": 25, "max": None}
    assert canslim["rsRating"] == {"min": 80, "max": None}


def test_seed_skips_presets_whose_names_already_exist():
    engine = sa.create_engine("sqlite:///:memory:")
    _make_filter_presets_table(engine)

    migration = _load_migration()

    user_minervini_payload = json.dumps({"customized": True})
    with engine.begin() as conn:
        conn.execute(
            sa.text(
                "INSERT INTO filter_presets "
                "(name, description, filters, sort_by, sort_order, position) "
                "VALUES "
                "('Minervini Trend Template', 'user copy', :filters, 'composite_score', 'desc', 0)"
            ),
            {"filters": user_minervini_payload},
        )
        _run_upgrade(migration, conn)

        existing = conn.execute(
            sa.text(
                "SELECT filters FROM filter_presets WHERE name = 'Minervini Trend Template'"
            )
        ).scalar_one()
        total = conn.execute(sa.text("SELECT COUNT(*) FROM filter_presets")).scalar_one()

    engine.dispose()

    assert existing == user_minervini_payload  # untouched
    # Original user row + every seeded preset except the skipped Minervini one.
    assert total == 1 + (len(migration.SEEDED_PRESETS) - 1)


def test_seed_starts_position_after_existing_max():
    engine = sa.create_engine("sqlite:///:memory:")
    _make_filter_presets_table(engine)

    migration = _load_migration()

    with engine.begin() as conn:
        conn.execute(
            sa.text(
                "INSERT INTO filter_presets "
                "(name, description, filters, sort_by, sort_order, position) "
                "VALUES "
                "('User Pinned', NULL, '{}', 'composite_score', 'desc', 7)"
            )
        )
        _run_upgrade(migration, conn)

        seeded_positions = conn.execute(
            sa.text(
                "SELECT position FROM filter_presets "
                "WHERE name <> 'User Pinned' ORDER BY position"
            )
        ).fetchall()

    engine.dispose()

    positions = [row[0] for row in seeded_positions]
    assert positions == list(range(8, 8 + len(migration.SEEDED_PRESETS)))


def test_seed_is_idempotent_when_run_twice():
    engine = sa.create_engine("sqlite:///:memory:")
    _make_filter_presets_table(engine)

    migration = _load_migration()

    with engine.begin() as conn:
        _run_upgrade(migration, conn)
        first_count = conn.execute(sa.text("SELECT COUNT(*) FROM filter_presets")).scalar_one()
        _run_upgrade(migration, conn)
        second_count = conn.execute(sa.text("SELECT COUNT(*) FROM filter_presets")).scalar_one()

    engine.dispose()

    assert first_count == len(migration.SEEDED_PRESETS)
    assert second_count == first_count


def test_downgrade_removes_seeded_presets_only():
    engine = sa.create_engine("sqlite:///:memory:")
    _make_filter_presets_table(engine)

    migration = _load_migration()

    with engine.begin() as conn:
        conn.execute(
            sa.text(
                "INSERT INTO filter_presets "
                "(name, description, filters, sort_by, sort_order, position) "
                "VALUES "
                "('User Custom', NULL, '{}', 'composite_score', 'desc', 0)"
            )
        )
        _run_upgrade(migration, conn)
        _run_downgrade(migration, conn)

        remaining = conn.execute(
            sa.text("SELECT name FROM filter_presets")
        ).fetchall()

    engine.dispose()

    assert [row[0] for row in remaining] == ["User Custom"]
