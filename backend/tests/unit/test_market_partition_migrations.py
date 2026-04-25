from __future__ import annotations

import importlib.util
from pathlib import Path

from alembic.migration import MigrationContext
from alembic.operations import Operations
import sqlalchemy as sa


MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "alembic" / "versions"


def _load_migration(filename: str):
    path = MIGRATIONS_DIR / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_upgrade(module, connection) -> None:
    context = MigrationContext.configure(connection)
    module.op = Operations(context)
    module.upgrade()


def test_ibd_group_rank_market_migration_drops_sqlite_date_group_unique():
    engine = sa.create_engine("sqlite:///:memory:")
    metadata = sa.MetaData()
    sa.Table(
        "ibd_group_ranks",
        metadata,
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("industry_group", sa.String(100), nullable=False),
        sa.Column("date", sa.Date, nullable=False),
        sa.UniqueConstraint("industry_group", "date"),
    )
    metadata.create_all(engine)

    migration = _load_migration("20260424_0015_add_market_to_ibd_group_ranks.py")
    with engine.begin() as conn:
        conn.execute(
            sa.text(
                "INSERT INTO ibd_group_ranks (industry_group, date) "
                "VALUES ('Internet Services', '2026-04-02')"
            )
        )
        _run_upgrade(migration, conn)
        conn.execute(
            sa.text(
                "INSERT INTO ibd_group_ranks (industry_group, date, market) "
                "VALUES ('Internet Services', '2026-04-02', 'HK')"
            )
        )

    engine.dispose()


def test_market_breadth_market_migration_drops_sqlite_date_unique():
    engine = sa.create_engine("sqlite:///:memory:")
    metadata = sa.MetaData()
    sa.Table(
        "market_breadth",
        metadata,
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("date", sa.Date, nullable=False),
        sa.UniqueConstraint("date"),
    )
    metadata.create_all(engine)

    migration = _load_migration("20260425_0016_add_market_to_market_breadth.py")
    with engine.begin() as conn:
        conn.execute(sa.text("INSERT INTO market_breadth (date) VALUES ('2026-04-02')"))
        _run_upgrade(migration, conn)
        conn.execute(
            sa.text(
                "INSERT INTO market_breadth (date, market) "
                "VALUES ('2026-04-02', 'HK')"
            )
        )

    engine.dispose()
