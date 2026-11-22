"""Upgrade/downgrade proof for the institutional_change drop migration."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations

MIGRATION_PATH = (
    Path(__file__).resolve().parents[2]
    / "alembic"
    / "versions"
    / "20261122_0075_drop_institutional_change.py"
)


def _run_revision(engine, fn_name: str) -> None:
    spec = importlib.util.spec_from_file_location("drop_institutional_change", MIGRATION_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with engine.begin() as connection:
        module.op = Operations(MigrationContext.configure(connection))
        getattr(module, fn_name)()


def _columns(engine) -> dict[str, tuple[type, bool]]:
    with engine.connect() as connection:
        return {
            column["name"]: (type(column["type"]), column["nullable"])
            for column in sa.inspect(connection).get_columns("stock_fundamentals")
        }


def test_drop_keeps_rows_and_downgrade_restores_a_nullable_float(tmp_path):
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'fundamentals.sqlite'}")
    metadata = sa.MetaData()
    sa.Table(
        "stock_fundamentals",
        metadata,
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("symbol", sa.String(20), nullable=False),
        sa.Column("institutional_transactions", sa.Float(), nullable=True),
        sa.Column("institutional_change", sa.Float(), nullable=True),
    )
    metadata.create_all(engine)
    with engine.begin() as connection:
        connection.execute(sa.text(
            "INSERT INTO stock_fundamentals (symbol, institutional_transactions, institutional_change) "
            "VALUES ('AAPL', 1.5, 2.5)"
        ))

    _run_revision(engine, "upgrade")

    assert list(_columns(engine)) == ["id", "symbol", "institutional_transactions"]
    with engine.connect() as connection:
        assert connection.execute(sa.text(
            "SELECT symbol, institutional_transactions FROM stock_fundamentals"
        )).all() == [("AAPL", 1.5)]

    _run_revision(engine, "downgrade")

    assert _columns(engine)["institutional_change"] == (sa.FLOAT, True)
    with engine.connect() as connection:
        assert connection.execute(sa.text(
            "SELECT symbol, institutional_change FROM stock_fundamentals"
        )).all() == [("AAPL", None)]
