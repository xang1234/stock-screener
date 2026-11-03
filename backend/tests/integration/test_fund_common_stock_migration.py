"""Upgrade/downgrade proof for the fund classification migration."""

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
    / "20261103_0074_mark_funds_not_common_stock.py"
)

ROWS = [
    ("SPY", "Exchange Traded Fund", True),
    ("PDI", "Closed-End Fund - Debt", True),
    ("ADX", "Closed-End Fund - Equity", True),
    ("NVDA", "Semiconductors", True),
    ("O", "REIT - Retail", True),
    ("PFD", "Banks - Regional", False),
    ("NEW", None, True),
]


def _run_revision(engine, fn_name: str) -> None:
    spec = importlib.util.spec_from_file_location("fund_common_stock_migration", MIGRATION_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with engine.begin() as connection:
        module.op = Operations(MigrationContext.configure(connection))
        getattr(module, fn_name)()


def _flags(engine) -> dict[str, bool]:
    with engine.connect() as connection:
        return dict(
            connection.execute(sa.text("SELECT symbol, is_common_stock FROM stock_universe")).all()
        )


def test_migration_marks_only_fund_rows_not_common_stock(tmp_path):
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'universe.sqlite'}")
    metadata = sa.MetaData()
    table = sa.Table(
        "stock_universe",
        metadata,
        sa.Column("symbol", sa.String(20), primary_key=True),
        sa.Column("industry", sa.String(100)),
        sa.Column("is_common_stock", sa.Boolean, nullable=False),
    )
    metadata.create_all(engine)
    with engine.begin() as connection:
        connection.execute(
            table.insert(),
            [{"symbol": s, "industry": i, "is_common_stock": c} for s, i, c in ROWS],
        )

    _run_revision(engine, "upgrade")

    expected = {
        "SPY": False,
        "PDI": False,
        "ADX": False,
        "NVDA": True,
        "O": True,
        "PFD": False,
        "NEW": True,
    }
    assert _flags(engine) == expected

    _run_revision(engine, "downgrade")

    assert _flags(engine) == expected
