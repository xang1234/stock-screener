"""Upgrade/downgrade proof for the canonical Market RS migration."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from alembic.migration import MigrationContext
from alembic.operations import Operations
import pytest
from sqlalchemy import (
    Column,
    Date,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    UniqueConstraint,
    create_engine,
    inspect,
    text,
)
from sqlalchemy.exc import IntegrityError

from app.domain.relative_strength import (
    BALANCED_RS_FORMULA_VERSION,
    LEGACY_RS_FORMULA_VERSION,
)


BACKEND_ROOT = Path(__file__).resolve().parents[2]
MIGRATION_PATH = (
    BACKEND_ROOT
    / "alembic"
    / "versions"
    / "20260718_0025_add_canonical_market_rs.py"
)


def _load_migration():
    spec = importlib.util.spec_from_file_location("canonical_rs_migration", MIGRATION_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_revision(engine, fn_name: str) -> None:
    module = _load_migration()
    with engine.begin() as connection:
        operations = Operations(MigrationContext.configure(connection))
        original_op = module.op
        module.op = operations
        try:
            getattr(module, fn_name)()
        finally:
            module.op = original_op


def _create_pre_canonical_schema(engine) -> None:
    metadata = MetaData()
    Table(
        "ibd_group_ranks",
        metadata,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("market", String(8), nullable=False, default="US"),
        Column("industry_group", String(100), nullable=False),
        Column("date", Date, nullable=False),
        Column("rank", Integer, nullable=False),
        Column("avg_rs_rating", Float, nullable=False),
        Column("median_rs_rating", Float),
        Column("weighted_avg_rs_rating", Float),
        Column("rs_std_dev", Float),
        Column("num_stocks", Integer, default=0),
        Column("num_stocks_rs_above_80", Integer, default=0),
        Column("top_symbol", String(20)),
        Column("top_rs_rating", Float),
        UniqueConstraint(
            "industry_group",
            "date",
            "market",
            name="uix_ibd_group_rank_market_date",
        ),
    )
    Table(
        "stock_universe",
        metadata,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("market", String(8), nullable=False),
    )
    metadata.create_all(engine)


def test_canonical_market_rs_migration_preserves_legacy_and_supports_coexistence(
    tmp_path,
):
    database_path = tmp_path / "canonical-rs-migration.sqlite"
    database_url = f"sqlite:///{database_path}"
    engine = create_engine(database_url)
    _create_pre_canonical_schema(engine)
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                INSERT INTO ibd_group_ranks (
                    market, industry_group, date, rank, avg_rs_rating,
                    num_stocks, num_stocks_rs_above_80
                ) VALUES (
                    'US', 'Software', '2026-04-10', 1, 95.0, 3, 3
                )
                """
            )
        )
    _run_revision(engine, "upgrade")
    with engine.begin() as connection:
        legacy = connection.execute(
            text(
                """
                SELECT rs_formula_version, avg_rs_rating_1m,
                       avg_rs_rating_3m, market_rs_run_id
                FROM ibd_group_ranks
                WHERE market='US' AND industry_group='Software'
                """
            )
        ).one()
        assert legacy[0] == LEGACY_RS_FORMULA_VERSION
        assert legacy[1:] == (None, None, None)
        assert connection.execute(
            text(
                "SELECT formula_version FROM market_rs_formula_pointers "
                "WHERE market='US'"
            )
        ).scalar_one() == LEGACY_RS_FORMULA_VERSION

        run_id = connection.execute(
            text(
                """
                INSERT INTO market_rs_runs (
                    market, as_of_date, formula_version, status,
                    benchmark_symbol, benchmark_as_of_date, universe_hash,
                    expected_symbol_count, eligible_symbol_count,
                    excluded_symbol_count, diagnostics_json
                ) VALUES (
                    'US', '2026-04-10', :formula, 'completed',
                    'SPY', '2026-04-10', :universe_hash,
                    3, 3, 0, '{}'
                )
                """
            ),
            {
                "formula": BALANCED_RS_FORMULA_VERSION,
                "universe_hash": "a" * 64,
            },
        ).lastrowid
        balanced_values = {
            "formula": BALANCED_RS_FORMULA_VERSION,
            "run_id": run_id,
        }
        connection.execute(
            text(
                """
                INSERT INTO ibd_group_ranks (
                    market, industry_group, date, rank, avg_rs_rating,
                    avg_rs_rating_1m, avg_rs_rating_3m, num_stocks,
                    num_stocks_rs_above_80, rs_formula_version,
                    market_rs_run_id
                ) VALUES (
                    'US', 'Software', '2026-04-10', 1, 70.0,
                    40.0, 50.0, 3, 1, :formula, :run_id
                )
                """
            ),
            balanced_values,
        )
        formulas = connection.execute(
            text(
                "SELECT rs_formula_version FROM ibd_group_ranks "
                "WHERE market='US' AND industry_group='Software' "
                "ORDER BY rs_formula_version"
            )
        ).scalars().all()
        assert set(formulas) == {
            LEGACY_RS_FORMULA_VERSION,
            BALANCED_RS_FORMULA_VERSION,
        }

    with engine.begin() as connection, pytest.raises(IntegrityError):
        connection.execute(
            text(
                """
                INSERT INTO ibd_group_ranks (
                    market, industry_group, date, rank, avg_rs_rating,
                    num_stocks, num_stocks_rs_above_80,
                    rs_formula_version, market_rs_run_id
                ) VALUES (
                    'US', 'Software', '2026-04-10', 2, 60.0,
                    3, 0, :formula, :run_id
                )
                """
            ),
            balanced_values,
        )
    _run_revision(engine, "downgrade")
    columns = {column["name"] for column in inspect(engine).get_columns("ibd_group_ranks")}
    assert {
        "avg_rs_rating_1m",
        "avg_rs_rating_3m",
        "rs_formula_version",
        "market_rs_run_id",
    }.isdisjoint(columns)
    with engine.connect() as connection:
        remaining = connection.execute(
            text(
                "SELECT industry_group, avg_rs_rating FROM ibd_group_ranks "
                "WHERE market='US' AND date='2026-04-10'"
            )
        ).all()
    engine.dispose()
    assert remaining == [("Software", 95.0)]
