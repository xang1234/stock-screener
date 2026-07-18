from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from app.infra.db.models.relative_strength import (
    MarketRsFormulaPointer,
    MarketRsRun,
    StockRsSnapshot,
)
from app.models.industry import IBDGroupRank


def test_market_rs_models_expose_versioned_snapshot_contract():
    assert MarketRsRun.__table__.name == "market_rs_runs"
    assert StockRsSnapshot.__table__.primary_key.columns.keys() == ["run_id", "symbol"]
    assert MarketRsFormulaPointer.__table__.primary_key.columns.keys() == ["market"]
    assert {
        "avg_rs_rating_1m",
        "avg_rs_rating_3m",
        "rs_formula_version",
        "market_rs_run_id",
    }.issubset(IBDGroupRank.__table__.columns.keys())


def test_group_unique_constraint_includes_formula_version():
    names = {
        constraint.name
        for constraint in IBDGroupRank.__table__.constraints
        if constraint.name
    }
    assert "uix_ibd_group_rank_market_date_formula" in names


def test_canonical_market_rs_migration_extends_current_head():
    migration_path = (
        Path(__file__).resolve().parents[2]
        / "alembic"
        / "versions"
        / "20260718_0025_add_canonical_market_rs.py"
    )
    spec = spec_from_file_location("canonical_market_rs_migration", migration_path)
    assert spec is not None and spec.loader is not None
    migration = module_from_spec(spec)
    spec.loader.exec_module(migration)

    assert migration.revision == "20260718_0025"
    assert migration.down_revision == "20260701_0024"
