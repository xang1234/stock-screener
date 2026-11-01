"""Upgrade/downgrade proof for the backtest replay picks migration."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.operations import Operations
from alembic.script import ScriptDirectory
from sqlalchemy.exc import IntegrityError

from app.infra.db.models.backtest_replay import BacktestReplayPick

BACKEND_ROOT = Path(__file__).resolve().parents[2]
MIGRATION_PATH = BACKEND_ROOT / "alembic" / "versions" / "20261101_0073_backtest_replay_picks.py"

_INSERT = sa.text(
    """
    INSERT INTO backtest_replay_picks (
        market, as_of_date, rank, symbol, minervini_score, rs_rating, stage,
        definition_version, market_rs_run_id
    ) VALUES ('US', '2026-05-01', :rank, :symbol, 91.5, 88, 2, :version, 7)
    """
)


def _load_migration():
    spec = importlib.util.spec_from_file_location("replay_picks_migration", MIGRATION_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_revision(engine, fn_name: str) -> None:
    module = _load_migration()
    with engine.begin() as connection:
        module.op = Operations(MigrationContext.configure(connection))
        getattr(module, fn_name)()


def test_replay_picks_migration_creates_one_pick_per_rank_and_drops_cleanly(tmp_path):
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'replay.sqlite'}")

    _run_revision(engine, "upgrade")

    with engine.begin() as connection:
        connection.execute(_INSERT, {"rank": 1, "symbol": "NVDA", "version": "minervini-top10-v1"})
        connection.execute(_INSERT, {"rank": 1, "symbol": "NVDA", "version": "minervini-top10-v2"})
        rows = connection.execute(
            sa.text(
                "SELECT definition_version, symbol, created_at IS NOT NULL "
                "FROM backtest_replay_picks ORDER BY definition_version"
            )
        ).all()
    assert rows == [("minervini-top10-v1", "NVDA", 1), ("minervini-top10-v2", "NVDA", 1)]
    with pytest.raises(IntegrityError), engine.begin() as connection:
        connection.execute(_INSERT, {"rank": 1, "symbol": "AMD", "version": "minervini-top10-v1"})
    with engine.connect() as connection:
        inspector = sa.inspect(connection)
        columns = [column["name"] for column in inspector.get_columns("backtest_replay_picks")]
        indexes = {
            index["name"]: index["column_names"]
            for index in inspector.get_indexes("backtest_replay_picks")
        }
    assert columns == [column.name for column in BacktestReplayPick.__table__.columns]
    assert indexes["ix_backtest_replay_picks_window"] == [
        "market", "definition_version", "as_of_date",
    ]

    _run_revision(engine, "downgrade")

    with engine.connect() as connection:
        assert "backtest_replay_picks" not in sa.inspect(connection).get_table_names()


def test_replay_picks_migration_extends_the_single_head():
    config = Config(str(BACKEND_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(BACKEND_ROOT / "alembic"))
    script = ScriptDirectory.from_config(config)

    assert script.get_heads() == [_load_migration().revision]
