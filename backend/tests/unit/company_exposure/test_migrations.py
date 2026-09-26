"""The company-exposure migration chain reproduces the ORM schema exactly."""

from __future__ import annotations

import importlib.util
from itertools import pairwise
from pathlib import Path

import pytest
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import MetaData, create_engine, inspect

import app.models  # noqa: F401
from app.database import Base

VERSIONS = Path(__file__).resolve().parents[3] / "alembic/versions"
PRECEDING_HEAD = "20260925_0057"
EXTERNAL_TABLES = ("stock_universe", "economic_themes")


def _chain():
    modules = []
    for path in sorted(VERSIONS.glob("*_exposure_*.py")):
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules.append(module)
    return modules


def _exposure_tables():
    return sorted(n for n in Base.metadata.tables if n.startswith("company_exposure_"))


def test_chain_is_linear_after_preceding_head():
    chain = _chain()
    assert chain, "no company-exposure migrations found"
    assert chain[0].down_revision == PRECEDING_HEAD
    for previous, current in pairwise(chain):
        assert current.down_revision == previous.revision


@pytest.mark.exposure_layer("schema")
def test_chain_matches_models_and_downgrades(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'chain.sqlite'}")
    chain = _chain()
    try:
        with engine.begin() as connection:
            for name in EXTERNAL_TABLES:
                Base.metadata.tables[name].create(connection)
            for migration in chain:
                migration.op = Operations(MigrationContext.configure(connection))
                migration.upgrade()
        subset = MetaData()
        for name in (*EXTERNAL_TABLES, *_exposure_tables()):
            Base.metadata.tables[name].to_metadata(subset)
        with engine.connect() as connection:
            diffs = compare_metadata(MigrationContext.configure(connection), subset)
        assert diffs == []
        with engine.begin() as connection:
            for migration in reversed(chain):
                migration.op = Operations(MigrationContext.configure(connection))
                migration.downgrade()
        remaining = set(inspect(engine).get_table_names())
        assert remaining.isdisjoint(_exposure_tables())
    finally:
        engine.dispose()
