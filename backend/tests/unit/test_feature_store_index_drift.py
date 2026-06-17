"""Drift guard for the feature-store preset-filter expression indexes.

Migration ``20260617_0021`` creates Postgres expression indexes whose SQL must
stay byte-identical (minus the table qualifier) to what the query builder
compiles for the same field — otherwise the planner silently declines the
index and the filter falls back to a full scan with no error. This test pins
that linkage so a change to ``json_number`` / ``_JSON_FIELD_MAP`` can't rot the
indexes unnoticed.

It also enforces that every indexed field is a *flat* top-level details_json
key, since the migration's ``_index_expr`` only emits the single-segment
``details_json ->> 'field'`` form (a nested path needs a different expression).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from sqlalchemy.dialects import postgresql

from app.infra.db.models.feature_store import StockFeatureDaily
from app.infra.db.portability import json_number
from app.infra.query.feature_store_query import _JSON_FIELD_MAP

_MIGRATION = (
    Path(__file__).parents[2]
    / "alembic"
    / "versions"
    / "20260617_0021_add_feature_store_preset_filter_indexes.py"
)


def _load_migration():
    spec = importlib.util.spec_from_file_location("_wsb_migration", _MIGRATION)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _builder_expr(field: str) -> str:
    """The flat extraction the query builder compiles for *field* on Postgres,
    minus the table qualifier (the index DDL is unqualified).

    Compiled WITHOUT ``literal_binds`` on purpose: this is the runtime form, so
    if the JSON key ever regresses to a bind parameter (``->> $1``) the string
    won't match the literal-key index expression and this test fails — that bind
    param is precisely what makes a generic plan skip the index.
    """
    compiled = str(
        json_number(StockFeatureDaily.details_json, (field,)).compile(
            dialect=postgresql.dialect(),
        )
    )
    return compiled.replace("stock_feature_daily.", "")


def test_indexed_fields_are_flat_top_level_keys():
    """A nested path would make the flat index expression index the wrong key."""
    migration = _load_migration()
    for field in migration._FIELDS:
        path = _JSON_FIELD_MAP.get(field)
        assert path is not None, f"{field} is not a JSON details field"
        assert len(path) == 1, (
            f"{field} maps to nested path {path}; the migration's flat "
            f"_index_expr would index the wrong key"
        )


def test_index_expr_matches_query_builder():
    """The index expression must match what the filter predicate compiles to."""
    migration = _load_migration()
    for field in migration._FIELDS:
        assert migration._index_expr(field) == _builder_expr(field), (
            f"index expression for {field} drifted from json_number(); the "
            f"Postgres planner will stop using ix_sfd_run_{field}"
        )
