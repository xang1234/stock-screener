"""Add expression indexes for feature-store preset filter fields.

The scan results endpoint filters/sorts ``stock_feature_daily`` rows on
screening fields that live inside the ``details_json`` blob (rs_rating,
minervini_score, stage, ...). Only ``composite_score`` / ``overall_rating``
were real indexed columns, so any preset that filtered a JSON field forced a
full scan that read every row's large ``details_json`` blob to evaluate the
``CAST(details_json ->> 'field' AS FLOAT)`` predicate. On the live US daily
run (~10k rows) that took >35s and tripped the 30s client timeout, so presets
appeared not to filter.

The query builder emits exactly ``CAST(details_json ->> '<field>' AS FLOAT)``
for these range filters (and always constrains ``run_id``). A composite
expression index ``(run_id, (CAST(details_json ->> '<field>' AS FLOAT)))``
matches that predicate, so the planner resolves the filter in the index
instead of reading every blob — verified via EXPLAIN to switch the plan from
a heap scan to a ``Bitmap Index Scan``.

No write-path or backfill change is needed: the index covers existing rows on
build and future rows on upsert. Postgres-only (the app requires PostgreSQL;
the unit-test harness builds the schema via metadata, not migrations).
"""

from __future__ import annotations

from alembic import op

revision = "20260617_0021"
down_revision = "20260601_0020"
branch_labels = None
depends_on = None

# Hot preset filter/sort fields stored at the TOP LEVEL of details_json. Each
# is filtered via FilterSpec.add_range -> json_number -> the flat
# CAST(details_json ->> 'f' AS FLOAT) expression below. rs_rating is the
# workhorse (nearly every preset constrains it); the rest seed the bitmap for
# score / growth / mover presets.
#
# Only flat top-level keys belong here — nested setup-engine fields (e.g.
# se_setup_score -> details_json -> 'setup_engine' ->> 'setup_score') would
# need a different expression and are deliberately excluded: SE presets already
# constrain rs_rating, so the filter uses that index and the residual sort runs
# over the narrowed set. test_feature_store_index_drift enforces the flat-path
# invariant for every entry here.
_FIELDS = [
    "rs_rating",
    "minervini_score",
    "canslim_score",
    "stage",
    "eps_growth_qq",
    "volume_breakthrough_score",
    "ipo_score",
    "perf_week",
    "price_change_1d",
]


def _index_name(field: str) -> str:
    return f"ix_sfd_run_{field}"


def _index_expr(field: str) -> str:
    """SQL for the indexed value.

    This MUST stay byte-identical (minus the table qualifier) to what
    ``feature_store_query.json_number()`` compiles to on Postgres for the same
    field — otherwise the planner silently declines the index and the filter
    falls back to a full scan. ``test_feature_store_index_drift`` pins that
    invariant so the linkage can't rot unnoticed.
    """
    return f"CAST(details_json ->> '{field}' AS FLOAT)"


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return
    for field in _FIELDS:
        op.execute(
            f"CREATE INDEX IF NOT EXISTS {_index_name(field)} "
            f"ON stock_feature_daily (run_id, ({_index_expr(field)}))"
        )


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return
    for field in _FIELDS:
        op.execute(f"DROP INDEX IF EXISTS {_index_name(field)}")
