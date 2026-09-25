"""Index the Daily Snapshot's volume and group-rank feature-store predicates.

``20260617_0021`` indexed the preset filter fields, but ``avg_dollar_volume``
and ``ibd_group_rank`` were not among them. Both sit on the Daily Snapshot read
path, which is not a saved preset:

* ``resolve_default_scan_filters`` supplies ``minVolume`` for every market
  (``US``: 100_000_000, ``DE``: 900_000), and that maps to
  ``_FIELD_BINDINGS["volume"] -> ("avg_dollar_volume",)``. The snapshot's
  top-candidate and leader queries therefore constrain it on every load.
* the leader query additionally constrains ``ibd_group_rank``.

Without an index the planner falls back to scanning every row of the run and
re-parsing ``details_json`` for each one. Measured on a 10,210-row run:
19,105 ms for the ``count(*)`` probe alone, versus 4.7 ms for ``rs_rating``
through its own index. The snapshot build then exceeds the client timeout, so
the payload is never cached and every reload rebuilds it.

``avg_dollar_volume`` resolves under two names: ``volume`` (the filter-facing
alias) and ``avg_dollar_volume`` (its own details_json key, added alongside this
migration so the drift guard can look the field up by the name indexed here).

PostgreSQL-only, matching ``20260617_0021`` and ``20260821_0028``: SQLite does
not parse these JSON operators. The indexes are built ``CONCURRENTLY`` inside an
autocommit block so a large run does not block writes while they are created.

Because the build runs outside a transaction, a failure partway through leaves
an *invalid* index behind under the target name. ``IF NOT EXISTS`` would then
treat the name as taken, so ``_rebuild_invalid_indexes`` drops any invalid
same-named index before the creates instead of silently finishing without a
usable one.

Deploy note: ``CONCURRENTLY`` avoids blocking writes but still reads every row's
``details_json`` — every run in ``stock_feature_daily``, not just the newest. On
a large table the first container start after this revision can spend a while
here, and a compose healthcheck tighter than that build would fail the rollout.
Build the two indexes ahead of the deploy if that becomes the case:

    CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_sfd_run_avg_dollar_volume
        ON stock_feature_daily (run_id, (CAST(details_json ->> 'avg_dollar_volume' AS FLOAT)));
    CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_sfd_run_ibd_group_rank
        ON stock_feature_daily (run_id, (CAST(details_json ->> 'ibd_group_rank' AS FLOAT)));
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20260925_0057"
down_revision = "20260925_0056"
branch_labels = None
depends_on = None

# Flat top-level details_json keys on the Daily Snapshot read path. The
# expression below is the single-segment form, so both fields must be flat —
# ``test_feature_store_index_drift`` asserts it.
_FIELDS = [
    "avg_dollar_volume",
    "ibd_group_rank",
]

_INVALID_INDEX_SQL = """
SELECT pg_catalog.format('%I.%I', n.nspname, idx.relname)
FROM pg_catalog.pg_index AS i
JOIN pg_catalog.pg_class AS idx ON idx.oid = i.indexrelid
JOIN pg_catalog.pg_namespace AS n ON n.oid = idx.relnamespace
WHERE i.indrelid = pg_catalog.to_regclass('stock_feature_daily')
  AND idx.relname = :index_name
  AND NOT i.indisvalid
"""


def _index_name(field: str) -> str:
    return f"ix_sfd_run_{field}"


def _index_expr(field: str) -> str:
    """SQL for the indexed value.

    Must stay byte-identical (minus the table qualifier) to what
    ``feature_store_query.json_number()`` compiles for the same field, or the
    planner silently declines the index. ``test_feature_store_index_drift``
    pins that linkage.
    """
    return f"CAST(details_json ->> '{field}' AS FLOAT)"


def _rebuild_invalid_indexes() -> None:
    """Drop same-named indexes a previous interrupted build left invalid.

    ``CREATE INDEX CONCURRENTLY`` cannot run inside a transaction and is not
    atomic: a failure leaves an invalid index under the target name. The
    following ``IF NOT EXISTS`` would consider the name taken and skip the
    build, so the snapshot would keep full-scanning with no error anywhere.

    Deliberately local to this revision. ``20260821_0028`` shares the lifecycle
    and has the same exposure, but it has already run in the field; rewriting a
    released migration is a different change from guarding a new one.
    """
    bind = op.get_bind()
    if op.get_context().as_sql:
        # Offline / ``--sql`` generation: there is no catalog to inspect, and
        # the emitted script is expected to contain only the CREATE statements.
        return
    for field in _FIELDS:
        name = _index_name(field)
        invalid = bind.execute(
            sa.text(_INVALID_INDEX_SQL), {"index_name": name}
        ).scalar()
        if invalid:
            with op.get_context().autocommit_block():
                op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {invalid}")


def _create_indexes() -> None:
    # Recovery runs before the creates: an invalid index left by an interrupted
    # CONCURRENTLY build would otherwise make ``IF NOT EXISTS`` skip the rebuild.
    _rebuild_invalid_indexes()
    with op.get_context().autocommit_block():
        for field in _FIELDS:
            op.execute(
                f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {_index_name(field)} "
                f"ON stock_feature_daily (run_id, ({_index_expr(field)}))"
            )


def _drop_indexes() -> None:
    with op.get_context().autocommit_block():
        for field in _FIELDS:
            op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {_index_name(field)}")


def upgrade() -> None:
    if op.get_bind().dialect.name == "postgresql":
        _create_indexes()


def downgrade() -> None:
    if op.get_bind().dialect.name == "postgresql":
        _drop_indexes()
