"""SQL aggregate readers for persisted opportunity-state projections."""

from __future__ import annotations

from collections import defaultdict

from sqlalchemy import (
    ColumnElement,
    Select,
    String,
    cast,
    func,
    literal_column,
    select,
)
from sqlalchemy.orm import Session, sessionmaker

from app.domain.scanning.opportunity_state import ActionState
from app.domain.scanning.opportunity_summary import OpportunityStateSummary
from app.infra.db.models.feature_store import StockFeatureDaily
from app.infra.db.portability import json_text
from app.models.scan_result import ScanResult

# The two details_json keys the opportunity projection carries, in the order the
# index ``ix_sfd_run_action_state_survivor`` (migration ``20260926_0058``) stores
# them. ``action_state`` is filtered, so it has to sit directly after the
# ``run_id`` prefix to be reachable as an index condition; ``correction_survivor``
# is only ever tested for truth and goes last.
ACTION_STATE_KEY = "action_state"
CORRECTION_SURVIVOR_KEY = "correction_survivor"

# The two spellings of JSON ``true`` that ``->>`` can yield, one per backend.
# PostgreSQL keeps the boolean and renders it as text ``'true'``; SQLite stores it
# as the integer ``1``, so ``->>`` returns ``'1'``. The counted predicate has to
# match both, or identical data counts differently depending on the backend --
# see ``counted_opportunity_predicates``.
_SURVIVOR_TRUE_TEXT = "true"
_SURVIVOR_TRUE_SQLITE = "1"


def survivor_predicate(details) -> ColumnElement:
    """The one survivor test, for any ``details_json``/``details`` column.

    Single definition on purpose. ``for_feature_run`` (counted) and
    ``for_scan``/``_aggregate`` (grouped) read the same projection through
    different shapes, and before this they decided the survivor flag with
    different SQL: the grouped path used ``as_boolean().is_(True)``, the counted
    path a text comparison. Identical data then classified differently depending
    on which path read it -- on PostgreSQL the cast also accepts ``'t'``,
    ``'yes'`` and ``'on'``, which the text comparison rejected, and it raises on
    a value like ``'2'`` while the text comparison does not.

    Cast-free by requirement: this expression is also the one migration
    ``20260926_0058`` indexes, and ``CREATE INDEX`` evaluates it against every
    row of every historical run. A cast that can raise would fail the startup
    migration on a single bad value.

    ``IN`` matching ``lower(...)`` needs both members because the two backends
    spell JSON ``true`` differently: PostgreSQL keeps the boolean and ``->>``
    renders the text ``'true'``; SQLite stores it as the integer ``1``, so its
    ``->>`` yields ``'1'``. ``'1'`` never occurs in the Postgres text form, so
    the extra member is inert there.

    Values outside the list -- ``'t'``, ``'yes'``, ``'on'``, ``'2'`` -- are not
    survivors. That is the deliberate reading: only the two canonical spellings
    of the flag count, and no input can make this raise.
    """
    return func.lower(json_text(details, (CORRECTION_SURVIVOR_KEY,))).in_(
        [_SURVIVOR_TRUE_TEXT, _SURVIVOR_TRUE_SQLITE]
    )


def counted_opportunity_predicates() -> tuple[ColumnElement, ColumnElement]:
    """The two predicates ``ix_sfd_run_action_state_survivor`` was built for.

    Returned as SQLAlchemy expressions for the *runtime* callers (the counted
    feature-run path and the drift guard) rather than only as SQL text, so a
    change here is what the migration is compared against -- one definition, two
    consumers, no second copy to forget.

    Both keys go through ``portability.json_text``, which inlines them as SQL
    literals. ``details["action_state"].as_string()`` would render
    ``->> %(details_json_1)s`` instead: psycopg2 interpolates that client-side,
    but a generic plan (and every server-side-binding driver) sees a parameter,
    cannot match it to the index's literal, and silently drops the index -- which
    on this table means a 154 s full scan instead of milliseconds.

    The survivor test is a cast-free ``lower(...) IN ('true', '1')`` rather than
    ``CAST(... AS BOOLEAN) IS true``. A boolean cast raises
    ``invalid input syntax for type boolean`` on non-boolean text, and inside the
    index that fails the startup migration (one bad value anywhere in the table)
    and every later insert/update carrying such a value. The ``IN`` list cannot
    raise for any input.

    Both list members are needed for the two backends to agree. PostgreSQL stores
    a JSON ``true`` and ``->>`` hands it back as the text ``'true'``; SQLite
    stores the same value as the integer ``1``, so its ``->>`` yields ``'1'``.
    Matching only ``'true'`` silently counts every survivor as a non-survivor on
    SQLite, where the grouped pivot used to read it as truthy -- so the survivor
    counts would differ by backend for identical data. ``'1'`` never occurs in
    the Postgres text form, so the extra member is inert there.
    """
    details = StockFeatureDaily.details_json
    return (
        cast(json_text(details, (ACTION_STATE_KEY,)), String),
        survivor_predicate(details),
    )


class SqlOpportunityStateSummaryRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def for_scan(self, scan_id: str) -> OpportunityStateSummary:
        """Aggregate the opportunity projection for one legacy scan run."""
        return self._aggregate(
            model=ScanResult,
            details=ScanResult.details,
            predicate=ScanResult.scan_id == scan_id,
        )

    def for_feature_run(self, run_id: int) -> OpportunityStateSummary:
        """Aggregate the opportunity projection for one feature-store run.

        Counted rather than grouped: the feature-store table is large enough per
        run that the grouped shape's per-row document reads dominate the Daily
        Snapshot build. See ``_count_feature_run``.
        """
        return self._count_feature_run(int(run_id))

    def _count_feature_run(self, run_id: int) -> OpportunityStateSummary:
        """Count the projection with both keys in the ``WHERE`` clause.

        The grouped pass (``_aggregate``) reads ``action_state`` and
        ``correction_survivor`` out of every row and re-parses each row's stored
        ``details_json`` to do it -- 10,079 rows at ~64 kB measured 19,357 ms on
        the production table. Neither ``Index Scan`` nor ``Bitmap Heap Scan`` is
        an index-only scan, so an expression index on the two *grouped* keys does
        not remove that work: the planner picked such an index up and the time
        did not move (19,768 ms).

        Counting per state instead of grouping moves both keys into ``WHERE``,
        where ``ix_sfd_run_action_state_survivor`` (migration ``20260926_0058``)
        serves them as index conditions and the stored document is read far less.
        Measured against a copy of the production table, with the index and its
        ``ANALYZE``: 6,281 / 6,162 / 8,667 / 9,646 / 12,537 ms disk-bound, and
        12.292 ms buffer-warm. In the same session the grouped pass stayed at
        19,176-19,465 ms.

        ``rows_total`` and ``survivor_count`` are served as index-only scans; the
        per-state counts fall back to a ``Bitmap Heap Scan`` when the state
        matches thousands of rows, so they still visit the heap. That residual
        cost is why this rewrite and the migration only make sense together --
        the grouped shape reads every document rather than a subset. Without the
        index this shape costs 154,462 ms, since each of the sixteen counts scans
        the run on its own.

        The keys are extracted through ``json_text`` rather than
        ``details["key"].as_string()``. That is not a stylistic choice: the
        subscript form renders the key as a *bind parameter*
        (``->> %(details_json_1)s``), which a generic plan cannot match to the
        index's inline literal, so the index would silently go unused and this
        shape would fall back to the 154 s full-scan case. ``json_text`` exists
        for exactly this and inlines the key.

        The unknown-state contract is preserved: an unrecognised ``action_state``
        lands in no bucket, as it did when the pivot skipped it, while still
        counting towards ``rows_total`` and, for a survivor, towards
        ``survivor_count``.
        """
        # Read through the shared definition, so the predicates the planner sees
        # and the ones the index was built from cannot drift apart.
        #
        # The survivor test stays in SQL, as it was upstream. Deciding it in
        # Python instead makes ``JSON_EXTRACT(...) IS 1`` collapse to
        # ``bool(...)`` on SQLite, where the string "false" and the integer 2
        # are both truthy -- so identical data would count differently
        # depending on the backend.
        action_state, survivor = counted_opportunity_predicates()
        predicate = StockFeatureDaily.run_id == run_id

        def counted(*conditions) -> Select:
            """``count(*)`` over the run, narrowed by *conditions*.

            Each count emits the same expression text ``_aggregate`` used, so
            the indexed predicate and the query predicate stay byte-identical
            (minus the table qualifier). ``test_opportunity_summary_index_drift``
            pins that against the migration.
            """
            return (
                select(func.count())
                .select_from(StockFeatureDaily)
                .where(predicate, *conditions)
                .scalar_subquery()
            )

        columns = [
            counted().label("rows_total"),
            counted(survivor).label("survivor_count"),
        ]
        for state in ActionState:
            state_predicate = action_state == state.value
            columns.append(counted(state_predicate).label(f"state_{state.value}"))
            columns.append(
                counted(state_predicate, survivor).label(f"survivor_{state.value}")
            )

        # One statement, one round trip: the sixteen counts are correlated
        # scalar subqueries, not sixteen separate queries.
        row = self._session.execute(select(*columns)).one()
        values = row._mapping

        action_state_counts: dict[ActionState, int] = defaultdict(int)
        survivor_action_state_counts: dict[ActionState, int] = defaultdict(int)
        for state in ActionState:
            action_state_counts[state] = values[f"state_{state.value}"]
            survivor_action_state_counts[state] = values[f"survivor_{state.value}"]

        return OpportunityStateSummary(
            rows_total=values["rows_total"],
            survivor_count=values["survivor_count"],
            action_state_counts=dict(action_state_counts),
            survivor_action_state_counts=dict(survivor_action_state_counts),
        )

    def _aggregate(self, *, model, details, predicate) -> OpportunityStateSummary:
        """Read ``action_state`` and ``correction_survivor`` once per row,
        group by the two, and pivot the buckets into a summary.

        Both columns are ``JSON`` rather than ``JSONB``, so each ``->>`` still
        parses the whole stored breakdown: two reads per row is the floor here
        without a schema change. The previous shape paid that per key *and*
        per action state, sixteen parses for the same answer.

        ``details`` is ``scan_results.details`` for a legacy scan run and
        ``stock_feature_daily.details_json`` for a feature-store run; the keys
        are the same in both.
        """
        action_state = details["action_state"].as_string()
        # The survivor test goes through the shared definition, so the grouped
        # path cannot classify a row differently from the counted one. It stays
        # in SQL -- deciding it in Python collapses ``JSON_EXTRACT(...) IS 1`` to
        # ``bool(...)`` on SQLite, where the string "false" and the integer 2 are
        # both truthy, so identical data would count differently by backend.
        survivor = survivor_predicate(details)
        buckets = (
            self._session.query(
                survivor.label("survivor"),
                action_state.label("action_state"),
                func.count().label("rows"),
            )
            .select_from(model)
            .filter(predicate)
            # Group by output position. Repeating the expressions would rely on
            # the driver inlining its bind parameters: psycopg2 interpolates on
            # the client, so the server sees identical text, but a driver that
            # binds server-side (asyncpg) emits $1 in the SELECT and $3 in the
            # GROUP BY, and PostgreSQL then rejects it as a grouping error.
            .group_by(literal_column("1"), literal_column("2"))
            .all()
        )

        rows_total = 0
        survivor_count = 0
        action_state_counts: dict[ActionState, int] = defaultdict(int)
        survivor_action_state_counts: dict[ActionState, int] = defaultdict(int)

        for is_survivor, raw_state, rows in buckets:
            rows_total += rows
            if is_survivor:
                survivor_count += rows
            # Unknown states stay out of every bucket; they are still counted
            # in rows_total so the totals never silently drift.
            try:
                state = ActionState(raw_state)
            except ValueError:
                continue
            action_state_counts[state] += rows
            if is_survivor:
                survivor_action_state_counts[state] += rows

        return OpportunityStateSummary(
            rows_total=rows_total,
            survivor_count=survivor_count,
            action_state_counts=dict(action_state_counts),
            survivor_action_state_counts=dict(survivor_action_state_counts),
        )


class SessionOpportunityStateSummaryReader:
    """Own a short-lived session for telemetry and other service consumers."""

    def __init__(self, session_factory: sessionmaker) -> None:
        self._session_factory = session_factory

    def for_scan(self, scan_id: str) -> OpportunityStateSummary:
        with self._session_factory() as session:
            return SqlOpportunityStateSummaryRepository(session).for_scan(scan_id)

    def for_feature_run(self, run_id: int) -> OpportunityStateSummary:
        with self._session_factory() as session:
            return SqlOpportunityStateSummaryRepository(session).for_feature_run(run_id)
