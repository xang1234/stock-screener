"""Regression tests for the lean count optimization in scan_result_query.

``apply_sort_and_paginate`` used ``Query.count()``, which SQLAlchemy emits as
``SELECT count(*) FROM (<full entity SELECT>)`` — wrapping the heavy
multi-column, double-outer-join SELECT (details JSON + sparkline blobs) in a
subquery. On large scans a *filtered* count then read every row's blobs and
dominated query time (25-90s observed live). ``lean_count`` emits a flat
``SELECT count(*) FROM ... WHERE ...`` instead.

These tests pin two invariants:
1. Correctness — the lean count equals the old ``Query.count()`` for every
   filter shape (the joins are 1:1 on symbol, so the total is unchanged).
2. Shape — the emitted SQL is flat: a ``count(*)`` with no subquery wrapper
   and no ``details`` / sparkline projection.
"""

from __future__ import annotations

from sqlalchemy import func

from app.domain.scanning.filter_spec import FilterSpec
from app.infra.db.portability import lean_count
from app.infra.db.repositories.scan_result_repo import _scan_results_query
from app.infra.query.scan_result_query import apply_filters
from app.models.scan_result import ScanResult

_SCAN_ID = "scan-lean-count"


def _seed(session):
    rows = [
        # symbol, composite, stage, rs_rating, minervini, ma_alignment
        ("AAA", 90.0, 2, 88.0, 81.0, True),
        ("BBB", 70.0, 2, 82.0, 60.0, True),
        ("CCC", 55.0, 4, 50.0, 10.0, False),
        ("DDD", 40.0, 1, 30.0, None, False),
        ("EEE", None, 3, 91.0, 5.0, True),   # null composite
        ("FFF", 65.0, 2, 79.0, 40.0, None),  # null ma_alignment
    ]
    for sym, comp, stage, rs, minv, ma in rows:
        session.add(
            ScanResult(
                scan_id=_SCAN_ID,
                symbol=sym,
                composite_score=comp,
                stage=stage,
                rs_rating=rs,
                minervini_score=minv,
                details={"ma_alignment": ma},
            )
        )
    session.commit()


def _filtered_query(session, spec: FilterSpec):
    return apply_filters(_scan_results_query(session, _SCAN_ID), spec)


def test_lean_count_matches_orm_count_no_filter(universe_session):
    _seed(universe_session)
    q = _filtered_query(universe_session, FilterSpec())
    assert lean_count(q) == q.count() == 6


def test_lean_count_matches_orm_count_sql_column_filter(universe_session):
    _seed(universe_session)
    spec = FilterSpec().add_range("rs_rating", 80, None)  # AAA, BBB, EEE
    q = _filtered_query(universe_session, spec)
    assert lean_count(q) == q.count() == 3


def test_lean_count_matches_orm_count_boolean_json_filter(universe_session):
    _seed(universe_session)
    spec = FilterSpec().add_boolean("ma_alignment", True)  # AAA, BBB, EEE
    q = _filtered_query(universe_session, spec)
    assert lean_count(q) == q.count() == 3


def test_lean_count_matches_orm_count_combined_filter(universe_session):
    _seed(universe_session)
    spec = (
        FilterSpec()
        .add_range("rs_rating", 75, None)
        .add_range("composite_score", 60, None)
        .add_boolean("ma_alignment", True)
    )  # AAA (90/88/T), BBB (70/82/T); EEE has null composite -> excluded
    q = _filtered_query(universe_session, spec)
    assert lean_count(q) == q.count() == 2


def test_lean_count_sql_is_flat(universe_session):
    """The lean count must not wrap the heavy SELECT or project blob columns."""
    _seed(universe_session)
    q = _filtered_query(universe_session, FilterSpec().add_range("rs_rating", 80, None))
    sql = str(q.order_by(None).with_entities(func.count())).lower()
    assert "count(" in sql
    # No subquery wrapper and no heavy-column projection.
    assert "from (select" not in sql
    assert "details" not in sql
    assert "sparkline" not in sql
