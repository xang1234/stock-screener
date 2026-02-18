"""Shared fixtures for repository integration tests.

Provides an in-memory SQLite engine with FK enforcement, a session
factory, and a query-counting context manager for N+1 detection.
"""

from __future__ import annotations

from contextlib import contextmanager

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from app.database import Base

# Force model registration so create_all picks up every table.
import app.models.scan_result  # noqa: F401
import app.infra.db.models.feature_store  # noqa: F401


@pytest.fixture
def engine():
    """Function-scoped :memory: SQLite engine with FK enforcement."""
    eng = create_engine("sqlite:///:memory:")

    @event.listens_for(eng, "connect")
    def _set_fk_pragma(dbapi_conn, _):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(eng)
    yield eng
    eng.dispose()


@pytest.fixture
def session(engine):
    """Function-scoped session bound to the in-memory engine."""
    factory = sessionmaker(bind=engine)
    sess = factory()
    yield sess
    sess.close()


@contextmanager
def count_queries(engine):
    """Context manager that counts SQL statements executed.

    Usage::

        with count_queries(engine) as counter:
            repo.query_run(run_id)
        assert counter["count"] <= 2
    """
    counter = {"count": 0}

    def _before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
        counter["count"] += 1

    event.listen(engine, "before_cursor_execute", _before_cursor_execute)
    try:
        yield counter
    finally:
        event.remove(engine, "before_cursor_execute", _before_cursor_execute)
