"""Shared query-counting context manager for N+1 detection.

Hooks into SQLAlchemy's ``before_cursor_execute`` event to count SQL
statements and optionally capture their text for debugging.
"""

from __future__ import annotations

from contextlib import contextmanager

from sqlalchemy import event
from sqlalchemy.engine import Engine


@contextmanager
def count_queries(engine: Engine):
    """Context manager that counts SQL statements executed.

    Yields a dict with:
      - ``count``: number of SQL statements executed
      - ``statements``: list of SQL statement strings (for debugging)

    Usage::

        with count_queries(engine) as counter:
            repo.query_run(run_id)
        assert counter["count"] <= 3
    """
    counter: dict = {"count": 0, "statements": []}

    def _before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
        counter["count"] += 1
        counter["statements"].append(statement)

    event.listen(engine, "before_cursor_execute", _before_cursor_execute)
    try:
        yield counter
    finally:
        event.remove(engine, "before_cursor_execute", _before_cursor_execute)
