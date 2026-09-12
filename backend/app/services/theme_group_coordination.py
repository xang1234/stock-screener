"""Serialize grouping mutations and derived publications across worker processes.

A session advisory lock survives the internal commits of existing metric and
snapshot services. Mutations use the same key with transaction lifetime. Nested
publication calls reuse the outer lock rather than deadlocking on another pool
connection. SQLite uses a process lock solely for the unit-test harness.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from threading import RLock
from time import sleep

from sqlalchemy import text

# One key keeps cross-pipeline publications and mutations from competing for the
# same derived tables, while versions themselves remain pipeline-scoped.
GROUPING_LOCK_KEY = 739284615
_publication_active = ContextVar("theme_publication_active", default=False)
_test_lock = RLock()


def lock_grouping_mutation(db):
    if db.get_bind().dialect.name == "postgresql" and not _publication_active.get():
        db.execute(
            text("SELECT pg_advisory_xact_lock(:key)"), {"key": GROUPING_LOCK_KEY}
        )


@contextmanager
def publication_scope(db):
    if _publication_active.get():
        yield
        return
    engine = db.get_bind().engine
    if engine.dialect.name == "postgresql":
        connection = None
        while connection is None:
            candidate = engine.connect()
            try:
                acquired = candidate.execute(
                    text("SELECT pg_try_advisory_lock(:key)"),
                    {"key": GROUPING_LOCK_KEY},
                ).scalar_one()
                candidate.commit()
            except Exception:
                candidate.close()
                raise
            if acquired:
                connection = candidate
            else:
                candidate.close()
                sleep(0.05)
        token = _publication_active.set(True)
        try:
            yield
        finally:
            _publication_active.reset(token)
            try:
                connection.execute(
                    text("SELECT pg_advisory_unlock(:key)"),
                    {"key": GROUPING_LOCK_KEY},
                )
                connection.commit()
            except Exception:
                connection.invalidate()
                raise
            finally:
                connection.close()
    else:
        with _test_lock:
            token = _publication_active.set(True)
            try:
                yield
            finally:
                _publication_active.reset(token)
