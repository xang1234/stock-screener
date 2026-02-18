"""Shared fixtures for repository integration tests.

Provides an in-memory SQLite engine with FK enforcement and a session
factory.  The ``count_queries`` context manager lives in
``tests.helpers.query_counter`` and is re-exported here for backward
compatibility.
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from app.database import Base

# Force model registration so create_all picks up every table.
import app.models.scan_result  # noqa: F401
import app.infra.db.models.feature_store  # noqa: F401

# Re-export shared helper so existing ``from .conftest import count_queries`` works.
from tests.helpers.query_counter import count_queries  # noqa: F401


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
