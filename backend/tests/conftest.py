"""
Shared pytest fixtures for backend tests.

Provides database session fixtures, mock data, and common test configuration.
"""
import os
import pytest
import sys
from pathlib import Path

# Add backend directory to path for imports
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

# Keep backend tests independent from a developer's local backend/.env and from
# CI job-level DATABASE_URL placeholders. Tests default to the shared SQLite
# harness unless a caller explicitly opts into using the supplied DATABASE_URL.
_allow_postgres = os.environ.get("STOCKSCANNER_TEST_ALLOW_POSTGRES") == "1"
_allow_postgres = _allow_postgres or (
    os.environ.get("STOCKSCANNER_TEST_USE_DATABASE_URL") == "1"
)

if _allow_postgres and os.environ.get("DATABASE_URL"):
    os.environ.pop("STOCKSCANNER_TEST_ALLOW_SQLITE", None)
else:
    os.environ["DATABASE_URL"] = "sqlite://"
    os.environ["STOCKSCANNER_TEST_ALLOW_SQLITE"] = "1"

pytest_plugins = ["tests.company_exposure_case_plugin"]

# Admin endpoint tests configure keys individually. Give those keys a stable
# test-only audit identity unless a case explicitly overrides it.
os.environ.setdefault("ADMIN_PRINCIPAL_ID", "test:admin")

import app.models  # noqa: F401
from app.database import SessionLocal, engine, Base


# Rebuilding ~200 tables costs ~0.1s per test on SQLite and ~2.5s on
# PostgreSQL, so the schema is rebuilt only when DDL may have changed it since
# the last reset; otherwise rows (and PostgreSQL sequences) are wiped.
_PG_DDL_EPOCH_SETUP = """
CREATE SCHEMA IF NOT EXISTS pytest_meta;
CREATE SEQUENCE IF NOT EXISTS pytest_meta.ddl_epoch;
CREATE OR REPLACE FUNCTION pytest_meta.bump_ddl_epoch() RETURNS event_trigger
LANGUAGE plpgsql AS $$
BEGIN PERFORM nextval('pytest_meta.ddl_epoch');
EXCEPTION WHEN OTHERS THEN NULL;
END $$;
DROP EVENT TRIGGER IF EXISTS pytest_ddl_epoch;
CREATE EVENT TRIGGER pytest_ddl_epoch ON ddl_command_end
EXECUTE FUNCTION pytest_meta.bump_ddl_epoch();
-- A new sequence reports last_value = start both before and after its first
-- nextval, which would hide the first DDL command; advance it once.
SELECT nextval('pytest_meta.ddl_epoch');
"""
_clean_schema_epoch = None


def _schema_epoch(conn):
    if conn.dialect.name == "sqlite":
        return conn.exec_driver_sql("PRAGMA schema_version").scalar()
    return conn.exec_driver_sql("SELECT last_value FROM pytest_meta.ddl_epoch").scalar()


def _wipe_rows(conn):
    tables = Base.metadata.sorted_tables
    if conn.dialect.name == "sqlite":
        for table in reversed(tables):
            conn.execute(table.delete())
        return
    # TRUNCATE bypasses the append-only row triggers and is not DDL, so it
    # leaves the epoch alone. Probe first: truncating all tables costs ~0.8s.
    probe = " UNION ALL ".join(
        f"SELECT '{t.name}' WHERE EXISTS (SELECT 1 FROM \"{t.name}\")" for t in tables
    )
    dirty = [row[0] for row in conn.exec_driver_sql(probe)]
    if dirty:
        names = ", ".join(f'"{name}"' for name in dirty)
        conn.exec_driver_sql(f"TRUNCATE {names} RESTART IDENTITY CASCADE")
    # Rolled-back inserts advance sequences without leaving rows behind.
    conn.exec_driver_sql(
        "SELECT setval((quote_ident(schemaname) || '.' || quote_ident(sequencename))::regclass, "
        "start_value, false) "
        "FROM pg_sequences WHERE schemaname = 'public' AND last_value IS NOT NULL"
    )


def _reset_test_database():
    global _clean_schema_epoch
    if _clean_schema_epoch is not None:
        with engine.begin() as conn:
            try:
                unchanged = _schema_epoch(conn) == _clean_schema_epoch
            except Exception:  # a test dropped the tracker itself
                unchanged = False
            if unchanged:
                _wipe_rows(conn)
                return
    _clean_schema_epoch = None
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    try:
        with engine.begin() as conn:
            if conn.dialect.name == "postgresql":
                conn.exec_driver_sql(_PG_DDL_EPOCH_SETUP)  # needs superuser
            _clean_schema_epoch = _schema_epoch(conn)
    except Exception:
        _clean_schema_epoch = None  # untracked: rebuild before every test


@pytest.fixture(autouse=True)
def isolated_session_secret(tmp_path_factory, monkeypatch):
    """Never create or cache the checkout's real data/.server_auth_session_secret."""
    from app.services import server_auth

    monkeypatch.setattr(
        server_auth,
        "_SESSION_SECRET_PATH",
        tmp_path_factory.mktemp("session-secret") / ".server_auth_session_secret",
    )
    server_auth._load_or_create_session_secret.cache_clear()
    yield
    server_auth._load_or_create_session_secret.cache_clear()


@pytest.fixture(autouse=True)
def shared_test_database():
    """Give each test an empty database with the current model schema."""
    _reset_test_database()
    yield


@pytest.fixture(autouse=True)
def runtime_services_context():
    """Provide a fresh runtime container for each test."""
    from app.wiring.bootstrap import (
        build_runtime_services,
        clear_runtime_services,
        set_runtime_services,
    )

    runtime_services = build_runtime_services(session_factory=SessionLocal)
    set_runtime_services(runtime_services, bind_process=True)
    try:
        yield runtime_services
    finally:
        try:
            runtime_services.reset_for_tests()
        finally:
            clear_runtime_services()


@pytest.fixture(scope="function")
def db_session():
    """
    Provides a database session for tests.

    Uses the existing database connection - tests should not modify
    production data unless explicitly intended.
    """
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture(scope="module")
def db_session_module():
    """
    Provides a database session scoped to the module level.

    More efficient for read-only tests that don't need isolation.
    """
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def test_symbols():
    """Common test symbols - mix of growth stocks."""
    return ['AAPL', 'NVDA', 'MSFT']


@pytest.fixture
def single_test_symbol():
    """Single test symbol for quick tests."""
    return 'AAPL'


@pytest.fixture
def scan_orchestrator():
    """Provides a ScanOrchestrator instance wired with production dependencies."""
    from app.wiring.bootstrap import get_scan_orchestrator
    return get_scan_orchestrator()


@pytest.fixture
def screener_registry():
    """Provides access to the screener registry."""
    from app.scanners.screener_registry import screener_registry
    return screener_registry
