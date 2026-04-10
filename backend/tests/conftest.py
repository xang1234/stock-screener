"""
Shared pytest fixtures for backend tests.

Provides database session fixtures, mock data, and common test configuration.
"""
import pytest
import sys
from pathlib import Path

# Add backend directory to path for imports
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from app.database import SessionLocal, engine, Base


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
