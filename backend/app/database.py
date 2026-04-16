"""Database setup and session management using SQLAlchemy."""

import os

from sqlalchemy import create_engine
from sqlalchemy.engine import make_url
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.pool import StaticPool

from .config import settings

_parsed_url = make_url(settings.database_url)
_backend_name = _parsed_url.get_backend_name()
_allow_test_sqlite = (
    _backend_name == "sqlite"
    and os.getenv("STOCKSCANNER_TEST_ALLOW_SQLITE") == "1"
)

if _backend_name != "postgresql" and not _allow_test_sqlite:
    raise ValueError(
        f"Only PostgreSQL is supported. Got DATABASE_URL with backend "
        f"'{_backend_name}'. Use: postgresql://user:pass@host/dbname"
    )

_engine_kwargs = {
    "echo": False,
}

if _allow_test_sqlite:
    # Shared in-memory SQLite is permitted only for the test harness.
    _engine_kwargs.update(
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
else:
    _engine_kwargs.update(
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=5,
    )

engine = create_engine(settings.database_url, **_engine_kwargs)

# Create SessionLocal class for database sessions
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for all database models
Base = declarative_base()


def get_db():
    """
    Dependency function to get database session.

    Yields:
        Session: Database session

    Usage:
        @app.get("/endpoint")
        def endpoint(db: Session = Depends(get_db)):
            ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """
    Initialize database by creating all tables.
    Should be called on application startup.
    """
    # Import all models here to ensure they are registered
    from . import models

    # Create all tables
    Base.metadata.create_all(bind=engine)


def is_corruption_error(exc: Exception) -> bool:
    """Detect low-level database corruption signatures."""
    msg = str(exc).lower()
    return any(s in msg for s in (
        "malformed", "disk image", "file is not a database", "disk i/o error",
        "could not read block", "invalid page", "database system is in recovery mode",
    ))


def safe_rollback(db):
    """Rollback that won't raise on a corrupted DB."""
    try:
        db.rollback()
    except Exception:
        pass
