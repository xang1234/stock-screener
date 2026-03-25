"""Database setup and session management using SQLAlchemy."""

from sqlalchemy import create_engine, event
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from .config import settings
from .infra.db.portability import is_postgres, is_sqlite


def _engine_kwargs() -> dict:
    kwargs = {
        "echo": False,
        "pool_pre_ping": True,
    }
    if is_sqlite(settings.database_url):
        kwargs["connect_args"] = {"check_same_thread": False}
    elif is_postgres(settings.database_url):
        # Keep the default conservative for the current multi-service Docker shape.
        kwargs["pool_size"] = 5
        kwargs["max_overflow"] = 5
    return kwargs


engine = create_engine(settings.database_url, **_engine_kwargs())


# Enable SQLite pragmas for WAL mode, concurrency, and foreign keys
if is_sqlite(settings.database_url):

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA busy_timeout=15000")
        cursor.execute("PRAGMA synchronous=FULL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

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
