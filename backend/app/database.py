"""
Database setup and session management using SQLAlchemy.
"""
from sqlalchemy import create_engine, event
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from .config import settings

# Create SQLAlchemy engine
engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False} if "sqlite" in settings.database_url else {},
    echo=False,  # Set to True for SQL query logging
    pool_pre_ping=True,  # Verify connections are alive before use (catches stale conns in Celery workers)
)


# Enable SQLite pragmas for WAL mode, concurrency, and foreign keys
if "sqlite" in settings.database_url:

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
    """Detect SQLite database corruption from exception messages."""
    msg = str(exc).lower()
    return any(s in msg for s in (
        "malformed", "disk image", "file is not a database", "disk i/o error",
    ))


def safe_rollback(db):
    """Rollback that won't raise on a corrupted DB."""
    try:
        db.rollback()
    except Exception:
        pass
