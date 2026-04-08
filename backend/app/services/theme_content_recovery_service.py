"""Recovery helpers for rebuildable theme-content storage."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from threading import Lock

from sqlalchemy import text

from ..database import engine
from ..infra.db.portability import is_postgres
from ..models.theme import ContentItem, ContentItemPipelineState, ThemeMention

logger = logging.getLogger(__name__)

_THEME_CONTENT_STORAGE_LOCK = Lock()
_THEME_CONTENT_RESET_LOOKBACK_DAYS = 365
_THEME_CONTENT_REINDEX_TARGETS = (
    "content_items",
    "content_sources",
    "theme_mentions",
    "content_item_pipeline_state",
)


def attempt_reindex_theme_content_storage(exc: Exception) -> bool:
    """Try to repair theme browser read paths by rebuilding relevant indexes."""
    logger.warning(
        "Attempting REINDEX for theme content browser after corruption signature: %s",
        exc,
    )
    try:
        with _THEME_CONTENT_STORAGE_LOCK:
            with engine.begin() as conn:
                for target in _THEME_CONTENT_REINDEX_TARGETS:
                    conn.execute(text(f'REINDEX TABLE "{target}"'))
    except Exception as reindex_exc:
        from ..database import is_corruption_error

        if not is_corruption_error(reindex_exc):
            raise
        logger.warning(
            "REINDEX did not clear theme content browser corruption: %s",
            reindex_exc,
        )
        return False
    logger.warning("REINDEX completed for theme content browser recovery")
    return True


def reset_corrupt_theme_content_storage(exc: Exception) -> None:
    """Drop and recreate rebuildable theme content tables after database corruption."""
    logger.warning(
        "Resetting theme content storage after database corruption signature: %s",
        exc,
    )
    with _THEME_CONTENT_STORAGE_LOCK:
        with engine.begin() as conn:
            drop_theme_content_tables(conn)
        with engine.begin() as conn:
            rewind_theme_content_source_cursors(conn)
        recreate_theme_content_tables()


def drop_theme_content_tables(conn) -> None:
    """Drop rebuildable theme content storage tables using normal DDL."""
    cascade = " CASCADE" if is_postgres(conn) else ""
    conn.execute(text(f"DROP TABLE IF EXISTS content_item_pipeline_state{cascade}"))
    conn.execute(text(f"DROP TABLE IF EXISTS theme_mentions{cascade}"))
    conn.execute(text(f"DROP TABLE IF EXISTS content_items{cascade}"))


def force_forget_theme_content_tables(conn) -> None:
    """Force-drop theme content storage tables (alias for drop)."""
    drop_theme_content_tables(conn)


def recreate_theme_content_tables() -> None:
    """Recreate theme content storage tables after a corruption reset."""
    ContentItem.__table__.create(bind=engine, checkfirst=True)
    ThemeMention.__table__.create(bind=engine, checkfirst=True)
    ContentItemPipelineState.__table__.create(bind=engine, checkfirst=True)


def rewind_theme_content_source_cursors(conn) -> None:
    """Rewind content-source cursors so the next poll repopulates article history."""
    rewind_at = datetime.utcnow() - timedelta(days=_THEME_CONTENT_RESET_LOOKBACK_DAYS)
    conn.execute(
        text(
            """
            UPDATE content_sources
            SET last_fetched_at = :rewind_at,
                total_items_fetched = 0
            """
        ),
        {"rewind_at": rewind_at},
    )
