"""Resolve a saved watchlist to its symbols in display order."""

from __future__ import annotations

from sqlalchemy.orm import Session

from ..models.user_watchlist import UserWatchlist, WatchlistItem


def watchlist_symbols(db: Session, watchlist_id: int) -> list[str] | None:
    """Return the watchlist's symbols in display order, or None if it does not exist."""

    rows = (
        db.query(WatchlistItem.symbol)
        .select_from(UserWatchlist)
        .outerjoin(WatchlistItem, WatchlistItem.watchlist_id == UserWatchlist.id)
        .filter(UserWatchlist.id == watchlist_id)
        .order_by(WatchlistItem.position, WatchlistItem.id)
        .all()
    )
    if not rows:
        return None
    return [symbol for (symbol,) in rows if symbol is not None]
