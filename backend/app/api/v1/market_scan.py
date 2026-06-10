"""
API endpoints for Market Scan feature.
Handles CRUD operations for market scan watchlists.
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List

from ...database import get_db
from ...domain.markets.key_markets import key_market_watchlist_defaults
from ...models.market_scan import ScanWatchlist
from ...schemas.market_scan import (
    WatchlistSymbolCreate,
    WatchlistSymbolUpdate,
    WatchlistSymbolResponse,
    WatchlistResponse,
    ReorderRequest,
)

router = APIRouter()

# Default symbols for the US Key Markets watchlist.
DEFAULT_KEY_MARKETS = key_market_watchlist_defaults("US")


@router.get("/watchlist/{list_name}", response_model=WatchlistResponse)
async def get_watchlist(
    list_name: str,
    db: Session = Depends(get_db)
):
    """
    Get all symbols in a watchlist, ordered by position.
    Auto-seeds default symbols for 'key_markets' list if empty.
    """
    symbols = db.query(ScanWatchlist).filter(
        ScanWatchlist.list_name == list_name
    ).order_by(ScanWatchlist.position).all()

    # Seed defaults if list is empty and is key_markets
    if not symbols and list_name == "key_markets":
        for idx, item in enumerate(DEFAULT_KEY_MARKETS):
            new_symbol = ScanWatchlist(
                list_name=list_name,
                symbol=item["symbol"],
                display_name=item["display_name"],
                position=idx
            )
            db.add(new_symbol)
        db.commit()

        # Re-fetch after seeding
        symbols = db.query(ScanWatchlist).filter(
            ScanWatchlist.list_name == list_name
        ).order_by(ScanWatchlist.position).all()

    return WatchlistResponse(
        list_name=list_name,
        symbols=[WatchlistSymbolResponse.model_validate(s) for s in symbols],
        total=len(symbols)
    )


@router.post("/watchlist/{list_name}", response_model=WatchlistSymbolResponse)
async def add_symbol(
    list_name: str,
    symbol_data: WatchlistSymbolCreate,
    db: Session = Depends(get_db)
):
    """
    Add a symbol to a watchlist.
    """
    # Check if symbol already exists in this list
    existing = db.query(ScanWatchlist).filter(
        ScanWatchlist.list_name == list_name,
        ScanWatchlist.symbol == symbol_data.symbol
    ).first()

    if existing:
        raise HTTPException(status_code=400, detail="Symbol already in watchlist")

    # Get max position
    max_pos = db.query(func.max(ScanWatchlist.position)).filter(
        ScanWatchlist.list_name == list_name
    ).scalar()
    max_pos = max_pos if max_pos is not None else -1

    new_symbol = ScanWatchlist(
        list_name=list_name,
        symbol=symbol_data.symbol,
        display_name=symbol_data.display_name,
        notes=symbol_data.notes,
        position=symbol_data.position if symbol_data.position is not None else max_pos + 1
    )
    db.add(new_symbol)
    db.commit()
    db.refresh(new_symbol)

    return WatchlistSymbolResponse.model_validate(new_symbol)


@router.put("/watchlist/{list_name}/{symbol_id}", response_model=WatchlistSymbolResponse)
async def update_symbol(
    list_name: str,
    symbol_id: int,
    symbol_data: WatchlistSymbolUpdate,
    db: Session = Depends(get_db)
):
    """
    Update a symbol's metadata or position.
    """
    symbol = db.query(ScanWatchlist).filter(
        ScanWatchlist.id == symbol_id,
        ScanWatchlist.list_name == list_name
    ).first()

    if not symbol:
        raise HTTPException(status_code=404, detail="Symbol not found")

    if symbol_data.display_name is not None:
        symbol.display_name = symbol_data.display_name
    if symbol_data.notes is not None:
        symbol.notes = symbol_data.notes
    if symbol_data.position is not None:
        symbol.position = symbol_data.position

    db.commit()
    db.refresh(symbol)

    return WatchlistSymbolResponse.model_validate(symbol)


@router.delete("/watchlist/{list_name}/{symbol_id}")
async def remove_symbol(
    list_name: str,
    symbol_id: int,
    db: Session = Depends(get_db)
):
    """
    Remove a symbol from a watchlist.
    """
    symbol = db.query(ScanWatchlist).filter(
        ScanWatchlist.id == symbol_id,
        ScanWatchlist.list_name == list_name
    ).first()

    if not symbol:
        raise HTTPException(status_code=404, detail="Symbol not found")

    deleted_symbol = symbol.symbol
    db.delete(symbol)
    db.commit()

    return {"status": "deleted", "symbol": deleted_symbol}


@router.put("/watchlist/{list_name}/reorder")
async def reorder_symbols(
    list_name: str,
    reorder_data: ReorderRequest,
    db: Session = Depends(get_db)
):
    """
    Reorder symbols in a watchlist.
    Accepts an array of symbol IDs in the desired order.
    """
    for idx, symbol_id in enumerate(reorder_data.symbol_ids):
        symbol = db.query(ScanWatchlist).filter(
            ScanWatchlist.id == symbol_id,
            ScanWatchlist.list_name == list_name
        ).first()
        if symbol:
            symbol.position = idx

    db.commit()

    return {"status": "reordered", "count": len(reorder_data.symbol_ids)}
