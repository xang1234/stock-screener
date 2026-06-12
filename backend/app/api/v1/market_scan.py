"""
API endpoints for Market Scan feature.
Handles CRUD operations for market scan watchlists and the aggregated
Daily Snapshot payload.
"""
import json
import logging
from typing import Any, List

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from sqlalchemy.orm import Session
from sqlalchemy import func

from ...database import get_db
from ...domain.markets.catalog import get_market_catalog
from ...domain.markets.key_markets import key_market_watchlist_defaults
from ...models.market_scan import ScanWatchlist
from ...schemas.market_scan import (
    DailySnapshotResponse,
    WatchlistSymbolCreate,
    WatchlistSymbolUpdate,
    WatchlistSymbolResponse,
    WatchlistResponse,
    ReorderRequest,
)
from ...services.daily_snapshot_service import (
    DAILY_SNAPSHOT_CACHE_TTL_SECONDS,
    build_daily_snapshot_payload,
    daily_snapshot_cache_key,
    daily_snapshot_etag,
    latest_completed_scan,
)
from ...services.redis_pool import get_redis_client
from ...wiring.bootstrap import get_get_scan_results_use_case, get_uow

logger = logging.getLogger(__name__)

router = APIRouter()

_market_catalog = get_market_catalog()

# Default symbols for the US Key Markets watchlist.
DEFAULT_KEY_MARKETS = key_market_watchlist_defaults("US")


def _if_none_match_matches(header_value: str | None, etag: str) -> bool:
    """RFC 7232 weak comparison against a (possibly comma-separated) If-None-Match."""
    if not header_value:
        return False

    def opaque(tag: str) -> str:
        return tag[2:] if tag.startswith("W/") else tag

    target = opaque(etag)
    for token in header_value.split(","):
        token = token.strip()
        if token and (token == "*" or opaque(token) == target):
            return True
    return False


# The handler returns a raw Response so the cached JSON string is served
# byte-for-byte (the ETag hashes those exact bytes); response_model documents
# the contract in OpenAPI, and the payload is validated against it on the
# cache-miss path below.
@router.get("/daily-snapshot", response_model=DailySnapshotResponse)
async def get_daily_snapshot(
    request: Request,
    market: str = Query("US", description="Market code (e.g. US, HK, JP, TW)"),
    db: Session = Depends(get_db),
    uow: Any = Depends(get_uow),
    use_case: Any = Depends(get_get_scan_results_use_case),
):
    """Aggregated Daily Snapshot payload for one market in a single request.

    Replaces the per-section fan-out (watchlist + per-symbol history + scan
    bootstrap + scan results + group rankings) with one cached payload.
    Served with an ETag; clients sending ``If-None-Match`` get a 304.
    """
    code = str(market or "US").strip().upper()
    if code not in _market_catalog.supported_market_codes():
        supported = ", ".join(_market_catalog.supported_market_codes())
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported market '{market}'. Expected one of: {supported}.",
        )

    # Keyed on the latest scan run: publishing a new run switches the key,
    # so cached snapshots invalidate immediately (TTL is just a backstop).
    scan = latest_completed_scan(db, code)
    cache_key = daily_snapshot_cache_key(code, scan.scan_id if scan else None)
    payload_json: str | None = None
    redis = get_redis_client()
    if redis is not None:
        try:
            cached = redis.get(cache_key)
            if cached:
                payload_json = cached.decode("utf-8") if isinstance(cached, bytes) else cached
        except Exception as exc:
            logger.warning("Daily snapshot cache read failed: %s", exc)

    if payload_json is None:
        payload = build_daily_snapshot_payload(
            db,
            market=code,
            market_display_name=_market_catalog.get(code).label,
            scan=scan,
            uow=uow,
            scan_results_use_case=use_case,
        )
        DailySnapshotResponse.model_validate(payload)
        payload_json = json.dumps(payload, separators=(",", ":"))
        if redis is not None:
            try:
                redis.setex(cache_key, DAILY_SNAPSHOT_CACHE_TTL_SECONDS, payload_json)
            except Exception as exc:
                logger.warning("Daily snapshot cache write failed: %s", exc)

    etag = daily_snapshot_etag(payload_json)
    if _if_none_match_matches(request.headers.get("if-none-match"), etag):
        return Response(status_code=304, headers={"ETag": etag})
    return Response(
        content=payload_json,
        media_type="application/json",
        headers={"ETag": etag, "Cache-Control": "private, max-age=60"},
    )


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
