"""
API endpoints for User-defined Watchlists feature.
Handles CRUD operations and data retrieval for watchlists and items.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date, timedelta
import logging
from typing import List, Dict

from fastapi import APIRouter, Depends, HTTPException, Query
import pandas as pd
from sqlalchemy import func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ...database import get_db
from ...models.user_watchlist import UserWatchlist, WatchlistItem
from ...models.stock import StockPrice
from ...models.stock_universe import StockUniverse
from ...models.industry import IBDIndustryGroup
from ...schemas.user_watchlist import (
    WatchlistCreate, WatchlistUpdate, WatchlistResponse,
    WatchlistItemCreate, WatchlistItemUpdate, WatchlistItemResponse,
    WatchlistDataResponse, WatchlistListResponse,
    WatchlistStockData, PriceChangeBounds,
    ReorderWatchlistsRequest, ReorderItemsRequest,
    BulkAddItemsRequest,
    WatchlistImportRequest, WatchlistImportResult,
    WatchlistStewardshipResponse,
)
from ...services.symbol_format import normalize_symbol
from ...services.watchlist_import_service import (
    parse_watchlist_import_symbols,
    split_import_results,
)
from ...services.watchlist_stewardship_service import WatchlistStewardshipService

logger = logging.getLogger(__name__)
router = APIRouter()
WATCHLIST_HISTORY_LOOKBACK_DAYS = 400


def _get_watchlist_stewardship_service() -> WatchlistStewardshipService:
    return WatchlistStewardshipService()


def _is_duplicate_watchlist_symbol_error(exc: IntegrityError) -> bool:
    message = str(getattr(exc, "orig", exc)).lower()
    return (
        "uix_watchlist_symbol" in message
        or (
            "unique constraint failed" in message
            and "watchlist_items.watchlist_id" in message
            and "watchlist_items.symbol" in message
        )
        or (
            "duplicate key value violates unique constraint" in message
            and "watchlist" in message
            and "symbol" in message
        )
    )


# ================= Watchlist CRUD =================

@router.get("", response_model=WatchlistListResponse, include_in_schema=False)
@router.get("/", response_model=WatchlistListResponse)
async def list_watchlists(db: Session = Depends(get_db)):
    """Get all user watchlists ordered by position."""
    watchlists = db.query(UserWatchlist).order_by(UserWatchlist.position).all()
    return WatchlistListResponse(
        watchlists=[WatchlistResponse.model_validate(w) for w in watchlists],
        total=len(watchlists)
    )


@router.post("", response_model=WatchlistResponse, include_in_schema=False)
@router.post("/", response_model=WatchlistResponse)
async def create_watchlist(data: WatchlistCreate, db: Session = Depends(get_db)):
    """Create a new watchlist."""
    existing = db.query(UserWatchlist).filter(UserWatchlist.name == data.name).first()
    if existing:
        raise HTTPException(status_code=400, detail="Watchlist with this name already exists")

    max_pos = db.query(func.max(UserWatchlist.position)).scalar() or -1

    watchlist = UserWatchlist(
        name=data.name,
        description=data.description,
        color=data.color,
        position=max_pos + 1
    )
    db.add(watchlist)
    db.commit()
    db.refresh(watchlist)
    return WatchlistResponse.model_validate(watchlist)


@router.put("/{watchlist_id}", response_model=WatchlistResponse)
async def update_watchlist(watchlist_id: int, updates: WatchlistUpdate, db: Session = Depends(get_db)):
    """Update watchlist properties."""
    watchlist = db.query(UserWatchlist).filter(UserWatchlist.id == watchlist_id).first()
    if not watchlist:
        raise HTTPException(status_code=404, detail="Watchlist not found")

    if updates.name is not None:
        watchlist.name = updates.name
    if updates.description is not None:
        watchlist.description = updates.description
    if updates.color is not None:
        watchlist.color = updates.color
    if updates.position is not None:
        watchlist.position = updates.position

    db.commit()
    db.refresh(watchlist)
    return WatchlistResponse.model_validate(watchlist)


@router.delete("/{watchlist_id}")
async def delete_watchlist(watchlist_id: int, db: Session = Depends(get_db)):
    """Delete a watchlist and all its items (cascade)."""
    watchlist = db.query(UserWatchlist).filter(UserWatchlist.id == watchlist_id).first()
    if not watchlist:
        raise HTTPException(status_code=404, detail="Watchlist not found")

    db.delete(watchlist)
    db.commit()
    return {"status": "deleted", "watchlist_id": watchlist_id}


@router.put("/reorder")
async def reorder_watchlists(
    reorder_data: ReorderWatchlistsRequest,
    db: Session = Depends(get_db)
):
    """Reorder watchlists by updating their position values."""
    for idx, watchlist_id in enumerate(reorder_data.watchlist_ids):
        watchlist = db.query(UserWatchlist).filter(UserWatchlist.id == watchlist_id).first()
        if watchlist:
            watchlist.position = idx
    db.commit()
    return {"status": "reordered"}


# ================= Watchlist Data (with sparklines and price changes) =================

@router.get("/{watchlist_id}/data", response_model=WatchlistDataResponse)
async def get_watchlist_data(watchlist_id: int, db: Session = Depends(get_db)):
    """
    Get complete watchlist data with items, sparklines, and price changes.
    Computes min/max bounds across ALL items for bar chart scaling.
    """
    watchlist = db.query(UserWatchlist).filter(UserWatchlist.id == watchlist_id).first()
    if not watchlist:
        raise HTTPException(status_code=404, detail="Watchlist not found")

    items = db.query(WatchlistItem).filter(
        WatchlistItem.watchlist_id == watchlist_id
    ).order_by(WatchlistItem.position).all()

    # Fetch market data for all symbols
    symbols = [item.symbol for item in items]
    stock_data_map = _fetch_stock_market_data(symbols, db)

    # Compute price change bounds
    price_change_bounds = _compute_price_change_bounds(stock_data_map)

    # Build response
    items_response = []
    for item in items:
        data = stock_data_map.get(item.symbol, {})
        items_response.append(WatchlistStockData(
            id=item.id,
            symbol=item.symbol,
            display_name=item.display_name,
            watchlist_id=item.watchlist_id,
            position=item.position,
            company_name=data.get("company_name"),
            ibd_industry=data.get("ibd_industry"),
            rs_data=data.get("rs_data"),
            rs_trend=data.get("rs_trend"),
            price_data=data.get("price_data"),
            price_trend=data.get("price_trend"),
            change_1d=data.get("change_1d"),
            change_5d=data.get("change_5d"),
            change_2w=data.get("change_2w"),
            change_1m=data.get("change_1m"),
            change_3m=data.get("change_3m"),
            change_6m=data.get("change_6m"),
            change_12m=data.get("change_12m"),
        ))

    return WatchlistDataResponse(
        id=watchlist.id,
        name=watchlist.name,
        description=watchlist.description,
        color=watchlist.color,
        items=items_response,
        price_change_bounds=price_change_bounds
    )


@router.get("/{watchlist_id}/stewardship", response_model=WatchlistStewardshipResponse)
async def get_watchlist_stewardship(
    watchlist_id: int,
    as_of_date: date | None = Query(None),
    profile: str | None = Query(None),
    db: Session = Depends(get_db),
    service: WatchlistStewardshipService = Depends(_get_watchlist_stewardship_service),
):
    """Get deterministic stewardship for one watchlist."""

    try:
        return service.get_watchlist_stewardship(
            db,
            watchlist_id=watchlist_id,
            as_of_date=as_of_date,
            profile=profile,
        )
    except ValueError as exc:
        if str(exc) == "watchlist_not_found":
            raise HTTPException(status_code=404, detail="Watchlist not found") from exc
        raise


def _fetch_stock_market_data(symbols: List[str], db: Session) -> Dict:
    """
    Fetch sparkline and price change data for a list of symbols.
    Uses stored database history so the watchlist page loads quickly without
    kicking off live refresh work.
    Also fetches company name and IBD industry.
    """
    if not symbols:
        return {}

    from ...scanners.criteria.price_sparkline import PriceSparklineCalculator
    from ...scanners.criteria.rs_sparkline import RSSparklineCalculator

    result = {}

    # Fetch company names from StockUniverse
    universe_data = db.query(StockUniverse).filter(
        StockUniverse.symbol.in_(symbols)
    ).all()
    company_names = {u.symbol: u.name for u in universe_data}

    # Fetch IBD industry groups
    ibd_data = db.query(IBDIndustryGroup).filter(
        IBDIndustryGroup.symbol.in_(symbols)
    ).all()
    ibd_industries = {i.symbol: i.industry_group for i in ibd_data}

    # Define lookback days for different periods
    periods = {
        "1d": 2,
        "5d": 6,
        "2w": 11,
        "1m": 22,
        "3m": 66,
        "6m": 132,
        "12m": 252,
    }

    price_frames = _load_price_frames_from_db([*symbols, "SPY"], db)
    spy_df = price_frames.get("SPY")
    spy_close = spy_df['Close'].tolist() if spy_df is not None and not spy_df.empty else []

    if not spy_close:
        logger.warning("No SPY price data found for RS calculation")

    rs_calc = RSSparklineCalculator()
    price_calc = PriceSparklineCalculator()

    for symbol in symbols:
        price_df = price_frames.get(symbol)

        if price_df is None or price_df.empty:
            result[symbol] = {}
            continue

        closes = price_df['Close'].tolist()

        # Calculate sparklines
        stock_series = pd.Series(closes)
        spy_series = pd.Series(spy_close[:len(closes)]) if spy_close else pd.Series([1.0] * len(closes))

        rs_result = rs_calc.calculate_rs_sparkline(stock_series, spy_series)
        price_result = price_calc.calculate_price_sparkline(stock_series)

        # Calculate price changes for each period
        changes = {}
        for period_name, days in periods.items():
            if len(closes) >= days:
                old_price = closes[-(days)]
                new_price = closes[-1]
                if old_price > 0:
                    changes[f"change_{period_name}"] = round(
                        ((new_price - old_price) / old_price) * 100, 2
                    )

        result[symbol] = {
            "rs_data": rs_result.get("rs_data"),
            "rs_trend": rs_result.get("rs_trend"),
            "price_data": price_result.get("price_data"),
            "price_trend": price_result.get("price_trend"),
            "company_name": company_names.get(symbol),
            "ibd_industry": ibd_industries.get(symbol),
            **changes
        }

    return result


def _load_price_frames_from_db(symbols: List[str], db: Session) -> Dict[str, pd.DataFrame]:
    """Bulk load recent price history for watchlist rendering from the local DB."""
    if not symbols:
        return {}

    unique_symbols = list(dict.fromkeys(symbols))
    start_date = date.today() - timedelta(days=WATCHLIST_HISTORY_LOOKBACK_DAYS)
    prices = (
        db.query(StockPrice)
        .filter(
            StockPrice.symbol.in_(unique_symbols),
            StockPrice.date >= start_date,
        )
        .order_by(StockPrice.symbol.asc(), StockPrice.date.asc())
        .all()
    )

    grouped_prices: dict[str, list[StockPrice]] = defaultdict(list)
    for price in prices:
        grouped_prices[price.symbol].append(price)

    frames: Dict[str, pd.DataFrame] = {}
    for symbol in unique_symbols:
        symbol_prices = grouped_prices.get(symbol)
        if not symbol_prices:
            continue
        df = pd.DataFrame(
            {
                "Date": [price.date for price in symbol_prices],
                "Open": [price.open for price in symbol_prices],
                "High": [price.high for price in symbol_prices],
                "Low": [price.low for price in symbol_prices],
                "Close": [price.close for price in symbol_prices],
                "Volume": [price.volume for price in symbol_prices],
            }
        )
        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)
        frames[symbol] = df

    return frames


def _compute_price_change_bounds(stock_data_map: Dict) -> Dict[str, PriceChangeBounds]:
    """
    Compute min/max price change bounds across all stocks for bar chart scaling.
    Returns bounds for each timeframe.
    """
    periods = ["1d", "5d", "2w", "1m", "3m", "6m", "12m"]
    bounds = {}

    for period in periods:
        key = f"change_{period}"
        values = [
            data.get(key) for data in stock_data_map.values()
            if data.get(key) is not None
        ]
        if values:
            bounds[period] = PriceChangeBounds(
                min=min(values),
                max=max(values)
            )
        else:
            bounds[period] = PriceChangeBounds(min=0, max=0)

    return bounds


# ================= Item CRUD =================

@router.post("/{watchlist_id}/items", response_model=WatchlistItemResponse)
async def add_item(
    watchlist_id: int,
    item_data: WatchlistItemCreate,
    db: Session = Depends(get_db)
):
    """Add a stock to a watchlist."""
    watchlist = db.query(UserWatchlist).filter(UserWatchlist.id == watchlist_id).first()
    if not watchlist:
        raise HTTPException(status_code=404, detail="Watchlist not found")

    symbol = normalize_symbol(item_data.symbol)
    if symbol is None:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid symbol format: {item_data.symbol!r}",
        )

    known = db.query(StockUniverse.symbol).filter(
        StockUniverse.active_filter(),
        StockUniverse.symbol == symbol,
    ).first()
    if known is None:
        raise HTTPException(
            status_code=400,
            detail=f"Symbol {symbol} is not in the active stock universe",
        )

    existing = db.query(WatchlistItem).filter(
        WatchlistItem.watchlist_id == watchlist_id,
        WatchlistItem.symbol == symbol
    ).first()
    if existing:
        raise HTTPException(status_code=400, detail="Stock already exists in watchlist")

    max_pos = db.query(func.max(WatchlistItem.position)).filter(
        WatchlistItem.watchlist_id == watchlist_id
    ).scalar() or -1

    item = WatchlistItem(
        watchlist_id=watchlist_id,
        symbol=symbol,
        display_name=item_data.display_name,
        notes=item_data.notes,
        position=max_pos + 1
    )
    db.add(item)
    db.commit()
    db.refresh(item)
    return WatchlistItemResponse.model_validate(item)


@router.post("/{watchlist_id}/items/bulk", response_model=List[WatchlistItemResponse])
async def bulk_add_items(
    watchlist_id: int,
    bulk_data: BulkAddItemsRequest,
    db: Session = Depends(get_db)
):
    """Add multiple stocks at once to a watchlist."""
    watchlist = db.query(UserWatchlist).filter(UserWatchlist.id == watchlist_id).first()
    if not watchlist:
        raise HTTPException(status_code=404, detail="Watchlist not found")

    # Get existing symbols
    existing_items = db.query(WatchlistItem).filter(
        WatchlistItem.watchlist_id == watchlist_id
    ).all()
    existing_symbols = {item.symbol for item in existing_items}

    # Gather every format-valid candidate, then batch-verify against the
    # active universe in a single query to avoid N+1 lookups. Insertion
    # order is preserved (parallel list) so the downstream position
    # assignment matches caller intent.
    candidates: list[str] = []
    seen: set[str] = set()
    for raw in bulk_data.symbols:
        normalized = normalize_symbol(raw)
        if normalized is None or normalized in existing_symbols or normalized in seen:
            continue
        seen.add(normalized)
        candidates.append(normalized)

    known_rows = (
        db.query(StockUniverse.symbol)
        .filter(
            StockUniverse.active_filter(),
            StockUniverse.symbol.in_(candidates),
        )
        .all()
        if candidates
        else []
    )
    known_symbols = {row.symbol for row in known_rows}

    max_pos = db.query(func.max(WatchlistItem.position)).filter(
        WatchlistItem.watchlist_id == watchlist_id
    ).scalar() or -1

    added_items = []
    for symbol in candidates:
        if symbol not in known_symbols:
            continue  # Skip unknown symbols silently; import endpoint surfaces them.
        max_pos += 1
        item = WatchlistItem(
            watchlist_id=watchlist_id,
            symbol=symbol,
            position=max_pos
        )
        db.add(item)
        existing_symbols.add(symbol)
        added_items.append(item)

    db.commit()

    # Refresh all items to get their IDs
    for item in added_items:
        db.refresh(item)

    return [WatchlistItemResponse.model_validate(item) for item in added_items]


@router.post("/{watchlist_id}/items/import", response_model=WatchlistImportResult)
async def import_items(
    watchlist_id: int,
    import_data: WatchlistImportRequest,
    db: Session = Depends(get_db),
):
    """Import symbols from pasted text or CSV with partial-success feedback."""

    watchlist = db.query(UserWatchlist).filter(UserWatchlist.id == watchlist_id).first()
    if not watchlist:
        raise HTTPException(status_code=404, detail="Watchlist not found")

    parsed_symbols = parse_watchlist_import_symbols(
        import_data.content,
        format_hint=import_data.format,
    )
    if not parsed_symbols:
        return WatchlistImportResult(
            requested_count=0,
            added=[],
            skipped_existing=[],
            invalid_symbols=[],
            added_items=[],
        )

    existing_items = db.query(WatchlistItem).filter(
        WatchlistItem.watchlist_id == watchlist_id
    ).all()
    existing_symbols = {item.symbol for item in existing_items}

    known_symbols = {
        row.symbol
        for row in db.query(StockUniverse.symbol)
        .filter(
            StockUniverse.active_filter(),
            StockUniverse.symbol.in_(parsed_symbols),
        )
        .all()
        if row.symbol
    }

    addable_symbols, skipped_existing, invalid_symbols = split_import_results(
        parsed_symbols,
        known_symbols,
        existing_symbols,
    )

    max_pos = db.query(func.max(WatchlistItem.position)).filter(
        WatchlistItem.watchlist_id == watchlist_id
    ).scalar() or -1

    added_symbols = []
    added_items = []
    for symbol in addable_symbols:
        max_pos += 1
        item = WatchlistItem(
            watchlist_id=watchlist_id,
            symbol=symbol,
            position=max_pos,
        )
        try:
            with db.begin_nested():
                db.add(item)
                db.flush()
        except IntegrityError as exc:
            if _is_duplicate_watchlist_symbol_error(exc):
                skipped_existing.append(symbol)
                continue
            raise
        added_symbols.append(symbol)
        added_items.append(item)

    db.commit()

    for item in added_items:
        db.refresh(item)

    return WatchlistImportResult(
        requested_count=len(parsed_symbols),
        added=added_symbols,
        skipped_existing=skipped_existing,
        invalid_symbols=invalid_symbols,
        added_items=[WatchlistItemResponse.model_validate(item) for item in added_items],
    )


@router.put("/items/{item_id}", response_model=WatchlistItemResponse)
async def update_item(
    item_id: int,
    updates: WatchlistItemUpdate,
    db: Session = Depends(get_db)
):
    """Update item properties."""
    item = db.query(WatchlistItem).filter(WatchlistItem.id == item_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")

    if updates.display_name is not None:
        item.display_name = updates.display_name
    if updates.notes is not None:
        item.notes = updates.notes
    if updates.position is not None:
        item.position = updates.position

    db.commit()
    db.refresh(item)
    return WatchlistItemResponse.model_validate(item)


@router.delete("/items/{item_id}")
async def remove_item(item_id: int, db: Session = Depends(get_db)):
    """Remove an item from a watchlist."""
    item = db.query(WatchlistItem).filter(WatchlistItem.id == item_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")

    db.delete(item)
    db.commit()
    return {"status": "deleted", "item_id": item_id}


@router.put("/{watchlist_id}/items/reorder")
async def reorder_items(
    watchlist_id: int,
    reorder_data: ReorderItemsRequest,
    db: Session = Depends(get_db)
):
    """Reorder items within a watchlist."""
    for idx, item_id in enumerate(reorder_data.item_ids):
        item = db.query(WatchlistItem).filter(
            WatchlistItem.id == item_id,
            WatchlistItem.watchlist_id == watchlist_id
        ).first()
        if item:
            item.position = idx
    db.commit()
    return {"status": "reordered"}
