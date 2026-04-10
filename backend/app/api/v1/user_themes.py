"""
API endpoints for User-defined Themes feature.
Handles CRUD operations and data retrieval for themes, subgroups, and stocks.
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List, Dict
import pandas as pd
import logging

from ...database import get_db
from ...models.user_theme import UserTheme, UserThemeSubgroup, UserThemeStock
from ...models.stock import StockPrice
from ...models.stock_universe import StockUniverse
from ...models.industry import IBDIndustryGroup
from ...wiring.bootstrap import get_price_cache
from ...schemas.user_theme import (
    UserThemeCreate, UserThemeUpdate, UserThemeResponse,
    SubgroupCreate, SubgroupUpdate, SubgroupResponse,
    ThemeStockCreate, ThemeStockUpdate, ThemeStockResponse,
    ThemeDataResponse, ThemeListResponse,
    SubgroupWithStocksResponse, StockDataResponse,
    ReorderThemesRequest, ReorderSubgroupsRequest, ReorderStocksRequest,
    PriceChangeBounds,
)
from ...scanners.criteria.rs_sparkline import RSSparklineCalculator
from ...scanners.criteria.price_sparkline import PriceSparklineCalculator

logger = logging.getLogger(__name__)
router = APIRouter()


# ================= Theme CRUD =================

@router.get("", response_model=ThemeListResponse, include_in_schema=False)
@router.get("/", response_model=ThemeListResponse)
async def list_themes(db: Session = Depends(get_db)):
    """Get all user themes ordered by position."""
    themes = db.query(UserTheme).order_by(UserTheme.position).all()
    return ThemeListResponse(
        themes=[UserThemeResponse.model_validate(t) for t in themes],
        total=len(themes)
    )


@router.post("", response_model=UserThemeResponse, include_in_schema=False)
@router.post("/", response_model=UserThemeResponse)
async def create_theme(theme_data: UserThemeCreate, db: Session = Depends(get_db)):
    """Create a new theme."""
    existing = db.query(UserTheme).filter(UserTheme.name == theme_data.name).first()
    if existing:
        raise HTTPException(status_code=400, detail="Theme with this name already exists")

    max_pos = db.query(func.max(UserTheme.position)).scalar() or -1

    theme = UserTheme(
        name=theme_data.name,
        description=theme_data.description,
        color=theme_data.color,
        position=max_pos + 1
    )
    db.add(theme)
    db.commit()
    db.refresh(theme)
    return UserThemeResponse.model_validate(theme)


@router.put("/{theme_id}", response_model=UserThemeResponse)
async def update_theme(theme_id: int, updates: UserThemeUpdate, db: Session = Depends(get_db)):
    """Update theme properties."""
    theme = db.query(UserTheme).filter(UserTheme.id == theme_id).first()
    if not theme:
        raise HTTPException(status_code=404, detail="Theme not found")

    if updates.name is not None:
        theme.name = updates.name
    if updates.description is not None:
        theme.description = updates.description
    if updates.color is not None:
        theme.color = updates.color
    if updates.position is not None:
        theme.position = updates.position

    db.commit()
    db.refresh(theme)
    return UserThemeResponse.model_validate(theme)


@router.delete("/{theme_id}")
async def delete_theme(theme_id: int, db: Session = Depends(get_db)):
    """Delete a theme and all its subgroups/stocks (cascade)."""
    theme = db.query(UserTheme).filter(UserTheme.id == theme_id).first()
    if not theme:
        raise HTTPException(status_code=404, detail="Theme not found")

    db.delete(theme)
    db.commit()
    return {"status": "deleted", "theme_id": theme_id}


@router.put("/reorder")
async def reorder_themes(
    reorder_data: ReorderThemesRequest,
    db: Session = Depends(get_db)
):
    """Reorder themes by updating their position values."""
    for idx, theme_id in enumerate(reorder_data.theme_ids):
        theme = db.query(UserTheme).filter(UserTheme.id == theme_id).first()
        if theme:
            theme.position = idx
    db.commit()
    return {"status": "reordered"}


# ================= Theme Data (with sparklines and price changes) =================

@router.get("/{theme_id}/data", response_model=ThemeDataResponse)
async def get_theme_data(theme_id: int, db: Session = Depends(get_db)):
    """
    Get complete theme data with subgroups, stocks, sparklines, and price changes.
    Computes min/max bounds across ALL stocks in theme for bar chart scaling.
    """
    theme = db.query(UserTheme).filter(UserTheme.id == theme_id).first()
    if not theme:
        raise HTTPException(status_code=404, detail="Theme not found")

    subgroups = db.query(UserThemeSubgroup).filter(
        UserThemeSubgroup.theme_id == theme_id
    ).order_by(UserThemeSubgroup.position).all()

    # Collect all stocks and map by subgroup
    all_stocks = []
    subgroup_stocks_map = {}

    for sg in subgroups:
        stocks = db.query(UserThemeStock).filter(
            UserThemeStock.subgroup_id == sg.id
        ).order_by(UserThemeStock.position).all()
        subgroup_stocks_map[sg.id] = stocks
        all_stocks.extend(stocks)

    # Fetch market data for all symbols
    symbols = [s.symbol for s in all_stocks]
    stock_data_map = _fetch_stock_market_data(symbols, db)

    # Compute price change bounds across all stocks
    price_change_bounds = _compute_price_change_bounds(stock_data_map)

    # Build response
    subgroups_response = []
    for sg in subgroups:
        stocks_response = []
        for stock in subgroup_stocks_map[sg.id]:
            data = stock_data_map.get(stock.symbol, {})
            stocks_response.append(StockDataResponse(
                id=stock.id,
                symbol=stock.symbol,
                display_name=stock.display_name,
                subgroup_id=stock.subgroup_id,
                position=stock.position,
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

        subgroups_response.append(SubgroupWithStocksResponse(
            id=sg.id,
            name=sg.name,
            position=sg.position,
            is_collapsed=sg.is_collapsed or False,
            stocks=stocks_response
        ))

    return ThemeDataResponse(
        id=theme.id,
        name=theme.name,
        description=theme.description,
        color=theme.color,
        subgroups=subgroups_response,
        price_change_bounds=price_change_bounds
    )


def _fetch_stock_market_data(symbols: List[str], db: Session) -> Dict:
    """
    Fetch sparkline and price change data for a list of symbols.
    Uses PriceCacheService for fresh data with automatic updates.
    Also fetches company name and IBD industry.
    """
    if not symbols:
        return {}

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

    # Use PriceCacheService for fresh data with automatic updates
    price_cache = get_price_cache()

    # Fetch SPY prices for RS calculation (with freshness check)
    spy_df = price_cache.get_historical_data("SPY", period="2y")
    spy_close = spy_df['Close'].tolist() if spy_df is not None and not spy_df.empty else []

    if not spy_close:
        logger.warning("No SPY price data found for RS calculation")

    # Bulk fetch all stock prices (with freshness check and auto-update)
    all_symbols = list(symbols)
    cached_prices = price_cache.get_many(all_symbols, period="2y")

    rs_calc = RSSparklineCalculator()
    price_calc = PriceSparklineCalculator()

    for symbol in symbols:
        price_df = cached_prices.get(symbol)

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


# ================= Subgroup CRUD =================

@router.post("/{theme_id}/subgroups", response_model=SubgroupResponse)
async def create_subgroup(
    theme_id: int,
    subgroup_data: SubgroupCreate,
    db: Session = Depends(get_db)
):
    """Create a new subgroup within a theme."""
    theme = db.query(UserTheme).filter(UserTheme.id == theme_id).first()
    if not theme:
        raise HTTPException(status_code=404, detail="Theme not found")

    existing = db.query(UserThemeSubgroup).filter(
        UserThemeSubgroup.theme_id == theme_id,
        UserThemeSubgroup.name == subgroup_data.name
    ).first()
    if existing:
        raise HTTPException(status_code=400, detail="Subgroup with this name already exists in theme")

    max_pos = db.query(func.max(UserThemeSubgroup.position)).filter(
        UserThemeSubgroup.theme_id == theme_id
    ).scalar() or -1

    subgroup = UserThemeSubgroup(
        theme_id=theme_id,
        name=subgroup_data.name,
        position=max_pos + 1
    )
    db.add(subgroup)
    db.commit()
    db.refresh(subgroup)
    return SubgroupResponse.model_validate(subgroup)


@router.put("/subgroups/{subgroup_id}", response_model=SubgroupResponse)
async def update_subgroup(
    subgroup_id: int,
    updates: SubgroupUpdate,
    db: Session = Depends(get_db)
):
    """Update subgroup properties."""
    subgroup = db.query(UserThemeSubgroup).filter(UserThemeSubgroup.id == subgroup_id).first()
    if not subgroup:
        raise HTTPException(status_code=404, detail="Subgroup not found")

    if updates.name is not None:
        subgroup.name = updates.name
    if updates.position is not None:
        subgroup.position = updates.position
    if updates.is_collapsed is not None:
        subgroup.is_collapsed = updates.is_collapsed

    db.commit()
    db.refresh(subgroup)
    return SubgroupResponse.model_validate(subgroup)


@router.delete("/subgroups/{subgroup_id}")
async def delete_subgroup(subgroup_id: int, db: Session = Depends(get_db)):
    """Delete a subgroup and all its stocks (cascade)."""
    subgroup = db.query(UserThemeSubgroup).filter(UserThemeSubgroup.id == subgroup_id).first()
    if not subgroup:
        raise HTTPException(status_code=404, detail="Subgroup not found")

    db.delete(subgroup)
    db.commit()
    return {"status": "deleted", "subgroup_id": subgroup_id}


@router.put("/{theme_id}/subgroups/reorder")
async def reorder_subgroups(
    theme_id: int,
    reorder_data: ReorderSubgroupsRequest,
    db: Session = Depends(get_db)
):
    """Reorder subgroups within a theme."""
    for idx, subgroup_id in enumerate(reorder_data.subgroup_ids):
        subgroup = db.query(UserThemeSubgroup).filter(
            UserThemeSubgroup.id == subgroup_id,
            UserThemeSubgroup.theme_id == theme_id
        ).first()
        if subgroup:
            subgroup.position = idx
    db.commit()
    return {"status": "reordered"}


# ================= Stock CRUD =================

@router.post("/subgroups/{subgroup_id}/stocks", response_model=ThemeStockResponse)
async def add_stock(
    subgroup_id: int,
    stock_data: ThemeStockCreate,
    db: Session = Depends(get_db)
):
    """Add a stock to a subgroup."""
    subgroup = db.query(UserThemeSubgroup).filter(UserThemeSubgroup.id == subgroup_id).first()
    if not subgroup:
        raise HTTPException(status_code=404, detail="Subgroup not found")

    existing = db.query(UserThemeStock).filter(
        UserThemeStock.subgroup_id == subgroup_id,
        UserThemeStock.symbol == stock_data.symbol.upper()
    ).first()
    if existing:
        raise HTTPException(status_code=400, detail="Stock already exists in subgroup")

    max_pos = db.query(func.max(UserThemeStock.position)).filter(
        UserThemeStock.subgroup_id == subgroup_id
    ).scalar() or -1

    stock = UserThemeStock(
        subgroup_id=subgroup_id,
        symbol=stock_data.symbol.upper(),
        display_name=stock_data.display_name,
        notes=stock_data.notes,
        position=max_pos + 1
    )
    db.add(stock)
    db.commit()
    db.refresh(stock)
    return ThemeStockResponse.model_validate(stock)


@router.put("/stocks/{stock_id}", response_model=ThemeStockResponse)
async def update_stock(
    stock_id: int,
    updates: ThemeStockUpdate,
    db: Session = Depends(get_db)
):
    """Update stock properties or move to different subgroup."""
    stock = db.query(UserThemeStock).filter(UserThemeStock.id == stock_id).first()
    if not stock:
        raise HTTPException(status_code=404, detail="Stock not found")

    if updates.display_name is not None:
        stock.display_name = updates.display_name
    if updates.notes is not None:
        stock.notes = updates.notes
    if updates.position is not None:
        stock.position = updates.position
    if updates.subgroup_id is not None:
        new_subgroup = db.query(UserThemeSubgroup).filter(
            UserThemeSubgroup.id == updates.subgroup_id
        ).first()
        if not new_subgroup:
            raise HTTPException(status_code=404, detail="Target subgroup not found")
        stock.subgroup_id = updates.subgroup_id

    db.commit()
    db.refresh(stock)
    return ThemeStockResponse.model_validate(stock)


@router.delete("/stocks/{stock_id}")
async def remove_stock(stock_id: int, db: Session = Depends(get_db)):
    """Remove a stock from its subgroup."""
    stock = db.query(UserThemeStock).filter(UserThemeStock.id == stock_id).first()
    if not stock:
        raise HTTPException(status_code=404, detail="Stock not found")

    db.delete(stock)
    db.commit()
    return {"status": "deleted", "stock_id": stock_id}


@router.put("/subgroups/{subgroup_id}/stocks/reorder")
async def reorder_stocks(
    subgroup_id: int,
    reorder_data: ReorderStocksRequest,
    db: Session = Depends(get_db)
):
    """Reorder stocks within a subgroup."""
    for idx, stock_id in enumerate(reorder_data.stock_ids):
        stock = db.query(UserThemeStock).filter(
            UserThemeStock.id == stock_id,
            UserThemeStock.subgroup_id == subgroup_id
        ).first()
        if stock:
            stock.position = idx
    db.commit()
    return {"status": "reordered"}
