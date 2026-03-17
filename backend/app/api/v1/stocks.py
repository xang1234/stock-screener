from __future__ import annotations

"""Stock data API endpoints"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
import logging

from ...database import get_db
from ...schemas.stock import StockInfo, StockTechnicals, StockData
from ...wiring.bootstrap import get_uow

logger = logging.getLogger(__name__)
router = APIRouter()


def _get_yfinance_service():
    from ...services.yfinance_service import yfinance_service

    return yfinance_service


def _build_data_fetcher(db: Session):
    from ...services.data_fetcher import DataFetcher

    return DataFetcher(db)


@router.get("/{symbol}/info", response_model=StockInfo)
async def get_stock_info(symbol: str):
    """
    Get basic stock information.

    Args:
        symbol: Stock ticker symbol (e.g., AAPL, MSFT)

    Returns:
        Basic stock information
    """
    info = _get_yfinance_service().get_stock_info(symbol.upper())

    if not info:
        logger.error(f"Failed to fetch stock info for {symbol} - check yfinance service logs for details")
        raise HTTPException(
            status_code=404,
            detail=f"Unable to fetch data for {symbol}. This could be due to: invalid symbol, network issues, or yfinance API problems. Check backend logs for details."
        )

    return info


@router.get("/{symbol}/fundamentals")
async def get_stock_fundamentals(
    symbol: str,
    force_refresh: bool = False,
    db: Session = Depends(get_db)
):
    """
    Get stock fundamental data.

    Args:
        symbol: Stock ticker symbol
        force_refresh: Force data refresh (ignore cache)

    Returns:
        Fundamental data including earnings, revenue, margins, description
    """
    from ...services.fundamentals_cache_service import FundamentalsCacheService

    cache = FundamentalsCacheService.get_instance()
    data = cache.get_fundamentals(symbol.upper(), force_refresh=force_refresh)

    if not data:
        raise HTTPException(
            status_code=404,
            detail=f"Fundamental data not available for {symbol}"
        )

    # Add symbol to response if not present
    if 'symbol' not in data:
        data['symbol'] = symbol.upper()

    return data


@router.get("/{symbol}/technicals", response_model=StockTechnicals)
async def get_stock_technicals(
    symbol: str,
    force_refresh: bool = False,
    db: Session = Depends(get_db)
):
    """
    Get stock technical indicators.

    Args:
        symbol: Stock ticker symbol
        force_refresh: Force data refresh (ignore cache)

    Returns:
        Technical indicators including MAs, RS rating, 52-week range
    """
    fetcher = _build_data_fetcher(db)
    data = fetcher.get_stock_technicals(symbol.upper(), force_refresh=force_refresh)

    if not data:
        raise HTTPException(
            status_code=404,
            detail=f"Technical data not available for {symbol}"
        )

    return data


@router.get("/{symbol}", response_model=StockData)
async def get_stock_data(
    symbol: str,
    include_fundamentals: bool = True,
    include_technicals: bool = True,
    force_refresh: bool = False,
    db: Session = Depends(get_db)
):
    """
    Get complete stock data (info + fundamentals + technicals).

    Args:
        symbol: Stock ticker symbol
        include_fundamentals: Include fundamental data
        include_technicals: Include technical indicators
        force_refresh: Force data refresh (ignore cache)

    Returns:
        Complete stock data
    """
    symbol = symbol.upper()

    # Get basic info
    info = _get_yfinance_service().get_stock_info(symbol)
    if not info:
        logger.error(f"Failed to fetch stock info for {symbol} - check yfinance service logs for details")
        raise HTTPException(
            status_code=404,
            detail=f"Unable to fetch data for {symbol}. This could be due to: invalid symbol, network issues, or yfinance API problems. Check backend logs for details."
        )

    result = {"info": info}

    # Get fundamentals if requested
    if include_fundamentals:
        fetcher = _build_data_fetcher(db)
        fundamentals = fetcher.get_stock_fundamentals(symbol, force_refresh=force_refresh)
        result["fundamentals"] = fundamentals

    # Get technicals if requested
    if include_technicals:
        fetcher = _build_data_fetcher(db)
        technicals = fetcher.get_stock_technicals(symbol, force_refresh=force_refresh)
        result["technicals"] = technicals

    return result


@router.get("/{symbol}/industry")
async def get_stock_industry(symbol: str, db: Session = Depends(get_db)):
    """
    Get stock industry classification.

    Args:
        symbol: Stock ticker symbol

    Returns:
        Industry classification (sector, industry, ibd_industry_group)
    """
    from ...services.ibd_industry_service import IBDIndustryService

    fetcher = _build_data_fetcher(db)
    classification = fetcher.get_industry_classification(symbol.upper())

    if not classification:
        raise HTTPException(
            status_code=404,
            detail=f"Industry classification not available for {symbol}"
        )

    # Add IBD industry group if available
    try:
        ibd_group = IBDIndustryService.get_industry_group(db, symbol.upper())
        classification['ibd_industry_group'] = ibd_group
    except Exception as e:
        logger.warning(f"Could not fetch IBD industry group for {symbol}: {e}")
        classification['ibd_industry_group'] = None

    return classification


@router.get("/{symbol}/chart-data")
async def get_chart_data(
    symbol: str,
    uow=Depends(get_uow),
):
    """
    Get all chart modal data in a single call from the feature store.

    Returns all data needed for the chart modal:
    - Basic info (symbol, name, price)
    - Industry classification (GICS sector/industry, IBD group)
    - RS ratings (overall, 1m, 3m, 12m, trend)
    - Technical data (stage, ADR, EPS rating)
    - Minervini/VCP data
    - Growth metrics
    """
    symbol = symbol.upper()

    with uow:
        # Find the latest published feature run
        latest_run = uow.feature_runs.get_latest_published()
        if latest_run is None:
            raise HTTPException(
                status_code=404,
                detail=f"No published scan data available for {symbol}",
            )

        item = uow.feature_store.get_by_symbol_for_run(
            latest_run.id, symbol
        )
        if item is None:
            raise HTTPException(
                status_code=404,
                detail=f"No scan data found for {symbol}",
            )

    ef = item.extended_fields or {}

    return {
        "source": "feature_store",
        "scan_date": latest_run.completed_at.isoformat() if latest_run.completed_at else None,
        # Basic info
        "symbol": item.symbol,
        "company_name": ef.get("company_name"),
        "current_price": item.current_price,
        # Industry classification
        "gics_sector": ef.get("gics_sector"),
        "gics_industry": ef.get("gics_industry"),
        "ibd_industry_group": ef.get("ibd_industry_group"),
        "ibd_group_rank": ef.get("ibd_group_rank"),
        # RS data
        "rs_rating": ef.get("rs_rating"),
        "rs_rating_1m": ef.get("rs_rating_1m"),
        "rs_rating_3m": ef.get("rs_rating_3m"),
        "rs_rating_12m": ef.get("rs_rating_12m"),
        "rs_trend": ef.get("rs_trend"),
        # Technical data
        "stage": ef.get("stage"),
        "adr_percent": ef.get("adr_percent"),
        "eps_rating": ef.get("eps_rating"),
        # Scores
        "minervini_score": ef.get("minervini_score"),
        "composite_score": item.composite_score,
        # VCP data
        "vcp_detected": ef.get("vcp_detected", False),
        "vcp_score": ef.get("vcp_score"),
        "vcp_pivot": ef.get("vcp_pivot"),
        "vcp_ready_for_breakout": ef.get("vcp_ready_for_breakout", False),
        # MA data
        "ma_alignment": ef.get("ma_alignment"),
        "passes_template": ef.get("passes_template"),
        # Growth metrics
        "eps_growth_qq": ef.get("eps_growth_qq"),
        "sales_growth_qq": ef.get("sales_growth_qq"),
        "eps_growth_yy": ef.get("eps_growth_yy"),
        "sales_growth_yy": ef.get("sales_growth_yy"),
    }


@router.get("/{symbol}/history")
async def get_price_history(symbol: str, period: str = "6mo"):
    """
    Get historical price data (OHLCV only) from cache.
    Uses cached data from database/Redis - does not call yfinance directly.
    """
    from ...services.price_cache_service import PriceCacheService
    import pandas as pd

    # Map period to days for filtering
    period_days = {
        "1mo": 30,
        "3mo": 90,
        "6mo": 180,
        "1y": 365,
        "2y": 730,
        "5y": 1825,
    }
    days = period_days.get(period, 180)

    # Get from cache (database) - no yfinance calls
    cache_service = PriceCacheService.get_instance()
    data = cache_service.get_cached_only(symbol.upper(), period="2y")

    if data is None or len(data) == 0:
        logger.warning(f"No cached data for {symbol}")
        raise HTTPException(
            status_code=404,
            detail=f"Historical data not available for {symbol}. Run a scan to populate cache."
        )

    logger.info(f"Retrieved {len(data)} rows from cache for {symbol}")

    # Filter to requested period using pandas Timestamp for timezone safety
    from datetime import datetime, timedelta
    cutoff_date = pd.Timestamp(datetime.now() - timedelta(days=days))

    # Handle timezone-aware index
    if data.index.tz is not None:
        cutoff_date = cutoff_date.tz_localize(data.index.tz)

    data = data[data.index >= cutoff_date]

    if len(data) == 0:
        raise HTTPException(
            status_code=404,
            detail=f"No data available for {symbol} in the last {period}"
        )

    # Convert to list of dicts for JSON response
    # Reset index and ensure column is named 'Date'
    df = data.reset_index()
    # The first column after reset_index is the former index
    date_col = df.columns[0]
    df = df.rename(columns={date_col: 'Date'})

    # Convert dates to string format
    df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')

    result = []
    for _, row in df.iterrows():
        result.append({
            'date': row['Date'],
            'open': round(float(row['Open']), 2),
            'high': round(float(row['High']), 2),
            'low': round(float(row['Low']), 2),
            'close': round(float(row['Close']), 2),
            'volume': int(row['Volume']),
        })

    logger.info(f"Returning {len(result)} price records for {symbol}")
    return result
