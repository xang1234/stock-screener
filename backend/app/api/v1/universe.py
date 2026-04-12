"""
Stock Universe API endpoints.

Manages the list of scannable stocks from NYSE/NASDAQ.
"""
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Body
from sqlalchemy.orm import Session
import logging
from typing import Any

from ...database import get_db
from ...wiring.bootstrap import get_stock_universe_service

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/refresh")
async def refresh_universe(
    exchange: str = None,
    db: Session = Depends(get_db)
):
    """
    Refresh stock universe from finviz.

    Fetches all stocks from NYSE/NASDAQ and updates the database.

    Args:
        exchange: Optional filter (nyse, nasdaq, amex) or None for all
        db: Database session

    Returns:
        Statistics about added/updated/deactivated stocks
    """
    try:
        logger.info(f"Refreshing stock universe (exchange={exchange})")
        stock_universe_service = get_stock_universe_service()

        stats = stock_universe_service.populate_universe(db, exchange)

        logger.info(
            f"Universe refresh completed: {stats['added']} added, "
            f"{stats['updated']} updated, {stats['deactivated']} deactivated, "
            f"{stats['total']} total"
        )

        return {
            "message": "Universe refreshed successfully",
            **stats
        }

    except Exception as e:
        logger.error(f"Error refreshing universe: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error refreshing universe: {str(e)}"
        )


@router.post("/import-csv")
async def import_csv(
    csv_content: str = Body(None, embed=True),
    db: Session = Depends(get_db)
):
    """
    Import stocks from CSV content.

    CSV format: symbol,name,exchange,sector,industry,market_cap
    Minimum required: symbol

    Args:
        csv_content: CSV content as string in request body
        db: Database session

    Returns:
        Statistics about added/updated stocks
    """
    try:
        # Get CSV content from body
        if not csv_content:
            raise HTTPException(
                status_code=400,
                detail="csv_content must be provided in request body"
            )

        logger.info("Importing CSV from request body")
        csv_text = csv_content

        # Import stocks
        stock_universe_service = get_stock_universe_service()
        stats = stock_universe_service.populate_from_csv(db, csv_text)

        logger.info(
            f"CSV import completed: {stats['added']} added, "
            f"{stats['updated']} updated, {stats['total']} total"
        )

        return {
            "message": "CSV imported successfully",
            **stats
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error importing CSV: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error importing CSV: {str(e)}"
        )


@router.post("/import-hk-csv")
async def import_hk_csv(
    csv_content: str = Body(None, embed=True),
    source_name: str = Body("hk_manual_csv", embed=True),
    snapshot_id: str | None = Body(None, embed=True),
    snapshot_as_of: str | None = Body(None, embed=True),
    source_metadata: dict[str, Any] | None = Body(None, embed=True),
    strict: bool = Body(True, embed=True),
    db: Session = Depends(get_db),
):
    """
    Import HK universe rows from CSV using HK-specific canonical normalization.

    This path supports local code variants (e.g. 700, 0700.HK) and applies
    deterministic zero-padding + canonical symbol generation.
    """
    try:
        if not csv_content:
            raise HTTPException(
                status_code=400,
                detail="csv_content must be provided in request body",
            )

        stock_universe_service = get_stock_universe_service()
        stats = stock_universe_service.ingest_hk_from_csv(
            db,
            csv_content,
            source_name=source_name,
            snapshot_id=snapshot_id,
            snapshot_as_of=snapshot_as_of,
            source_metadata=source_metadata,
            strict=strict,
        )
        return {
            "message": "HK CSV imported successfully",
            **stats,
        }
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error importing HK CSV: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error importing HK CSV: {str(e)}",
        )


@router.post("/import-jp-csv")
async def import_jp_csv(
    csv_content: str = Body(None, embed=True),
    source_name: str = Body("jp_manual_csv", embed=True),
    snapshot_id: str | None = Body(None, embed=True),
    snapshot_as_of: str | None = Body(None, embed=True),
    source_metadata: dict[str, Any] | None = Body(None, embed=True),
    strict: bool = Body(True, embed=True),
    db: Session = Depends(get_db),
):
    """
    Import JP universe rows from CSV using JP-specific canonical normalization.

    This path supports exchange-specific code formats (e.g. 7203, 7203.T,
    JPX:7203) and emits canonical `.T` symbols.
    """
    try:
        if not csv_content:
            raise HTTPException(
                status_code=400,
                detail="csv_content must be provided in request body",
            )

        stock_universe_service = get_stock_universe_service()
        stats = stock_universe_service.ingest_jp_from_csv(
            db,
            csv_content,
            source_name=source_name,
            snapshot_id=snapshot_id,
            snapshot_as_of=snapshot_as_of,
            source_metadata=source_metadata,
            strict=strict,
        )
        return {
            "message": "JP CSV imported successfully",
            **stats,
        }
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error importing JP CSV: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error importing JP CSV: {str(e)}",
        )


@router.post("/import-tw-csv")
async def import_tw_csv(
    csv_content: str = Body(None, embed=True),
    source_name: str = Body("tw_manual_csv", embed=True),
    snapshot_id: str | None = Body(None, embed=True),
    snapshot_as_of: str | None = Body(None, embed=True),
    source_metadata: dict[str, Any] | None = Body(None, embed=True),
    strict: bool = Body(True, embed=True),
    db: Session = Depends(get_db),
):
    """
    Import TW universe rows from CSV using TW-specific canonical normalization.

    This path supports TWSE and TPEX variants (e.g. 2330, 2330.TW, 3008.TWO,
    TWSE:2330) and preserves exchange-level distinctions under market TW.
    """
    try:
        if not csv_content:
            raise HTTPException(
                status_code=400,
                detail="csv_content must be provided in request body",
            )

        stock_universe_service = get_stock_universe_service()
        stats = stock_universe_service.ingest_tw_from_csv(
            db,
            csv_content,
            source_name=source_name,
            snapshot_id=snapshot_id,
            snapshot_as_of=snapshot_as_of,
            source_metadata=source_metadata,
            strict=strict,
        )
        return {
            "message": "TW CSV imported successfully",
            **stats,
        }
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error importing TW CSV: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error importing TW CSV: {str(e)}",
        )


@router.get("/stats")
async def get_universe_stats(db: Session = Depends(get_db)):
    """
    Get stock universe statistics.

    Returns:
        Total stocks, active stocks, breakdown by exchange
    """
    try:
        stock_universe_service = get_stock_universe_service()
        stats = stock_universe_service.get_stats(db)

        return {
            "total": stats['total'],
            "active": stats['active'],
            "by_exchange": stats['by_exchange'],
            "sp500": stats.get('sp500', 0),
            "by_status": stats.get('by_status', {}),
            "by_market": stats.get('by_market', {}),
            "market_checks": stats.get('market_checks', {}),
            "recent_deactivations": stats.get('recent_deactivations', []),
        }

    except Exception as e:
        logger.error(f"Error getting universe stats: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error getting universe stats: {str(e)}"
        )


@router.get("/stats/by-market")
async def get_universe_market_audit(db: Session = Depends(get_db)):
    """Get market-level universe audit summary for ops dashboards and launch gates."""
    try:
        stock_universe_service = get_stock_universe_service()
        return stock_universe_service.get_market_audit(db)
    except Exception as e:
        logger.error(f"Error getting universe market audit: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error getting universe market audit: {str(e)}"
        )


@router.post("/symbols")
async def add_symbol(
    symbol: str,
    name: str = "",
    db: Session = Depends(get_db)
):
    """
    Manually add a symbol to the universe.

    Args:
        symbol: Stock symbol
        name: Company name (optional)
        db: Database session

    Returns:
        Success message
    """
    try:
        stock_universe_service = get_stock_universe_service()
        success = stock_universe_service.add_manual_symbol(db, symbol, name)

        if success:
            return {"message": f"Symbol {symbol} added successfully"}
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to add symbol {symbol}"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding symbol: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error adding symbol: {str(e)}"
        )


@router.delete("/symbols/{symbol}")
async def deactivate_symbol(
    symbol: str,
    db: Session = Depends(get_db)
):
    """
    Deactivate a symbol (remove from scanning).

    Args:
        symbol: Stock symbol
        db: Database session

    Returns:
        Success message
    """
    try:
        stock_universe_service = get_stock_universe_service()
        success = stock_universe_service.deactivate_symbol(db, symbol)

        if success:
            return {"message": f"Symbol {symbol} deactivated successfully"}
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Symbol {symbol} not found"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deactivating symbol: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error deactivating symbol: {str(e)}"
        )


@router.post("/refresh-sp500")
async def refresh_sp500(db: Session = Depends(get_db)):
    """
    Refresh S&P 500 membership for all stocks.

    Fetches the current S&P 500 list from Wikipedia and updates
    the is_sp500 flag for all stocks in the universe.

    Returns:
        Statistics about S&P 500 update
    """
    try:
        logger.info("Refreshing S&P 500 membership")
        stock_universe_service = get_stock_universe_service()

        stats = stock_universe_service.update_sp500_membership(db)

        logger.info(
            f"S&P 500 refresh completed: {stats['sp500_count']} symbols fetched, "
            f"{stats['updated']} stocks marked as S&P 500"
        )

        return {
            "message": "S&P 500 membership updated successfully",
            **stats
        }

    except Exception as e:
        logger.error(f"Error refreshing S&P 500: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error refreshing S&P 500: {str(e)}"
        )
