"""
Celery tasks for stock universe management.

Provides scheduled tasks for:
- Weekly universe refresh from finviz (adds new stocks, deactivates removed)
- S&P 500 membership refresh
"""
import logging
from datetime import datetime
from typing import Any

from ..celery_app import celery_app
from ..database import SessionLocal
from ..wiring.bootstrap import get_stock_universe_service
from .data_fetch_lock import serialized_data_fetch

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name='app.tasks.universe_tasks.refresh_stock_universe')
@serialized_data_fetch('refresh_stock_universe')
def refresh_stock_universe(self, exchange_filter: str = None, market: str | None = None):
    """
    Weekly task to refresh stock universe from finviz.

    - Adds new stocks listed on NYSE/NASDAQ/AMEX
    - Deactivates stocks removed from exchanges
    - Updates metadata (sector, industry, market_cap)

    Scheduled: Sunday 3 AM ET (after weekly-full-refresh)

    Args:
        exchange_filter: Optional filter to only refresh specific exchange
        market: Optional market code (US/HK/JP/TW) for per-market routing and
            logging. The actual per-market universe refresh logic lives in the
            market-specific ingestion services (e.g. ingest_hk_universe_csv);
            this task's `market` kwarg currently acts as a log label and queue
            router. When market is non-US, refresh is skipped so we don't
            accidentally run the US finviz path against an HK market context.

    Returns:
        Dict with refresh statistics
    """
    from .market_queues import market_tag, log_extra, normalize_market
    _log_extra = log_extra(market)
    _market = normalize_market(market) if market is not None else None
    logger.info("=" * 60)
    logger.info("TASK: Stock Universe Refresh %s", market_tag(market), extra=_log_extra)
    logger.info("Timestamp: %s", datetime.now().strftime('%Y-%m-%d %H:%M:%S'), extra=_log_extra)
    if exchange_filter:
        logger.info("Exchange filter: %s", exchange_filter, extra=_log_extra)
    logger.info("=" * 60)

    # Non-US universe refreshes use dedicated CSV/provider-specific ingestion
    # tasks (see ingest_hk_universe_csv). Skip the finviz path for non-US
    # markets so a per-market beat entry doesn't accidentally hit finviz for HK.
    if _market is not None and _market != "US":
        logger.info(
            "Skipping finviz universe refresh for non-US market %s; use the "
            "market-specific ingestion task instead.",
            _market,
            extra=_log_extra,
        )
        return {
            'status': 'skipped',
            'reason': f'finviz universe refresh does not apply to market {_market}',
            'market': _market,
            'timestamp': datetime.now().isoformat(),
        }

    db = SessionLocal()
    try:
        stock_universe_service = get_stock_universe_service()
        stats = stock_universe_service.populate_universe(db, exchange_filter=exchange_filter)

        logger.info("=" * 60)
        logger.info("Universe Refresh Complete!")
        logger.info(f"Added: {stats.get('added', 0)}")
        logger.info(f"Updated: {stats.get('updated', 0)}")
        logger.info(f"Deactivated: {stats.get('deactivated', 0)}")
        logger.info(f"Total in finviz: {stats.get('total', 0)}")
        logger.info("=" * 60)

        return {
            'status': 'success',
            **stats,
            'timestamp': datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error refreshing universe: {e}", exc_info=True)
        return {
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }
    finally:
        db.close()


@celery_app.task(bind=True, name='app.tasks.universe_tasks.ingest_hk_universe_csv')
@serialized_data_fetch('ingest_hk_universe_csv')
def ingest_hk_universe_csv(
    self,
    csv_content: str,
    source_name: str = "hk_manual_csv",
    snapshot_id: str | None = None,
    snapshot_as_of: str | None = None,
    source_metadata: dict[str, Any] | None = None,
    strict: bool = True,
):
    """
    Ingest HK universe rows from CSV using canonical HK normalization.

    This task applies deterministic HK canonicalization (local-code variant
    handling and zero-padding) before upserting rows into stock_universe.
    """
    logger.info("=" * 60)
    logger.info("TASK: HK Universe Ingestion")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Source: {source_name}")
    if snapshot_id:
        logger.info(f"Snapshot ID: {snapshot_id}")
    logger.info("=" * 60)

    db = SessionLocal()
    try:
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
        logger.info("=" * 60)
        logger.info("HK Universe Ingestion Complete!")
        logger.info(f"Added: {stats.get('added', 0)}")
        logger.info(f"Updated: {stats.get('updated', 0)}")
        logger.info(f"Canonical rows: {stats.get('total', 0)}")
        logger.info(f"Rejected rows: {stats.get('rejected', 0)}")
        logger.info("=" * 60)
        return {
            'status': 'success',
            **stats,
            'timestamp': datetime.now().isoformat(),
        }
    except Exception:
        logger.exception("Error ingesting HK universe CSV")
        raise
    finally:
        db.close()


@celery_app.task(bind=True, name='app.tasks.universe_tasks.ingest_jp_universe_csv')
@serialized_data_fetch('ingest_jp_universe_csv')
def ingest_jp_universe_csv(
    self,
    csv_content: str,
    source_name: str = "jp_manual_csv",
    snapshot_id: str | None = None,
    snapshot_as_of: str | None = None,
    source_metadata: dict[str, Any] | None = None,
    strict: bool = True,
):
    """
    Ingest JP universe rows from CSV using canonical JP normalization.

    This task applies deterministic JP canonicalization (local-code format
    normalization + exchange alias mapping) before upserting rows.
    """
    logger.info("=" * 60)
    logger.info("TASK: JP Universe Ingestion")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Source: {source_name}")
    if snapshot_id:
        logger.info(f"Snapshot ID: {snapshot_id}")
    logger.info("=" * 60)

    db = SessionLocal()
    try:
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
        logger.info("=" * 60)
        logger.info("JP Universe Ingestion Complete!")
        logger.info(f"Added: {stats.get('added', 0)}")
        logger.info(f"Updated: {stats.get('updated', 0)}")
        logger.info(f"Canonical rows: {stats.get('total', 0)}")
        logger.info(f"Rejected rows: {stats.get('rejected', 0)}")
        logger.info("=" * 60)
        return {
            'status': 'success',
            **stats,
            'timestamp': datetime.now().isoformat(),
        }
    except Exception:
        logger.exception("Error ingesting JP universe CSV")
        raise
    finally:
        db.close()


@celery_app.task(bind=True, name='app.tasks.universe_tasks.ingest_tw_universe_csv')
@serialized_data_fetch('ingest_tw_universe_csv')
def ingest_tw_universe_csv(
    self,
    csv_content: str,
    source_name: str = "tw_manual_csv",
    snapshot_id: str | None = None,
    snapshot_as_of: str | None = None,
    source_metadata: dict[str, Any] | None = None,
    strict: bool = True,
):
    """
    Ingest TW universe rows from CSV using canonical TW normalization.

    This task applies deterministic TW canonicalization (TWSE/TPEX exchange
    alias handling and .TW/.TWO normalization) before upserting rows.
    """
    logger.info("=" * 60)
    logger.info("TASK: TW Universe Ingestion")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Source: {source_name}")
    if snapshot_id:
        logger.info(f"Snapshot ID: {snapshot_id}")
    logger.info("=" * 60)

    db = SessionLocal()
    try:
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
        logger.info("=" * 60)
        logger.info("TW Universe Ingestion Complete!")
        logger.info(f"Added: {stats.get('added', 0)}")
        logger.info(f"Updated: {stats.get('updated', 0)}")
        logger.info(f"Canonical rows: {stats.get('total', 0)}")
        logger.info(f"Rejected rows: {stats.get('rejected', 0)}")
        logger.info("=" * 60)
        return {
            'status': 'success',
            **stats,
            'timestamp': datetime.now().isoformat(),
        }
    except Exception:
        logger.exception("Error ingesting TW universe CSV")
        raise
    finally:
        db.close()


@celery_app.task(bind=True, name='app.tasks.universe_tasks.refresh_sp500_membership')
@serialized_data_fetch('refresh_sp500_membership')
def refresh_sp500_membership(self):
    """
    Weekly task to update S&P 500 membership flags.

    Fetches current S&P 500 list from Wikipedia and updates
    the is_sp500 flag for all stocks in the universe.

    Returns:
        Dict with update statistics
    """
    logger.info("=" * 60)
    logger.info("TASK: S&P 500 Membership Refresh")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    db = SessionLocal()
    try:
        stock_universe_service = get_stock_universe_service()
        stats = stock_universe_service.update_sp500_membership(db)

        logger.info("=" * 60)
        logger.info("S&P 500 Membership Refresh Complete!")
        logger.info(f"S&P 500 symbols found: {stats.get('sp500_count', 0)}")
        logger.info(f"Stocks updated: {stats.get('updated', 0)}")
        logger.info("=" * 60)

        return {
            'status': 'success',
            **stats,
            'timestamp': datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error refreshing S&P 500 membership: {e}", exc_info=True)
        return {
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }
    finally:
        db.close()
