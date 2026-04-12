"""
Celery tasks for stock universe management.

Provides scheduled tasks for:
- Weekly universe refresh from finviz (adds new stocks, deactivates removed)
- S&P 500 membership refresh
"""
import logging
from datetime import datetime

from ..celery_app import celery_app
from ..wiring.bootstrap import get_session_factory, get_stock_universe_service
from .data_fetch_lock import serialized_data_fetch

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name='app.tasks.universe_tasks.refresh_stock_universe')
@serialized_data_fetch('refresh_stock_universe')
def refresh_stock_universe(self, exchange_filter: str = None):
    """
    Weekly task to refresh stock universe from finviz.

    - Adds new stocks listed on NYSE/NASDAQ/AMEX
    - Deactivates stocks removed from exchanges
    - Updates metadata (sector, industry, market_cap)

    Scheduled: Sunday 3 AM ET (after weekly-full-refresh)

    Args:
        exchange_filter: Optional filter to only refresh specific exchange

    Returns:
        Dict with refresh statistics
    """
    logger.info("=" * 60)
    logger.info("TASK: Stock Universe Refresh")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if exchange_filter:
        logger.info(f"Exchange filter: {exchange_filter}")
    logger.info("=" * 60)

    db = get_session_factory()()
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

    db = get_session_factory()()
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
