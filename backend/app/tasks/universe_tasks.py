"""
Celery tasks for stock universe management.

Provides scheduled tasks for:
- Weekly universe refresh from finviz / official exchange sources
- S&P 500 membership refresh
"""
import logging
from datetime import datetime
from typing import Any, Optional

from ..celery_app import celery_app
from ..database import SessionLocal
from ..services.market_activity_service import (
    mark_market_activity_completed,
    mark_market_activity_failed,
    mark_market_activity_started,
)
from ..wiring.bootstrap import get_stock_universe_service
from .data_fetch_lock import serialized_data_fetch

logger = logging.getLogger(__name__)

_OFFICIAL_SOURCE_MARKETS = frozenset({"HK", "JP", "TW"})
_OFFICIAL_UNIVERSE_LOCK_RETRY_BASE_SECONDS = 300
_OFFICIAL_UNIVERSE_LOCK_RETRY_MAX_SECONDS = 1800
_OFFICIAL_UNIVERSE_LOCK_MAX_RETRIES = 12


def _count_active_universe(market: Optional[str]) -> Optional[int]:
    """Return active-universe size for ``market`` (None on any failure).

    Used as a pre-ingest snapshot so :func:`_emit_universe_drift` can compute
    real drift instead of guessing from the stats dict — HK/JP/TW ingest
    paths don't carry a ``deactivated`` key, so an after-the-fact heuristic
    would miss real drift.
    """
    try:
        from ..models.stock_universe import StockUniverse
        from .market_queues import normalize_market

        m = normalize_market(market) if market is not None else None
        db = SessionLocal()
        try:
            q = db.query(StockUniverse).filter(StockUniverse.is_active == True)
            if m is not None:
                q = q.filter(StockUniverse.market == m)
            return int(q.count())
        finally:
            db.close()
    except Exception as exc:
        logger.debug("telemetry: pre-count failed (%s)", exc)
        return None


def _emit_universe_drift(market: Optional[str], prior_size: Optional[int]) -> None:
    """Emit drift telemetry using a real pre-ingest snapshot."""
    try:
        from ..services.telemetry import get_telemetry

        current_size = _count_active_universe(market)
        if current_size is None:
            return
        get_telemetry().record_universe_drift(
            market, current_size=current_size, prior_size=prior_size,
        )
    except Exception as exc:
        logger.debug("telemetry: universe_drift emit failed (%s)", exc)


def _official_lock_retry_delay(retry_count: int) -> int:
    return min(
        _OFFICIAL_UNIVERSE_LOCK_RETRY_BASE_SECONDS * max(1, retry_count + 1),
        _OFFICIAL_UNIVERSE_LOCK_RETRY_MAX_SECONDS,
    )


def _ingest_official_snapshot(snapshot: Any) -> dict[str, Any]:
    """Dispatch an official-source snapshot into the market-specific ingest path."""
    db = SessionLocal()
    try:
        stock_universe_service = get_stock_universe_service()
        if snapshot.market == "HK":
            return stock_universe_service.ingest_hk_snapshot_rows(
                db,
                rows=snapshot.rows,
                source_name=snapshot.source_name,
                snapshot_id=snapshot.snapshot_id,
                snapshot_as_of=snapshot.snapshot_as_of,
                source_metadata=snapshot.source_metadata,
            )
        if snapshot.market == "JP":
            return stock_universe_service.ingest_jp_snapshot_rows(
                db,
                rows=snapshot.rows,
                source_name=snapshot.source_name,
                snapshot_id=snapshot.snapshot_id,
                snapshot_as_of=snapshot.snapshot_as_of,
                source_metadata=snapshot.source_metadata,
            )
        if snapshot.market == "TW":
            return stock_universe_service.ingest_tw_snapshot_rows(
                db,
                rows=snapshot.rows,
                source_name=snapshot.source_name,
                snapshot_id=snapshot.snapshot_id,
                snapshot_as_of=snapshot.snapshot_as_of,
                source_metadata=snapshot.source_metadata,
            )
        raise ValueError(f"Unsupported official universe snapshot market {snapshot.market!r}")
    finally:
        db.close()


@celery_app.task(bind=True, name='app.tasks.universe_tasks.refresh_stock_universe')
@serialized_data_fetch('refresh_stock_universe')
def refresh_stock_universe(
    self,
    exchange_filter: str = None,
    market: str | None = None,
    activity_lifecycle: str | None = None,
):
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
    from ..services.runtime_preferences_service import is_market_enabled_now
    _log_extra = log_extra(market)
    _market = normalize_market(market) if market is not None else None
    logger.info("=" * 60)
    logger.info("TASK: Stock Universe Refresh %s", market_tag(market), extra=_log_extra)
    logger.info("Timestamp: %s", datetime.now().strftime('%Y-%m-%d %H:%M:%S'), extra=_log_extra)
    if exchange_filter:
        logger.info("Exchange filter: %s", exchange_filter, extra=_log_extra)
    logger.info("=" * 60)

    effective_market = _market or "US"
    activity_lifecycle = activity_lifecycle or "weekly_refresh"
    if not is_market_enabled_now(effective_market):
        logger.info("Skipping universe refresh for disabled market %s", effective_market, extra=_log_extra)
        return {
            'status': 'skipped',
            'reason': f'market {effective_market} is disabled in local runtime preferences',
            'market': effective_market,
            'timestamp': datetime.now().isoformat(),
        }

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

    prior_size = _count_active_universe(effective_market)
    db = SessionLocal()
    try:
        mark_market_activity_started(
            db,
            market=effective_market,
            stage_key="universe",
            lifecycle=activity_lifecycle,
            task_name=getattr(self, "name", "refresh_stock_universe"),
            task_id=getattr(getattr(self, "request", None), "id", None),
            message="Refreshing stock universe",
        )
        stock_universe_service = get_stock_universe_service()
        stats = stock_universe_service.populate_universe(db, exchange_filter=exchange_filter)

        logger.info("=" * 60)
        logger.info("Universe Refresh Complete!")
        logger.info(f"Added: {stats.get('added', 0)}")
        logger.info(f"Updated: {stats.get('updated', 0)}")
        logger.info(f"Deactivated: {stats.get('deactivated', 0)}")
        logger.info(f"Total in finviz: {stats.get('total', 0)}")
        logger.info("=" * 60)

        _emit_universe_drift(effective_market, prior_size)
        mark_market_activity_completed(
            db,
            market=effective_market,
            stage_key="universe",
            lifecycle=activity_lifecycle,
            task_name=getattr(self, "name", "refresh_stock_universe"),
            task_id=getattr(getattr(self, "request", None), "id", None),
            message="Universe refresh completed",
        )

        return {
            'status': 'success',
            **stats,
            'timestamp': datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error refreshing universe: {e}", exc_info=True)
        mark_market_activity_failed(
            db,
            market=effective_market,
            stage_key="universe",
            lifecycle=activity_lifecycle,
            task_name=getattr(self, "name", "refresh_stock_universe"),
            task_id=getattr(getattr(self, "request", None), "id", None),
            message=str(e),
        )
        return {
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }
    finally:
        db.close()


@celery_app.task(bind=True, name='app.tasks.universe_tasks.refresh_official_market_universe')
def refresh_official_market_universe(
    self,
    market: str,
    activity_lifecycle: str | None = None,
):
    """Refresh HK/JP/TW universe snapshots from official exchange sources."""
    from ..services.official_market_universe_source_service import (
        OfficialMarketUniverseSourceService,
    )
    from ..services.runtime_preferences_service import is_market_enabled_now
    from ..wiring.bootstrap import get_data_fetch_lock
    from .market_queues import log_extra, market_tag, normalize_market

    _market = normalize_market(market)
    activity_lifecycle = activity_lifecycle or "weekly_refresh"
    if not is_market_enabled_now(_market):
        logger.info("Skipping official universe refresh for disabled market %s", _market)
        return {
            "status": "skipped",
            "reason": f"market {_market} is disabled in local runtime preferences",
            "market": _market,
            "timestamp": datetime.now().isoformat(),
        }
    if _market not in _OFFICIAL_SOURCE_MARKETS:
        raise ValueError(
            f"refresh_official_market_universe only supports {sorted(_OFFICIAL_SOURCE_MARKETS)}, "
            f"got {_market!r}"
        )

    _log_extra = log_extra(_market)
    task_id = getattr(self.request, "id", None) or "unknown"
    task_name = "refresh_official_market_universe"
    logger.info("=" * 60)
    logger.info("TASK: Official Universe Refresh %s", market_tag(_market), extra=_log_extra)
    logger.info("Timestamp: %s", datetime.now().strftime('%Y-%m-%d %H:%M:%S'), extra=_log_extra)
    logger.info("=" * 60)

    lock = get_data_fetch_lock()
    acquired, is_reentrant = lock.acquire(task_name, task_id, market=_market)
    if not acquired:
        holder = lock.get_current_holder(market=_market) or {}
        delay_seconds = _official_lock_retry_delay(getattr(self.request, "retries", 0))
        logger.info(
            "Official universe refresh lock busy for %s; held by %s (%s). Retrying in %ss.",
            _market,
            holder.get("task_name", "unknown"),
            holder.get("task_id", "unknown"),
            delay_seconds,
            extra=_log_extra,
        )
        raise self.retry(
            countdown=delay_seconds,
            max_retries=_OFFICIAL_UNIVERSE_LOCK_MAX_RETRIES,
            exc=RuntimeError(
                f"Market data fetch lock busy for {_market}; retrying official universe refresh"
            ),
        )

    try:
        activity_db = SessionLocal()
        try:
            mark_market_activity_started(
                activity_db,
                market=_market,
                stage_key="universe",
                lifecycle=activity_lifecycle,
                task_name=getattr(self, "name", task_name),
                task_id=task_id,
                message="Refreshing official market universe",
            )
        finally:
            activity_db.close()
        prior_size = _count_active_universe(_market)
        snapshot = OfficialMarketUniverseSourceService().fetch_market_snapshot(_market)
        stats = _ingest_official_snapshot(snapshot)
        _emit_universe_drift(_market, prior_size)

        logger.info("=" * 60)
        logger.info("Official Universe Refresh Complete %s", market_tag(_market), extra=_log_extra)
        logger.info("Source: %s", snapshot.source_name, extra=_log_extra)
        logger.info("Snapshot ID: %s", snapshot.snapshot_id, extra=_log_extra)
        logger.info("Canonical rows: %s", stats.get('total', 0), extra=_log_extra)
        logger.info("Added: %s", stats.get('added', 0), extra=_log_extra)
        logger.info("Updated: %s", stats.get('updated', 0), extra=_log_extra)
        logger.info("Rejected rows: %s", stats.get('rejected', 0), extra=_log_extra)
        logger.info("=" * 60)
        activity_db = SessionLocal()
        try:
            mark_market_activity_completed(
                activity_db,
                market=_market,
                stage_key="universe",
                lifecycle=activity_lifecycle,
                task_name=getattr(self, "name", task_name),
                task_id=task_id,
                current=stats.get("total"),
                total=stats.get("total"),
                message="Official universe refresh completed",
            )
        finally:
            activity_db.close()

        return {
            'status': 'success',
            'market': _market,
            'source_name': snapshot.source_name,
            'snapshot_id': snapshot.snapshot_id,
            'snapshot_as_of': snapshot.snapshot_as_of,
            **stats,
            'timestamp': datetime.now().isoformat(),
        }
    except Exception as exc:
        activity_db = SessionLocal()
        try:
            mark_market_activity_failed(
                activity_db,
                market=_market,
                stage_key="universe",
                lifecycle=activity_lifecycle,
                task_name=getattr(self, "name", task_name),
                task_id=task_id,
                message=str(exc),
            )
        finally:
            activity_db.close()
        logger.exception("Error refreshing official universe for %s", _market)
        raise
    finally:
        if acquired and not is_reentrant:
            lock.release(task_id, market=_market)


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

    _hk_prior_size = _count_active_universe("HK")
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
        _emit_universe_drift("HK", _hk_prior_size)
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

    _jp_prior_size = _count_active_universe("JP")
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
        _emit_universe_drift("JP", _jp_prior_size)
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

    _tw_prior_size = _count_active_universe("TW")
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
        _emit_universe_drift("TW", _tw_prior_size)
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
