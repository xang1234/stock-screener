"""Celery task wrapper for the daily feature snapshot use case.

Thin shim — all business logic lives in
:class:`~app.use_cases.feature_store.build_daily_snapshot.BuildDailyFeatureSnapshotUseCase`.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from uuid import uuid4

from celery.exceptions import SoftTimeLimitExceeded

from app.celery_app import celery_app
from app.config import settings
from app.domain.scanning.signature import (
    build_scan_signature_payload,
    hash_scan_signature,
    hash_universe_symbols,
)
from app.services.market_activity_service import (
    mark_market_activity_completed,
    mark_market_activity_failed,
    mark_market_activity_started,
)
from app.tasks.workload_coordination import serialized_market_workload

logger = logging.getLogger(__name__)


def _upsert_feature_run_pointer(*, session_factory, pointer_key: str, run_id: int) -> None:
    """Ensure a published-run pointer references *run_id*."""
    from sqlalchemy.exc import IntegrityError

    from app.infra.db.models.feature_store import FeatureRunPointer

    with session_factory() as db:
        try:
            pointer = (
                db.query(FeatureRunPointer)
                .filter(FeatureRunPointer.key == pointer_key)
                .first()
            )
            if pointer is None:
                db.add(FeatureRunPointer(key=pointer_key, run_id=run_id))
            else:
                pointer.run_id = run_id
            db.commit()
        except IntegrityError:
            db.rollback()
            pointer = (
                db.query(FeatureRunPointer)
                .filter(FeatureRunPointer.key == pointer_key)
                .first()
            )
            if pointer is None:
                db.add(FeatureRunPointer(key=pointer_key, run_id=run_id))
            else:
                pointer.run_id = run_id
            db.commit()


def _fail_stale_feature_runs(*, session_factory, stale_after_minutes: int) -> int:
    """Fail abandoned RUNNING feature runs and emit a warning log."""
    from sqlalchemy import func

    from app.domain.feature_store.models import RunStats, RunStatus
    from app.infra.db.models.feature_store import (
        FeatureRun,
        FeatureRunUniverseSymbol,
        StockFeatureDaily,
    )
    from app.infra.db.uow import SqlUnitOfWork

    if stale_after_minutes <= 0:
        return 0

    now_utc = datetime.now(timezone.utc)
    cutoff = now_utc - timedelta(minutes=stale_after_minutes)
    try:
        with SqlUnitOfWork(session_factory) as uow:
            session = getattr(uow, "session", None)
            if session is None:
                logger.debug("Skipping stale feature-run cleanup: no SQLAlchemy session")
                return 0

            stale_runs = (
                session.query(FeatureRun.id, FeatureRun.created_at)
                .filter(
                    FeatureRun.status == RunStatus.RUNNING.value,
                    FeatureRun.created_at < cutoff,
                )
                .order_by(FeatureRun.created_at.asc())
                .all()
            )
            if not stale_runs:
                return 0

            cleaned = 0
            for run_id, created_at in stale_runs:
                created_at_utc = (
                    created_at.astimezone(timezone.utc)
                    if getattr(created_at, "tzinfo", None) is not None
                    else created_at.replace(tzinfo=timezone.utc)
                )
                total_symbols = (
                    session.query(func.count(FeatureRunUniverseSymbol.symbol))
                    .filter(FeatureRunUniverseSymbol.run_id == run_id)
                    .scalar()
                    or 0
                )
                row_count = (
                    session.query(func.count(StockFeatureDaily.symbol))
                    .filter(StockFeatureDaily.run_id == run_id)
                    .scalar()
                    or 0
                )
                duration_seconds = max(
                    (now_utc - created_at_utc).total_seconds(),
                    0.0,
                )
                uow.feature_runs.mark_failed(
                    run_id,
                    RunStats(
                        total_symbols=total_symbols,
                        processed_symbols=min(row_count, total_symbols),
                        failed_symbols=0,
                        duration_seconds=round(duration_seconds, 2),
                        passed_symbols=None,
                    ),
                    warnings=(
                        "Marked failed by stale-run cleanup after exceeding "
                        f"{stale_after_minutes} minutes in RUNNING state",
                    ),
                )
                cleaned += 1

            uow.commit()
    except Exception:
        logger.exception("Stale feature-run cleanup failed; continuing without cleanup")
        return 0

    logger.warning(
        "Marked %d stale feature run(s) failed before starting a new snapshot",
        cleaned,
    )
    return cleaned


def _enrich_feature_run_with_ibd_metadata(
    *,
    feature_run_id: int,
    ranking_date: date,
    session_factory=None,
    taxonomy_service=None,
    market_group_ranking_service=None,
) -> dict[str, int | str]:
    """Backfill IBD industry/rank metadata into persisted feature-row details."""
    from app.database import SessionLocal
    from app.infra.db.models.feature_store import FeatureRun, StockFeatureDaily
    from app.models.industry import IBDGroupRank, IBDIndustryGroup
    from app.services.market_group_ranking_service import MarketGroupRankingService
    from app.services.market_taxonomy_service import get_market_taxonomy_service

    session_factory = session_factory or SessionLocal
    db = session_factory()
    taxonomy_service = taxonomy_service or get_market_taxonomy_service()
    market_group_ranking_service = (
        market_group_ranking_service or MarketGroupRankingService()
    )
    try:
        feature_run = (
            db.query(FeatureRun)
            .filter(FeatureRun.id == feature_run_id)
            .first()
        )
        config = (feature_run.config_json or {}) if feature_run is not None else {}
        universe = config.get("universe") if isinstance(config, dict) else {}
        if not isinstance(universe, dict):
            universe = {}
        market = str((universe or {}).get("market") or "US").strip().upper()

        rows = (
            db.query(StockFeatureDaily)
            .filter(StockFeatureDaily.run_id == feature_run_id)
            .all()
        )
        if not rows:
            return {
                "run_id": feature_run_id,
                "ranking_date": ranking_date.isoformat(),
                "total_rows": 0,
                "updated_rows": 0,
                "missing_industry_rows": 0,
                "missing_rank_rows": 0,
            }

        industries_by_symbol: dict[str, str | None] = {}
        ranks_by_group: dict[str, int] = {}
        market_themes_by_symbol: dict[str, list[str]] = {}
        sector_by_symbol: dict[str, str | None] = {}

        if market == "US":
            industries_by_symbol = {
                symbol: industry_group
                for symbol, industry_group in (
                    db.query(StockFeatureDaily.symbol, IBDIndustryGroup.industry_group)
                    .join(
                        IBDIndustryGroup,
                        IBDIndustryGroup.symbol == StockFeatureDaily.symbol,
                    )
                    .filter(StockFeatureDaily.run_id == feature_run_id)
                    .all()
                )
            }
            ranks_by_group = {
                industry_group: rank
                for industry_group, rank in (
                    db.query(IBDGroupRank.industry_group, IBDGroupRank.rank)
                    .filter(IBDGroupRank.date == ranking_date)
                    .all()
                )
            }
            market_themes_by_symbol = {row.symbol: [] for row in rows}
        else:
            serialized_rows: list[dict[str, object]] = []
            for row in rows:
                details = dict(row.details_json or {})
                entry = taxonomy_service.get(row.symbol, market=market)
                industries_by_symbol[row.symbol] = entry.industry_group if entry else None
                sector_by_symbol[row.symbol] = entry.sector if entry else None
                market_themes_by_symbol[row.symbol] = entry.themes_list() if entry else []
                serialized_rows.append(
                    {
                        "symbol": row.symbol,
                        "composite_score": row.composite_score,
                        "current_price": details.get("current_price"),
                        "rs_rating": details.get("rs_rating"),
                        "rs_rating_1m": details.get("rs_rating_1m"),
                        "rs_rating_3m": details.get("rs_rating_3m"),
                        "rs_rating_12m": details.get("rs_rating_12m"),
                        "eps_growth_qq": details.get("eps_growth_qq"),
                        "eps_growth_yy": details.get("eps_growth_yy"),
                        "sales_growth_qq": details.get("sales_growth_qq"),
                        "sales_growth_yy": details.get("sales_growth_yy"),
                        "stage": details.get("stage"),
                        "market_cap": details.get("market_cap"),
                        "market_cap_usd": details.get("market_cap_usd"),
                        "ibd_industry_group": industries_by_symbol[row.symbol],
                        "price_sparkline_data": details.get("price_sparkline_data"),
                        "price_trend": details.get("price_trend"),
                        "price_change_1d": details.get("price_change_1d"),
                        "rs_sparkline_data": details.get("rs_sparkline_data"),
                        "rs_trend": details.get("rs_trend"),
                    }
                )
            ranks_by_group = {
                str(row["industry_group"]): int(row["rank"])
                for row in market_group_ranking_service.compute_group_rankings_from_serialized_rows(
                    serialized_rows,
                    ranking_date=ranking_date,
                )
                if row.get("industry_group") and row.get("rank") is not None
            }

        updated_rows = 0
        missing_industry_rows = 0
        missing_rank_rows = 0

        for row in rows:
            details = dict(row.details_json or {})
            industry_group = industries_by_symbol.get(row.symbol)
            group_rank = ranks_by_group.get(industry_group) if industry_group else None
            market_themes = list(market_themes_by_symbol.get(row.symbol) or [])
            sector = sector_by_symbol.get(row.symbol)
            sector_changed = bool(
                market != "US"
                and sector
                and not details.get("gics_sector")
            )
            if industry_group is None:
                missing_industry_rows += 1
            elif group_rank is None:
                missing_rank_rows += 1

            if (
                details.get("ibd_industry_group") != industry_group
                or details.get("ibd_group_rank") != group_rank
                or list(details.get("market_themes") or []) != market_themes
                or sector_changed
            ):
                details["ibd_industry_group"] = industry_group
                details["ibd_group_rank"] = group_rank
                details["market_themes"] = market_themes
                if sector_changed:
                    details["gics_sector"] = sector
                row.details_json = details
                updated_rows += 1

        db.commit()
        return {
            "run_id": feature_run_id,
            "ranking_date": ranking_date.isoformat(),
            "total_rows": len(rows),
            "updated_rows": updated_rows,
            "missing_industry_rows": missing_industry_rows,
            "missing_rank_rows": missing_rank_rows,
        }
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def _create_auto_scan_for_published_run(
    *,
    feature_run_id: int,
    universe_name: str,
    screeners: list[str],
    criteria: dict | None,
    composite_method: str,
) -> str:
    """Create a completed scan row bound to a published feature run."""
    from sqlalchemy import func

    from app.database import SessionLocal
    from app.infra.db.models.feature_store import FeatureRunUniverseSymbol
    from app.infra.db.uow import SqlUnitOfWork
    from app.models.scan_result import SCAN_TRIGGER_SOURCE_AUTO
    from app.services.ui_snapshot_service import safe_publish_scan_bootstrap
    from app.services.scan_execution import cleanup_old_scans
    from app.services.universe_resolver import normalize_universe_definition

    auto_idempotency_key = f"auto-feature-run:{feature_run_id}"
    universe_key: str | None = None

    with SqlUnitOfWork(SessionLocal) as uow:
        existing = uow.scans.get_by_idempotency_key(auto_idempotency_key)
        if existing is not None:
            scan_id = existing.scan_id
        else:
            universe_def = normalize_universe_definition(universe_name)
            feature_run = uow.feature_runs.get_run(feature_run_id)
            run_universe_count = 0
            session = getattr(uow, "session", None)
            if session is not None:
                run_universe_count = (
                    session.query(func.count(FeatureRunUniverseSymbol.symbol))
                    .filter(FeatureRunUniverseSymbol.run_id == feature_run_id)
                    .scalar()
                    or 0
                )
            symbols = []
            if run_universe_count <= 0:
                symbols = uow.universe.resolve_symbols(universe_def)
            ran_at = feature_run.published_at or datetime.now(timezone.utc)
            passed_stocks = (
                feature_run.stats.passed_symbols
                if feature_run.stats and feature_run.stats.passed_symbols is not None
                else 0
            )
            scan = uow.scans.create(
                scan_id=str(uuid4()),
                criteria=criteria or {},
                universe=universe_def.label(),
                universe_key=universe_def.key(),
                universe_type=universe_def.type.value,
                universe_market=(
                    universe_def.market.value if getattr(universe_def, "market", None) else None
                ),
                universe_exchange=(
                    universe_def.exchange.value if universe_def.exchange else None
                ),
                universe_index=(
                    universe_def.index.value if universe_def.index else None
                ),
                universe_symbols=universe_def.symbols,
                screener_types=screeners,
                composite_method=composite_method,
                total_stocks=run_universe_count or len(symbols),
                passed_stocks=passed_stocks,
                status="completed",
                task_id=None,
                idempotency_key=auto_idempotency_key,
                feature_run_id=feature_run_id,
                trigger_source=SCAN_TRIGGER_SOURCE_AUTO,
                started_at=ran_at,
                completed_at=ran_at,
            )
            scan_id = scan.scan_id
            universe_key = scan.universe_key
            uow.commit()

    if universe_key:
        db = SessionLocal()
        try:
            cleanup_old_scans(db, universe_key)
        finally:
            db.close()

    safe_publish_scan_bootstrap(scan_id)
    safe_publish_scan_bootstrap()
    return scan_id


def _is_market_trading_day(as_of: date, *, market: str | None) -> bool:
    if (market or "US").upper() == "US":
        from app.use_cases.feature_store.build_daily_snapshot import _is_us_trading_day

        return _is_us_trading_day(as_of)

    from app.services.market_calendar_service import MarketCalendarService

    return MarketCalendarService().is_trading_day((market or "US").upper(), as_of)


@celery_app.task(
    bind=True,
    name="app.interfaces.tasks.feature_store_tasks.build_daily_snapshot",
    autoretry_for=(ConnectionError, TimeoutError, OSError),
    retry_backoff=60,
    retry_backoff_max=600,
    max_retries=3,
    soft_time_limit=settings.feature_snapshot_soft_time_limit_seconds,
)
@serialized_market_workload("build_daily_snapshot")
def build_daily_snapshot(
    self,
    as_of_date_str: str | None = None,
    screener_names: list[str] | None = None,
    universe_name: str | None = None,
    criteria: dict | None = None,
    composite_method: str | None = None,
    skip_if_published: bool = True,
    static_daily_mode: bool = False,
    market: str | None = None,
    publish_pointer_key: str = "latest_published",
    ignore_runtime_market_gate: bool = False,
    activity_lifecycle: str | None = None,
) -> dict:
    """Build a full feature snapshot for a trading day.

    Parameters
    ----------
    as_of_date_str:
        ISO date string (YYYY-MM-DD). Defaults to today.
    screener_names:
        Screener list, e.g. ``["minervini", "canslim"]``.
    universe_name:
        Universe filter name. Defaults to the shared scan default profile.
    skip_if_published:
        If True (default), skip when a PUBLISHED run already exists for the
        date. Pass False to force a rebuild.
    """
    from app.database import SessionLocal
    from app.domain.scanning.ports import NeverCancelledToken, NullProgressSink
    from app.infra.db.uow import SqlUnitOfWork
    from app.infra.tasks.progress_sink import CeleryProgressSink
    from app.services.runtime_preferences_service import is_market_enabled_now
    from app.services.universe_resolver import normalize_universe_definition
    from app.utils.symbol_support import split_supported_price_symbols
    from app.use_cases.feature_store.build_daily_snapshot import (
        BuildDailySnapshotCommand,
    )
    from app.wiring.bootstrap import get_build_daily_snapshot_use_case

    def _publish_activity(activity_fn, **kwargs) -> None:
        activity_db = SessionLocal()
        try:
            activity_fn(activity_db, **kwargs)
        except Exception:
            logger.warning(
                "Failed to publish market activity for feature snapshot",
                extra={
                    "market": kwargs.get("market"),
                    "stage_key": kwargs.get("stage_key"),
                    "task_id": kwargs.get("task_id"),
                },
                exc_info=True,
            )
        finally:
            activity_db.close()

    as_of = date.fromisoformat(as_of_date_str) if as_of_date_str else date.today()
    from app.domain.scanning.defaults import get_default_scan_profile

    defaults = get_default_scan_profile()
    screeners = defaults["screeners"] if screener_names is None else screener_names
    universe_name = defaults["universe"] if universe_name is None else universe_name
    criteria = defaults["criteria"] if criteria is None else criteria
    composite_method = (
        defaults["composite_method"]
        if composite_method is None
        else composite_method
    )
    correlation_id = self.request.id
    effective_market = market.upper() if isinstance(market, str) else "US"
    activity_lifecycle = activity_lifecycle or "daily_refresh"

    # `market` is a routing/log label here; real per-market scoping comes
    # from `universe_name` via universe_resolver.
    from app.tasks.market_queues import log_extra
    _log_extra = log_extra(market)
    if market is not None:
        bypass_runtime_market_gate = static_daily_mode and ignore_runtime_market_gate
        if not bypass_runtime_market_gate and not is_market_enabled_now(market):
            logger.info("Skipping feature snapshot for disabled market %s", market, extra=_log_extra)
            return {
                "status": "skipped",
                "reason": f"market {effective_market} is disabled in local runtime preferences",
                "market": effective_market,
                "timestamp": datetime.now().isoformat(),
            }
        logger.debug("build_daily_snapshot market label=%s", market, extra=_log_extra)

    logger.info(
        "┌─── build_daily_snapshot ───────────────────┐\n"
        "│  date=%s  correlation_id=%s\n"
        "│  screeners=%s  universe=%s  composite=%s\n"
        "│  soft_time_limit=%ss\n"
        "└────────────────────────────────────────────┘",
        as_of,
        correlation_id,
        screeners,
        universe_name,
        composite_method,
        settings.feature_snapshot_soft_time_limit_seconds,
    )

    cleaned_stale_runs = _fail_stale_feature_runs(
        session_factory=SessionLocal,
        stale_after_minutes=settings.feature_snapshot_stale_after_minutes,
    )

    # ── Trading-day guard (fast exit — lock released immediately) ─
    if not _is_market_trading_day(as_of, market=market):
        logger.info("Skipping build_daily_snapshot: %s is not a trading day for %s", as_of, effective_market)
        return {
            "status": "skipped",
            "reason": "not_trading_day",
            "as_of_date": str(as_of),
            "cleaned_stale_runs": cleaned_stale_runs,
        }

    # ── Skip-if-published guard (cheap DB check) ─────────────
    if skip_if_published:
        uow_check = SqlUnitOfWork(SessionLocal)
        with uow_check:
            universe_def = normalize_universe_definition(universe_name)
            symbols = uow_check.universe.resolve_symbols(universe_def)
            skipped_symbols: list[str] = []
            if static_daily_mode:
                symbols, skipped_symbols = split_supported_price_symbols(symbols)
            signature_payload = build_scan_signature_payload(
                universe_type=getattr(universe_def, "type", "all"),
                screeners=screeners,
                composite_method=composite_method,
                criteria=criteria,
            )
            input_hash = hash_scan_signature(signature_payload)
            universe_hash = hash_universe_symbols(symbols)
            matching_run = uow_check.feature_runs.find_latest_published_exact(
                input_hash=input_hash,
                universe_hash=universe_hash,
                as_of_date=as_of,
            )
            if matching_run is not None:
                _upsert_feature_run_pointer(
                    session_factory=SessionLocal,
                    pointer_key=publish_pointer_key,
                    run_id=matching_run.id,
                )
                auto_scan_id = _create_auto_scan_for_published_run(
                    feature_run_id=matching_run.id,
                    universe_name=universe_name,
                    screeners=screeners,
                    criteria=criteria,
                    composite_method=composite_method,
                )
                logger.info(
                    "Skipping build_daily_snapshot: run %d already published for %s "
                    "(auto scan %s ensured)",
                    matching_run.id, as_of, auto_scan_id,
                )
                return {
                    "status": "skipped",
                    "reason": "already_published",
                    "existing_run_id": matching_run.id,
                    "as_of_date": str(as_of),
                    "cleaned_stale_runs": cleaned_stale_runs,
                    "skipped_symbols": len(skipped_symbols),
                }

    # ── Execute use case ─────────────────────────────────────
    use_case = get_build_daily_snapshot_use_case()
    uow = SqlUnitOfWork(SessionLocal)
    cmd = BuildDailySnapshotCommand(
        as_of_date=as_of,
        screener_names=screeners,
        universe_def=normalize_universe_definition(universe_name),
        criteria=criteria,
        composite_method=composite_method,
        correlation_id=correlation_id,
        exclude_unsupported_price_symbols=static_daily_mode,
        batch_only_prices=static_daily_mode,
        batch_only_fundamentals=static_daily_mode,
        require_bulk_prefetch=static_daily_mode,
        static_chunk_size=(
            settings.static_snapshot_chunk_size if static_daily_mode else None
        ),
        static_parallel_workers=(
            settings.static_snapshot_parallel_workers if static_daily_mode else 1
        ),
        publish_pointer_key=publish_pointer_key,
    )

    try:
        _publish_activity(
            mark_market_activity_started,
            market=effective_market,
            stage_key="snapshot",
            lifecycle=activity_lifecycle,
            task_name=getattr(self, "name", "build_daily_snapshot"),
            task_id=correlation_id,
            message="Building feature snapshot",
        )
        result = use_case.execute(
            uow=uow,
            cmd=cmd,
            progress=NullProgressSink() if static_daily_mode else CeleryProgressSink(self),
            cancel=NeverCancelledToken(),
        )
    except SoftTimeLimitExceeded:
        logger.error(
            "Soft time limit exceeded in build_daily_snapshot for %s (correlation_id=%s)",
            as_of,
            correlation_id,
            exc_info=True,
        )
        _publish_activity(
            mark_market_activity_failed,
            market=effective_market,
            stage_key="snapshot",
            lifecycle=activity_lifecycle,
            task_name=getattr(self, "name", "build_daily_snapshot"),
            task_id=correlation_id,
            message="Soft time limit exceeded",
        )
        raise
    except Exception as exc:
        _publish_activity(
            mark_market_activity_failed,
            market=effective_market,
            stage_key="snapshot",
            lifecycle=activity_lifecycle,
            task_name=getattr(self, "name", "build_daily_snapshot"),
            task_id=correlation_id,
            message=str(exc),
        )
        raise

    auto_scan_id = None
    metadata_refresh_stats = None
    logger.info(
        "build_daily_snapshot completed: run_id=%d status=%s "
        "total=%d processed=%d failed=%d skipped=%d row_count=%d duration=%.2fs dq_passed=%s",
        result.run_id, result.status,
        result.total_symbols, result.processed_symbols,
        result.failed_symbols, result.skipped_symbols, result.row_count,
        result.duration_seconds, result.dq_passed,
    )

    if result.status == "published":
        metadata_refresh_stats = _enrich_feature_run_with_ibd_metadata(
            feature_run_id=result.run_id,
            ranking_date=as_of,
        )
        logger.info(
            "Enriched feature run %d with IBD metadata for %s: %s rows updated",
            result.run_id,
            as_of,
            metadata_refresh_stats["updated_rows"],
        )
        auto_scan_id = _create_auto_scan_for_published_run(
            feature_run_id=result.run_id,
            universe_name=universe_name,
            screeners=screeners,
            criteria=criteria,
            composite_method=composite_method,
        )
        logger.info(
            "Created auto scan %s for published feature run %d",
            auto_scan_id,
            result.run_id,
        )
    _publish_activity(
        mark_market_activity_completed,
        market=effective_market,
        stage_key="snapshot",
        lifecycle=activity_lifecycle,
        task_name=getattr(self, "name", "build_daily_snapshot"),
        task_id=correlation_id,
        current=result.processed_symbols,
        total=result.total_symbols,
        message=f"Feature snapshot {result.status}",
    )

    return {
        "run_id": result.run_id,
        "status": result.status,
        "as_of_date": str(as_of),
        "total_symbols": result.total_symbols,
        "processed_symbols": result.processed_symbols,
        "failed_symbols": result.failed_symbols,
        "skipped_symbols": result.skipped_symbols,
        "row_count": result.row_count,
        "duration_seconds": result.duration_seconds,
        "dq_passed": result.dq_passed,
        "auto_scan_id": auto_scan_id,
        "metadata_refresh": metadata_refresh_stats if result.status == "published" else None,
        "cleaned_stale_runs": cleaned_stale_runs,
        "warnings": list(result.warnings),
    }
