"""Local-default bootstrap orchestration tasks."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable

from celery import chain
from celery.exceptions import Retry

from ..database import SessionLocal
from ..domain.bootstrap.plan import (
    BootstrapOperation,
    BootstrapQueueKind,
    MarketBootstrapPlan,
    build_bootstrap_plan,
)
from ..services.market_activity_service import (
    mark_current_market_activity_failed,
    mark_market_activity_completed,
    mark_market_activity_failed,
    mark_market_activity_progress,
)
from ..services.bootstrap_run_manifest import (
    BootstrapRunManifest,
    BootstrapQueueState,
    BootstrapRunManifestRepository,
)
from ..tasks.market_queues import (
    data_fetch_queue_for_market,
    market_jobs_queue_for_market,
    normalize_market,
)
from ..celery_app import celery_app

logger = logging.getLogger(__name__)
WAIT_FOR_BOOTSTRAP_PRICE_WARMUP_TASK_NAME = (
    "app.tasks.runtime_bootstrap_tasks.wait_for_bootstrap_price_warmup"
)
BOOTSTRAP_PRICE_WARMUP_RETRY_COUNTDOWN_SECONDS = 30
BOOTSTRAP_PRICE_WARMUP_MAX_RETRIES = 120
BOOTSTRAP_PRICE_WARMUP_METADATA_MAX_AGE = timedelta(hours=12)


@dataclass(frozen=True)
class ReadinessFailure:
    stage_key: str
    activity_message: str
    result_reason: str


@dataclass(frozen=True)
class MarketReadinessCompletion:
    market: str
    ready: bool
    failure: ReadinessFailure | None = None


class BootstrapDispatchError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        primary_market: str,
        enabled_markets: Iterable[str],
        primary_task_id: str | None = None,
        market_task_ids: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.primary_market = primary_market
        self.enabled_markets = list(enabled_markets)
        self.primary_task_id = primary_task_id
        self.market_task_ids = dict(market_task_ids or {})

    @property
    def dispatched_any(self) -> bool:
        return self.primary_task_id is not None or bool(self.market_task_ids)


@dataclass
class BootstrapQueueManifestRecorder:
    primary_market: str
    enabled_markets: list[str]
    market_task_ids: dict[str, str]
    primary_task_id: str | None = None

    @classmethod
    def create(
        cls,
        *,
        primary_market: str,
        enabled_markets: Iterable[str],
    ) -> "BootstrapQueueManifestRecorder":
        return cls(
            primary_market=primary_market,
            enabled_markets=list(enabled_markets),
            market_task_ids={},
        )

    def record_queueing(self) -> None:
        self._record(BootstrapQueueState.QUEUEING)

    def record_dispatched_market(self, *, market: str, task_id: str) -> None:
        market_code = normalize_market(market)
        if market_code == self.primary_market:
            self.primary_task_id = task_id
        self.market_task_ids[market_code] = task_id
        self.record_partial_safely()

    def record_partial_safely(self) -> None:
        try:
            self._record(BootstrapQueueState.PARTIAL)
        except Exception:
            logger.warning(
                "Failed to record partial bootstrap task manifest",
                extra=self.log_extra(),
                exc_info=True,
            )

    def record_queued(self) -> None:
        self._record(BootstrapQueueState.QUEUED)

    def record_dispatch_failed_safely(self) -> None:
        try:
            self._record(BootstrapQueueState.DISPATCH_FAILED)
        except Exception:
            logger.warning(
                "Failed to record bootstrap dispatch failure",
                extra=self.log_extra(),
                exc_info=True,
            )

    def _record(self, queue_state: BootstrapQueueState) -> None:
        record_runtime_bootstrap_run(
            primary_market=self.primary_market,
            enabled_markets=self.enabled_markets,
            primary_task_id=self.primary_task_id,
            market_task_ids=self.market_task_ids,
            queue_state=queue_state.value,
        )

    def log_extra(self) -> dict:
        return {
            "primary_market": self.primary_market,
            "enabled_markets": self.enabled_markets,
            "market_task_ids": self.market_task_ids,
        }

    def dispatch_error(self, exc: Exception) -> BootstrapDispatchError:
        return BootstrapDispatchError(
            str(exc),
            primary_market=self.primary_market,
            enabled_markets=self.enabled_markets,
            primary_task_id=self.primary_task_id,
            market_task_ids=self.market_task_ids,
        )


def _bootstrap_universe_name(market: str) -> str:
    return f"market:{market.upper()}"


def _queue_for_stage(stage) -> str:
    if stage.queue_kind == BootstrapQueueKind.DATA_FETCH:
        return data_fetch_queue_for_market(stage.kwargs["market"])
    if stage.queue_kind == BootstrapQueueKind.MARKET_JOBS:
        return market_jobs_queue_for_market(stage.kwargs["market"])
    raise ValueError(f"Unsupported bootstrap queue kind: {stage.queue_kind}")


def _warmup_int(metadata: dict | None, key: str) -> int | None:
    if not metadata:
        return None
    value = metadata.get(key)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _warmup_summary(metadata: dict | None) -> str:
    if not metadata:
        return "missing metadata"
    status = str(metadata.get("status") or "missing status")
    current = _warmup_int(metadata, "count")
    total = _warmup_int(metadata, "total")
    if current is None and total is None:
        return status
    return f"{status}, {current}/{total}"


def _warmup_percent(metadata: dict | None) -> float | None:
    current = _warmup_int(metadata, "count")
    total = _warmup_int(metadata, "total")
    if current is None or total is None or total <= 0:
        return None
    return round((current / total) * 100, 4)


def _bootstrap_price_warmup_wait_reason(metadata: dict | None) -> str | None:
    if not metadata:
        return "Missing price cache warmup metadata for bootstrap"

    if metadata.get("status") != "completed":
        return (
            "Cache warmup not complete for bootstrap price run "
            f"({_warmup_summary(metadata)})"
        )

    completed_at_raw = metadata.get("completed_at")
    if completed_at_raw:
        try:
            completed_at = datetime.fromisoformat(str(completed_at_raw))
        except ValueError:
            return "Price cache warmup metadata timestamp is invalid"
        now = datetime.now(completed_at.tzinfo) if completed_at.tzinfo else datetime.now()
        if now - completed_at > BOOTSTRAP_PRICE_WARMUP_METADATA_MAX_AGE:
            return "Price cache warmup metadata is stale for bootstrap"

    return None


def _build_market_bootstrap_signatures(market_plan: MarketBootstrapPlan) -> list:
    from app.interfaces.tasks.feature_store_tasks import build_daily_snapshot
    from app.tasks.breadth_tasks import calculate_daily_breadth_with_gapfill
    from app.tasks.cache_tasks import smart_refresh_cache
    from app.tasks.fundamentals_tasks import refresh_all_fundamentals
    from app.tasks.group_rank_tasks import calculate_daily_group_rankings_with_gapfill
    from app.tasks.industry_tasks import load_tracked_ibd_industry_groups
    from app.tasks.universe_tasks import (
        refresh_official_market_universe,
        refresh_stock_universe,
    )

    task_by_operation = {
        BootstrapOperation.REFRESH_STOCK_UNIVERSE: refresh_stock_universe,
        BootstrapOperation.REFRESH_OFFICIAL_MARKET_UNIVERSE: refresh_official_market_universe,
        BootstrapOperation.LOAD_TRACKED_IBD_INDUSTRY_GROUPS: load_tracked_ibd_industry_groups,
        BootstrapOperation.SMART_REFRESH_CACHE: smart_refresh_cache,
        BootstrapOperation.WAIT_FOR_BOOTSTRAP_PRICE_WARMUP: wait_for_bootstrap_price_warmup,
        BootstrapOperation.REFRESH_ALL_FUNDAMENTALS: refresh_all_fundamentals,
        BootstrapOperation.CALCULATE_DAILY_BREADTH_WITH_GAPFILL: (
            calculate_daily_breadth_with_gapfill
        ),
        BootstrapOperation.CALCULATE_DAILY_GROUP_RANKINGS_WITH_GAPFILL: (
            calculate_daily_group_rankings_with_gapfill
        ),
        BootstrapOperation.BUILD_DAILY_SNAPSHOT: build_daily_snapshot,
    }
    return [
        task_by_operation[stage.operation]
        .si(**stage.kwargs)
        .set(queue=_queue_for_stage(stage))
        for stage in market_plan.stages
    ]


@celery_app.task(
    bind=True,
    name=WAIT_FOR_BOOTSTRAP_PRICE_WARMUP_TASK_NAME,
    queue="market_jobs",
    soft_time_limit=300,
    max_retries=BOOTSTRAP_PRICE_WARMUP_MAX_RETRIES,
)
def wait_for_bootstrap_price_warmup(
    self,
    market: str,
    activity_lifecycle: str | None = None,
) -> dict:
    from ..wiring.bootstrap import get_price_cache

    market_code = normalize_market(market)
    lifecycle = activity_lifecycle or "bootstrap"
    task_name = getattr(self, "name", WAIT_FOR_BOOTSTRAP_PRICE_WARMUP_TASK_NAME)
    task_id = getattr(getattr(self, "request", None), "id", None)
    db = SessionLocal()
    try:
        warmup_metadata = get_price_cache().get_warmup_metadata(market=market_code)
        wait_reason = _bootstrap_price_warmup_wait_reason(warmup_metadata)
        if wait_reason is None:
            mark_market_activity_completed(
                db,
                market=market_code,
                stage_key="prices",
                lifecycle=lifecycle,
                task_name=task_name,
                task_id=task_id,
                message="Price cache warmup ready",
            )
            return {
                "status": "ready",
                "market": market_code,
                "warmup": warmup_metadata,
            }

        retries = getattr(getattr(self, "request", None), "retries", 0) or 0
        if retries >= BOOTSTRAP_PRICE_WARMUP_MAX_RETRIES:
            mark_market_activity_failed(
                db,
                market=market_code,
                stage_key="prices",
                lifecycle=lifecycle,
                task_name=task_name,
                task_id=task_id,
                message=f"Price cache warmup unavailable: {wait_reason}",
            )
            raise RuntimeError(wait_reason)

        mark_market_activity_progress(
            db,
            market=market_code,
            stage_key="prices",
            lifecycle=lifecycle,
            task_name=task_name,
            task_id=task_id,
            percent=_warmup_percent(warmup_metadata),
            current=_warmup_int(warmup_metadata, "count"),
            total=_warmup_int(warmup_metadata, "total"),
            message=f"Waiting for price cache warmup: {_warmup_summary(warmup_metadata)}",
        )
        raise self.retry(
            exc=RuntimeError(
                f"waiting_for_bootstrap_price_warmup:{market_code}: {wait_reason}"
            ),
            countdown=BOOTSTRAP_PRICE_WARMUP_RETRY_COUNTDOWN_SECONDS,
            max_retries=BOOTSTRAP_PRICE_WARMUP_MAX_RETRIES,
        )
    except Retry:
        raise
    finally:
        db.close()


def _apply_bootstrap_workflow(workflow, errback):
    return workflow.apply_async(link_error=errback)


def _queue_market_bootstrap_workflow(
    market_plan: MarketBootstrapPlan,
    *,
    completion_task,
    completion_kwargs: dict,
    errback_task,
    errback_kwargs: dict,
):
    completion = completion_task.si(**completion_kwargs).set(queue="celery")
    errback = errback_task.s(**errback_kwargs).set(queue="celery")
    return _apply_bootstrap_workflow(
        chain(*_build_market_bootstrap_signatures(market_plan), completion),
        errback,
    )


def record_runtime_bootstrap_run(
    *,
    primary_market: str,
    enabled_markets: Iterable[str],
    primary_task_id: str | None = None,
    market_task_ids: dict[str, str | None] | None = None,
    queue_state: BootstrapQueueState | str = BootstrapQueueState.QUEUED,
) -> dict:
    db = SessionLocal()
    try:
        return BootstrapRunManifestRepository().save(
            db,
            BootstrapRunManifest.create(
                primary_market=primary_market,
                enabled_markets=enabled_markets,
                primary_task_id=primary_task_id,
                market_task_ids=market_task_ids or {},
                queue_state=queue_state,
                queued_at=datetime.now(timezone.utc).isoformat(),
            ),
        )
    finally:
        db.close()


def _readiness_failure(market_result) -> ReadinessFailure:
    if market_result is None or not market_result.core_ready:
        return ReadinessFailure(
            stage_key="core",
            activity_message="Bootstrap core data incomplete",
            result_reason="missing core market data",
        )
    return ReadinessFailure(
        stage_key="scan",
        activity_message="Bootstrap scan did not publish",
        result_reason="missing published auto scan",
    )


def _evaluate_market_readiness(db, *, market: str, bootstrap_started_at=None) -> MarketReadinessCompletion:
    from ..services.bootstrap_readiness_service import BootstrapReadinessService

    market_code = normalize_market(market)
    readiness = BootstrapReadinessService().evaluate(
        db,
        enabled_markets=[market_code],
        bootstrap_started_at=bootstrap_started_at,
    )
    market_result = readiness.market_results.get(market_code)
    result_market = market_result.market if market_result else market_code
    if market_result and market_result.ready:
        return MarketReadinessCompletion(market=result_market, ready=True)
    return MarketReadinessCompletion(
        market=result_market,
        ready=False,
        failure=_readiness_failure(market_result),
    )


def _mark_readiness_failure(db, completion: MarketReadinessCompletion) -> str:
    failure = completion.failure or _readiness_failure(None)
    mark_market_activity_failed(
        db,
        market=completion.market,
        stage_key=failure.stage_key,
        lifecycle="bootstrap",
        task_name="runtime_bootstrap",
        task_id=None,
        message=failure.activity_message,
    )
    return failure.result_reason


def queue_local_runtime_bootstrap(*, primary_market: str, enabled_markets: Iterable[str]) -> str:
    plan = build_bootstrap_plan(
        primary_market=primary_market,
        enabled_markets=enabled_markets,
    )
    primary = plan.primary_market
    enabled = list(plan.enabled_markets)

    market_plans_by_code = {market_plan.market: market_plan for market_plan in plan.market_plans}
    primary_plan = market_plans_by_code[primary]
    manifest_recorder = BootstrapQueueManifestRecorder.create(
        primary_market=primary,
        enabled_markets=enabled,
    )
    manifest_recorder.record_queueing()

    try:
        primary_task = _queue_market_bootstrap_workflow(
            primary_plan,
            completion_task=complete_local_runtime_bootstrap,
            completion_kwargs={"primary_market": primary},
            errback_task=fail_local_runtime_bootstrap,
            errback_kwargs={"primary_market": primary},
        )
        manifest_recorder.record_dispatched_market(
            market=primary,
            task_id=primary_task.id,
        )

        for market_plan in plan.market_plans:
            if market_plan.market == primary:
                continue
            background_task = _queue_market_bootstrap_workflow(
                market_plan,
                completion_task=complete_background_market_bootstrap,
                completion_kwargs={"market": market_plan.market},
                errback_task=fail_background_market_bootstrap,
                errback_kwargs={"market": market_plan.market},
            )
            manifest_recorder.record_dispatched_market(
                market=market_plan.market,
                task_id=background_task.id,
            )

    except Exception as exc:
        manifest_recorder.record_dispatch_failed_safely()
        raise manifest_recorder.dispatch_error(exc) from exc

    try:
        manifest_recorder.record_queued()
    except Exception:
        logger.warning(
            "Queued bootstrap tasks but failed to record task manifest",
            extra=manifest_recorder.log_extra(),
            exc_info=True,
        )

    logger.info(
        "Queued local runtime bootstrap",
        extra={
            "primary_market": primary,
            "enabled_markets": enabled,
            "task_id": manifest_recorder.primary_task_id,
        },
    )
    return manifest_recorder.primary_task_id


@celery_app.task(
    name="app.tasks.runtime_bootstrap_tasks.complete_local_runtime_bootstrap",
    queue="celery",
)
def complete_local_runtime_bootstrap(
    primary_market: str,
    enabled_markets: Iterable[str] | None = None,
) -> dict:
    from ..services.runtime_preferences_service import (
        get_runtime_preferences,
        set_bootstrap_state,
    )

    del enabled_markets  # Legacy Celery payload compatibility; readiness is primary-only.
    primary_market = normalize_market(primary_market)
    db = SessionLocal()
    try:
        prefs = get_runtime_preferences(db)
        completion = _evaluate_market_readiness(
            db,
            market=primary_market,
            bootstrap_started_at=prefs.bootstrap_started_at,
        )
        if not completion.ready:
            reason = _mark_readiness_failure(db, completion)
            set_bootstrap_state(db, "failed")
            return {
                "status": "failed",
                "primary_market": primary_market,
                "market": completion.market,
                "reason": reason,
            }
        set_bootstrap_state(db, "ready")
    finally:
        db.close()

    logger.info(
        "Completed local runtime bootstrap",
        extra={
            "primary_market": primary_market,
        },
    )
    return {
        "status": "ready",
        "primary_market": primary_market,
        "market": completion.market,
    }


@celery_app.task(
    name="app.tasks.runtime_bootstrap_tasks.complete_background_market_bootstrap",
    queue="celery",
)
def complete_background_market_bootstrap(market: str) -> dict:
    from ..services.runtime_preferences_service import get_runtime_preferences

    db = SessionLocal()
    try:
        prefs = get_runtime_preferences(db)
        completion = _evaluate_market_readiness(
            db,
            market=market,
            bootstrap_started_at=prefs.bootstrap_started_at,
        )
        if completion.ready:
            return {
                "status": "ready",
                "market": completion.market,
            }
        reason = _mark_readiness_failure(db, completion)
        return {
            "status": "failed",
            "market": completion.market,
            "reason": reason,
        }
    finally:
        db.close()


@celery_app.task(
    name="app.tasks.runtime_bootstrap_tasks.fail_local_runtime_bootstrap",
    queue="celery",
)
def fail_local_runtime_bootstrap(*_celery_errback_args, primary_market: str) -> dict:
    from ..services.runtime_preferences_service import set_bootstrap_state

    primary_market = normalize_market(primary_market)
    db = SessionLocal()
    try:
        set_bootstrap_state(db, "failed")
        mark_current_market_activity_failed(
            db,
            market=str(primary_market).upper(),
            lifecycle="bootstrap",
            message="Bootstrap failed",
        )
    finally:
        db.close()

    logger.warning(
        "Marked local runtime bootstrap failed",
        extra={
            "primary_market": primary_market,
            "callback_args": _celery_errback_args,
        },
    )
    return {
        "status": "failed",
        "primary_market": primary_market,
        "market": str(primary_market).upper(),
    }


@celery_app.task(
    name="app.tasks.runtime_bootstrap_tasks.fail_background_market_bootstrap",
    queue="celery",
)
def fail_background_market_bootstrap(*_celery_errback_args, market: str) -> dict:
    market = normalize_market(market)
    db = SessionLocal()
    try:
        mark_current_market_activity_failed(
            db,
            market=str(market).upper(),
            lifecycle="bootstrap",
            message="Bootstrap failed",
        )
    finally:
        db.close()

    logger.warning(
        "Marked background market bootstrap failed",
        extra={
            "market": market,
            "callback_args": _celery_errback_args,
        },
    )
    return {
        "status": "failed",
        "market": str(market).upper(),
    }
