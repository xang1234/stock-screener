"""Celery tasks for canonical, point-in-time Market RS snapshots."""

from __future__ import annotations

from datetime import datetime
import logging

from sqlalchemy.exc import DBAPIError

from app.celery_app import celery_app
from app.database import SessionLocal
from app.domain.relative_strength import BALANCED_RS_FORMULA_VERSION
from app.services.market_rs_inputs import MarketRsInputUnavailable
from app.tasks.market_queues import normalize_market
from app.tasks.transient_database import retry_transient_database_error
from app.wiring.bootstrap import (
    get_market_calendar_service,
    get_market_rs_snapshot_service,
)


logger = logging.getLogger(__name__)
_TRANSIENT_CONNECTION_ERRORS = (ConnectionError, TimeoutError, OSError)


def _failed_result(
    *,
    market: str,
    as_of_date: str,
    formula_version: str,
    reason_code: str,
    diagnostics: dict[str, object],
) -> dict[str, object]:
    return {
        "status": "failed",
        "market": market,
        "as_of_date": as_of_date,
        "formula_version": formula_version,
        "reason_code": reason_code,
        "diagnostics": diagnostics,
    }


def _retry_connection_failure(task, exc: Exception) -> None:
    retries = getattr(getattr(task, "request", None), "retries", 0) or 0
    countdown = min(5 * (2**retries), 60)
    raise task.retry(exc=exc, countdown=countdown, max_retries=2)


@celery_app.task(
    bind=True,
    name="app.tasks.market_rs_tasks.calculate_market_rs_snapshot",
    soft_time_limit=3600,
    max_retries=2,
)
def calculate_market_rs_snapshot(
    self,
    market: str,
    calculation_date: str | None = None,
    formula_version: str = BALANCED_RS_FORMULA_VERSION,
    activity_lifecycle: str | None = None,
) -> dict[str, object]:
    """Idempotently publish one exact-date balanced Market RS snapshot.

    This task deliberately does not update the active formula pointer. During
    rollout it can therefore run as a shadow stage for legacy Markets.
    """
    raw_market = "" if market is None else str(market).strip().upper()
    try:
        market_code = normalize_market(market)
        if market_code == "SHARED":
            raise ValueError("Canonical Market RS requires an explicit market")
    except ValueError as exc:
        return _failed_result(
            market=raw_market or "SHARED",
            as_of_date=str(calculation_date),
            formula_version=formula_version,
            reason_code="invalid_market",
            diagnostics={"error": str(exc)},
        )

    calendar = get_market_calendar_service()
    try:
        as_of_date = (
            calendar.last_completed_trading_day(market_code)
            if calculation_date is None
            else datetime.strptime(calculation_date, "%Y-%m-%d").date()
        )
    except (TypeError, ValueError) as exc:
        return _failed_result(
            market=market_code,
            as_of_date=str(calculation_date),
            formula_version=formula_version,
            reason_code="invalid_date",
            diagnostics={"error": str(exc)},
        )

    if formula_version != BALANCED_RS_FORMULA_VERSION:
        return _failed_result(
            market=market_code,
            as_of_date=as_of_date.isoformat(),
            formula_version=formula_version,
            reason_code="unsupported_formula",
            diagnostics={
                "supported_formula_version": BALANCED_RS_FORMULA_VERSION,
            },
        )

    if not calendar.is_trading_day(market_code, as_of_date):
        return _failed_result(
            market=market_code,
            as_of_date=as_of_date.isoformat(),
            formula_version=formula_version,
            reason_code="not_trading_day",
            diagnostics={"error": f"{as_of_date} is not a {market_code} trading day"},
        )

    db = SessionLocal()
    try:
        run = get_market_rs_snapshot_service().calculate(
            db,
            market=market_code,
            as_of_date=as_of_date,
            formula_version=formula_version,
        )
        if (
            run.status != "completed"
            or run.market != market_code
            or run.as_of_date != as_of_date
            or run.formula_version != formula_version
            or run.id is None
        ):
            raise RuntimeError("Snapshot service returned a non-matching Market RS run")
        return {
            "status": "completed",
            "market": market_code,
            "as_of_date": as_of_date.isoformat(),
            "formula_version": formula_version,
            "market_rs_run_id": run.id,
            "eligible_symbol_count": run.eligible_symbol_count,
        }
    except MarketRsInputUnavailable as exc:
        return _failed_result(
            market=market_code,
            as_of_date=as_of_date.isoformat(),
            formula_version=formula_version,
            reason_code=exc.reason_code,
            diagnostics={
                **exc.diagnostics,
                "benchmark_symbol": exc.benchmark_symbol,
                "universe_hash": exc.universe_hash,
                "expected_symbol_count": exc.expected_symbol_count,
            },
        )
    except _TRANSIENT_CONNECTION_ERRORS as exc:
        _retry_connection_failure(self, exc)
        raise AssertionError("unreachable")
    except DBAPIError as exc:
        retry_transient_database_error(
            self,
            "calculate_market_rs_snapshot",
            exc,
            logger=logger,
            max_retries=2,
        )
        return _failed_result(
            market=market_code,
            as_of_date=as_of_date.isoformat(),
            formula_version=formula_version,
            reason_code="calculation_failed",
            diagnostics={"error_type": type(exc).__name__, "error": str(exc)},
        )
    except Exception as exc:
        logger.exception(
            "Canonical Market RS calculation failed for market=%s date=%s",
            market_code,
            as_of_date,
        )
        return _failed_result(
            market=market_code,
            as_of_date=as_of_date.isoformat(),
            formula_version=formula_version,
            reason_code="calculation_failed",
            diagnostics={"error_type": type(exc).__name__, "error": str(exc)},
        )
    finally:
        db.close()
