"""Desktop-local update service for scheduled and manual refresh work."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, time, timedelta, timezone
import os
from pathlib import Path
from typing import Any, Callable

from sqlalchemy.orm import Session, sessionmaker

from app.config import settings
from app.database import SessionLocal
from app.models.industry import IBDGroupRank, IBDIndustryGroup
from app.models.market_breadth import MarketBreadth
from app.services.breadth_calculator_service import BreadthCalculatorService
from app.services.fundamentals_cache_service import FundamentalsCacheService
from app.services.ibd_group_rank_service import IBDGroupRankService
from app.services.job_backend import LocalJobBackend
from app.services.price_cache_service import PriceCacheService
from app.services.stock_universe_service import stock_universe_service
from app.utils.market_hours import get_eastern_now, get_last_trading_day, is_trading_day

from .desktop_runtime_state import (
    UPDATE_STATE_KEY,
    build_data_status,
    default_update_state,
    load_json_setting,
    save_json_setting,
    utc_now_iso,
)


class DesktopUpdateService:
    """Persist and run desktop-local refresh work without Celery or Redis."""

    SETTING_KEY = UPDATE_STATE_KEY
    SETTING_DESCRIPTION = "Desktop runtime update state"
    LOCK_FILE = "desktop_update.lock"

    def __init__(
        self,
        *,
        session_factory: sessionmaker = SessionLocal,
        job_backend: LocalJobBackend,
    ) -> None:
        self._session_factory = session_factory
        self._job_backend = job_backend

    def get_status(self) -> dict[str, Any]:
        with self._session_factory() as db:
            state = load_json_setting(db, key=self.SETTING_KEY, default=default_update_state())
            state = self._merge_snapshot(state)
            state["data_status"] = build_data_status(db)
            save_json_setting(db, key=self.SETTING_KEY, payload=state, description=self.SETTING_DESCRIPTION)
            return state

    def start_update(self, *, scope: str = "manual", triggered_by: str = "manual", force: bool = False) -> dict[str, Any]:
        with self._session_factory() as db:
            current = self._merge_snapshot(load_json_setting(db, key=self.SETTING_KEY, default=default_update_state()))
            if not force and current["status"] in {"queued", "running"}:
                current["data_status"] = build_data_status(db)
                return current

            job_id = self._job_backend.submit_job(
                "desktop_update",
                lambda queued_job_id: self._run_update_job(queued_job_id, scope=scope, triggered_by=triggered_by),
                message="Preparing desktop updates",
                total=0,
            )
            state = default_update_state()
            state.update(
                {
                    "status": "queued",
                    "scope": scope,
                    "triggered_by": triggered_by,
                    "job_id": job_id,
                    "message": "Desktop update queued",
                    "started_at": utc_now_iso(),
                    "data_status": build_data_status(db),
                }
            )
            save_json_setting(db, key=self.SETTING_KEY, payload=state, description=self.SETTING_DESCRIPTION)
            return state

    def run_due_updates(self) -> dict[str, Any]:
        due_scope = self._resolve_due_scope()
        if due_scope is None:
            with self._session_factory() as db:
                state = load_json_setting(db, key=self.SETTING_KEY, default=default_update_state())
                state["message"] = "No scheduled desktop update is due"
                state["data_status"] = build_data_status(db)
                save_json_setting(db, key=self.SETTING_KEY, payload=state, description=self.SETTING_DESCRIPTION)
                return state
        return self.run_update_now(scope=due_scope, triggered_by="scheduler")

    def run_update_now(self, *, scope: str, triggered_by: str) -> dict[str, Any]:
        return self._run_update_job(None, scope=scope, triggered_by=triggered_by)

    def _run_update_job(self, job_id: str | None, *, scope: str, triggered_by: str) -> dict[str, Any]:
        with self._acquire_lock() as locked:
            if not locked:
                with self._session_factory() as db:
                    state = load_json_setting(db, key=self.SETTING_KEY, default=default_update_state())
                    state.update(
                        {
                            "status": "completed",
                            "scope": scope,
                            "triggered_by": triggered_by,
                            "message": "Another desktop update is already running",
                            "completed_at": utc_now_iso(),
                            "error": None,
                            "data_status": build_data_status(db),
                        }
                    )
                    save_json_setting(db, key=self.SETTING_KEY, payload=state, description=self.SETTING_DESCRIPTION)
                    return state

            return self._run_update_cycle(job_id=job_id, scope=scope, triggered_by=triggered_by)

    def _run_update_cycle(self, *, job_id: str | None, scope: str, triggered_by: str) -> dict[str, Any]:
        warnings: list[str] = []
        steps = self._resolve_steps(scope)
        with self._session_factory() as db:
            state = default_update_state()
            state.update(
                {
                    "status": "running",
                    "scope": scope,
                    "triggered_by": triggered_by,
                    "job_id": job_id,
                    "message": "Refreshing desktop data",
                    "started_at": utc_now_iso(),
                    "completed_at": None,
                    "current": 0,
                    "total": len(steps),
                    "percent": 0.0,
                    "steps": [
                        {"name": name, "label": label, "status": "pending", "message": None, "details": None}
                        for name, label, _runner in steps
                    ],
                    "warnings": [],
                    "error": None,
                    "data_status": build_data_status(db),
                }
            )
            save_json_setting(db, key=self.SETTING_KEY, payload=state, description=self.SETTING_DESCRIPTION)

        for index, (name, label, runner) in enumerate(steps, start=1):
            self._update_step(job_id, name=name, label=label, status="running", message=label, current=index - 1)
            try:
                with self._session_factory() as db:
                    details = runner(db)
                    details_status = details.get("status", "completed")
                    message = details.get("message") or label
                    if details_status == "warning":
                        warnings.append(f"{label}: {message}")
                        details_status = "completed"
                self._update_step(
                    job_id,
                    name=name,
                    label=label,
                    status=details_status,
                    message=message,
                    current=index,
                    details=details,
                    warnings=warnings,
                )
            except Exception as exc:  # noqa: BLE001
                return self._complete(
                    job_id,
                    status="failed",
                    message=f"{label} failed",
                    error=str(exc),
                    warnings=warnings,
                )

        return self._complete(
            job_id,
            status="completed",
            message="Desktop data refresh completed" if not warnings else "Desktop update completed with warnings",
            warnings=warnings,
        )

    def _resolve_steps(self, scope: str) -> list[tuple[str, str, Callable[[Session], dict[str, Any]]]]:
        if scope == "core":
            return [
                ("refresh_universe", "Refresh stock universe", self._refresh_universe),
                ("refresh_prices", "Refresh price data", self._refresh_prices),
                ("calculate_breadth", "Update market breadth", self._calculate_breadth),
                ("calculate_groups", "Update group rankings", self._calculate_groups),
                ("refresh_fundamentals", "Refresh core fundamentals", self._refresh_core_fundamentals),
            ]
        if scope == "weekly":
            return [
                ("refresh_universe", "Refresh stock universe", self._refresh_universe),
                ("refresh_prices", "Refresh price data", self._refresh_prices),
                ("calculate_breadth", "Update market breadth", self._calculate_breadth),
                ("calculate_groups", "Update group rankings", self._calculate_groups),
                ("refresh_fundamentals", "Refresh core fundamentals", self._refresh_core_fundamentals),
            ]
        return [
            ("refresh_prices", "Refresh price data", self._refresh_prices),
            ("calculate_breadth", "Update market breadth", self._calculate_breadth),
            ("calculate_groups", "Update group rankings", self._calculate_groups),
        ]

    def _resolve_due_scope(self) -> str | None:
        now_et = get_eastern_now()
        with self._session_factory() as db:
            status = build_data_status(db)
            if not status["local_data_present"]:
                return None

            fundamentals_ts = status["fundamentals"]["last_success_at"]
            if now_et.weekday() >= 5 and self._is_weekly_due(fundamentals_ts):
                return "weekly"

            latest_trading_day = get_last_trading_day(now_et.date())
            market_refresh_time = time(settings.cache_warm_hour, settings.cache_warm_minute)
            refreshed_date = status["prices"]["last_success_at"]
            if (
                is_trading_day(latest_trading_day)
                and now_et.time() >= market_refresh_time
                and refreshed_date != latest_trading_day.isoformat()
            ):
                return "daily"
        return None

    @staticmethod
    def _is_weekly_due(last_success_at: str | None) -> bool:
        if not last_success_at:
            return True
        try:
            last_dt = datetime.fromisoformat(last_success_at)
        except ValueError:
            return True
        return datetime.now(last_dt.tzinfo or timezone.utc) - last_dt > timedelta(days=6)

    def _update_step(
        self,
        job_id: str | None,
        *,
        name: str,
        label: str,
        status: str,
        message: str,
        current: int,
        details: dict[str, Any] | None = None,
        warnings: list[str] | None = None,
        error: str | None = None,
    ) -> None:
        with self._session_factory() as db:
            state = load_json_setting(db, key=self.SETTING_KEY, default=default_update_state())
            state["current_step"] = name if status == "running" else None
            state["message"] = message
            state["current"] = current
            total = max(state.get("total") or 0, len(state.get("steps") or []))
            state["total"] = total
            state["percent"] = round((current / total) * 100, 2) if total else 100.0
            if warnings is not None:
                state["warnings"] = warnings
            if error is not None:
                state["error"] = error
            for step in state["steps"]:
                if step["name"] == name:
                    step["label"] = label
                    step["status"] = status
                    step["message"] = message
                    step["details"] = details
                    break
            state["data_status"] = build_data_status(db)
            save_json_setting(db, key=self.SETTING_KEY, payload=state, description=self.SETTING_DESCRIPTION)

        if job_id:
            self._job_backend.update(
                job_id,
                status="running" if status == "running" else None,
                message=message,
                current=current,
                total=total,
                percent=round((current / total) * 100, 2) if total else 100.0,
                error=error,
            )

    def _complete(
        self,
        job_id: str | None,
        *,
        status: str,
        message: str,
        warnings: list[str],
        error: str | None = None,
    ) -> dict[str, Any]:
        with self._session_factory() as db:
            state = load_json_setting(db, key=self.SETTING_KEY, default=default_update_state())
            state["status"] = status
            state["message"] = message
            state["current_step"] = None
            state["completed_at"] = utc_now_iso()
            state["current"] = state.get("total") or len(state.get("steps") or [])
            state["percent"] = 100.0
            state["warnings"] = warnings
            state["error"] = error
            if status == "completed":
                state["last_success_at"] = state["completed_at"]
            state["data_status"] = build_data_status(db)
            save_json_setting(db, key=self.SETTING_KEY, payload=state, description=self.SETTING_DESCRIPTION)
            completed = state

        if job_id:
            self._job_backend.update(
                job_id,
                status=status,
                message=message,
                current=completed["current"],
                total=completed["total"],
                percent=completed["percent"],
                result=completed,
                error=error,
            )
        return completed

    def _merge_snapshot(self, state: dict[str, Any]) -> dict[str, Any]:
        job_id = state.get("job_id")
        if not job_id:
            return state
        snapshot = self._job_backend.get_status(job_id)
        if snapshot is None:
            return state
        state["current"] = snapshot.current
        state["total"] = snapshot.total
        state["percent"] = snapshot.percent
        state["message"] = snapshot.message or state.get("message")
        if snapshot.started_at:
            state["started_at"] = snapshot.started_at
        if snapshot.completed_at:
            state["completed_at"] = snapshot.completed_at
        if snapshot.status:
            state["status"] = snapshot.status
        if snapshot.error:
            state["error"] = snapshot.error
        return state

    @contextmanager
    def _acquire_lock(self):
        lock_dir = settings.desktop_data_path / "locks"
        lock_dir.mkdir(parents=True, exist_ok=True)
        lock_path = lock_dir / self.LOCK_FILE
        handle = lock_path.open("a+", encoding="utf-8")
        locked = False
        try:
            try:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                locked = True
            except (ImportError, BlockingIOError):
                locked = False
            yield locked
        finally:
            if locked:
                try:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                except Exception:
                    pass
            handle.close()

    def _refresh_universe(self, db: Session) -> dict[str, Any]:
        stats = stock_universe_service.populate_universe(db)
        return {
            "status": "completed",
            "message": f"Universe refresh processed {stats.get('total', 0)} symbols",
            **stats,
        }

    def _refresh_prices(self, db: Session) -> dict[str, Any]:
        symbols = stock_universe_service.get_active_symbols(db)
        if not symbols:
            return {"status": "warning", "message": "No active symbols available for price refresh"}

        price_cache = PriceCacheService.get_instance()
        batch_size = max(int(settings.desktop_price_refresh_batch_size), 25)
        warmed = 0
        requested = 0
        for start in range(0, len(symbols), batch_size):
            batch_symbols = symbols[start:start + batch_size]
            batch = price_cache.get_many(batch_symbols, period="2y")
            requested += len(batch_symbols)
            warmed += sum(1 for frame in batch.values() if frame is not None and not frame.empty)
        return {
            "status": "completed",
            "message": f"Refreshed prices for {warmed}/{requested} symbols",
            "symbols_requested": requested,
            "symbols_warmed": warmed,
        }

    def _refresh_core_fundamentals(self, db: Session) -> dict[str, Any]:
        limit = max(int(settings.desktop_background_fundamentals_limit), 25)
        symbols = stock_universe_service.get_active_symbols(db, limit=limit)
        if not symbols:
            return {"status": "warning", "message": "No active symbols available for fundamentals refresh"}

        cache = FundamentalsCacheService.get_instance()
        updated = 0
        for symbol in symbols:
            if cache.get_fundamentals(symbol, force_refresh=True):
                updated += 1
        return {
            "status": "completed",
            "message": f"Refreshed fundamentals for {updated}/{len(symbols)} core symbols",
            "symbols_requested": len(symbols),
            "symbols_warmed": updated,
        }

    def _calculate_breadth(self, db: Session) -> dict[str, Any]:
        calc_date = get_last_trading_day(get_eastern_now().date())
        metrics = BreadthCalculatorService(db).calculate_daily_breadth(calculation_date=calc_date)
        record = db.query(MarketBreadth).filter(MarketBreadth.date == calc_date).first()
        if record is None:
            record = MarketBreadth(date=calc_date)
            db.add(record)

        record.stocks_up_4pct = metrics["stocks_up_4pct"]
        record.stocks_down_4pct = metrics["stocks_down_4pct"]
        record.ratio_5day = metrics["ratio_5day"]
        record.ratio_10day = metrics["ratio_10day"]
        record.stocks_up_25pct_quarter = metrics["stocks_up_25pct_quarter"]
        record.stocks_down_25pct_quarter = metrics["stocks_down_25pct_quarter"]
        record.stocks_up_25pct_month = metrics["stocks_up_25pct_month"]
        record.stocks_down_25pct_month = metrics["stocks_down_25pct_month"]
        record.stocks_up_50pct_month = metrics["stocks_up_50pct_month"]
        record.stocks_down_50pct_month = metrics["stocks_down_50pct_month"]
        record.stocks_up_13pct_34days = metrics["stocks_up_13pct_34days"]
        record.stocks_down_13pct_34days = metrics["stocks_down_13pct_34days"]
        record.total_stocks_scanned = metrics["total_stocks_scanned"]
        db.commit()

        from app.services.ui_snapshot_service import safe_publish_breadth_bootstrap

        safe_publish_breadth_bootstrap()
        return {
            "status": "completed",
            "message": f"Stored breadth baseline for {calc_date.isoformat()}",
            "date": calc_date.isoformat(),
            "total_stocks_scanned": metrics["total_stocks_scanned"],
        }

    def _calculate_groups(self, db: Session) -> dict[str, Any]:
        if db.query(IBDIndustryGroup).count() == 0:
            return {
                "status": "warning",
                "message": "Group ranking refresh skipped because no industry mappings are available",
            }

        calc_date = get_last_trading_day(get_eastern_now().date())
        results = IBDGroupRankService.get_instance().calculate_group_rankings(db, calc_date)
        ranked = db.query(IBDGroupRank).filter(IBDGroupRank.date == calc_date).count()

        from app.services.ui_snapshot_service import safe_publish_groups_bootstrap

        safe_publish_groups_bootstrap()
        return {
            "status": "completed",
            "message": f"Stored {ranked or len(results)} group rankings for {calc_date.isoformat()}",
            "date": calc_date.isoformat(),
            "groups_ranked": ranked or len(results),
        }
