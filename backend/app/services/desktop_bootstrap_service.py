"""Desktop bootstrap orchestration for first-run local installs."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Any, Callable

from sqlalchemy.orm import Session, sessionmaker

from app.config import settings
from app.database import SessionLocal
from app.models.app_settings import AppSetting
from app.models.industry import IBDGroupRank, IBDIndustryGroup
from app.models.market_breadth import MarketBreadth
from app.models.stock_universe import StockUniverse
from app.services.breadth_calculator_service import BreadthCalculatorService
from app.services.fundamentals_cache_service import FundamentalsCacheService
from app.services.ibd_group_rank_service import IBDGroupRankService
from app.services.ibd_industry_service import IBDIndustryService
from app.services.job_backend import LocalJobBackend
from app.services.price_cache_service import PriceCacheService
from app.services.stock_universe_service import stock_universe_service
from app.utils.market_hours import get_eastern_now, get_last_trading_day

logger = logging.getLogger(__name__)


class DesktopBootstrapService:
    """Persist and run desktop bootstrap work through the local job backend."""

    SETTING_KEY = "desktop_bootstrap_state"
    SETTING_CATEGORY = "desktop"
    INTERRUPTED_ERROR = "Previous desktop bootstrap was interrupted before it finished."
    INTERRUPTED_MESSAGE = "Previous desktop bootstrap was interrupted. Retry setup to continue."
    INTERRUPTED_STEP_MESSAGE = "Bootstrap interrupted before this step completed."
    STEPS: tuple[tuple[str, str, bool], ...] = (
        ("seed_universe", "Import starter universe", True),
        ("load_industries", "Load starter industry groups", False),
        ("refresh_universe", "Refresh stock universe", False),
        ("warm_prices", "Warm price cache", False),
        ("warm_fundamentals", "Warm fundamentals cache", False),
        ("calculate_breadth", "Generate breadth baseline", False),
        ("calculate_groups", "Generate group ranking baseline", False),
    )

    def __init__(
        self,
        *,
        session_factory: sessionmaker = SessionLocal,
        job_backend: LocalJobBackend,
    ) -> None:
        self._session_factory = session_factory
        self._job_backend = job_backend

    def get_status(self) -> dict[str, Any]:
        """Return the current persisted bootstrap state."""
        with self._session_factory() as db:
            state = self._load_state(db)
            state = self._merge_snapshot(state)
            self._persist_state(db, state)
            return state

    def start_bootstrap(self, *, force: bool = False) -> dict[str, Any]:
        """Queue the bootstrap job if needed and return the current state."""
        if not settings.desktop_mode:
            raise RuntimeError("Desktop bootstrap is only available in desktop mode")

        with self._session_factory() as db:
            current = self._merge_snapshot(self._load_state(db))
            if not force and current["status"] in {"queued", "running"}:
                return current
            if not force and current["status"] == "completed":
                return current

            job_id = self._job_backend.submit_job(
                "desktop_bootstrap",
                self._run_bootstrap_job,
                message="Preparing desktop data",
                total=len(self.STEPS),
            )
            state = self._new_state(
                status="queued",
                message="Desktop bootstrap queued",
                job_id=job_id,
            )
            self._persist_state(db, state)
            return state

    def _run_bootstrap_job(self, job_id: str) -> dict[str, Any]:
        warnings: list[str] = []
        with self._session_factory() as db:
            state = self._load_state(db)
            state.update(
                {
                    "status": "running",
                    "job_id": job_id,
                    "message": "Preparing desktop data",
                    "started_at": datetime.utcnow().isoformat(),
                    "completed_at": None,
                    "current_step": None,
                    "current": 0,
                    "total": len(self.STEPS),
                    "percent": 0.0,
                    "warnings": [],
                    "error": None,
                }
            )
            self._persist_state(db, state)

        step_handlers: dict[str, Callable[[Session], dict[str, Any]]] = {
            "seed_universe": self._seed_universe,
            "load_industries": self._load_industries,
            "refresh_universe": self._refresh_universe,
            "warm_prices": self._warm_prices,
            "warm_fundamentals": self._warm_fundamentals,
            "calculate_breadth": self._calculate_breadth,
            "calculate_groups": self._calculate_groups,
        }

        for index, (name, label, required) in enumerate(self.STEPS, start=1):
            self._update_step(
                job_id,
                name=name,
                label=label,
                status="running",
                message=label,
                current=index - 1,
            )

            try:
                with self._session_factory() as db:
                    details = step_handlers[name](db)
                message = details.get("message") or label
                step_status = details.get("status", "completed")
                if step_status == "warning":
                    warning = f"{label}: {message}"
                    warnings.append(warning)
                    step_status = "completed"
                self._update_step(
                    job_id,
                    name=name,
                    label=label,
                    status=step_status,
                    message=message,
                    details=details,
                    current=index,
                )
            except Exception as exc:  # noqa: BLE001 - bootstrap should downgrade optional failures
                logger.warning("Desktop bootstrap step %s failed", name, exc_info=True)
                step_message = str(exc)
                if required:
                    self._update_step(
                        job_id,
                        name=name,
                        label=label,
                        status="failed",
                        message=step_message,
                        current=index,
                        error=step_message,
                    )
                    return self._complete_state(
                        job_id,
                        status="failed",
                        message=f"{label} failed",
                        warnings=warnings,
                        error=step_message,
                    )

                warnings.append(f"{label}: {step_message}")
                self._update_step(
                    job_id,
                    name=name,
                    label=label,
                    status="failed",
                    message=step_message,
                    details={"status": "failed", "error": step_message},
                    current=index,
                    warnings=warnings,
                )

        message = "Desktop data bootstrap completed"
        if warnings:
            message = "Desktop bootstrap completed with warnings"
        return self._complete_state(
            job_id,
            status="completed",
            message=message,
            warnings=warnings,
        )

    def _complete_state(
        self,
        job_id: str,
        *,
        status: str,
        message: str,
        warnings: list[str],
        error: str | None = None,
    ) -> dict[str, Any]:
        with self._session_factory() as db:
            state = self._load_state(db)
            state["status"] = status
            state["message"] = message
            state["completed_at"] = datetime.utcnow().isoformat()
            state["current_step"] = None
            state["current"] = len(self.STEPS)
            state["total"] = len(self.STEPS)
            state["percent"] = 100.0
            state["warnings"] = warnings
            state["error"] = error
            self._persist_state(db, state)
            return deepcopy(state)

    def _update_step(
        self,
        job_id: str,
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
        total = len(self.STEPS)
        percent = round((current / total) * 100, 2) if total else 100.0
        self._job_backend.update(
            job_id,
            status="running" if status == "running" else None,
            message=message,
            current=current,
            total=total,
            percent=percent,
            error=error,
        )

        with self._session_factory() as db:
            state = self._load_state(db)
            state["current_step"] = name if status == "running" else state.get("current_step")
            state["message"] = message
            state["current"] = current
            state["total"] = total
            state["percent"] = percent
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

            if status != "running":
                state["current_step"] = None

            self._persist_state(db, state)

    def _seed_universe(self, db: Session) -> dict[str, Any]:
        seed_path = Path(settings.desktop_bootstrap_seed_path)
        if not seed_path.exists():
            raise FileNotFoundError(f"Universe seed CSV not found at {seed_path}")

        existing = db.query(StockUniverse).filter(StockUniverse.is_active.is_(True)).count()
        csv_content = seed_path.read_text(encoding="utf-8")
        stats = stock_universe_service.populate_from_csv(db, csv_content)
        if stats.get("total", 0) <= 0:
            raise RuntimeError(f"Universe seed CSV at {seed_path} did not import any symbols")

        active_symbols = db.query(StockUniverse).filter(StockUniverse.is_active.is_(True)).count()
        return {
            "status": "completed",
            "message": (
                f"Seeded starter universe to {active_symbols} active symbols "
                f"({stats.get('added', 0)} added, {stats.get('updated', 0)} updated)"
            ),
            "existing_symbols": existing,
            "active_symbols": active_symbols,
            **stats,
        }

    def _load_industries(self, db: Session) -> dict[str, Any]:
        existing = db.query(IBDIndustryGroup).count()
        if existing > 0:
            return {
                "status": "completed",
                "message": f"Industry seed already present ({existing} mappings)",
                "existing_mappings": existing,
            }

        seed_path = Path(settings.desktop_bootstrap_industry_seed_path)
        if not seed_path.exists():
            return {
                "status": "warning",
                "message": f"Industry seed not found at {seed_path}",
            }

        loaded = IBDIndustryService.load_from_csv(db, str(seed_path))
        return {
            "status": "completed",
            "message": f"Loaded {loaded} industry mappings",
            "loaded": loaded,
        }

    def _refresh_universe(self, db: Session) -> dict[str, Any]:
        if not settings.desktop_bootstrap_refresh_universe:
            return {
                "status": "completed",
                "message": "Universe refresh skipped by configuration",
                "skipped": True,
            }

        stats = stock_universe_service.populate_universe(db)
        return {
            "status": "completed",
            "message": f"Universe refresh processed {stats.get('total', 0)} symbols",
            **stats,
        }

    def _warm_prices(self, db: Session) -> dict[str, Any]:
        symbols = stock_universe_service.get_active_symbols(
            db,
            limit=max(settings.desktop_bootstrap_fundamentals_limit, 25),
        )
        if "SPY" not in symbols:
            symbols = ["SPY", *symbols]
        symbols = list(dict.fromkeys(symbols))

        if not symbols:
            return {
                "status": "warning",
                "message": "Price warmup skipped because no active symbols are available",
            }

        price_cache = PriceCacheService.get_instance()
        batch = price_cache.get_many(symbols, period="2y")
        warmed = sum(1 for frame in batch.values() if frame is not None and not frame.empty)
        return {
            "status": "completed",
            "message": f"Warmed price cache for {warmed}/{len(symbols)} symbols",
            "symbols_requested": len(symbols),
            "symbols_warmed": warmed,
        }

    def _warm_fundamentals(self, db: Session) -> dict[str, Any]:
        limit = settings.desktop_bootstrap_fundamentals_limit
        if limit <= 0:
            return {
                "status": "completed",
                "message": "Fundamentals warmup skipped by configuration",
                "skipped": True,
            }

        symbols = stock_universe_service.get_active_symbols(db, limit=limit)
        if not symbols:
            return {
                "status": "warning",
                "message": "Fundamentals warmup skipped because no active symbols are available",
            }

        cache = FundamentalsCacheService.get_instance()
        updated = 0
        for symbol in symbols:
            data = cache.get_fundamentals(symbol, force_refresh=True)
            if data:
                updated += 1

        return {
            "status": "completed",
            "message": f"Warmed fundamentals for {updated}/{len(symbols)} symbols",
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

        return {
            "status": "completed",
            "message": f"Stored breadth baseline for {calc_date.isoformat()}",
            "date": calc_date.isoformat(),
            "total_stocks_scanned": metrics["total_stocks_scanned"],
        }

    def _calculate_groups(self, db: Session) -> dict[str, Any]:
        industry_count = db.query(IBDIndustryGroup).count()
        if industry_count == 0:
            return {
                "status": "warning",
                "message": "Group ranking baseline skipped because no industry mappings are available",
            }

        calc_date = get_last_trading_day(get_eastern_now().date())
        results = IBDGroupRankService.get_instance().calculate_group_rankings(db, calc_date)
        ranked = db.query(IBDGroupRank).filter(IBDGroupRank.date == calc_date).count()
        if not results and ranked == 0:
            return {
                "status": "warning",
                "message": "Group ranking baseline did not produce any results",
                "date": calc_date.isoformat(),
            }

        return {
            "status": "completed",
            "message": f"Stored {ranked or len(results)} group rankings for {calc_date.isoformat()}",
            "date": calc_date.isoformat(),
            "groups_ranked": ranked or len(results),
        }

    def _load_state(self, db: Session) -> dict[str, Any]:
        setting = db.query(AppSetting).filter(AppSetting.key == self.SETTING_KEY).first()
        if setting is None:
            return self._new_state()

        try:
            state = json.loads(setting.value)
        except json.JSONDecodeError:
            logger.warning("Invalid desktop bootstrap state payload, resetting")
            return self._new_state()

        merged = self._new_state()
        merged.update(state)
        if "steps" in state:
            step_map = {step["name"]: step for step in state["steps"]}
            merged["steps"] = [
                {
                    **template,
                    **deepcopy(step_map.get(template["name"], {})),
                }
                for template in self._default_steps()
            ]
        return merged

    def _persist_state(self, db: Session, state: dict[str, Any]) -> None:
        setting = db.query(AppSetting).filter(AppSetting.key == self.SETTING_KEY).first()
        payload = json.dumps(state)
        if setting is None:
            setting = AppSetting(
                key=self.SETTING_KEY,
                value=payload,
                category=self.SETTING_CATEGORY,
                description="Desktop runtime bootstrap state",
            )
            db.add(setting)
        else:
            setting.value = payload
            setting.category = self.SETTING_CATEGORY
            setting.description = "Desktop runtime bootstrap state"
        db.commit()

    def _merge_snapshot(self, state: dict[str, Any]) -> dict[str, Any]:
        job_id = state.get("job_id")
        if not job_id:
            return state

        snapshot = self._job_backend.get_status(job_id)
        if snapshot is None:
            if state.get("status") in {"queued", "running"}:
                return self._mark_interrupted_state(state)
            return state

        state["current"] = snapshot.current
        state["total"] = snapshot.total
        state["percent"] = snapshot.percent
        state["message"] = snapshot.message or state.get("message")
        if snapshot.started_at:
            state["started_at"] = snapshot.started_at
        if snapshot.completed_at:
            state["completed_at"] = snapshot.completed_at
        if snapshot.status in {"queued", "running"}:
            state["status"] = snapshot.status
        elif snapshot.status in {"completed", "failed", "cancelled"} and state["status"] not in {"completed", "failed"}:
            state["status"] = snapshot.status
        if snapshot.error and not state.get("error"):
            state["error"] = snapshot.error
        return state

    def _mark_interrupted_state(self, state: dict[str, Any]) -> dict[str, Any]:
        current_step = state.get("current_step")
        interrupted = deepcopy(state)
        interrupted["status"] = "failed"
        interrupted["message"] = self.INTERRUPTED_MESSAGE
        interrupted["error"] = self.INTERRUPTED_ERROR
        interrupted["current_step"] = None
        interrupted["completed_at"] = interrupted.get("completed_at") or datetime.utcnow().isoformat()

        for step in interrupted["steps"]:
            if step["status"] == "running" or (current_step and step["name"] == current_step):
                step["status"] = "failed"
                step["message"] = self.INTERRUPTED_STEP_MESSAGE
                break

        return interrupted

    def _new_state(
        self,
        *,
        status: str = "idle",
        message: str = "Desktop bootstrap has not started",
        job_id: str | None = None,
    ) -> dict[str, Any]:
        return {
            "status": status,
            "job_id": job_id,
            "message": message,
            "current_step": None,
            "started_at": None,
            "completed_at": None,
            "current": 0,
            "total": len(self.STEPS),
            "percent": 0.0,
            "steps": self._default_steps(),
            "warnings": [],
            "error": None,
        }

    def _default_steps(self) -> list[dict[str, Any]]:
        return [
            {
                "name": name,
                "label": label,
                "status": "pending",
                "message": None,
                "details": None,
            }
            for name, label, _required in self.STEPS
        ]
