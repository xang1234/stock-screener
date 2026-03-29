"""Desktop setup service for first-run local installs."""

from __future__ import annotations

from copy import deepcopy
from datetime import date
import json
from pathlib import Path
import sqlite3
from typing import Any

from sqlalchemy.orm import Session, sessionmaker

from app.config import settings
from app.database import SessionLocal, engine
from app.models.industry import IBDGroupRank, IBDIndustryGroup
from app.models.market_breadth import MarketBreadth
from app.models.stock import StockFundamental, StockPrice
from app.services.job_backend import LocalJobBackend
from app.services.stock_universe_service import stock_universe_service

from .desktop_runtime_state import (
    SETUP_STATE_KEY,
    build_data_status,
    default_setup_state,
    load_json_setting,
    local_data_present,
    save_json_setting,
    utc_now_iso,
)


class DesktopSetupService:
    """Orchestrate first-run starter data install and optional core download."""

    SETTING_KEY = SETUP_STATE_KEY
    SETTING_DESCRIPTION = "Desktop runtime setup state"
    INTERRUPTED_ERROR = "Previous desktop setup was interrupted before it finished."
    INTERRUPTED_MESSAGE = "Previous desktop setup was interrupted. Retry setup to continue."
    INTERRUPTED_STEP_MESSAGE = "Interrupted before setup finished"
    OPTIONS = (
        {
            "id": "quick_start",
            "label": "Quick Start",
            "description": "Install starter data, open immediately, and continue updates in the background.",
            "recommended": True,
        },
        {
            "id": "download_latest",
            "label": "Download Latest Before Opening",
            "description": "Install starter data, then wait for the first core local refresh to complete.",
            "recommended": False,
        },
    )

    def __init__(
        self,
        *,
        session_factory: sessionmaker = SessionLocal,
        job_backend: LocalJobBackend,
        update_service,
    ) -> None:
        self._session_factory = session_factory
        self._job_backend = job_backend
        self._update_service = update_service

    def get_options(self) -> list[dict[str, Any]]:
        return [deepcopy(option) for option in self.OPTIONS]

    def get_status(self) -> dict[str, Any]:
        with self._session_factory() as db:
            state = load_json_setting(db, key=self.SETTING_KEY, default=default_setup_state())
            state = self._merge_snapshot(state)
            state["app_ready"] = local_data_present(db)
            state["data_status"] = build_data_status(db)
            save_json_setting(db, key=self.SETTING_KEY, payload=state, description=self.SETTING_DESCRIPTION)
            return state

    def start_setup(self, *, mode: str = "quick_start", force: bool = False) -> dict[str, Any]:
        if mode not in {option["id"] for option in self.OPTIONS}:
            raise ValueError(f"Unknown desktop setup mode: {mode}")

        with self._session_factory() as db:
            current = self._merge_snapshot(load_json_setting(db, key=self.SETTING_KEY, default=default_setup_state()))
            if not force and current["status"] in {"queued", "running"}:
                current["data_status"] = build_data_status(db)
                return current

        if mode == "quick_start":
            return self._run_quick_start()

        return self._queue_download_latest()

    def get_legacy_bootstrap_status(self) -> dict[str, Any]:
        status = self.get_status()
        return {
            "status": status["status"],
            "job_id": status.get("job_id"),
            "message": status.get("message"),
            "current_step": status.get("current_step"),
            "started_at": status.get("started_at"),
            "completed_at": status.get("completed_at"),
            "current": status.get("current"),
            "total": status.get("total"),
            "percent": status.get("percent"),
            "steps": status.get("steps", []),
            "warnings": status.get("warnings", []),
            "error": status.get("error"),
        }

    def _run_quick_start(self) -> dict[str, Any]:
        with self._session_factory() as db:
            state = default_setup_state()
            state.update(
                {
                    "status": "running",
                    "mode": "quick_start",
                    "message": "Installing starter data",
                    "started_at": utc_now_iso(),
                    "steps": [
                        {"name": "install_starter", "label": "Install starter data", "status": "running", "message": "Installing starter data", "details": None},
                        {"name": "queue_updates", "label": "Queue background updates", "status": "pending", "message": None, "details": None},
                    ],
                    "current": 0,
                    "total": 2,
                    "percent": 0.0,
                }
            )
            save_json_setting(db, key=self.SETTING_KEY, payload=state, description=self.SETTING_DESCRIPTION)

            starter_details = self._install_starter_data(db)
            state["steps"][0].update(status="completed", message=starter_details["message"], details=starter_details)
            state["current"] = 1
            state["percent"] = 50.0
            state["message"] = starter_details["message"]
            save_json_setting(db, key=self.SETTING_KEY, payload=state, description=self.SETTING_DESCRIPTION)

        update_state = self._update_service.start_update(scope="daily", triggered_by="setup")

        with self._session_factory() as db:
            state = load_json_setting(db, key=self.SETTING_KEY, default=default_setup_state())
            state["status"] = "completed"
            state["mode"] = "quick_start"
            state["current"] = 2
            state["total"] = 2
            state["percent"] = 100.0
            state["completed_at"] = utc_now_iso()
            state["starter_baseline_active"] = True
            state["app_ready"] = True
            state["message"] = "Starter data installed. Background updates are running."
            state["steps"][1].update(
                status="completed",
                message=update_state.get("message") or "Background updates queued",
                details={"update_job_id": update_state.get("job_id"), "update_status": update_state.get("status")},
            )
            state["data_status"] = build_data_status(db)
            save_json_setting(db, key=self.SETTING_KEY, payload=state, description=self.SETTING_DESCRIPTION)
            return state

    def _queue_download_latest(self) -> dict[str, Any]:
        with self._session_factory() as db:
            job_id = self._job_backend.submit_job(
                "desktop_setup",
                self._run_download_latest,
                message="Downloading initial desktop data",
                total=2,
            )
            state = default_setup_state()
            state.update(
                {
                    "status": "queued",
                    "mode": "download_latest",
                    "job_id": job_id,
                    "message": "Desktop setup queued",
                    "started_at": utc_now_iso(),
                    "steps": [
                        {"name": "install_starter", "label": "Install starter data", "status": "pending", "message": None, "details": None},
                        {"name": "download_core", "label": "Download core market data", "status": "pending", "message": None, "details": None},
                    ],
                }
            )
            state["data_status"] = build_data_status(db)
            save_json_setting(db, key=self.SETTING_KEY, payload=state, description=self.SETTING_DESCRIPTION)
            return state

    def _run_download_latest(self, job_id: str) -> dict[str, Any]:
        with self._session_factory() as db:
            state = load_json_setting(db, key=self.SETTING_KEY, default=default_setup_state())
            state["status"] = "running"
            state["job_id"] = job_id
            state["message"] = "Installing starter data"
            state["started_at"] = state.get("started_at") or utc_now_iso()
            state["steps"][0]["status"] = "running"
            save_json_setting(db, key=self.SETTING_KEY, payload=state, description=self.SETTING_DESCRIPTION)

            starter_details = self._install_starter_data(db)
            state["steps"][0].update(status="completed", message=starter_details["message"], details=starter_details)
            state["steps"][1].update(status="running", message="Downloading core market data")
            state["current"] = 1
            state["percent"] = 50.0
            state["message"] = "Downloading core market data"
            save_json_setting(db, key=self.SETTING_KEY, payload=state, description=self.SETTING_DESCRIPTION)

        update_state = self._update_service.run_update_now(scope="core", triggered_by="setup")

        with self._session_factory() as db:
            state = load_json_setting(db, key=self.SETTING_KEY, default=default_setup_state())
            if update_state["status"] == "completed":
                state["status"] = "completed"
                state["message"] = "Desktop setup completed with live core market data."
                state["starter_baseline_active"] = False
                state["error"] = None
            else:
                state["status"] = "failed"
                state["message"] = update_state.get("message") or "Desktop setup failed"
                state["error"] = update_state.get("error")
            state["steps"][1].update(
                status="completed" if update_state["status"] == "completed" else "failed",
                message=update_state.get("message"),
                details={"update_state": update_state},
            )
            state["current"] = 2
            state["total"] = 2
            state["percent"] = 100.0
            state["completed_at"] = utc_now_iso()
            state["app_ready"] = local_data_present(db)
            state["data_status"] = build_data_status(db)
            save_json_setting(db, key=self.SETTING_KEY, payload=state, description=self.SETTING_DESCRIPTION)
            return state

    def _install_starter_data(self, db: Session) -> dict[str, Any]:
        if local_data_present(db):
            return {
                "status": "completed",
                "message": "Local desktop data already exists",
                "source": "existing",
            }

        starter_db = Path(settings.desktop_starter_db_path)
        if starter_db.exists():
            self._copy_starter_db(starter_db)
            return {
                "status": "completed",
                "message": f"Installed starter snapshot from {starter_db.name}",
                "source": "starter_db",
            }

        universe_seed = Path(settings.desktop_bootstrap_seed_path)
        if universe_seed.exists():
            csv_content = universe_seed.read_text(encoding="utf-8")
            stock_universe_service.populate_from_csv(db, csv_content)

        industry_seed = Path(settings.desktop_bootstrap_industry_seed_path)
        if industry_seed.exists() and db.query(IBDIndustryGroup).count() == 0:
            from app.services.ibd_industry_service import IBDIndustryService

            IBDIndustryService.load_from_csv(db, str(industry_seed))

        manifest_path = Path(settings.desktop_starter_manifest_path)
        imported = {"prices": 0, "fundamentals": 0, "groups": 0, "breadth": 0}
        if manifest_path.exists():
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            imported["prices"] = self._import_price_seed(db, payload.get("prices", []))
            imported["fundamentals"] = self._import_fundamental_seed(db, payload.get("fundamentals", []))
            imported["breadth"] = self._import_breadth_seed(db, payload.get("breadth"))
            imported["groups"] = self._import_group_seed(db, payload.get("groups", []))

        from app.services.ui_snapshot_service import safe_publish_all_bootstraps

        safe_publish_all_bootstraps()
        return {
            "status": "completed",
            "message": "Installed starter data from bundled seed files",
            "source": "seed_files",
            **imported,
        }

    def _copy_starter_db(self, starter_db: Path) -> None:
        from app.config import settings as runtime_settings

        target_path = Path(runtime_settings.database_url.replace("sqlite:///", ""))
        target_path.parent.mkdir(parents=True, exist_ok=True)
        engine.dispose()
        with sqlite3.connect(starter_db) as source, sqlite3.connect(target_path) as destination:
            source.backup(destination)

    @staticmethod
    def _import_price_seed(db: Session, entries: list[dict[str, Any]]) -> int:
        imported = 0
        for entry in entries:
            symbol = entry["symbol"]
            for bar in entry.get("bars", []):
                bar_date = date.fromisoformat(bar["date"])
                existing = db.query(StockPrice).filter(StockPrice.symbol == symbol, StockPrice.date == bar_date).first()
                if existing is None:
                    existing = StockPrice(symbol=symbol, date=bar_date)
                    db.add(existing)
                existing.open = bar["open"]
                existing.high = bar["high"]
                existing.low = bar["low"]
                existing.close = bar["close"]
                existing.adj_close = bar.get("adj_close", bar["close"])
                existing.volume = bar["volume"]
                imported += 1
        db.commit()
        return imported

    @staticmethod
    def _import_fundamental_seed(db: Session, entries: list[dict[str, Any]]) -> int:
        imported = 0
        for entry in entries:
            symbol = entry["symbol"]
            record = db.query(StockFundamental).filter(StockFundamental.symbol == symbol).first()
            if record is None:
                record = StockFundamental(symbol=symbol)
                db.add(record)
            for key, value in entry.items():
                if key == "symbol" or not hasattr(record, key):
                    continue
                setattr(record, key, value)
            imported += 1
        db.commit()
        return imported

    @staticmethod
    def _import_breadth_seed(db: Session, entry: dict[str, Any] | None) -> int:
        if not entry:
            return 0
        breadth_date = date.fromisoformat(entry["date"])
        record = db.query(MarketBreadth).filter(MarketBreadth.date == breadth_date).first()
        if record is None:
            record = MarketBreadth(date=breadth_date)
            db.add(record)
        for key, value in entry.items():
            if key == "date" or not hasattr(record, key):
                continue
            setattr(record, key, value)
        db.commit()
        return 1

    @staticmethod
    def _import_group_seed(db: Session, entries: list[dict[str, Any]]) -> int:
        imported = 0
        for entry in entries:
            rank_date = date.fromisoformat(entry["date"])
            record = db.query(IBDGroupRank).filter(
                IBDGroupRank.industry_group == entry["industry_group"],
                IBDGroupRank.date == rank_date,
            ).first()
            if record is None:
                record = IBDGroupRank(industry_group=entry["industry_group"], date=rank_date, rank=entry["rank"], avg_rs_rating=entry["avg_rs_rating"])
                db.add(record)
            for key, value in entry.items():
                if key in {"industry_group", "date"} or not hasattr(record, key):
                    continue
                setattr(record, key, value)
            imported += 1
        db.commit()
        return imported

    def _merge_snapshot(self, state: dict[str, Any]) -> dict[str, Any]:
        job_id = state.get("job_id")
        if not job_id:
            return state
        snapshot = self._job_backend.get_status(job_id)
        if snapshot is None:
            if state.get("status") in {"queued", "running"}:
                state["status"] = "failed"
                state["message"] = self.INTERRUPTED_MESSAGE
                state["error"] = self.INTERRUPTED_ERROR
                state["current_step"] = None
                for step in state.get("steps", []):
                    if step.get("status") in {"pending", "running"}:
                        step["status"] = "failed"
                        step["message"] = self.INTERRUPTED_STEP_MESSAGE
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
