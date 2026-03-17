from __future__ import annotations

import json
from pathlib import Path
import uuid

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models.app_settings import AppSetting
from app.models.industry import IBDGroupRank
from app.models.market_breadth import MarketBreadth
from app.models.stock_universe import StockUniverse
from app.services.desktop_bootstrap_service import DesktopBootstrapService
from app.services.job_backend import JobSnapshot


class DeferredLocalJobBackend:
    def __init__(self) -> None:
        self._jobs: dict[str, dict] = {}
        self._snapshots: dict[str, JobSnapshot] = {}

    def submit_job(self, job_type, runner, *, message, total=None):
        job_id = str(uuid.uuid4())
        self._jobs[job_id] = {"runner": runner}
        self._snapshots[job_id] = JobSnapshot(
            job_id=job_id,
            job_type=job_type,
            status="queued",
            message=message,
            total=total,
        )
        return job_id

    def update(self, job_id, *, status=None, message=None, current=None, total=None, percent=None, result=None, error=None):
        snapshot = self._snapshots[job_id]
        if status is not None:
            snapshot.status = status
        if message is not None:
            snapshot.message = message
        if current is not None:
            snapshot.current = current
        if total is not None:
            snapshot.total = total
        if percent is not None:
            snapshot.percent = percent
        if result is not None:
            snapshot.result = result
        if error is not None:
            snapshot.error = error

    def get_status(self, job_id):
        snapshot = self._snapshots.get(job_id)
        if snapshot is None:
            return None
        return JobSnapshot(**snapshot.to_dict())

    def run_all(self):
        for job_id, job in list(self._jobs.items()):
            result = job["runner"](job_id)
            snapshot = self._snapshots[job_id]
            snapshot.status = result.get("status", "completed")
            snapshot.message = result.get("message")
            snapshot.current = result.get("current")
            snapshot.total = result.get("total", snapshot.total)
            snapshot.percent = result.get("percent")
            snapshot.result = result
            snapshot.error = result.get("error")
            self._jobs.pop(job_id, None)


def _make_session_factory(tmp_path: Path):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'desktop-bootstrap.db'}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)


def test_desktop_bootstrap_service_runs_seeded_bootstrap(tmp_path, monkeypatch):
    from app.services import desktop_bootstrap_service as module

    session_factory = _make_session_factory(tmp_path)
    job_backend = DeferredLocalJobBackend()

    universe_seed = tmp_path / "universe_seed.csv"
    universe_seed.write_text(
        "symbol,name,exchange,sector,industry,market_cap\n"
        "AAPL,Apple Inc,NASDAQ,Technology,Consumer Electronics,2900B\n"
        "MSFT,Microsoft Corp,NASDAQ,Technology,Software - Infrastructure,3100B\n"
        "NVDA,NVIDIA Corp,NASDAQ,Technology,Semiconductors,2200B\n",
        encoding="utf-8",
    )
    industry_seed = tmp_path / "ibd_industry_seed.csv"
    industry_seed.write_text(
        "AAPL,Consumer Electronics\nMSFT,Enterprise Software\nNVDA,Semiconductor Fabless\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(module.settings, "desktop_mode", True)
    monkeypatch.setattr(module.settings, "desktop_bootstrap_seed_path", str(universe_seed))
    monkeypatch.setattr(module.settings, "desktop_bootstrap_industry_seed_path", str(industry_seed))
    monkeypatch.setattr(module.settings, "desktop_bootstrap_refresh_universe", False)
    monkeypatch.setattr(module.settings, "desktop_bootstrap_fundamentals_limit", 0)

    class FakePriceCache:
        def get_many(self, symbols, period="2y"):
            frame = pd.DataFrame(
                {"Close": [100.0, 101.0, 102.0]},
                index=pd.date_range("2025-01-01", periods=3),
            )
            return {symbol: frame for symbol in symbols}

    class FakeBreadthCalculator:
        def __init__(self, db):
            self.db = db

        def calculate_daily_breadth(self, calculation_date=None):
            return {
                "stocks_up_4pct": 1,
                "stocks_down_4pct": 0,
                "ratio_5day": 1.5,
                "ratio_10day": 2.5,
                "stocks_up_25pct_quarter": 1,
                "stocks_down_25pct_quarter": 0,
                "stocks_up_25pct_month": 1,
                "stocks_down_25pct_month": 0,
                "stocks_up_50pct_month": 0,
                "stocks_down_50pct_month": 0,
                "stocks_up_13pct_34days": 1,
                "stocks_down_13pct_34days": 0,
                "total_stocks_scanned": 3,
            }

    class FakeFundamentalsCache:
        def get_fundamentals(self, symbol, force_refresh=True):
            return {"symbol": symbol}

    class FakeGroupRankService:
        def calculate_group_rankings(self, db, calc_date):
            db.add(
                IBDGroupRank(
                    industry_group="Enterprise Software",
                    date=calc_date,
                    rank=1,
                    avg_rs_rating=95.0,
                    num_stocks=3,
                    num_stocks_rs_above_80=2,
                    top_symbol="MSFT",
                    top_rs_rating=99.0,
                )
            )
            db.commit()
            return [{"industry_group": "Enterprise Software", "rank": 1}]

    class FakeGroupRankServiceFacade:
        @staticmethod
        def get_instance():
            return FakeGroupRankService()

    monkeypatch.setattr(module.PriceCacheService, "get_instance", staticmethod(lambda: FakePriceCache()))
    monkeypatch.setattr(module.FundamentalsCacheService, "get_instance", staticmethod(lambda: FakeFundamentalsCache()))
    monkeypatch.setattr(module, "BreadthCalculatorService", FakeBreadthCalculator)
    monkeypatch.setattr(module, "IBDGroupRankService", FakeGroupRankServiceFacade)
    monkeypatch.setattr(module, "get_eastern_now", lambda: pd.Timestamp("2025-01-10T12:00:00", tz="US/Eastern").to_pydatetime())
    monkeypatch.setattr(module, "get_last_trading_day", lambda d: d)

    service = DesktopBootstrapService(
        session_factory=session_factory,
        job_backend=job_backend,
    )

    queued = service.start_bootstrap()
    assert queued["status"] == "queued"

    job_backend.run_all()
    completed = service.get_status()

    assert completed["status"] == "completed"
    assert completed["warnings"] == []
    assert all(step["status"] == "completed" for step in completed["steps"])

    with session_factory() as db:
        assert db.query(StockUniverse).count() == 3
        assert db.query(MarketBreadth).count() == 1
        assert db.query(IBDGroupRank).count() == 1
        persisted = db.query(AppSetting).filter(AppSetting.key == DesktopBootstrapService.SETTING_KEY).one()
        assert json.loads(persisted.value)["status"] == "completed"


def test_desktop_bootstrap_service_fails_when_seed_missing(tmp_path, monkeypatch):
    from app.services import desktop_bootstrap_service as module

    session_factory = _make_session_factory(tmp_path)
    job_backend = DeferredLocalJobBackend()

    monkeypatch.setattr(module.settings, "desktop_mode", True)
    monkeypatch.setattr(module.settings, "desktop_bootstrap_seed_path", str(tmp_path / "missing_seed.csv"))
    monkeypatch.setattr(module.settings, "desktop_bootstrap_industry_seed_path", str(tmp_path / "missing_industry.csv"))
    monkeypatch.setattr(module.settings, "desktop_bootstrap_refresh_universe", False)
    monkeypatch.setattr(module.settings, "desktop_bootstrap_fundamentals_limit", 0)

    service = DesktopBootstrapService(
        session_factory=session_factory,
        job_backend=job_backend,
    )

    service.start_bootstrap()
    job_backend.run_all()
    failed = service.get_status()

    assert failed["status"] == "failed"
    assert "Universe seed CSV not found" in failed["error"]
