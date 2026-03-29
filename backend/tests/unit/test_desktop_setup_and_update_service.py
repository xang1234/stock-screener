from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models.industry import IBDGroupRank, IBDIndustryGroup
from app.models.market_breadth import MarketBreadth
from app.models.stock import StockFundamental
from app.models.stock_universe import StockUniverse
from app.services.desktop_runtime_state import (
    SETUP_STATE_KEY,
    UPDATE_STATE_KEY,
    default_setup_state,
    default_update_state,
    load_json_setting,
    local_data_present,
    save_json_setting,
)
from app.services.desktop_setup_service import DesktopSetupService
from app.services.desktop_update_service import DesktopUpdateService


class _NoopJobBackend:
    def submit_job(self, *_args, **_kwargs):
        return "job-1"

    def update(self, *_args, **_kwargs):
        return None

    def get_status(self, _job_id):
        return None


class _FakeUpdateService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, bool]] = []

    def start_update(self, *, scope: str = "manual", triggered_by: str = "manual", force: bool = False):
        self.calls.append((scope, triggered_by, force))
        return {
            "status": "queued",
            "job_id": "update-job",
            "message": "Background updates queued",
        }


def _make_session_factory(tmp_path: Path):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'desktop-runtime.db'}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)


def _write_seed_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    universe_seed = tmp_path / "universe_seed.csv"
    universe_seed.write_text(
        "symbol,name,exchange,sector,industry,market_cap\n"
        "MSFT,Microsoft Corp,NASDAQ,Technology,Software - Infrastructure,3100B\n"
        "NVDA,NVIDIA Corp,NASDAQ,Technology,Semiconductors,2200B\n"
        "META,Meta Platforms Inc,NASDAQ,Communication Services,Internet Content & Information,1300B\n",
        encoding="utf-8",
    )
    industry_seed = tmp_path / "ibd_industry_seed.csv"
    industry_seed.write_text(
        "MSFT,Enterprise Software\nNVDA,Semiconductor Fabless\nMETA,Internet Content\n",
        encoding="utf-8",
    )
    starter_manifest = tmp_path / "starter_manifest.json"
    starter_manifest.write_text(
        json.dumps(
            {
                "prices": [
                    {
                        "symbol": "MSFT",
                        "bars": [
                            {
                                "date": "2025-01-10",
                                "open": 430.0,
                                "high": 435.0,
                                "low": 428.0,
                                "close": 434.0,
                                "adj_close": 434.0,
                                "volume": 25000000,
                            }
                        ],
                    }
                ],
                "fundamentals": [
                    {
                        "symbol": "MSFT",
                        "market_cap": 3100000000000,
                        "avg_volume": 25000000,
                        "sector": "Technology",
                        "industry": "Software - Infrastructure",
                    }
                ],
                "breadth": {
                    "date": "2025-01-10",
                    "stocks_up_4pct": 10,
                    "stocks_down_4pct": 2,
                    "ratio_5day": 1.8,
                    "ratio_10day": 1.5,
                    "stocks_up_25pct_quarter": 8,
                    "stocks_down_25pct_quarter": 1,
                    "stocks_up_25pct_month": 7,
                    "stocks_down_25pct_month": 1,
                    "stocks_up_50pct_month": 2,
                    "stocks_down_50pct_month": 0,
                    "stocks_up_13pct_34days": 6,
                    "stocks_down_13pct_34days": 1,
                    "total_stocks_scanned": 3,
                },
                "groups": [
                    {
                        "industry_group": "Enterprise Software",
                        "date": "2025-01-10",
                        "rank": 1,
                        "avg_rs_rating": 94.0,
                        "num_stocks": 1,
                        "num_stocks_rs_above_80": 1,
                        "top_symbol": "MSFT",
                        "top_rs_rating": 98.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return universe_seed, industry_seed, starter_manifest


def test_desktop_setup_service_quick_start_installs_starter_data(tmp_path, monkeypatch):
    from app.services import desktop_setup_service as module
    from app.services import ui_snapshot_service as snapshot_module

    session_factory = _make_session_factory(tmp_path)
    update_service = _FakeUpdateService()
    universe_seed, industry_seed, starter_manifest = _write_seed_files(tmp_path)

    monkeypatch.setattr(module.settings, "desktop_mode", True)
    monkeypatch.setattr(module.settings, "desktop_bootstrap_seed_path", str(universe_seed))
    monkeypatch.setattr(module.settings, "desktop_bootstrap_industry_seed_path", str(industry_seed))
    monkeypatch.setattr(module.settings, "desktop_starter_manifest_path", str(starter_manifest))
    monkeypatch.setattr(module.settings, "desktop_starter_db_path", str(tmp_path / "missing.sqlite3"))
    monkeypatch.setattr(snapshot_module, "safe_publish_all_bootstraps", lambda: None)

    service = DesktopSetupService(
        session_factory=session_factory,
        job_backend=_NoopJobBackend(),
        update_service=update_service,
    )

    completed = service.start_setup(mode="quick_start")

    assert completed["status"] == "completed"
    assert completed["app_ready"] is True
    assert update_service.calls == [("core", "setup", False)]

    with session_factory() as db:
        assert db.query(StockUniverse).count() == 3
        assert db.query(IBDIndustryGroup).count() == 3
        assert db.query(StockFundamental).count() == 1
        assert db.query(MarketBreadth).count() == 1
        assert db.query(IBDGroupRank).count() == 1


def test_desktop_update_service_runs_daily_refresh_without_celery_or_redis(tmp_path, monkeypatch):
    from app.services import desktop_update_service as module
    from app.services import ui_snapshot_service as snapshot_module

    session_factory = _make_session_factory(tmp_path)

    with session_factory() as db:
        db.add_all(
            [
                StockUniverse(symbol="MSFT", name="Microsoft", exchange="NASDAQ", sector="Technology", industry="Software", market_cap=1, is_active=True, source="seed"),
                StockUniverse(symbol="NVDA", name="NVIDIA", exchange="NASDAQ", sector="Technology", industry="Semiconductors", market_cap=1, is_active=True, source="seed"),
            ]
        )
        db.add_all(
            [
                IBDIndustryGroup(symbol="MSFT", industry_group="Enterprise Software"),
                IBDIndustryGroup(symbol="NVDA", industry_group="Semiconductor Fabless"),
            ]
        )
        db.commit()

    class _FakePriceCache:
        def get_many(self, symbols, period="2y"):
            frame = pd.DataFrame({"Close": [100.0, 101.0]}, index=pd.date_range("2025-01-09", periods=2))
            return {symbol: frame for symbol in symbols}

    class _FakeBreadthCalculator:
        def __init__(self, db):
            self.db = db

        def calculate_daily_breadth(self, calculation_date=None):
            return {
                "stocks_up_4pct": 2,
                "stocks_down_4pct": 0,
                "ratio_5day": 2.0,
                "ratio_10day": 1.7,
                "stocks_up_25pct_quarter": 2,
                "stocks_down_25pct_quarter": 0,
                "stocks_up_25pct_month": 2,
                "stocks_down_25pct_month": 0,
                "stocks_up_50pct_month": 1,
                "stocks_down_50pct_month": 0,
                "stocks_up_13pct_34days": 2,
                "stocks_down_13pct_34days": 0,
                "total_stocks_scanned": 2,
            }

    class _FakeGroupRankService:
        def calculate_group_rankings(self, db, calc_date):
            db.add(
                IBDGroupRank(
                    industry_group="Enterprise Software",
                    date=calc_date,
                    rank=1,
                    avg_rs_rating=95.0,
                    num_stocks=1,
                    num_stocks_rs_above_80=1,
                    top_symbol="MSFT",
                    top_rs_rating=98.0,
                )
            )
            db.commit()
            return [{"industry_group": "Enterprise Software", "rank": 1}]

    class _FakeGroupRankServiceFacade:
        @staticmethod
        def get_instance():
            return _FakeGroupRankService()

    monkeypatch.setattr(module.settings, "desktop_mode", True)
    monkeypatch.setattr(module.settings, "desktop_data_dir", str(tmp_path / "desktop-data"))
    monkeypatch.setattr(module.settings, "desktop_price_refresh_batch_size", 25)
    monkeypatch.setattr(module.PriceCacheService, "get_instance", staticmethod(lambda: _FakePriceCache()))
    monkeypatch.setattr(module, "BreadthCalculatorService", _FakeBreadthCalculator)
    monkeypatch.setattr(module, "IBDGroupRankService", _FakeGroupRankServiceFacade)
    monkeypatch.setattr(module, "get_eastern_now", lambda: pd.Timestamp("2025-01-10T18:00:00", tz="US/Eastern").to_pydatetime())
    monkeypatch.setattr(module, "get_last_trading_day", lambda d: d)
    monkeypatch.setattr(snapshot_module, "safe_publish_breadth_bootstrap", lambda: None)
    monkeypatch.setattr(snapshot_module, "safe_publish_groups_bootstrap", lambda: None)

    with session_factory() as db:
        setup_state = default_setup_state()
        setup_state.update(
            {
                "status": "completed",
                "completed_at": "2025-01-10T10:00:00+00:00",
                "starter_baseline_active": True,
                "app_ready": True,
            }
        )
        save_json_setting(db, key=SETUP_STATE_KEY, payload=setup_state, description="Desktop runtime setup state")

    service = DesktopUpdateService(
        session_factory=session_factory,
        job_backend=_NoopJobBackend(),
    )

    completed = service.run_update_now(scope="daily", triggered_by="test")

    assert completed["status"] == "completed"
    assert [step["name"] for step in completed["steps"]] == [
        "refresh_prices",
        "calculate_breadth",
        "calculate_groups",
    ]

    with session_factory() as db:
        assert db.query(MarketBreadth).count() == 1
        assert db.query(IBDGroupRank).count() == 1
        persisted_setup = load_json_setting(db, key=SETUP_STATE_KEY, default=default_setup_state())
        assert persisted_setup["starter_baseline_active"] is False


def test_desktop_setup_requires_completed_setup_state_before_marking_data_ready(tmp_path, monkeypatch):
    from app.services import desktop_setup_service as module

    session_factory = _make_session_factory(tmp_path)

    monkeypatch.setattr(module.settings, "desktop_mode", True)

    with session_factory() as db:
        db.add(
            StockUniverse(
                symbol="MSFT",
                name="Microsoft",
                exchange="NASDAQ",
                sector="Technology",
                industry="Software",
                market_cap=1,
                is_active=True,
                source="seed",
            )
        )
        db.commit()
        assert local_data_present(db) is False

    service = DesktopSetupService(
        session_factory=session_factory,
        job_backend=_NoopJobBackend(),
        update_service=_FakeUpdateService(),
    )

    status = service.get_status()

    assert status["app_ready"] is False
    assert status["data_status"]["local_data_present"] is False


def test_desktop_update_service_marks_missing_running_job_as_interrupted(tmp_path):
    session_factory = _make_session_factory(tmp_path)

    with session_factory() as db:
        state = default_update_state()
        state.update(
            {
                "status": "running",
                "scope": "daily",
                "triggered_by": "scheduler",
                "job_id": "missing-job",
                "current_step": "calculate_breadth",
                "total": 3,
                "steps": [
                    {"name": "refresh_prices", "label": "Refresh price data", "status": "completed", "message": "done", "details": None},
                    {"name": "calculate_breadth", "label": "Update market breadth", "status": "running", "message": "working", "details": None},
                    {"name": "calculate_groups", "label": "Update group rankings", "status": "pending", "message": None, "details": None},
                ],
            }
        )
        save_json_setting(db, key=UPDATE_STATE_KEY, payload=state, description="Desktop runtime update state")

    service = DesktopUpdateService(
        session_factory=session_factory,
        job_backend=_NoopJobBackend(),
    )

    status = service.get_status()

    assert status["status"] == "failed"
    assert status["error"] == service.INTERRUPTED_ERROR
    assert status["current_step"] is None
    assert status["completed_at"] is not None
    assert [step["status"] for step in status["steps"]] == ["completed", "failed", "failed"]

    with session_factory() as db:
        persisted = load_json_setting(db, key=UPDATE_STATE_KEY, default=default_update_state())
        assert persisted["status"] == "failed"
