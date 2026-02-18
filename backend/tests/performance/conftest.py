"""Fixtures for query performance tests.

Seeds an in-memory SQLite database (via StaticPool) with 500 rows in both
the feature store and legacy scan_results tables.  Module-scoped engine
keeps data alive across function-scoped sessions.
"""

from __future__ import annotations

from datetime import date

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base
from app.domain.feature_store.models import RATING_TO_INT
from app.infra.db.models.feature_store import FeatureRun, StockFeatureDaily
from app.infra.db.repositories.feature_store_repo import SqlFeatureStoreRepository
from app.infra.db.repositories.scan_result_repo import (
    SqlScanResultRepository,
    _map_orchestrator_result,
)
from app.models.scan_result import Scan, ScanResult  # noqa: F401 — register models
from app.models.stock_universe import StockUniverse

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_ROWS = 500
PERF_AS_OF_DATE = date(2026, 2, 15)
PERF_SCAN_ID = "perf-test-00000000-0000-0000-0000"
PERF_FEATURE_RUN_ID = 1

SECTORS = ["Technology", "Healthcare", "Financials", "Consumer Discretionary", "Industrials"]
RATINGS = ["Strong Buy", "Buy", "Watch", "Pass"]
INDUSTRIES = [
    "Software", "Semiconductors", "Pharmaceuticals", "Banks",
    "Retail", "Aerospace", "Managed Health Care", "Payment Processing",
    "E-Commerce", "Biotechnology",
]


def _build_perf_result(i: int) -> dict:
    """Build a deterministic orchestrator output dict for row *i*."""
    score = (i * 100.0) / NUM_ROWS  # 0.0 .. 99.8
    sector = SECTORS[i % len(SECTORS)]
    industry = INDUSTRIES[i % len(INDUSTRIES)]
    rating_str = RATINGS[i % len(RATINGS)]

    return {
        "composite_score": score,
        "rating": rating_str,
        "current_price": 50.0 + (i % 200),
        "minervini_score": score * 0.9,
        "canslim_score": score * 0.8 if i % 3 == 0 else None,
        "ipo_score": None,
        "custom_score": None,
        "volume_breakthrough_score": None,
        "rs_rating": 20.0 + (i % 80),
        "rs_rating_1m": 15.0 + (i % 70),
        "rs_rating_3m": 18.0 + (i % 75),
        "rs_rating_12m": 12.0 + (i % 65),
        "stage": (i % 4) + 1,
        "stage_name": f"Stage {(i % 4) + 1}",
        "avg_dollar_volume": 1_000_000 * ((i % 50) + 1),
        "market_cap": 1_000_000_000 * ((i % 100) + 1),
        "ma_alignment": i % 2 == 0,
        "vcp_detected": False,
        "vcp_score": None,
        "vcp_pivot": None,
        "vcp_ready_for_breakout": None,
        "vcp_contraction_ratio": None,
        "vcp_atr_score": None,
        "passes_template": i % 3 == 0,
        "adr_percent": 1.0 + (i % 10) * 0.5,
        "eps_growth_qq": -10.0 + (i % 60),
        "sales_growth_qq": -5.0 + (i % 40),
        "eps_growth_yy": 5.0 + (i % 50),
        "sales_growth_yy": 3.0 + (i % 35),
        "peg_ratio": 0.5 + (i % 20) * 0.3,
        "eps_rating": 30 + (i % 70),
        "ibd_industry_group": f"IBD-Group-{i % 25}",
        "ibd_group_rank": (i % 50) + 1,
        "gics_sector": sector,
        "gics_industry": industry,
        "rs_sparkline_data": [50 + j + (i % 10) for j in range(5)],
        "rs_trend": 1 if i % 2 == 0 else -1,
        "price_sparkline_data": [100 + j + (i % 20) for j in range(5)],
        "price_change_1d": -2.0 + (i % 40) * 0.1,
        "price_trend": 1 if i % 3 == 0 else 0,
        "ipo_date": "2015-06-01",
        "beta": 0.5 + (i % 15) * 0.1,
        "beta_adj_rs": 20.0 + (i % 60),
        "beta_adj_rs_1m": 18.0 + (i % 55),
        "beta_adj_rs_3m": 19.0 + (i % 58),
        "beta_adj_rs_12m": 16.0 + (i % 50),
        "perf_week": -5.0 + (i % 20) * 0.5,
        "perf_month": -8.0 + (i % 30) * 0.6,
        "perf_3m": -10.0 + (i % 40) * 0.8,
        "perf_6m": -12.0 + (i % 50) * 1.0,
        "gap_percent": (i % 10) * 0.2,
        "volume_surge": 0.5 + (i % 8) * 0.3,
        "ema_10_distance": -3.0 + (i % 15) * 0.4,
        "ema_20_distance": -5.0 + (i % 20) * 0.5,
        "ema_50_distance": -8.0 + (i % 30) * 0.6,
        "from_52w_high_pct": -(i % 30),
        "above_52w_low_pct": 10.0 + (i % 80),
        "screeners_run": ["minervini", "canslim"] if i % 3 == 0 else ["minervini"],
        "composite_method": "weighted_average",
        "screeners_passed": 2 if i % 3 == 0 else 1,
        "screeners_total": 2 if i % 3 == 0 else 1,
    }


# ---------------------------------------------------------------------------
# Engine: module-scoped with StaticPool
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def perf_engine():
    """Module-scoped in-memory SQLite engine with StaticPool.

    StaticPool ensures a single connection is reused, so seeded data
    persists across function-scoped sessions.
    """
    eng = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @event.listens_for(eng, "connect")
    def _set_fk_pragma(dbapi_conn, _):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(eng)

    # ── Seed data ─────────────────────────────────────────────────────
    factory = sessionmaker(bind=eng)
    session = factory()

    # 1. Universe rows
    for i in range(NUM_ROWS):
        session.add(
            StockUniverse(
                symbol=f"PERF{i:03d}",
                name=f"PerfCo {i}",
                exchange="NASDAQ",
            )
        )

    # 2. Feature store: FeatureRun + StockFeatureDaily rows
    session.add(
        FeatureRun(
            id=PERF_FEATURE_RUN_ID,
            as_of_date=PERF_AS_OF_DATE,
            run_type="daily_snapshot",
            status="published",
        )
    )
    session.flush()

    for i in range(NUM_ROWS):
        result_dict = _build_perf_result(i)
        session.add(
            StockFeatureDaily(
                run_id=PERF_FEATURE_RUN_ID,
                symbol=f"PERF{i:03d}",
                as_of_date=PERF_AS_OF_DATE,
                composite_score=result_dict["composite_score"],
                overall_rating=RATING_TO_INT.get(result_dict["rating"], 2),
                passes_count=result_dict.get("screeners_passed", 0),
                details_json=result_dict,
            )
        )

    # 3. Legacy: Scan + ScanResult rows
    session.add(
        Scan(
            scan_id=PERF_SCAN_ID,
            status="completed",
            screener_types=["minervini", "canslim"],
            composite_method="weighted_average",
        )
    )
    session.flush()

    legacy_rows = [
        _map_orchestrator_result(PERF_SCAN_ID, f"PERF{i:03d}", _build_perf_result(i))
        for i in range(NUM_ROWS)
    ]
    session.bulk_save_objects([ScanResult(**r) for r in legacy_rows])

    session.commit()
    session.close()

    yield eng
    eng.dispose()


# ---------------------------------------------------------------------------
# Session: function-scoped (reads from the shared StaticPool engine)
# ---------------------------------------------------------------------------


@pytest.fixture
def perf_session(perf_engine):
    """Function-scoped session reading from the module-scoped engine."""
    factory = sessionmaker(bind=perf_engine)
    sess = factory()
    yield sess
    sess.close()


# ---------------------------------------------------------------------------
# Seed integrity check: fail fast if seeding was wrong
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True, scope="module")
def _verify_seed_integrity(perf_engine):
    """Fail fast if seeded row counts are wrong — prevents silent false greens."""
    factory = sessionmaker(bind=perf_engine)
    session = factory()
    try:
        fs_count = session.query(StockFeatureDaily).count()
        legacy_count = session.query(ScanResult).count()
        assert fs_count == NUM_ROWS, f"Feature store has {fs_count} rows, expected {NUM_ROWS}"
        assert legacy_count == NUM_ROWS, f"Legacy has {legacy_count} rows, expected {NUM_ROWS}"
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Repository fixtures: function-scoped
# ---------------------------------------------------------------------------


@pytest.fixture
def fs_repo(perf_session) -> SqlFeatureStoreRepository:
    return SqlFeatureStoreRepository(perf_session)


@pytest.fixture
def legacy_repo(perf_session) -> SqlScanResultRepository:
    return SqlScanResultRepository(perf_session)
