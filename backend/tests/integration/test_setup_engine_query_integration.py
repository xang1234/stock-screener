"""Integration tests for Setup Engine fields through the full query pipeline.

Proves SE fields survive: Scanner → Orchestrator promotion → Persistence
(JSON column) → Query builders (apply_filters / apply_sort_and_paginate)
→ Domain mappers (extended_fields).

Uses real repository .query() methods with QuerySpec/FilterSpec/SortSpec,
exercising the actual SQLAlchemy query builder code (not raw SQL).

Seed data flows through the real orchestrator path:
  _call_combine_results() → _map_orchestrator_result() → bulk_insert()
  _call_combine_results() → _map_orchestrator_to_feature_row() → ORM insert
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.database import Base
from app.domain.common.query import (
    BooleanFilter,
    CategoricalFilter,
    FilterSpec,
    QuerySpec,
    RangeFilter,
    SortOrder,
    SortSpec,
)
from app.domain.feature_store.models import RATING_TO_INT
from app.domain.scanning.models import ResultPage, ScanResultItemDomain
from app.infra.db.models.feature_store import FeatureRun, StockFeatureDaily
from app.infra.db.repositories.feature_store_repo import SqlFeatureStoreRepository
from app.infra.db.repositories.scan_result_repo import (
    SqlScanResultRepository,
    _map_orchestrator_result,
)
from app.models.scan_result import Scan, ScanResult
from app.models.stock_universe import StockUniverse
from app.scanners.base_screener import ScreenerResult, StockData
from app.scanners.scan_orchestrator import ScanOrchestrator
from app.use_cases.feature_store.build_daily_snapshot import (
    _map_orchestrator_to_feature_row,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCAN_ID = "se-g3-test-scan"
AS_OF = date(2026, 2, 20)

# Canonical SE payloads per symbol — scores chosen to catch lexicographic bugs.
# String "9.5" sorts differently than float 9.5 when compared to 50/82.5/100.
_SE_PAYLOADS: dict[str, dict[str, Any]] = {
    "AAPL": {
        "schema_version": "1.0",
        "timeframe": "daily",
        "setup_score": 82.5,
        "quality_score": 75.0,
        "readiness_score": 90.0,
        "setup_ready": True,
        "pattern_primary": "VCP",
        "pattern_confidence": 0.88,
        "pivot_price": 142.50,
        "pivot_type": "breakout",
        "pivot_date": "2026-02-15",
        "distance_to_pivot_pct": 2.3,
        "atr14_pct": 3.1,
        "atr14_pct_trend": -0.2,
        "bb_width_pct": 5.5,
        "bb_width_pctile_252": 22.0,
        "volume_vs_50d": 1.5,
        "rs": 1.12,
        "rs_line_new_high": True,
        "rs_vs_spy_65d": 8.5,
        "rs_vs_spy_trend_20d": 0.03,
        "stage": 2,
        "ma_alignment_score": 75.0,
        "rs_rating": 62.0,
        "candidates": [],
        "explain": {
            "passed_checks": ["check_a"],
            "failed_checks": [],
            "invalidation_flags": [],
            "key_levels": {"pivot": 142.50},
        },
    },
    "MSFT": {
        "schema_version": "1.0",
        "timeframe": "daily",
        "setup_score": 9.5,
        "quality_score": 30.0,
        "readiness_score": 40.0,
        "setup_ready": False,
        "pattern_primary": "Cup-with-Handle",
        "pattern_confidence": 0.55,
        "pivot_price": 450.00,
        "pivot_type": "handle",
        "pivot_date": "2026-02-10",
        "distance_to_pivot_pct": 8.1,
        "atr14_pct": 2.0,
        "atr14_pct_trend": 0.1,
        "bb_width_pct": 7.0,
        "bb_width_pctile_252": 45.0,
        "volume_vs_50d": 0.8,
        "rs": 0.95,
        "rs_line_new_high": False,
        "rs_vs_spy_65d": -2.0,
        "rs_vs_spy_trend_20d": -0.01,
        "stage": 2,
        "ma_alignment_score": 50.0,
        "rs_rating": 55.0,
        "candidates": [],
        "explain": {
            "passed_checks": [],
            "failed_checks": ["check_b"],
            "invalidation_flags": [],
            "key_levels": {},
        },
    },
    "GOOGL": {
        "schema_version": "1.0",
        "timeframe": "daily",
        "setup_score": 100.0,
        "quality_score": 95.0,
        "readiness_score": 95.0,
        "setup_ready": True,
        "pattern_primary": "High Tight Flag",
        "pattern_confidence": 0.95,
        "pivot_price": 165.00,
        "pivot_type": "flag",
        "pivot_date": "2026-02-18",
        "distance_to_pivot_pct": 0.5,
        "atr14_pct": 4.0,
        "atr14_pct_trend": 0.3,
        "bb_width_pct": 3.0,
        "bb_width_pctile_252": 10.0,
        "volume_vs_50d": 2.1,
        "rs": 1.35,
        "rs_line_new_high": True,
        "rs_vs_spy_65d": 15.0,
        "rs_vs_spy_trend_20d": 0.05,
        "stage": 2,
        "ma_alignment_score": 90.0,
        "rs_rating": 92.0,
        "candidates": [],
        "explain": {
            "passed_checks": ["check_a", "check_c"],
            "failed_checks": [],
            "invalidation_flags": [],
            "key_levels": {"pivot": 165.00},
        },
    },
    # TSLA: NO setup_engine — Minervini only
    "NVDA": {
        "schema_version": "1.0",
        "timeframe": "daily",
        "setup_score": 50.0,
        "quality_score": 60.0,
        "readiness_score": 70.0,
        "setup_ready": True,
        "pattern_primary": "VCP",
        "pattern_confidence": 0.72,
        "pivot_price": 900.00,
        "pivot_type": "breakout",
        "pivot_date": "2026-02-12",
        "distance_to_pivot_pct": 5.0,
        "atr14_pct": 3.5,
        "atr14_pct_trend": 0.0,
        "bb_width_pct": 6.0,
        "bb_width_pctile_252": 30.0,
        "volume_vs_50d": 1.1,
        "rs": 1.05,
        "rs_line_new_high": False,
        "rs_vs_spy_65d": 5.0,
        "rs_vs_spy_trend_20d": 0.02,
        "stage": 2,
        "ma_alignment_score": 65.0,
        "rs_rating": 70.0,
        "candidates": [],
        "explain": {
            "passed_checks": [],
            "failed_checks": [],
            "invalidation_flags": [],
            "key_levels": {},
        },
    },
}


# ---------------------------------------------------------------------------
# Shared helpers (adapted from test_setup_engine_persistence.py)
# ---------------------------------------------------------------------------


def _make_stub_stock_data(symbol: str) -> StockData:
    """Minimal StockData for _combine_results()."""
    dates = pd.date_range(end="2026-02-20", periods=10, freq="B")
    df = pd.DataFrame(
        {"Open": 100.0, "High": 105.0, "Low": 99.0, "Close": 102.0, "Volume": 1_000_000},
        index=dates,
    )
    return StockData(symbol=symbol, price_data=df, benchmark_data=df)


def _make_se_screener_result(payload: dict[str, Any]) -> ScreenerResult:
    """Build a ScreenerResult mimicking SetupEngineScanner output."""
    return ScreenerResult(
        score=payload.get("setup_score", 0) or 0,
        passes=bool(payload.get("setup_ready")),
        rating="Buy" if payload.get("setup_ready") else "Pass",
        breakdown={"setup_score": payload.get("setup_score", 0) or 0},
        details={"setup_engine": payload},
        screener_name="setup_engine",
    )


def _make_minervini_screener_result() -> ScreenerResult:
    """Build a minimal Minervini ScreenerResult."""
    return ScreenerResult(
        score=78.0,
        passes=True,
        rating="Buy",
        breakdown={"rs": 85, "stage": 90},
        details={
            "rs_rating": 85,
            "stage": 2,
            "stage_name": "Stage 2 - Uptrend",
        },
        screener_name="minervini",
    )


def _call_combine_results(
    screener_results: dict[str, ScreenerResult],
    stock_data: StockData,
) -> dict[str, Any]:
    """Call ScanOrchestrator._combine_results() without running a full scan."""
    orch = ScanOrchestrator.__new__(ScanOrchestrator)
    return orch._combine_results(
        symbol=stock_data.symbol,
        screener_results=screener_results,
        stock_data=stock_data,
        composite_score=80.0,
        overall_rating="Buy",
        composite_method="weighted_average",
    )


def _build_test_orchestrator_results() -> list[tuple[str, dict[str, Any]]]:
    """Build 5 test orchestrator result dicts through the real pipeline.

    Returns (symbol, result_dict) tuples ready for persistence mapping.
    """
    results: list[tuple[str, dict[str, Any]]] = []
    for symbol in ("AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"):
        sd = _make_stub_stock_data(symbol)
        se_payload = _SE_PAYLOADS.get(symbol)
        if se_payload is not None:
            screeners = {
                "minervini": _make_minervini_screener_result(),
                "setup_engine": _make_se_screener_result(se_payload),
            }
        else:
            # TSLA: Minervini only
            screeners = {"minervini": _make_minervini_screener_result()}
        result_dict = _call_combine_results(screeners, sd)
        results.append((symbol, result_dict))
    return results


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_UNIVERSE_ENTRIES = [
    ("AAPL", "Apple Inc"),
    ("MSFT", "Microsoft Corp"),
    ("GOOGL", "Alphabet Inc"),
    ("TSLA", "Tesla Inc"),
    ("NVDA", "NVIDIA Corp"),
]


def _make_session() -> Session:
    """Create a fresh in-memory SQLite session with full ORM schema."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return Session(engine)


def _seed_universe(session: Session) -> None:
    """Insert StockUniverse entries for company name lookup."""
    for sym, name in _UNIVERSE_ENTRIES:
        session.add(StockUniverse(symbol=sym, name=name, exchange="NASDAQ"))


@pytest.fixture
def seeded_scan_session():
    """Seed scan_results via the real orchestrator → mapper → ORM pipeline."""
    session = _make_session()
    scan = Scan(scan_id=SCAN_ID, status="completed", total_stocks=5, passed_stocks=4)
    session.add(scan)
    _seed_universe(session)

    orchestrator_results = _build_test_orchestrator_results()
    mapped_rows = [
        _map_orchestrator_result(SCAN_ID, symbol, result_dict)
        for symbol, result_dict in orchestrator_results
    ]
    repo = SqlScanResultRepository(session)
    repo.bulk_insert(mapped_rows)
    session.commit()
    yield session
    session.close()


@pytest.fixture
def seeded_feature_session():
    """Seed feature store via the real orchestrator → feature mapper → ORM pipeline."""
    session = _make_session()
    run = FeatureRun(
        id=1,
        as_of_date=AS_OF,
        run_type="daily_snapshot",
        status="published",
    )
    session.add(run)
    _seed_universe(session)

    orchestrator_results = _build_test_orchestrator_results()
    for symbol, result_dict in orchestrator_results:
        feature_row = _map_orchestrator_to_feature_row(symbol, AS_OF, result_dict)
        session.add(
            StockFeatureDaily(
                run_id=1,
                symbol=symbol,
                as_of_date=AS_OF,
                composite_score=feature_row.composite_score,
                overall_rating=feature_row.overall_rating,
                passes_count=feature_row.passes_count,
                details_json=feature_row.details,
            )
        )
    session.commit()
    yield session
    session.close()


# ---------------------------------------------------------------------------
# Class 1: ScanResult path — full stack through repository .query()
# ---------------------------------------------------------------------------


class TestScanResultSEQueryIntegration:
    """Exercises SqlScanResultRepository.query() with SE filter/sort/mapping."""

    # -- Filter tests -------------------------------------------------------

    def test_range_filter_se_setup_score_min_max(self, seeded_scan_session):
        """Range filter on se_setup_score min=40, max=90 → AAPL(82.5), NVDA(50)."""
        repo = SqlScanResultRepository(seeded_scan_session)
        spec = QuerySpec(
            filters=FilterSpec(
                range_filters=[
                    RangeFilter(field="se_setup_score", min_value=40, max_value=90)
                ]
            )
        )
        page = repo.query(SCAN_ID, spec)
        symbols = {item.symbol for item in page.items}
        assert symbols == {"AAPL", "NVDA"}

    def test_range_filter_se_readiness_score_min(self, seeded_scan_session):
        """Range filter on se_readiness_score min=80 → AAPL(90), GOOGL(95)."""
        repo = SqlScanResultRepository(seeded_scan_session)
        spec = QuerySpec(
            filters=FilterSpec(
                range_filters=[
                    RangeFilter(field="se_readiness_score", min_value=80)
                ]
            )
        )
        page = repo.query(SCAN_ID, spec)
        symbols = {item.symbol for item in page.items}
        assert symbols == {"AAPL", "GOOGL"}

    def test_boolean_filter_se_setup_ready_true(self, seeded_scan_session):
        """Boolean filter se_setup_ready=True → AAPL, GOOGL, NVDA."""
        repo = SqlScanResultRepository(seeded_scan_session)
        spec = QuerySpec(
            filters=FilterSpec(
                boolean_filters=[BooleanFilter(field="se_setup_ready", value=True)]
            )
        )
        page = repo.query(SCAN_ID, spec)
        symbols = {item.symbol for item in page.items}
        # MSFT has setup_ready=False, TSLA has no SE data (NULL)
        assert symbols == {"AAPL", "GOOGL", "NVDA"}

    def test_boolean_filter_se_setup_ready_false(self, seeded_scan_session):
        """Boolean filter se_setup_ready=False → MSFT only."""
        repo = SqlScanResultRepository(seeded_scan_session)
        spec = QuerySpec(
            filters=FilterSpec(
                boolean_filters=[BooleanFilter(field="se_setup_ready", value=False)]
            )
        )
        page = repo.query(SCAN_ID, spec)
        symbols = {item.symbol for item in page.items}
        assert symbols == {"MSFT"}

    def test_categorical_filter_se_pattern_primary(self, seeded_scan_session):
        """Categorical filter se_pattern_primary in ('VCP') → AAPL, NVDA."""
        repo = SqlScanResultRepository(seeded_scan_session)
        spec = QuerySpec(
            filters=FilterSpec(
                categorical_filters=[
                    CategoricalFilter(field="se_pattern_primary", values=("VCP",))
                ]
            )
        )
        page = repo.query(SCAN_ID, spec)
        symbols = {item.symbol for item in page.items}
        assert symbols == {"AAPL", "NVDA"}

    # -- Sort tests ---------------------------------------------------------

    def test_sort_se_setup_score_desc(self, seeded_scan_session):
        """Sort se_setup_score DESC → GOOGL(100), AAPL(82.5), NVDA(50), MSFT(9.5), TSLA(null)."""
        repo = SqlScanResultRepository(seeded_scan_session)
        spec = QuerySpec(sort=SortSpec(field="se_setup_score", order=SortOrder.DESC))
        page = repo.query(SCAN_ID, spec)
        symbols = [item.symbol for item in page.items]
        assert symbols == ["GOOGL", "AAPL", "NVDA", "MSFT", "TSLA"]

    def test_sort_se_setup_score_asc(self, seeded_scan_session):
        """Sort se_setup_score ASC → MSFT(9.5), NVDA(50), AAPL(82.5), GOOGL(100), TSLA(null last)."""
        repo = SqlScanResultRepository(seeded_scan_session)
        spec = QuerySpec(sort=SortSpec(field="se_setup_score", order=SortOrder.ASC))
        page = repo.query(SCAN_ID, spec)
        symbols = [item.symbol for item in page.items]
        # 9.5 < 50 < 82.5 < 100 — numeric sort, not string sort
        assert symbols == ["MSFT", "NVDA", "AAPL", "GOOGL", "TSLA"]

    def test_sort_se_pattern_confidence_desc(self, seeded_scan_session):
        """Sort se_pattern_confidence DESC → GOOGL(0.95), AAPL(0.88), NVDA(0.72), MSFT(0.55), TSLA(null)."""
        repo = SqlScanResultRepository(seeded_scan_session)
        spec = QuerySpec(
            sort=SortSpec(field="se_pattern_confidence", order=SortOrder.DESC)
        )
        page = repo.query(SCAN_ID, spec)
        symbols = [item.symbol for item in page.items]
        assert symbols == ["GOOGL", "AAPL", "NVDA", "MSFT", "TSLA"]

    # -- Combined filter + sort ---------------------------------------------

    def test_filter_setup_ready_true_sort_score_desc(self, seeded_scan_session):
        """Filter se_setup_ready=True + sort se_setup_score DESC → GOOGL, AAPL, NVDA."""
        repo = SqlScanResultRepository(seeded_scan_session)
        spec = QuerySpec(
            filters=FilterSpec(
                boolean_filters=[BooleanFilter(field="se_setup_ready", value=True)]
            ),
            sort=SortSpec(field="se_setup_score", order=SortOrder.DESC),
        )
        page = repo.query(SCAN_ID, spec)
        symbols = [item.symbol for item in page.items]
        assert symbols == ["GOOGL", "AAPL", "NVDA"]

    # -- Domain mapper tests ------------------------------------------------

    def test_aapl_extended_fields_se_populated(self, seeded_scan_session):
        """AAPL extended_fields has all expected se_* keys with correct values."""
        repo = SqlScanResultRepository(seeded_scan_session)
        page = repo.query(SCAN_ID, QuerySpec())
        aapl = next(i for i in page.items if i.symbol == "AAPL")
        ef = aapl.extended_fields

        assert ef["se_setup_score"] == pytest.approx(82.5)
        assert ef["se_pattern_primary"] == "VCP"
        assert ef["se_setup_ready"] is True
        assert ef["se_quality_score"] == pytest.approx(75.0)
        assert ef["se_readiness_score"] == pytest.approx(90.0)
        assert ef["se_pattern_confidence"] == pytest.approx(0.88)
        assert ef["se_pivot_price"] == pytest.approx(142.50)
        assert ef["se_pivot_type"] == "breakout"
        assert ef["se_pivot_date"] == "2026-02-15"
        assert ef["se_timeframe"] == "daily"
        assert ef["se_atr14_pct"] == pytest.approx(3.1)
        assert ef["se_explain"] is not None
        assert ef["se_candidates"] is not None

    def test_tsla_extended_fields_se_none(self, seeded_scan_session):
        """TSLA (no SE data) has se_* keys as None — backward compat."""
        repo = SqlScanResultRepository(seeded_scan_session)
        page = repo.query(SCAN_ID, QuerySpec())
        tsla = next(i for i in page.items if i.symbol == "TSLA")
        ef = tsla.extended_fields

        assert ef["se_setup_score"] is None
        assert ef["se_pattern_primary"] is None
        assert ef["se_setup_ready"] is None
        assert ef["se_quality_score"] is None
        assert ef["se_readiness_score"] is None
        assert ef["se_pattern_confidence"] is None
        assert ef["se_pivot_price"] is None
        assert ef["se_pivot_type"] is None


# ---------------------------------------------------------------------------
# Class 2: Feature Store path — full stack through repository
# ---------------------------------------------------------------------------


class TestFeatureStoreSEQueryIntegration:
    """Exercises SqlFeatureStoreRepository.query_run_as_scan_results() with SE fields."""

    # -- Filter tests -------------------------------------------------------

    def test_range_filter_se_setup_score(self, seeded_feature_session):
        """Range filter on se_setup_score min=40, max=90 → AAPL, NVDA."""
        repo = SqlFeatureStoreRepository(seeded_feature_session)
        spec = QuerySpec(
            filters=FilterSpec(
                range_filters=[
                    RangeFilter(field="se_setup_score", min_value=40, max_value=90)
                ]
            )
        )
        page = repo.query_run_as_scan_results(1, spec)
        symbols = {item.symbol for item in page.items}
        assert symbols == {"AAPL", "NVDA"}

    def test_boolean_filter_se_setup_ready_true(self, seeded_feature_session):
        """Boolean filter se_setup_ready=True → AAPL, GOOGL, NVDA."""
        repo = SqlFeatureStoreRepository(seeded_feature_session)
        spec = QuerySpec(
            filters=FilterSpec(
                boolean_filters=[BooleanFilter(field="se_setup_ready", value=True)]
            )
        )
        page = repo.query_run_as_scan_results(1, spec)
        symbols = {item.symbol for item in page.items}
        assert symbols == {"AAPL", "GOOGL", "NVDA"}

    def test_categorical_filter_se_pattern_primary(self, seeded_feature_session):
        """Categorical filter se_pattern_primary in ('VCP') → AAPL, NVDA."""
        repo = SqlFeatureStoreRepository(seeded_feature_session)
        spec = QuerySpec(
            filters=FilterSpec(
                categorical_filters=[
                    CategoricalFilter(field="se_pattern_primary", values=("VCP",))
                ]
            )
        )
        page = repo.query_run_as_scan_results(1, spec)
        symbols = {item.symbol for item in page.items}
        assert symbols == {"AAPL", "NVDA"}

    # -- Sort tests ---------------------------------------------------------

    def test_sort_se_setup_score_desc(self, seeded_feature_session):
        """Sort se_setup_score DESC → GOOGL, AAPL, NVDA, MSFT, TSLA(null last)."""
        repo = SqlFeatureStoreRepository(seeded_feature_session)
        spec = QuerySpec(sort=SortSpec(field="se_setup_score", order=SortOrder.DESC))
        page = repo.query_run_as_scan_results(1, spec)
        symbols = [item.symbol for item in page.items]
        assert symbols == ["GOOGL", "AAPL", "NVDA", "MSFT", "TSLA"]

    def test_sort_se_setup_score_asc_nulls_last(self, seeded_feature_session):
        """Sort se_setup_score ASC → MSFT(9.5), NVDA(50), AAPL(82.5), GOOGL(100), TSLA(null)."""
        repo = SqlFeatureStoreRepository(seeded_feature_session)
        spec = QuerySpec(sort=SortSpec(field="se_setup_score", order=SortOrder.ASC))
        page = repo.query_run_as_scan_results(1, spec)
        symbols = [item.symbol for item in page.items]
        assert symbols == ["MSFT", "NVDA", "AAPL", "GOOGL", "TSLA"]

    # -- Combined filter + sort ---------------------------------------------

    def test_filter_setup_ready_sort_score_desc(self, seeded_feature_session):
        """Filter se_setup_ready=True + sort se_setup_score DESC → GOOGL, AAPL, NVDA."""
        repo = SqlFeatureStoreRepository(seeded_feature_session)
        spec = QuerySpec(
            filters=FilterSpec(
                boolean_filters=[BooleanFilter(field="se_setup_ready", value=True)]
            ),
            sort=SortSpec(field="se_setup_score", order=SortOrder.DESC),
        )
        page = repo.query_run_as_scan_results(1, spec)
        symbols = [item.symbol for item in page.items]
        assert symbols == ["GOOGL", "AAPL", "NVDA"]

    # -- Domain mapper tests ------------------------------------------------

    def test_aapl_extended_fields_populated(self, seeded_feature_session):
        """AAPL extended_fields has all expected se_* keys with correct values."""
        repo = SqlFeatureStoreRepository(seeded_feature_session)
        page = repo.query_run_as_scan_results(1, QuerySpec())
        aapl = next(i for i in page.items if i.symbol == "AAPL")
        ef = aapl.extended_fields

        assert ef["se_setup_score"] == pytest.approx(82.5)
        assert ef["se_pattern_primary"] == "VCP"
        assert ef["se_setup_ready"] is True
        assert ef["se_quality_score"] == pytest.approx(75.0)
        assert ef["se_readiness_score"] == pytest.approx(90.0)
        assert ef["se_pattern_confidence"] == pytest.approx(0.88)
        assert ef["se_pivot_price"] == pytest.approx(142.50)
        assert ef["se_pivot_type"] == "breakout"
        assert ef["se_pivot_date"] == "2026-02-15"
        assert ef["se_timeframe"] == "daily"
        assert ef["se_atr14_pct"] == pytest.approx(3.1)
        assert ef["se_explain"] is not None
        assert ef["se_candidates"] is not None

    def test_tsla_extended_fields_none(self, seeded_feature_session):
        """TSLA (no SE data) has se_* keys as None — backward compat."""
        repo = SqlFeatureStoreRepository(seeded_feature_session)
        page = repo.query_run_as_scan_results(1, QuerySpec())
        tsla = next(i for i in page.items if i.symbol == "TSLA")
        ef = tsla.extended_fields

        assert ef["se_setup_score"] is None
        assert ef["se_pattern_primary"] is None
        assert ef["se_setup_ready"] is None

    # -- Path parity --------------------------------------------------------

    def test_path_parity_aapl_se_fields(
        self, seeded_scan_session, seeded_feature_session
    ):
        """Both paths produce identical se_* extended_fields for AAPL."""
        sr_repo = SqlScanResultRepository(seeded_scan_session)
        sr_page = sr_repo.query(SCAN_ID, QuerySpec())
        sr_aapl = next(i for i in sr_page.items if i.symbol == "AAPL")

        fs_repo = SqlFeatureStoreRepository(seeded_feature_session)
        fs_page = fs_repo.query_run_as_scan_results(1, QuerySpec())
        fs_aapl = next(i for i in fs_page.items if i.symbol == "AAPL")

        se_keys = [k for k in sr_aapl.extended_fields if k.startswith("se_")]
        assert len(se_keys) > 0, "No se_* keys found in extended_fields"
        for key in se_keys:
            sr_val = sr_aapl.extended_fields[key]
            fs_val = fs_aapl.extended_fields[key]
            if isinstance(sr_val, float):
                assert sr_val == pytest.approx(fs_val), (
                    f"Path parity mismatch for {key}: scan={sr_val} vs feature={fs_val}"
                )
            else:
                assert sr_val == fs_val, (
                    f"Path parity mismatch for {key}: scan={sr_val} vs feature={fs_val}"
                )
