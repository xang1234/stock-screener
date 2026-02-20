"""Round-trip persistence & backward-compatibility tests for setup_engine.

Verifies:
1. ScanOrchestrator._combine_results() promotes setup_engine to top level
2. _map_orchestrator_result() preserves the promoted dict through to details
3. json_extract(details, '$.setup_engine.*') resolves non-NULL in SQLite
4. Feature store mapper preserves the promoted dict and json_extract works
5. Validation guard in scan_result_repo accepts/rejects payloads correctly
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict

import pandas as pd
import pytest
from sqlalchemy import Column, Date, Float, Integer, MetaData, String, Table, Text, create_engine, text
from sqlalchemy.orm import Session

from app.scanners.base_screener import ScreenerResult, StockData
from app.scanners.scan_orchestrator import ScanOrchestrator
from app.infra.db.repositories.scan_result_repo import _map_orchestrator_result
from app.infra.query.scan_result_query import _JSON_FIELD_MAP as SR_JSON_FIELD_MAP
from app.infra.query.feature_store_query import _JSON_FIELD_MAP as FS_JSON_FIELD_MAP
from app.use_cases.feature_store.build_daily_snapshot import _map_orchestrator_to_feature_row

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# All se_* keys from both query builders
_SR_SE_KEYS = {k: v for k, v in SR_JSON_FIELD_MAP.items() if k.startswith("se_")}
_FS_SE_KEYS = {k: v for k, v in FS_JSON_FIELD_MAP.items() if k.startswith("se_")}

# Canonical test payload matching SetupEnginePayload schema
_SE_PAYLOAD: dict[str, Any] = {
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
    "candidates": [],
    "explain": {
        "passed_checks": [],
        "failed_checks": [],
        "invalidation_flags": [],
        "key_levels": {},
    },
}

# Lookup from se_* field name to the inner payload key
_SE_FIELD_TO_PAYLOAD_KEY: dict[str, str] = {}
for _k, _path in SR_JSON_FIELD_MAP.items():
    if _k.startswith("se_") and _path.startswith("$.setup_engine."):
        _SE_FIELD_TO_PAYLOAD_KEY[_k] = _path.split(".")[-1]


def _make_stub_stock_data(symbol: str = "TEST") -> StockData:
    """Minimal StockData for _combine_results()."""
    dates = pd.date_range(end="2026-02-20", periods=10, freq="B")
    df = pd.DataFrame(
        {"Open": 100.0, "High": 105.0, "Low": 99.0, "Close": 102.0, "Volume": 1_000_000},
        index=dates,
    )
    return StockData(symbol=symbol, price_data=df, benchmark_data=df)


def _make_se_screener_result(payload: dict[str, Any] | None = None) -> ScreenerResult:
    """Build a ScreenerResult mimicking SetupEngineScanner's normal output."""
    se = payload or dict(_SE_PAYLOAD)
    return ScreenerResult(
        score=se.get("setup_score", 0) or 0,
        passes=bool(se.get("setup_ready")),
        rating="Buy" if se.get("setup_ready") else "Pass",
        breakdown={"setup_score": se.get("setup_score", 0) or 0},
        details={"setup_engine": se},
        screener_name="setup_engine",
    )


def _make_minervini_screener_result() -> ScreenerResult:
    """Build a minimal minervini ScreenerResult."""
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
    stock_data: StockData | None = None,
) -> dict[str, Any]:
    """Call ScanOrchestrator._combine_results() without running a full scan."""
    orch = ScanOrchestrator.__new__(ScanOrchestrator)
    sd = stock_data or _make_stub_stock_data()
    return orch._combine_results(
        symbol=sd.symbol,
        screener_results=screener_results,
        stock_data=sd,
        composite_score=80.0,
        overall_rating="Buy",
        composite_method="weighted_average",
    )


# ---------------------------------------------------------------------------
# Test Class 1: Orchestrator Promotion
# ---------------------------------------------------------------------------


class TestOrchestratorPromotion:
    """Pure unit tests on _combine_results() output structure."""

    def test_setup_engine_promoted_to_top_level(self):
        result = _call_combine_results({"setup_engine": _make_se_screener_result()})
        assert "setup_engine" in result, "setup_engine should be promoted to top level"
        assert result["setup_engine"]["setup_score"] == 82.5
        assert result["setup_engine"]["pattern_primary"] == "VCP"
        assert result["setup_engine"]["setup_ready"] is True

    def test_promoted_dict_is_shallow_copy(self):
        """Promoted dict must not share identity with nested details."""
        result = _call_combine_results({"setup_engine": _make_se_screener_result()})
        nested = result["details"]["screeners"]["setup_engine"]["details"]["setup_engine"]
        promoted = result["setup_engine"]
        assert promoted is not nested, "Promoted dict should be a shallow copy"
        assert promoted == nested, "Values should be equal"

    def test_promotion_skipped_when_scanner_absent(self):
        result = _call_combine_results({"minervini": _make_minervini_screener_result()})
        assert "setup_engine" not in result

    def test_promotion_skipped_for_insufficient_data(self):
        sr = ScreenerResult(
            score=0.0,
            passes=False,
            rating="Insufficient Data",
            breakdown={},
            details={"reason": "Not enough price history"},
            screener_name="setup_engine",
        )
        result = _call_combine_results({"setup_engine": sr})
        assert "setup_engine" not in result, (
            "Insufficient data result has no 'setup_engine' key in details"
        )

    def test_promotion_skipped_for_error_result(self):
        sr = ScreenerResult(
            score=0.0,
            passes=False,
            rating="Error",
            breakdown={},
            details={"error": "Something went wrong"},
            screener_name="setup_engine",
        )
        result = _call_combine_results({"setup_engine": sr})
        assert "setup_engine" not in result

    def test_coexists_with_minervini(self):
        result = _call_combine_results({
            "minervini": _make_minervini_screener_result(),
            "setup_engine": _make_se_screener_result(),
        })
        # Minervini fields promoted
        assert result["rs_rating"] == 85
        assert result["stage"] == 2
        # SE fields promoted
        assert result["setup_engine"]["setup_score"] == 82.5
        assert result["setup_engine"]["pattern_primary"] == "VCP"


# ---------------------------------------------------------------------------
# Test Class 2: ScanResult Round-Trip (orchestrator → SQLite → json_extract)
# ---------------------------------------------------------------------------


@pytest.fixture()
def sr_db_session():
    """In-memory SQLite with minimal scan_results schema."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    meta = MetaData()

    Table("scans", meta, Column("scan_id", String(36), primary_key=True))
    Table(
        "scan_results", meta,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("scan_id", String(36)),
        Column("symbol", String(10), nullable=False),
        Column("composite_score", Float),
        Column("details", Text),
    )
    meta.create_all(engine)

    with Session(engine) as session:
        session.execute(text("INSERT INTO scans (scan_id) VALUES ('scan-1')"))
        session.commit()
        yield session


def _insert_sr_row(session: Session, symbol: str, result_dict: dict) -> None:
    """Map through _map_orchestrator_result and insert into scan_results."""
    mapped = _map_orchestrator_result("scan-1", symbol, result_dict)
    details_json = json.dumps(mapped["details"]) if mapped.get("details") else None
    session.execute(
        text(
            "INSERT INTO scan_results (scan_id, symbol, composite_score, details) "
            "VALUES (:sid, :sym, :score, :details)"
        ),
        {
            "sid": "scan-1",
            "sym": symbol,
            "score": mapped.get("composite_score"),
            "details": details_json,
        },
    )
    session.flush()


# Parametrize over all se_* fields from scan_result_query
_SR_SE_PARAMS = [
    pytest.param(field_name, json_path, id=field_name)
    for field_name, json_path in sorted(_SR_SE_KEYS.items())
]


class TestScanResultRoundTrip:
    """Tests through _map_orchestrator_result() → in-memory SQLite → json_extract()."""

    @pytest.mark.parametrize("field_name,json_path", _SR_SE_PARAMS)
    def test_all_se_fields_resolve_non_null(self, sr_db_session, field_name, json_path):
        result_dict = _call_combine_results({"setup_engine": _make_se_screener_result()})
        _insert_sr_row(sr_db_session, "AAPL", result_dict)

        rows = sr_db_session.execute(
            text(f"SELECT json_extract(details, '{json_path}') FROM scan_results WHERE symbol = 'AAPL'")
        ).fetchall()
        assert len(rows) == 1
        assert rows[0][0] is not None, f"{field_name} ({json_path}) should be non-NULL"

    def test_numeric_values_match_input(self, sr_db_session):
        result_dict = _call_combine_results({"setup_engine": _make_se_screener_result()})
        _insert_sr_row(sr_db_session, "AAPL", result_dict)

        checks = [
            ("$.setup_engine.setup_score", 82.5),
            ("$.setup_engine.pivot_price", 142.50),
            ("$.setup_engine.distance_to_pivot_pct", 2.3),
            ("$.setup_engine.rs", 1.12),
        ]
        for json_path, expected in checks:
            row = sr_db_session.execute(
                text(f"SELECT json_extract(details, '{json_path}') FROM scan_results WHERE symbol = 'AAPL'")
            ).fetchone()
            assert row[0] == pytest.approx(expected), f"{json_path} mismatch"

    def test_boolean_fields_roundtrip(self, sr_db_session):
        result_dict = _call_combine_results({"setup_engine": _make_se_screener_result()})
        _insert_sr_row(sr_db_session, "AAPL", result_dict)

        # setup_ready = True → stored as 1 in SQLite JSON
        row = sr_db_session.execute(
            text("SELECT json_extract(details, '$.setup_engine.setup_ready') FROM scan_results WHERE symbol = 'AAPL'")
        ).fetchone()
        assert row[0] == 1, "True should be stored as 1 in SQLite JSON"

        # rs_line_new_high = True → stored as 1
        row = sr_db_session.execute(
            text("SELECT json_extract(details, '$.setup_engine.rs_line_new_high') FROM scan_results WHERE symbol = 'AAPL'")
        ).fetchone()
        assert row[0] == 1

    def test_string_fields_roundtrip(self, sr_db_session):
        result_dict = _call_combine_results({"setup_engine": _make_se_screener_result()})
        _insert_sr_row(sr_db_session, "AAPL", result_dict)

        row = sr_db_session.execute(
            text("SELECT json_extract(details, '$.setup_engine.pattern_primary') FROM scan_results WHERE symbol = 'AAPL'")
        ).fetchone()
        assert row[0] == "VCP"

        row = sr_db_session.execute(
            text("SELECT json_extract(details, '$.setup_engine.pivot_type') FROM scan_results WHERE symbol = 'AAPL'")
        ).fetchone()
        assert row[0] == "breakout"

    def test_validation_guard_accepts_valid_payload(self, sr_db_session):
        result_dict = _call_combine_results({"setup_engine": _make_se_screener_result()})
        mapped = _map_orchestrator_result("scan-1", "AAPL", result_dict)
        # Valid payload should be preserved in details
        assert "setup_engine" in mapped["details"], (
            "Valid setup_engine should not be dropped by validation guard"
        )
        assert "setup_engine_validation_errors" not in mapped["details"]

    def test_row_without_setup_engine_returns_null(self, sr_db_session):
        """Old rows without setup_engine should return NULL for SE json paths."""
        # Insert a new-style row with SE data
        se_result = _call_combine_results({"setup_engine": _make_se_screener_result()})
        _insert_sr_row(sr_db_session, "AAPL", se_result)

        # Insert an old-style row without SE data
        old_result = _call_combine_results({"minervini": _make_minervini_screener_result()})
        _insert_sr_row(sr_db_session, "MSFT", old_result)

        sr_db_session.flush()

        # AAPL should have non-NULL SE values
        row = sr_db_session.execute(
            text("SELECT json_extract(details, '$.setup_engine.setup_score') FROM scan_results WHERE symbol = 'AAPL'")
        ).fetchone()
        assert row[0] is not None

        # MSFT should have NULL SE values (backward compat)
        row = sr_db_session.execute(
            text("SELECT json_extract(details, '$.setup_engine.setup_score') FROM scan_results WHERE symbol = 'MSFT'")
        ).fetchone()
        assert row[0] is None

    def test_domain_mapper_handles_setup_engine_dict(self):
        """_map_row_to_domain requires a ScanResult ORM row; verify _map_orchestrator_result
        produces details with a setup_engine dict that doesn't break reader-side expectations."""
        result_dict = _call_combine_results({"setup_engine": _make_se_screener_result()})
        mapped = _map_orchestrator_result("scan-1", "AAPL", result_dict)
        details = mapped["details"]
        # The details dict should be json-serializable (no weird types)
        serialized = json.dumps(details)
        deserialized = json.loads(serialized)
        assert "setup_engine" in deserialized
        assert deserialized["setup_engine"]["setup_score"] == pytest.approx(82.5)


# ---------------------------------------------------------------------------
# Test Class 3: Feature Store Round-Trip
# ---------------------------------------------------------------------------


@pytest.fixture()
def fs_db_session():
    """In-memory SQLite with minimal feature store schema."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    meta = MetaData()

    Table(
        "feature_runs", meta,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("as_of_date", Date, nullable=False),
        Column("run_type", Text, nullable=False),
        Column("status", Text, nullable=False),
    )
    Table(
        "stock_feature_daily", meta,
        Column("run_id", Integer, primary_key=True),
        Column("symbol", Text, nullable=False, primary_key=True),
        Column("as_of_date", Date, nullable=False),
        Column("composite_score", Float, nullable=True),
        Column("overall_rating", Integer, nullable=True),
        Column("passes_count", Integer, nullable=True),
        Column("details_json", Text, nullable=True),
    )
    meta.create_all(engine)

    with Session(engine) as session:
        session.execute(
            text("INSERT INTO feature_runs (id, as_of_date, run_type, status) VALUES (1, '2026-02-20', 'daily_snapshot', 'published')")
        )
        session.commit()
        yield session


def _insert_fs_row(session: Session, symbol: str, result_dict: dict) -> None:
    """Map through _map_orchestrator_to_feature_row and insert."""
    from datetime import date
    feature_row = _map_orchestrator_to_feature_row(symbol, date(2026, 2, 20), result_dict)
    details_json = json.dumps(feature_row.details) if feature_row.details else None
    session.execute(
        text(
            "INSERT INTO stock_feature_daily "
            "(run_id, symbol, as_of_date, composite_score, overall_rating, passes_count, details_json) "
            "VALUES (:rid, :sym, :dt, :score, :rating, :passes, :details)"
        ),
        {
            "rid": 1,
            "sym": symbol,
            "dt": "2026-02-20",
            "score": feature_row.composite_score,
            "rating": feature_row.overall_rating,
            "passes": feature_row.passes_count,
            "details": details_json,
        },
    )
    session.flush()


_FS_SE_PARAMS = [
    pytest.param(field_name, json_path, id=field_name)
    for field_name, json_path in sorted(_FS_SE_KEYS.items())
]


class TestFeatureStoreRoundTrip:
    """Tests through _map_orchestrator_to_feature_row() → SQLite → json_extract()."""

    @pytest.mark.parametrize("field_name,json_path", _FS_SE_PARAMS)
    def test_all_se_fields_resolve_non_null(self, fs_db_session, field_name, json_path):
        result_dict = _call_combine_results({"setup_engine": _make_se_screener_result()})
        _insert_fs_row(fs_db_session, "AAPL", result_dict)

        rows = fs_db_session.execute(
            text(f"SELECT json_extract(details_json, '{json_path}') FROM stock_feature_daily WHERE symbol = 'AAPL'")
        ).fetchall()
        assert len(rows) == 1
        assert rows[0][0] is not None, f"{field_name} ({json_path}) should be non-NULL"

    def test_non_se_fields_also_resolve(self, fs_db_session):
        """Existing fields like $.rs_rating and $.stage still work alongside SE promotion."""
        result_dict = _call_combine_results({
            "minervini": _make_minervini_screener_result(),
            "setup_engine": _make_se_screener_result(),
        })
        _insert_fs_row(fs_db_session, "AAPL", result_dict)

        # rs_rating from minervini
        row = fs_db_session.execute(
            text("SELECT json_extract(details_json, '$.rs_rating') FROM stock_feature_daily WHERE symbol = 'AAPL'")
        ).fetchone()
        assert row[0] == 85

        # stage from minervini
        row = fs_db_session.execute(
            text("SELECT json_extract(details_json, '$.stage') FROM stock_feature_daily WHERE symbol = 'AAPL'")
        ).fetchone()
        assert row[0] == 2

        # SE still works
        row = fs_db_session.execute(
            text("SELECT json_extract(details_json, '$.setup_engine.setup_score') FROM stock_feature_daily WHERE symbol = 'AAPL'")
        ).fetchone()
        assert row[0] == pytest.approx(82.5)


# ---------------------------------------------------------------------------
# Test Class 4: Validation Guard
# ---------------------------------------------------------------------------


class TestValidationGuard:
    """Pure function tests on _map_orchestrator_result() validation guard."""

    def test_valid_payload_passes_through(self):
        result_dict = _call_combine_results({"setup_engine": _make_se_screener_result()})
        mapped = _map_orchestrator_result("scan-1", "AAPL", result_dict)
        assert "setup_engine" in mapped["details"]
        assert mapped["details"]["setup_engine"]["setup_score"] == pytest.approx(82.5)

    def test_invalid_payload_dropped(self):
        """A setup_engine dict missing required keys should be dropped."""
        result_dict = _call_combine_results({"setup_engine": _make_se_screener_result()})
        # Corrupt the promoted setup_engine to be invalid
        result_dict["setup_engine"] = {"setup_score": 50.0}  # missing many required keys
        mapped = _map_orchestrator_result("scan-1", "AAPL", result_dict)
        # The validation guard should have dropped it
        assert "setup_engine" not in mapped["details"] or "setup_engine_validation_errors" in mapped["details"]

    def test_invalid_payload_no_cascade(self):
        """Even when SE payload is invalid, other fields are still populated."""
        result_dict = _call_combine_results({
            "minervini": _make_minervini_screener_result(),
            "setup_engine": _make_se_screener_result(),
        })
        result_dict["setup_engine"] = {"setup_score": 50.0}  # corrupt
        mapped = _map_orchestrator_result("scan-1", "AAPL", result_dict)
        # Other fields should be fine
        assert mapped["symbol"] == "AAPL"
        assert mapped["composite_score"] is not None
        assert mapped["rs_rating"] == 85

    def test_non_dict_ignored(self):
        """When setup_engine is a non-dict (e.g. string), validation is skipped."""
        result_dict = _call_combine_results({"setup_engine": _make_se_screener_result()})
        result_dict["setup_engine"] = "some_string"
        mapped = _map_orchestrator_result("scan-1", "AAPL", result_dict)
        # Should not crash; non-dict is not validated
        assert mapped["symbol"] == "AAPL"
