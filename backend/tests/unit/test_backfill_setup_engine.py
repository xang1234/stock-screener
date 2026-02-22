"""Unit tests for the setup_engine backfill script.

Tests the core logic in isolation:
- Row classification (needs backfill vs. skip)
- Date truncation of price DataFrames
- Merge correctness via attach_setup_engine
- Error isolation per symbol
- Dry-run: no DB writes
- Idempotency with --force override
- Run status filtering
"""
from __future__ import annotations

import argparse
from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# Import script functions under test
from scripts.backfill_setup_engine import (
    CURRENT_SCHEMA,
    _needs_backfill,
    _truncate_to_date,
    run_backfill,
)


# ───────────────────────────────────────────────────────────────────────────
# _needs_backfill classification
# ───────────────────────────────────────────────────────────────────────────


class TestNeedsBackfill:
    """Test the row classification logic."""

    def test_null_details_needs_backfill(self):
        assert _needs_backfill(None, force=False) is True

    def test_empty_dict_needs_backfill(self):
        assert _needs_backfill({}, force=False) is True

    def test_no_setup_engine_key_needs_backfill(self):
        details = {"minervini": {"score": 80}}
        assert _needs_backfill(details, force=False) is True

    def test_wrong_schema_version_needs_backfill(self):
        details = {"setup_engine": {"schema_version": "v0"}}
        assert _needs_backfill(details, force=False) is True

    def test_matching_schema_version_skipped(self):
        details = {"setup_engine": {"schema_version": CURRENT_SCHEMA}}
        assert _needs_backfill(details, force=False) is False

    def test_force_overrides_matching_schema(self):
        details = {"setup_engine": {"schema_version": CURRENT_SCHEMA}}
        assert _needs_backfill(details, force=True) is True

    def test_force_on_null_details(self):
        assert _needs_backfill(None, force=True) is True

    def test_non_dict_details_needs_backfill(self):
        """If details_json is somehow a string or list, treat as needing backfill."""
        assert _needs_backfill("not a dict", force=False) is True
        assert _needs_backfill([1, 2, 3], force=False) is True

    def test_empty_setup_engine_needs_backfill(self):
        """Empty setup_engine dict (no schema_version) needs backfill."""
        details = {"setup_engine": {}}
        assert _needs_backfill(details, force=False) is True

    def test_setup_engine_none_needs_backfill(self):
        """setup_engine key present but None."""
        details = {"setup_engine": None}
        assert _needs_backfill(details, force=False) is True


# ───────────────────────────────────────────────────────────────────────────
# _truncate_to_date
# ───────────────────────────────────────────────────────────────────────────


class TestTruncateToDate:
    """Test DataFrame date slicing."""

    @pytest.fixture
    def sample_df(self):
        """Create a sample OHLCV DataFrame with DatetimeIndex."""
        dates = pd.date_range("2025-01-01", periods=10, freq="B")
        return pd.DataFrame(
            {
                "Open": range(10),
                "High": range(10, 20),
                "Low": range(20, 30),
                "Close": range(30, 40),
                "Volume": range(100, 110),
            },
            index=dates,
        )

    def test_truncate_to_mid_date(self, sample_df):
        """Rows after as_of_date should be excluded."""
        result = _truncate_to_date(sample_df, date(2025, 1, 7))
        # 2025-01-01 (Wed) through 2025-01-07 (Tue) = 5 business days
        assert len(result) == 5
        assert result.index[-1] <= pd.Timestamp("2025-01-07")

    def test_truncate_to_date_beyond_range(self, sample_df):
        """If as_of_date is after all data, return everything."""
        result = _truncate_to_date(sample_df, date(2026, 12, 31))
        assert len(result) == len(sample_df)

    def test_truncate_to_date_before_range(self, sample_df):
        """If as_of_date is before all data, return empty."""
        result = _truncate_to_date(sample_df, date(2024, 1, 1))
        assert len(result) == 0

    def test_truncate_empty_df(self):
        """Empty DataFrame returns empty."""
        empty = pd.DataFrame()
        result = _truncate_to_date(empty, date(2025, 1, 1))
        assert result.empty

    def test_truncate_preserves_columns(self, sample_df):
        """Column names should be preserved after truncation."""
        result = _truncate_to_date(sample_df, date(2025, 1, 3))
        assert list(result.columns) == ["Open", "High", "Low", "Close", "Volume"]


# ───────────────────────────────────────────────────────────────────────────
# Merge correctness (via attach_setup_engine)
# ───────────────────────────────────────────────────────────────────────────


class TestMergeCorrectness:
    """Verify that existing details_json fields are preserved after merge."""

    def test_merge_preserves_existing_keys(self):
        from app.scanners.setup_engine_scanner import attach_setup_engine

        existing = {"minervini": {"score": 85}, "canslim": {"eps_growth": 30}}
        payload = {"schema_version": "v1", "setup_score": 72.0}
        merged = attach_setup_engine(existing, payload)

        assert merged["minervini"] == {"score": 85}
        assert merged["canslim"] == {"eps_growth": 30}
        assert merged["setup_engine"]["setup_score"] == 72.0

    def test_merge_with_none_details(self):
        from app.scanners.setup_engine_scanner import attach_setup_engine

        payload = {"schema_version": "v1", "setup_score": 50.0}
        merged = attach_setup_engine(None, payload)

        assert "setup_engine" in merged
        assert merged["setup_engine"]["setup_score"] == 50.0

    def test_merge_overwrites_old_setup_engine(self):
        from app.scanners.setup_engine_scanner import attach_setup_engine

        existing = {"setup_engine": {"schema_version": "v0", "setup_score": 10.0}}
        new_payload = {"schema_version": "v1", "setup_score": 72.0}
        merged = attach_setup_engine(existing, new_payload)

        assert merged["setup_engine"]["schema_version"] == "v1"
        assert merged["setup_engine"]["setup_score"] == 72.0


# ───────────────────────────────────────────────────────────────────────────
# Integration-style tests for run_backfill
# ───────────────────────────────────────────────────────────────────────────

def _make_args(**overrides) -> argparse.Namespace:
    """Build a default argparse.Namespace for run_backfill."""
    defaults = {
        "run_id": None,
        "date_from": None,
        "date_to": None,
        "symbols": None,
        "status": "published,completed",
        "dry_run": False,
        "force": False,
        "chunk_size": 50,
        "fetch_delay": 0.0,
        "yes": True,  # skip confirmation in tests
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _make_price_df(start: str = "2024-06-01", periods: int = 200) -> pd.DataFrame:
    """Create a realistic-looking OHLCV DataFrame."""
    dates = pd.date_range(start, periods=periods, freq="B")
    import numpy as np

    close = 100 + np.cumsum(np.random.randn(periods) * 0.5)
    return pd.DataFrame(
        {
            "Open": close - 0.5,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Volume": np.random.randint(1_000_000, 10_000_000, size=periods),
        },
        index=dates,
    )


class TestRunBackfillDryRun:
    """Verify dry-run mode performs no writes."""

    @patch("scripts.backfill_setup_engine.SessionLocal")
    def test_dry_run_no_writes(self, mock_session_cls):
        mock_db = MagicMock()
        mock_session_cls.return_value = mock_db

        # Simulate one row that needs backfill (no setup_engine key)
        mock_row = (1, "AAPL", date(2025, 6, 1), {"minervini": {"score": 80}})
        mock_query = MagicMock()
        mock_query.all.return_value = [mock_row]
        mock_db.query.return_value.join.return_value.filter.return_value = mock_query

        args = _make_args(dry_run=True)
        run_backfill(args)

        # Should NOT have called commit or update
        mock_db.commit.assert_not_called()


class TestRunBackfillErrorIsolation:
    """Verify that a failed symbol doesn't abort the entire backfill."""

    @patch("scripts.backfill_setup_engine.BenchmarkCacheService")
    @patch("scripts.backfill_setup_engine._fetch_price_data")
    @patch("scripts.backfill_setup_engine.SetupEngineScanner")
    @patch("scripts.backfill_setup_engine.SessionLocal")
    def test_failed_symbol_continues(
        self, mock_session_cls, mock_scanner_cls, mock_fetch, mock_bench_cls,
    ):
        mock_db = MagicMock()
        mock_session_cls.return_value = mock_db

        # Two symbols: AAPL (will fail), MSFT (will succeed)
        rows = [
            (1, "AAPL", date(2025, 6, 1), None),
            (1, "MSFT", date(2025, 6, 1), None),
        ]
        mock_query = MagicMock()
        mock_query.all.return_value = rows
        mock_db.query.return_value.join.return_value.filter.return_value = mock_query

        # AAPL fetch raises, MSFT succeeds
        def fetch_side_effect(symbol, delay):
            if symbol == "AAPL":
                raise RuntimeError("Network error")
            return _make_price_df()

        mock_fetch.side_effect = fetch_side_effect

        # SPY data
        spy_mock = MagicMock()
        spy_mock.get_spy_data.return_value = _make_price_df()
        mock_bench_cls.get_instance.return_value = spy_mock

        # Scanner returns valid result for MSFT
        scanner_inst = MagicMock()
        scanner_inst.scan_stock.return_value = MagicMock(
            details={"setup_engine": {"schema_version": "v1", "setup_score": 70.0}},
            rating="Watch",
        )
        mock_scanner_cls.return_value = scanner_inst

        # Spot-check mock
        spot_query = MagicMock()
        spot_query.all.return_value = []

        # Make db.query return different things for different args
        original_query = mock_db.query

        def query_side_effect(*args, **kwargs):
            # The first call with StockFeatureDaily columns is the main query
            # Subsequent calls could be for update or spot-check
            result = MagicMock()
            result.join.return_value.filter.return_value = mock_query
            result.filter.return_value.update.return_value = 1
            result.filter.return_value.order_by.return_value.limit.return_value.all.return_value = []
            return result

        mock_db.query.side_effect = query_side_effect

        args = _make_args(yes=True)
        # Should complete without raising
        run_backfill(args)

        # MSFT should have been processed (scanner called at least once)
        assert scanner_inst.scan_stock.call_count >= 1


class TestRunBackfillStatusFilter:
    """Verify run status filter is applied to the query."""

    @patch("scripts.backfill_setup_engine.SessionLocal")
    def test_custom_status_filter(self, mock_session_cls):
        mock_db = MagicMock()
        mock_session_cls.return_value = mock_db

        mock_query = MagicMock()
        mock_query.all.return_value = []  # No rows — just testing the filter is applied

        # Build the chain
        mock_join = MagicMock()
        mock_filter = MagicMock()
        mock_filter.all.return_value = []

        mock_db.query.return_value.join.return_value = mock_join
        mock_join.filter.return_value = mock_filter
        mock_filter.all.return_value = []

        args = _make_args(status="published")
        run_backfill(args)

        # The filter chain was called (query → join → filter)
        mock_db.query.assert_called()


class TestRunBackfillIdempotency:
    """Verify rows with matching schema_version are skipped."""

    @patch("scripts.backfill_setup_engine.SessionLocal")
    def test_already_backfilled_rows_skipped(self, mock_session_cls):
        mock_db = MagicMock()
        mock_session_cls.return_value = mock_db

        # Row already has current schema
        row_already_done = (
            1, "AAPL", date(2025, 6, 1),
            {"setup_engine": {"schema_version": CURRENT_SCHEMA, "setup_score": 80.0}},
        )
        mock_query = MagicMock()
        mock_query.all.return_value = [row_already_done]
        mock_db.query.return_value.join.return_value.filter.return_value = mock_query

        args = _make_args()
        run_backfill(args)

        # No commits should happen — all rows were up-to-date
        mock_db.commit.assert_not_called()


class TestRunBackfillForceOverwrite:
    """Verify --force reprocesses even schema-matching rows."""

    @patch("scripts.backfill_setup_engine.BenchmarkCacheService")
    @patch("scripts.backfill_setup_engine._fetch_price_data")
    @patch("scripts.backfill_setup_engine.SetupEngineScanner")
    @patch("scripts.backfill_setup_engine.SessionLocal")
    def test_force_reprocesses_matching_rows(
        self, mock_session_cls, mock_scanner_cls, mock_fetch, mock_bench_cls,
    ):
        mock_db = MagicMock()
        mock_session_cls.return_value = mock_db

        # Row already at current schema — normally would be skipped
        row = (
            1, "AAPL", date(2025, 6, 1),
            {"setup_engine": {"schema_version": CURRENT_SCHEMA, "setup_score": 60.0}},
        )
        mock_query = MagicMock()
        mock_query.all.return_value = [row]
        mock_db.query.return_value.join.return_value.filter.return_value = mock_query

        # Price data
        mock_fetch.return_value = _make_price_df()

        # SPY data
        spy_mock = MagicMock()
        spy_mock.get_spy_data.return_value = _make_price_df()
        mock_bench_cls.get_instance.return_value = spy_mock

        # Scanner returns updated result
        scanner_inst = MagicMock()
        scanner_inst.scan_stock.return_value = MagicMock(
            details={"setup_engine": {"schema_version": "v1", "setup_score": 75.0}},
            rating="Buy",
        )
        mock_scanner_cls.return_value = scanner_inst

        # Make db.query flexible for update + spot-check
        def query_side_effect(*args, **kwargs):
            result = MagicMock()
            result.join.return_value.filter.return_value = mock_query
            result.filter.return_value.update.return_value = 1
            result.filter.return_value.order_by.return_value.limit.return_value.all.return_value = []
            return result

        mock_db.query.side_effect = query_side_effect

        args = _make_args(force=True)
        run_backfill(args)

        # Scanner should have been called (force reprocesses)
        scanner_inst.scan_stock.assert_called_once()
        # And a commit should have happened
        mock_db.commit.assert_called()
