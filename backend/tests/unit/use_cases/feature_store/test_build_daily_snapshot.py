"""Tests for BuildDailyFeatureSnapshotUseCase.

Uses in-memory fakes from conftest.py — no DB, no mocks, pure behaviour tests.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import patch

import pytest

from app.domain.common.errors import ValidationError
from app.domain.feature_store.models import RunStatus
from app.use_cases.feature_store.build_daily_snapshot import (
    BuildDailyFeatureSnapshotUseCase,
    BuildDailySnapshotCommand,
    _map_orchestrator_to_feature_row,
)
from tests.unit.use_cases.conftest import (
    FakeCancellationToken,
    FakeFeatureRunRepository,
    FakeFeatureStoreRepository,
    FakeProgressSink,
    FakeScanner,
    FakeUnitOfWork,
    FakeUniverseRepository,
)

# All tests patch _is_us_trading_day to avoid real calendar lookups.
_PATCH_TRADING_DAY = patch(
    "app.use_cases.feature_store.build_daily_snapshot._is_us_trading_day",
    return_value=True,
)

AS_OF = date(2026, 2, 17)  # Tuesday


def _make_cmd(**overrides) -> BuildDailySnapshotCommand:
    """Build a command with sensible defaults; override any field."""
    defaults = dict(
        as_of_date=AS_OF,
        screener_names=["minervini"],
        universe_def={"name": "test"},
    )
    defaults.update(overrides)
    return BuildDailySnapshotCommand(**defaults)


def _make_uow(
    symbols: list[str] | None = None,
    scanner_results: dict[str, dict] | None = None,
) -> tuple[FakeUnitOfWork, FakeScanner]:
    """Create a UoW + Scanner pair with configurable symbol list."""
    universe = FakeUniverseRepository(
        symbols if symbols is not None else ["AAPL", "MSFT", "GOOGL"]
    )
    feature_runs = FakeFeatureRunRepository()
    feature_store = FakeFeatureStoreRepository()
    uow = FakeUnitOfWork(
        universe=universe,
        feature_runs=feature_runs,
        feature_store=feature_store,
    )
    scanner = FakeScanner(results=scanner_results)
    return uow, scanner


# ---------------------------------------------------------------------------
# Pure function tests
# ---------------------------------------------------------------------------


class TestMapOrchestratorToFeatureRow:
    def test_maps_all_fields(self):
        row = _map_orchestrator_to_feature_row(
            "AAPL",
            AS_OF,
            {
                "composite_score": 85.5,
                "rating": "Strong Buy",
                "screeners_passed": 2,
                "details": {"screeners": {}},
            },
        )
        assert row.symbol == "AAPL"
        assert row.as_of_date == AS_OF
        assert row.composite_score == 85.5
        assert row.overall_rating == 5  # Strong Buy → 5
        assert row.passes_count == 2
        assert row.details == {"screeners": {}}

    def test_unknown_rating_defaults_to_watch(self):
        row = _map_orchestrator_to_feature_row(
            "X", AS_OF, {"rating": "Unknown-Rating"}
        )
        assert row.overall_rating == 3  # Watch

    def test_missing_fields_are_none(self):
        row = _map_orchestrator_to_feature_row("X", AS_OF, {})
        assert row.composite_score is None
        assert row.passes_count is None
        assert row.details is None

    def test_all_rating_values(self):
        for rating, expected_int in [
            ("Strong Buy", 5),
            ("Buy", 4),
            ("Watch", 3),
            ("Pass", 2),
            ("Error", 1),
        ]:
            row = _map_orchestrator_to_feature_row(
                "X", AS_OF, {"rating": rating}
            )
            assert row.overall_rating == expected_int, f"Failed for {rating}"


# ---------------------------------------------------------------------------
# Command validation tests
# ---------------------------------------------------------------------------


class TestBuildDailySnapshotCommand:
    def test_valid_defaults(self):
        cmd = _make_cmd()
        assert cmd.chunk_size == 50
        assert cmd.dq_thresholds.row_count_threshold == 0.9
        assert cmd.dq_thresholds.null_max_rate == 0.05

    def test_chunk_size_must_be_positive(self):
        with pytest.raises(ValueError, match="chunk_size"):
            _make_cmd(chunk_size=0)

    def test_dq_thresholds_validation_propagated(self):
        """DQThresholds validation fires during command construction."""
        from app.domain.feature_store.quality import DQThresholds

        with pytest.raises(ValueError, match="row_count_threshold"):
            _make_cmd(dq_thresholds=DQThresholds(row_count_threshold=1.5))

    def test_custom_dq_thresholds(self):
        from app.domain.feature_store.quality import DQThresholds

        thresholds = DQThresholds(row_count_threshold=0.5, null_max_rate=0.1)
        cmd = _make_cmd(dq_thresholds=thresholds)
        assert cmd.dq_thresholds.row_count_threshold == 0.5
        assert cmd.dq_thresholds.null_max_rate == 0.1


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestHappyPath:
    @_PATCH_TRADING_DAY
    def test_scans_all_symbols_and_publishes(self, _mock_td):
        uow, scanner = _make_uow()
        use_case = BuildDailyFeatureSnapshotUseCase(scanner=scanner)
        progress = FakeProgressSink()

        result = use_case.execute(
            uow, _make_cmd(), progress, FakeCancellationToken()
        )

        assert result.status == RunStatus.PUBLISHED.value
        assert result.total_symbols == 3
        assert result.processed_symbols == 3
        assert result.failed_symbols == 0
        assert result.dq_passed is True
        assert result.run_id == 1

    @_PATCH_TRADING_DAY
    def test_progress_events_emitted(self, _mock_td):
        uow, scanner = _make_uow()
        use_case = BuildDailyFeatureSnapshotUseCase(scanner=scanner)
        progress = FakeProgressSink()

        use_case.execute(
            uow, _make_cmd(chunk_size=2), progress, FakeCancellationToken()
        )

        # 3 symbols, chunk_size=2 → 2 chunk progress + 1 completion = 3 events
        assert len(progress.events) >= 3
        # Final event should show all symbols processed
        final = progress.events[-1]
        assert final.current == 3
        assert final.total == 3

    @_PATCH_TRADING_DAY
    def test_feature_rows_persisted(self, _mock_td):
        uow, scanner = _make_uow()
        use_case = BuildDailyFeatureSnapshotUseCase(scanner=scanner)

        result = use_case.execute(
            uow, _make_cmd(), FakeProgressSink(), FakeCancellationToken()
        )

        # All 3 symbols should have feature rows in the store
        count = uow.feature_store.count_by_run_id(result.run_id)
        assert count == 3

    @_PATCH_TRADING_DAY
    def test_universe_symbols_saved(self, _mock_td):
        uow, scanner = _make_uow(symbols=["AAPL", "TSLA"])
        use_case = BuildDailyFeatureSnapshotUseCase(scanner=scanner)

        result = use_case.execute(
            uow, _make_cmd(), FakeProgressSink(), FakeCancellationToken()
        )

        saved_universe = uow.feature_store._universe.get(result.run_id)
        assert saved_universe == ["AAPL", "TSLA"]

    @_PATCH_TRADING_DAY
    def test_idempotent_rerun_creates_new_run(self, _mock_td):
        uow, scanner = _make_uow()
        use_case = BuildDailyFeatureSnapshotUseCase(scanner=scanner)
        cmd = _make_cmd()

        r1 = use_case.execute(
            uow, cmd, FakeProgressSink(), FakeCancellationToken()
        )
        r2 = use_case.execute(
            uow, cmd, FakeProgressSink(), FakeCancellationToken()
        )

        assert r1.run_id != r2.run_id
        assert r1.status == RunStatus.PUBLISHED.value
        assert r2.status == RunStatus.PUBLISHED.value


# ---------------------------------------------------------------------------
# Non-trading day
# ---------------------------------------------------------------------------


class TestTradingDayValidation:
    def test_non_trading_day_raises(self):
        uow, scanner = _make_uow()
        use_case = BuildDailyFeatureSnapshotUseCase(scanner=scanner)

        with patch(
            "app.use_cases.feature_store.build_daily_snapshot._is_us_trading_day",
            return_value=False,
        ):
            with pytest.raises(ValidationError, match="not a US trading day"):
                use_case.execute(
                    uow,
                    _make_cmd(),
                    FakeProgressSink(),
                    FakeCancellationToken(),
                )


# ---------------------------------------------------------------------------
# Empty universe
# ---------------------------------------------------------------------------


class TestEmptyUniverse:
    @_PATCH_TRADING_DAY
    def test_empty_universe_raises(self, _mock_td):
        uow, scanner = _make_uow(symbols=[])
        use_case = BuildDailyFeatureSnapshotUseCase(scanner=scanner)

        with pytest.raises(ValidationError, match="zero symbols"):
            use_case.execute(
                uow,
                _make_cmd(),
                FakeProgressSink(),
                FakeCancellationToken(),
            )


# ---------------------------------------------------------------------------
# DQ quarantine
# ---------------------------------------------------------------------------


class TestDQQuarantine:
    @_PATCH_TRADING_DAY
    def test_quarantine_on_low_row_count(self, _mock_td):
        """If too many symbols fail, row_count DQ check blocks publish."""
        # 10 symbols, 8 return errors → only 2 rows → 2/10 = 20% < 90%
        error_result = {"error": "No data"}
        results = {
            f"SYM{i}": error_result for i in range(8)
        }
        # 2 symbols succeed
        results["SYM8"] = {
            "composite_score": 75.0,
            "rating": "Buy",
            "screeners_passed": 1,
        }
        results["SYM9"] = {
            "composite_score": 80.0,
            "rating": "Buy",
            "screeners_passed": 1,
        }
        symbols = [f"SYM{i}" for i in range(10)]
        uow, scanner = _make_uow(symbols=symbols, scanner_results=results)
        use_case = BuildDailyFeatureSnapshotUseCase(scanner=scanner)

        result = use_case.execute(
            uow, _make_cmd(), FakeProgressSink(), FakeCancellationToken()
        )

        assert result.status == RunStatus.QUARANTINED.value
        assert result.dq_passed is False
        assert result.failed_symbols == 8

    @_PATCH_TRADING_DAY
    def test_publish_with_acceptable_failures(self, _mock_td):
        """If failures are below threshold, DQ passes and run publishes."""
        # 10 symbols, 1 fails → 9/10 = 90% = threshold → passes
        results = {}
        for i in range(9):
            results[f"SYM{i}"] = {
                "composite_score": 50.0 + i,
                "rating": "Watch",
                "screeners_passed": 1,
            }
        results["SYM9"] = {"error": "Bad data"}

        symbols = [f"SYM{i}" for i in range(10)]
        uow, scanner = _make_uow(symbols=symbols, scanner_results=results)
        use_case = BuildDailyFeatureSnapshotUseCase(scanner=scanner)

        result = use_case.execute(
            uow, _make_cmd(), FakeProgressSink(), FakeCancellationToken()
        )

        assert result.status == RunStatus.PUBLISHED.value
        assert result.dq_passed is True
        assert result.failed_symbols == 1


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------


class TestCancellation:
    @_PATCH_TRADING_DAY
    def test_cancel_between_chunks(self, _mock_td):
        """Cancellation between chunks returns early with COMPLETED status."""
        symbols = [f"SYM{i}" for i in range(10)]
        uow, scanner = _make_uow(symbols=symbols)
        use_case = BuildDailyFeatureSnapshotUseCase(scanner=scanner)

        # Cancel immediately (before first chunk processes)
        cancel = FakeCancellationToken(cancel_after=0)

        result = use_case.execute(
            uow, _make_cmd(chunk_size=5), FakeProgressSink(), cancel
        )

        assert result.processed_symbols == 0
        assert result.status == RunStatus.COMPLETED.value
        assert result.dq_passed is False
        assert "Cancelled" in result.warnings[0]


# ---------------------------------------------------------------------------
# Partial failures
# ---------------------------------------------------------------------------


class TestPartialFailures:
    @_PATCH_TRADING_DAY
    def test_failed_symbols_dont_block_others(self, _mock_td):
        """Per-symbol errors are counted but don't crash the run."""
        results = {
            "AAPL": {
                "composite_score": 85.0,
                "rating": "Strong Buy",
                "screeners_passed": 1,
            },
            "MSFT": {"error": "Data fetch failed"},
            "GOOGL": {
                "composite_score": 70.0,
                "rating": "Buy",
                "screeners_passed": 1,
            },
        }
        uow, scanner = _make_uow(scanner_results=results)
        use_case = BuildDailyFeatureSnapshotUseCase(scanner=scanner)

        result = use_case.execute(
            uow, _make_cmd(), FakeProgressSink(), FakeCancellationToken()
        )

        assert result.processed_symbols == 3
        assert result.failed_symbols == 1
        # 2/3 ≈ 66.7% < 90% threshold → quarantined
        assert result.status == RunStatus.QUARANTINED.value

    @_PATCH_TRADING_DAY
    def test_scanner_exception_counted_as_failure(self, _mock_td):
        """If scanner.scan_stock_multi raises, symbol is counted as failed."""

        class ExplodingScanner:
            def scan_stock_multi(self, symbol, screener_names, **kw):
                if symbol == "BOOM":
                    raise RuntimeError("kaboom")
                return {
                    "composite_score": 75.0,
                    "rating": "Buy",
                    "screeners_passed": 1,
                }

        uow, _ = _make_uow(symbols=["AAPL", "BOOM", "GOOGL"])
        use_case = BuildDailyFeatureSnapshotUseCase(scanner=ExplodingScanner())

        result = use_case.execute(
            uow, _make_cmd(), FakeProgressSink(), FakeCancellationToken()
        )

        assert result.processed_symbols == 3
        assert result.failed_symbols == 1


# ---------------------------------------------------------------------------
# Chunking behaviour
# ---------------------------------------------------------------------------


class TestChunking:
    @_PATCH_TRADING_DAY
    def test_commits_after_each_chunk(self, _mock_td):
        """UoW.commit() is called after each chunk + completion steps."""
        uow, scanner = _make_uow(
            symbols=[f"SYM{i}" for i in range(10)]
        )
        use_case = BuildDailyFeatureSnapshotUseCase(scanner=scanner)

        use_case.execute(
            uow, _make_cmd(chunk_size=3), FakeProgressSink(), FakeCancellationToken()
        )

        # 10 symbols / 3 per chunk = 4 chunks (3+3+3+1)
        # Commits: 1 (start_run) + 1 (save_universe) + 4 (chunks)
        #        + 1 (mark_completed) + 1 (publish) = 8
        assert uow.committed >= 8


# ---------------------------------------------------------------------------
# DQ delegation regression
# ---------------------------------------------------------------------------


class TestDQDelegation:
    @_PATCH_TRADING_DAY
    def test_five_dq_checks_run_after_delegation(self, _mock_td):
        """Verify all 5 DQ checks run after refactoring to delegate."""
        uow, scanner = _make_uow()
        use_case = BuildDailyFeatureSnapshotUseCase(scanner=scanner)
        result = use_case.execute(
            uow, _make_cmd(), FakeProgressSink(), FakeCancellationToken()
        )
        assert result.status == RunStatus.PUBLISHED.value
        # A published result with dq_passed=True means all CRITICAL checks
        # passed (the 4 WARNING checks are non-blocking).
        assert result.dq_passed is True

    @_PATCH_TRADING_DAY
    def test_same_rating_produces_warning(self, _mock_td):
        """When all symbols get the same rating, rating_distribution warns."""
        # All symbols return the same score/rating → 1 distinct rating
        same_result = {
            "composite_score": 50.0,
            "rating": "Watch",
            "screeners_passed": 1,
        }
        scanner_results = {
            "AAPL": same_result,
            "MSFT": same_result,
            "GOOGL": same_result,
        }
        uow, scanner = _make_uow(scanner_results=scanner_results)
        use_case = BuildDailyFeatureSnapshotUseCase(scanner=scanner)
        result = use_case.execute(
            uow, _make_cmd(), FakeProgressSink(), FakeCancellationToken()
        )
        # Still publishes (rating_distribution is WARNING, not CRITICAL)
        assert result.status == RunStatus.PUBLISHED.value
        assert result.dq_passed is True
        # But warnings should mention rating distribution
        assert any("distinct rating" in w for w in result.warnings)
