"""Unit tests for RunBulkScanUseCase using in-memory fakes.

No DB, no Redis, no Celery — pure domain logic testing.
"""

from __future__ import annotations

import pytest

from app.domain.common.errors import EntityNotFoundError
from app.domain.scanning.models import ScanStatus
from app.use_cases.scanning.run_bulk_scan import (
    RunBulkScanCommand,
    RunBulkScanResult,
    RunBulkScanUseCase,
)

from tests.unit.use_cases.conftest import (
    FakeCancellationToken,
    FakeProgressSink,
    FakeScan,
    FakeScanRepository,
    FakeScanResultRepository,
    FakeScanner,
    FakeStockDataProvider,
    FakeUnitOfWork,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_scan(
    scan_id: str = "test-scan-001",
    screener_types: list[str] | None = None,
    composite_method: str = "weighted_average",
) -> FakeScan:
    return FakeScan(
        scan_id=scan_id,
        screener_types=screener_types or ["minervini"],
        composite_method=composite_method,
    )


def _make_use_case(scanner: FakeScanner | None = None) -> RunBulkScanUseCase:
    return RunBulkScanUseCase(scanner=scanner or FakeScanner())


class _SpyDataProvider(FakeStockDataProvider):
    """FakeStockDataProvider that records keyword args passed to prepare_data_bulk."""

    def __init__(self):
        super().__init__()
        self.captured_bulk_kwargs: list[dict] = []

    def prepare_data_bulk(self, symbols, requirements, **kwargs):
        self.captured_bulk_kwargs.append(kwargs)
        return super().prepare_data_bulk(symbols, requirements, **kwargs)


class _BulkAwareScanner:
    """Minimal scanner that advertises merged-requirements support so the
    use case exercises the prepare_data_bulk path."""

    def get_merged_requirements(self, screener_names, criteria=None):
        return {"needs": "all"}

    def scan_stock_multi(self, symbol, screener_names, **_):
        return {
            "composite_score": 80.0,
            "rating": "Buy",
            "passes_template": True,
            "current_price": 100.0,
        }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRunBulkScanHappyPath:
    """The normal completion path."""

    def test_scans_all_symbols_and_completes(self):
        scan_repo = FakeScanRepository()
        scan_repo.scans["s1"] = _make_scan("s1")
        result_repo = FakeScanResultRepository()
        uow = FakeUnitOfWork(scans=scan_repo, scan_results=result_repo)

        scanner = FakeScanner()
        uc = _make_use_case(scanner)
        progress = FakeProgressSink()
        cancel = FakeCancellationToken()

        cmd = RunBulkScanCommand(
            scan_id="s1",
            symbols=["AAPL", "MSFT", "GOOG"],
            chunk_size=2,
        )

        result = uc.execute(uow, cmd, progress, cancel)

        assert result.status == ScanStatus.COMPLETED.value
        assert result.total_scanned == 3
        assert result.passed == 3
        assert result.failed == 0
        assert result.scan_id == "s1"

    def test_scanner_called_with_correct_params(self):
        scan_repo = FakeScanRepository()
        scan_repo.scans["s1"] = _make_scan(
            "s1",
            screener_types=["minervini", "canslim"],
            composite_method="maximum",
        )
        uow = FakeUnitOfWork(scans=scan_repo)
        scanner = FakeScanner()
        uc = _make_use_case(scanner)

        cmd = RunBulkScanCommand(
            scan_id="s1",
            symbols=["AAPL"],
            criteria={"include_vcp": True},
        )
        uc.execute(uow, cmd, FakeProgressSink(), FakeCancellationToken())

        assert scanner.calls == ["AAPL"]

    def test_results_persisted_per_chunk(self):
        scan_repo = FakeScanRepository()
        scan_repo.scans["s1"] = _make_scan("s1")
        result_repo = FakeScanResultRepository()
        uow = FakeUnitOfWork(scans=scan_repo, scan_results=result_repo)

        uc = _make_use_case()
        cmd = RunBulkScanCommand(
            scan_id="s1",
            symbols=["A", "B", "C", "D", "E"],
            chunk_size=2,
        )
        uc.execute(uow, cmd, FakeProgressSink(), FakeCancellationToken())

        # 5 symbols -> 3 chunks (2+2+1) -> 5 results persisted
        assert result_repo.count_by_scan_id("s1") == 5

    def test_progress_events_emitted_per_chunk(self):
        scan_repo = FakeScanRepository()
        scan_repo.scans["s1"] = _make_scan("s1")
        uow = FakeUnitOfWork(scans=scan_repo)

        progress = FakeProgressSink()
        cmd = RunBulkScanCommand(
            scan_id="s1",
            symbols=["A", "B", "C", "D", "E"],
            chunk_size=2,
        )
        _make_use_case().execute(uow, cmd, progress, FakeCancellationToken())

        # 3 chunks -> 3 progress events
        assert len(progress.events) == 3
        assert progress.events[-1].current == 5
        assert progress.events[-1].total == 5

    def test_uses_bulk_data_prep_when_scanner_supports_merged_requirements(self):
        scan_repo = FakeScanRepository()
        scan_repo.scans["s1"] = _make_scan("s1", screener_types=["minervini"])
        uow = FakeUnitOfWork(scans=scan_repo)
        data_provider = FakeStockDataProvider()

        class BulkAwareScanner:
            def __init__(self):
                self.calls = []
                self.requirements_calls = []

            def get_merged_requirements(self, screener_names, criteria=None):
                self.requirements_calls.append((tuple(screener_names), criteria))
                return {"needs": "price+fundamentals"}

            def scan_stock_multi(
                self,
                symbol,
                screener_names,
                criteria=None,
                composite_method="weighted_average",
                pre_merged_requirements=None,
                pre_fetched_data=None,
            ):
                self.calls.append(
                    {
                        "symbol": symbol,
                        "pre_merged_requirements": pre_merged_requirements,
                        "pre_fetched_data": pre_fetched_data,
                    }
                )
                return {
                    "composite_score": 82.0,
                    "rating": "Buy",
                    "passes_template": True,
                    "current_price": 100.0,
                }

        scanner = BulkAwareScanner()
        uc = RunBulkScanUseCase(scanner=scanner, data_provider=data_provider)
        result = uc.execute(
            uow,
            RunBulkScanCommand(scan_id="s1", symbols=["AAPL", "MSFT"], chunk_size=2),
            FakeProgressSink(),
            FakeCancellationToken(),
        )

        assert result.status == ScanStatus.COMPLETED.value
        assert data_provider.prepare_calls == ["AAPL", "MSFT"]
        assert scanner.requirements_calls == [(("minervini",), {})]
        assert len(scanner.calls) == 2
        assert all(
            call["pre_merged_requirements"] == {"needs": "price+fundamentals"}
            for call in scanner.calls
        )
        assert all(call["pre_fetched_data"] is not None for call in scanner.calls)

    def test_cache_only_flag_propagates_to_bulk_data_fetch(self):
        """cache_only=True on the command must disable live-fetch fallback inside
        prepare_data_bulk. This is the teeth behind the API-level staleness gate:
        even if a scan slips past the 409, the data layer refuses to hit yfinance.
        """
        scan_repo = FakeScanRepository()
        scan_repo.scans["s1"] = _make_scan("s1", screener_types=["minervini"])
        uow = FakeUnitOfWork(scans=scan_repo)

        data_provider = _SpyDataProvider()
        uc = RunBulkScanUseCase(scanner=_BulkAwareScanner(), data_provider=data_provider)

        uc.execute(
            uow,
            RunBulkScanCommand(
                scan_id="s1",
                symbols=["AAPL", "MSFT"],
                chunk_size=10,
                cache_only=True,
            ),
            FakeProgressSink(),
            FakeCancellationToken(),
        )

        assert data_provider.captured_bulk_kwargs, "prepare_data_bulk was not invoked"
        call = data_provider.captured_bulk_kwargs[0]
        assert call["batch_only_prices"] is True
        assert call["batch_only_fundamentals"] is True

    def test_cache_only_skips_per_symbol_fallback_when_bulk_fetch_fails(self):
        """When prepare_data_bulk raises AND cache_only=True, the per-symbol
        scan path must NOT be invoked — otherwise the orchestrator would call
        prepare_data() which falls through to yfinance on cache miss, defeating
        the cache-only guarantee for manual scans.
        """
        scan_repo = FakeScanRepository()
        scan_repo.scans["s1"] = _make_scan("s1", screener_types=["minervini"])
        uow = FakeUnitOfWork(scans=scan_repo)

        class FailingBulkProvider(_SpyDataProvider):
            def prepare_data_bulk(self, symbols, requirements, **kwargs):
                self.captured_bulk_kwargs.append(kwargs)
                raise RuntimeError("simulated bulk fetch failure")

        scan_calls: list[str] = []

        class TrackingScanner(_BulkAwareScanner):
            def scan_stock_multi(self, symbol, screener_names, **kwargs):
                scan_calls.append(symbol)
                return super().scan_stock_multi(symbol, screener_names, **kwargs)

        data_provider = FailingBulkProvider()
        uc = RunBulkScanUseCase(scanner=TrackingScanner(), data_provider=data_provider)

        result = uc.execute(
            uow,
            RunBulkScanCommand(
                scan_id="s1",
                symbols=["AAPL", "MSFT", "GOOG"],
                chunk_size=10,
                cache_only=True,
            ),
            FakeProgressSink(),
            FakeCancellationToken(),
        )

        assert data_provider.captured_bulk_kwargs, "bulk fetch was not attempted"
        assert scan_calls == [], (
            "scan_stock_multi must not be called when bulk fails in cache-only mode "
            "(per-symbol prepare_data would re-enter live yfinance fallback)"
        )
        assert result.failed == 3
        assert result.passed == 0

    def test_cache_only_skips_per_symbol_when_get_merged_requirements_raises(self):
        """When get_merged_requirements() raises, merged_requirements stays None
        and the bulk fetch block is never entered. The cache-only invariant must
        still hold — symbols must NOT flow through the per-symbol prepare_data()
        path which can still hit live yfinance/Finviz.
        """
        scan_repo = FakeScanRepository()
        scan_repo.scans["s1"] = _make_scan("s1", screener_types=["minervini"])
        uow = FakeUnitOfWork(scans=scan_repo)

        scan_calls: list[str] = []
        bulk_calls: list[list[str]] = []

        class FailingMergeScanner:
            def get_merged_requirements(self, screener_names, criteria=None):
                raise RuntimeError("simulated merge failure")

            def scan_stock_multi(self, symbol, screener_names, **kwargs):
                scan_calls.append(symbol)
                return {"composite_score": 0.0, "rating": "Buy", "passes_template": False}

        class TrackingProvider(_SpyDataProvider):
            def prepare_data_bulk(self, symbols, requirements, **kwargs):
                bulk_calls.append(list(symbols))
                return super().prepare_data_bulk(symbols, requirements, **kwargs)

        uc = RunBulkScanUseCase(scanner=FailingMergeScanner(), data_provider=TrackingProvider())
        result = uc.execute(
            uow,
            RunBulkScanCommand(
                scan_id="s1",
                symbols=["AAPL", "MSFT"],
                chunk_size=10,
                cache_only=True,
            ),
            FakeProgressSink(),
            FakeCancellationToken(),
        )

        assert bulk_calls == [], "bulk fetch should be skipped when merged_requirements is None"
        assert scan_calls == [], (
            "scan_stock_multi must not be called when cache_only=True and bulk path was never attempted"
        )
        assert result.failed == 2
        assert result.passed == 0

    def test_bulk_fetch_failure_falls_back_to_per_symbol_when_not_cache_only(self):
        """When cache_only=False (default), bulk failure SHOULD fall back to
        per-symbol scanning — preserves legacy behavior for any non-manual caller.
        """
        scan_repo = FakeScanRepository()
        scan_repo.scans["s1"] = _make_scan("s1", screener_types=["minervini"])
        uow = FakeUnitOfWork(scans=scan_repo)

        class FailingBulkProvider(_SpyDataProvider):
            def prepare_data_bulk(self, symbols, requirements, **kwargs):
                self.captured_bulk_kwargs.append(kwargs)
                raise RuntimeError("simulated bulk fetch failure")

        scan_calls: list[str] = []

        class TrackingScanner(_BulkAwareScanner):
            def scan_stock_multi(self, symbol, screener_names, **kwargs):
                scan_calls.append(symbol)
                return super().scan_stock_multi(symbol, screener_names, **kwargs)

        uc = RunBulkScanUseCase(scanner=TrackingScanner(), data_provider=FailingBulkProvider())
        uc.execute(
            uow,
            RunBulkScanCommand(scan_id="s1", symbols=["AAPL", "MSFT"], chunk_size=10),
            FakeProgressSink(),
            FakeCancellationToken(),
        )

        assert scan_calls == ["AAPL", "MSFT"]

    def test_cache_only_false_leaves_live_fetch_enabled(self):
        scan_repo = FakeScanRepository()
        scan_repo.scans["s1"] = _make_scan("s1", screener_types=["minervini"])
        uow = FakeUnitOfWork(scans=scan_repo)

        data_provider = _SpyDataProvider()
        uc = RunBulkScanUseCase(scanner=_BulkAwareScanner(), data_provider=data_provider)
        uc.execute(
            uow,
            RunBulkScanCommand(scan_id="s1", symbols=["AAPL"]),
            FakeProgressSink(),
            FakeCancellationToken(),
        )

        assert data_provider.captured_bulk_kwargs[0]["batch_only_prices"] is False
        assert data_provider.captured_bulk_kwargs[0]["batch_only_fundamentals"] is False

    def test_scan_status_transitions(self):
        scan_repo = FakeScanRepository()
        scan_repo.scans["s1"] = _make_scan("s1")
        uow = FakeUnitOfWork(scans=scan_repo)

        cmd = RunBulkScanCommand(scan_id="s1", symbols=["AAPL"])
        _make_use_case().execute(uow, cmd, FakeProgressSink(), FakeCancellationToken())

        # Should transition: running -> completed
        assert scan_repo.status_history == [
            ("s1", "running"),
            ("s1", "completed"),
        ]

    def test_scan_not_found_raises(self):
        uow = FakeUnitOfWork()

        cmd = RunBulkScanCommand(scan_id="nonexistent", symbols=["AAPL"])
        with pytest.raises(EntityNotFoundError):
            _make_use_case().execute(
                uow, cmd, FakeProgressSink(), FakeCancellationToken()
            )


class TestCancellation:
    """Cancellation token stops processing between chunks."""

    def test_cancellation_stops_after_first_chunk(self):
        scan_repo = FakeScanRepository()
        scan_repo.scans["s1"] = _make_scan("s1")
        result_repo = FakeScanResultRepository()
        uow = FakeUnitOfWork(scans=scan_repo, scan_results=result_repo)

        # Cancel after 1 check (i.e. after the first chunk processes
        # and the second chunk's cancellation gate fires)
        cancel = FakeCancellationToken(cancel_after=1)
        scanner = FakeScanner()

        cmd = RunBulkScanCommand(
            scan_id="s1",
            symbols=["A", "B", "C", "D"],
            chunk_size=2,
        )
        result = _make_use_case(scanner).execute(
            uow, cmd, FakeProgressSink(), cancel
        )

        assert result.status == ScanStatus.CANCELLED.value
        # First chunk (A, B) processed; second chunk cancelled before processing
        assert result.total_scanned == 2
        assert len(scanner.calls) == 2

    def test_cancelled_scan_status_set(self):
        scan_repo = FakeScanRepository()
        scan_repo.scans["s1"] = _make_scan("s1")
        uow = FakeUnitOfWork(scans=scan_repo)

        cancel = FakeCancellationToken(cancel_after=1)
        cmd = RunBulkScanCommand(
            scan_id="s1",
            symbols=["A", "B", "C", "D"],
            chunk_size=2,
        )
        _make_use_case().execute(uow, cmd, FakeProgressSink(), cancel)

        assert scan_repo.scans["s1"].status == ScanStatus.CANCELLED.value


class TestResume:
    """Checkpoint recovery skips already-processed symbols."""

    def test_resume_skips_already_processed(self):
        scan_repo = FakeScanRepository()
        scan_repo.scans["s1"] = _make_scan("s1")

        # Pre-populate 2 existing results (simulating a previous partial run)
        result_repo = FakeScanResultRepository()
        result_repo._persisted_results = [
            ("s1", "AAPL", {}),
            ("s1", "MSFT", {}),
        ]
        uow = FakeUnitOfWork(scans=scan_repo, scan_results=result_repo)

        scanner = FakeScanner()
        cmd = RunBulkScanCommand(
            scan_id="s1",
            symbols=["AAPL", "MSFT", "GOOG", "AMZN"],
            chunk_size=10,
        )
        result = _make_use_case(scanner).execute(
            uow, cmd, FakeProgressSink(), FakeCancellationToken()
        )

        # Only GOOG and AMZN should be scanned (first 2 skipped)
        assert scanner.calls == ["GOOG", "AMZN"]
        assert result.total_scanned == 4  # total including resumed
        assert result.status == ScanStatus.COMPLETED.value

    def test_resume_preserves_passed_count(self):
        """On resume, the passed counter should include prior passes."""
        scan_repo = FakeScanRepository()
        scan = _make_scan("s1")
        scan.passed_stocks = 5  # 5 passed in the previous partial run
        scan_repo.scans["s1"] = scan

        result_repo = FakeScanResultRepository()
        result_repo._persisted_results = [("s1", f"SYM{i}", {}) for i in range(10)]
        uow = FakeUnitOfWork(scans=scan_repo, scan_results=result_repo)

        # Scanner returns 2 more passing symbols
        scanner = FakeScanner()
        cmd = RunBulkScanCommand(
            scan_id="s1",
            symbols=[f"SYM{i}" for i in range(12)],  # 10 done + 2 new
            chunk_size=50,
        )
        result = _make_use_case(scanner).execute(
            uow, cmd, FakeProgressSink(), FakeCancellationToken()
        )

        # 5 from before + 2 new passes = 7
        assert result.passed == 7
        assert scan_repo.scans["s1"].passed_stocks == 7

    def test_resume_with_all_done_completes_immediately(self):
        scan_repo = FakeScanRepository()
        scan_repo.scans["s1"] = _make_scan("s1")

        result_repo = FakeScanResultRepository()
        result_repo._persisted_results = [
            ("s1", "AAPL", {}),
            ("s1", "MSFT", {}),
        ]
        uow = FakeUnitOfWork(scans=scan_repo, scan_results=result_repo)

        scanner = FakeScanner()
        cmd = RunBulkScanCommand(
            scan_id="s1",
            symbols=["AAPL", "MSFT"],
            chunk_size=10,
        )
        result = _make_use_case(scanner).execute(
            uow, cmd, FakeProgressSink(), FakeCancellationToken()
        )

        assert scanner.calls == []
        assert result.status == ScanStatus.COMPLETED.value


class TestErrorHandling:
    """Per-stock errors and fatal errors."""

    def test_per_stock_error_counted_as_failed(self):
        scan_repo = FakeScanRepository()
        scan_repo.scans["s1"] = _make_scan("s1")

        # MSFT returns an error result
        scanner = FakeScanner(
            results={
                "AAPL": {"composite_score": 80, "rating": "Buy", "passes_template": True},
                "MSFT": {"error": "No data available"},
                "GOOG": {"composite_score": 70, "rating": "Watch", "passes_template": False},
            }
        )

        uow = FakeUnitOfWork(scans=scan_repo)
        cmd = RunBulkScanCommand(scan_id="s1", symbols=["AAPL", "MSFT", "GOOG"])
        result = _make_use_case(scanner).execute(
            uow, cmd, FakeProgressSink(), FakeCancellationToken()
        )

        assert result.passed == 1  # AAPL only
        assert result.failed == 1  # MSFT
        assert result.total_scanned == 3
        assert result.status == ScanStatus.COMPLETED.value

    def test_scanner_exception_counted_as_failed(self):
        scan_repo = FakeScanRepository()
        scan_repo.scans["s1"] = _make_scan("s1")

        class ExplodingScanner:
            calls = []

            def scan_stock_multi(
                self, symbol, screener_names, criteria=None, composite_method="weighted_average"
            ):
                self.calls.append(symbol)
                if symbol == "MSFT":
                    raise RuntimeError("boom")
                return {"composite_score": 80, "rating": "Buy", "passes_template": True}

        scanner = ExplodingScanner()
        uow = FakeUnitOfWork(scans=scan_repo)
        cmd = RunBulkScanCommand(scan_id="s1", symbols=["AAPL", "MSFT", "GOOG"])
        result = RunBulkScanUseCase(scanner=scanner).execute(
            uow, cmd, FakeProgressSink(), FakeCancellationToken()
        )

        assert result.failed == 1
        assert result.passed == 2
        assert result.status == ScanStatus.COMPLETED.value

    def test_insufficient_history_result_is_persisted_in_legacy_scan_results(self):
        scan_repo = FakeScanRepository()
        scan_repo.scans["s1"] = _make_scan("s1")
        result_repo = FakeScanResultRepository()
        uow = FakeUnitOfWork(scans=scan_repo, scan_results=result_repo)

        scanner = FakeScanner(
            results={
                "AAPL": {
                    "composite_score": None,
                    "rating": "Insufficient Data",
                    "reason": "Insufficient price history",
                    "result_status": "insufficient_history",
                    "scan_mode": "ipo_weighted",
                    "is_scannable": False,
                    "rs_rating_1m": 50.0,
                    "adr_percent": 3.5,
                    "price_change_1d": 2.1,
                },
            }
        )

        cmd = RunBulkScanCommand(scan_id="s1", symbols=["AAPL"])
        result = _make_use_case(scanner).execute(
            uow, cmd, FakeProgressSink(), FakeCancellationToken()
        )

        assert result.failed == 0
        assert result.passed == 0
        assert result_repo.count_by_scan_id("s1") == 1
        _, symbol, persisted = result_repo._persisted_results[0]
        assert symbol == "AAPL"
        assert persisted["composite_score"] is None
        assert persisted["scan_mode"] == "ipo_weighted"
        assert persisted["is_scannable"] is False
        assert persisted["rs_rating_1m"] == 50.0
        assert persisted["adr_percent"] == 3.5
        assert persisted["price_change_1d"] == 2.1

    def test_fatal_error_marks_scan_failed(self):
        """A fatal error (e.g. commit failure) marks the scan as FAILED."""
        scan_repo = FakeScanRepository()
        scan_repo.scans["s1"] = _make_scan("s1")

        # Simulate a fatal error during result persistence
        class ExplodingResultRepo(FakeScanResultRepository):
            def persist_orchestrator_results(self, scan_id, results):
                raise RuntimeError("disk full")

        uow = FakeUnitOfWork(
            scans=scan_repo,
            scan_results=ExplodingResultRepo(),
        )
        cmd = RunBulkScanCommand(scan_id="s1", symbols=["AAPL"])

        with pytest.raises(RuntimeError, match="disk full"):
            _make_use_case().execute(
                uow, cmd, FakeProgressSink(), FakeCancellationToken()
            )

        assert scan_repo.scans["s1"].status == ScanStatus.FAILED.value


class TestProgressEvents:
    """Progress event contents are accurate."""

    def test_progress_tracks_passed_and_failed(self):
        scan_repo = FakeScanRepository()
        scan_repo.scans["s1"] = _make_scan("s1")

        scanner = FakeScanner(
            results={
                "AAPL": {"composite_score": 80, "rating": "Buy", "passes_template": True},
                "MSFT": {"error": "No data"},
            }
        )
        progress = FakeProgressSink()
        uow = FakeUnitOfWork(scans=scan_repo)

        cmd = RunBulkScanCommand(scan_id="s1", symbols=["AAPL", "MSFT"], chunk_size=10)
        _make_use_case(scanner).execute(uow, cmd, progress, FakeCancellationToken())

        assert len(progress.events) == 1
        evt = progress.events[0]
        assert evt.current == 2
        assert evt.total == 2
        assert evt.passed == 1
        assert evt.failed == 1
        assert evt.throughput is not None

    def test_no_progress_events_on_empty_resume(self):
        scan_repo = FakeScanRepository()
        scan_repo.scans["s1"] = _make_scan("s1")
        result_repo = FakeScanResultRepository()
        result_repo._persisted_results = [("s1", "A", {})]
        uow = FakeUnitOfWork(scans=scan_repo, scan_results=result_repo)

        progress = FakeProgressSink()
        cmd = RunBulkScanCommand(scan_id="s1", symbols=["A"])
        _make_use_case().execute(uow, cmd, progress, FakeCancellationToken())

        # All symbols already processed -> no chunks -> no progress events
        assert len(progress.events) == 0


class TestInputValidation:
    """Command-level validation."""

    def test_chunk_size_zero_raises(self):
        with pytest.raises(ValueError, match="chunk_size must be >= 1"):
            RunBulkScanCommand(scan_id="s1", symbols=["A"], chunk_size=0)

    def test_chunk_size_negative_raises(self):
        with pytest.raises(ValueError, match="chunk_size must be >= 1"):
            RunBulkScanCommand(scan_id="s1", symbols=["A"], chunk_size=-5)
