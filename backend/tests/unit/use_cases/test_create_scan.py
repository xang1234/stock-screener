"""Unit tests for CreateScanUseCase — pure in-memory, no infrastructure."""

import pytest

from app.domain.common.errors import ValidationError
from app.use_cases.scanning.create_scan import (
    ActiveScanConflictError,
    CreateScanCommand,
    CreateScanResult,
    CreateScanUseCase,
    StaleMarketDataError,
)
from app.domain.scanning.errors import SingleActiveScanViolation

from tests.unit.use_cases.conftest import (
    FakeFeatureRunRepository,
    FakeScanRepository,
    FakeTaskDispatcher,
    FakeUnitOfWork,
    FakeUniverseRepository,
)


# ── Helpers ──────────────────────────────────────────────────────────────


def _make_command(**overrides) -> CreateScanCommand:
    defaults = dict(
        universe_def="all",
        universe_label="All Stocks",
        universe_key="all",
        universe_type="all",
        screeners=["minervini"],
        composite_method="weighted_average",
    )
    defaults.update(overrides)
    return CreateScanCommand(**defaults)


def _make_uow(symbols: list[str] | None = None) -> FakeUnitOfWork:
    """Build a UoW with a configurable universe."""
    return FakeUnitOfWork(universe=FakeUniverseRepository(symbols or []))


# ── Tests ────────────────────────────────────────────────────────────────


class TestCreateScanUseCase:
    """Core business logic for scan creation."""

    def test_creates_scan_and_dispatches_task(self):
        uow = _make_uow(["AAPL", "MSFT", "GOOGL"])
        dispatcher = FakeTaskDispatcher()
        uc = CreateScanUseCase(dispatcher=dispatcher)

        result = uc.execute(uow, _make_command())

        assert result.is_duplicate is False
        assert result.status == "queued"
        assert result.total_stocks == 3
        assert len(result.scan_id) == 36  # UUID format
        assert len(dispatcher.dispatched) == 1
        assert dispatcher.dispatched[0][1] == ["AAPL", "MSFT", "GOOGL"]

    def test_scan_record_persisted_before_dispatch(self):
        uow = _make_uow(["AAPL"])
        dispatcher = FakeTaskDispatcher()
        uc = CreateScanUseCase(dispatcher=dispatcher)

        uc.execute(uow, _make_command())

        assert len(uow.scans.rows) == 1
        scan = uow.scans.rows[0]
        assert scan.status == "queued"
        assert scan.trigger_source == "manual"
        assert scan.total_stocks == 1
        assert scan.task_id == "fake-task-123"

    def test_stores_universe_metadata(self):
        uow = _make_uow(["AAPL"])
        dispatcher = FakeTaskDispatcher()
        uc = CreateScanUseCase(dispatcher=dispatcher)

        cmd = _make_command(
            universe_label="NYSE",
            universe_key="exchange:NYSE",
            universe_type="exchange",
            universe_exchange="NYSE",
        )
        uc.execute(uow, cmd)

        scan = uow.scans.rows[0]
        assert scan.universe == "NYSE"
        assert scan.universe_key == "exchange:NYSE"
        assert scan.universe_type == "exchange"
        assert scan.universe_exchange == "NYSE"

    def test_stores_market_universe_metadata(self):
        uow = _make_uow(["0700.HK"])
        dispatcher = FakeTaskDispatcher()
        uc = CreateScanUseCase(dispatcher=dispatcher)

        cmd = _make_command(
            universe_label="Hong Kong Market",
            universe_key="market:HK",
            universe_type="market",
            universe_market="HK",
        )
        uc.execute(uow, cmd)

        scan = uow.scans.rows[0]
        assert scan.universe == "Hong Kong Market"
        assert scan.universe_key == "market:HK"
        assert scan.universe_type == "market"
        assert scan.universe_market == "HK"

    def test_empty_universe_raises_validation_error(self):
        uow = _make_uow([])
        dispatcher = FakeTaskDispatcher()
        uc = CreateScanUseCase(dispatcher=dispatcher)

        with pytest.raises(ValidationError, match="No symbols found"):
            uc.execute(uow, _make_command())

        assert len(dispatcher.dispatched) == 0

    def test_rejects_second_active_scan(self):
        uow = _make_uow(["AAPL", "MSFT"])
        uow.scans.create(
            scan_id="scan-active",
            status="running",
            total_stocks=2,
            trigger_source="manual",
        )
        dispatcher = FakeTaskDispatcher()
        uc = CreateScanUseCase(dispatcher=dispatcher)

        with pytest.raises(ActiveScanConflictError) as exc_info:
            uc.execute(uow, _make_command())

        assert exc_info.value.active_scan.scan_id == "scan-active"
        assert len(dispatcher.dispatched) == 0

    def test_dispatch_failure_marks_scan_failed(self):
        uow = _make_uow(["AAPL"])
        dispatcher = FakeTaskDispatcher(should_fail=True)
        uc = CreateScanUseCase(dispatcher=dispatcher)

        with pytest.raises(RuntimeError, match="Celery is down"):
            uc.execute(uow, _make_command())

        scan = uow.scans.rows[0]
        assert scan.status == "failed"

    def test_unique_constraint_violation_maps_to_active_scan_conflict(self):
        class _RaceyScanRepository(FakeScanRepository):
            def create(self, *, scan_id: str, **fields):
                super().create(
                    scan_id="scan-active",
                    status="queued",
                    total_stocks=2,
                    trigger_source="manual",
                )
                raise SingleActiveScanViolation("duplicate active scan")

        uow = FakeUnitOfWork(
            scans=_RaceyScanRepository(),
            universe=FakeUniverseRepository(["AAPL", "MSFT"]),
        )
        dispatcher = FakeTaskDispatcher()
        uc = CreateScanUseCase(dispatcher=dispatcher)

        with pytest.raises(ActiveScanConflictError) as exc_info:
            uc.execute(uow, _make_command())

        assert exc_info.value.active_scan.scan_id == "scan-active"
        assert uow.rolled_back >= 1
        assert len(dispatcher.dispatched) == 0

    def test_commits_at_least_twice_on_success(self):
        """First commit persists scan, second stores task_id."""
        uow = _make_uow(["AAPL"])
        dispatcher = FakeTaskDispatcher()
        uc = CreateScanUseCase(dispatcher=dispatcher)

        uc.execute(uow, _make_command())

        assert uow.committed >= 2


class TestIdempotency:
    """Idempotency key prevents duplicate scans."""

    def test_duplicate_key_returns_existing_scan(self):
        uow = _make_uow(["AAPL"])
        dispatcher = FakeTaskDispatcher()
        uc = CreateScanUseCase(dispatcher=dispatcher)

        cmd = _make_command(idempotency_key="abc-123")
        result1 = uc.execute(uow, cmd)
        assert result1.is_duplicate is False

        result2 = uc.execute(uow, cmd)
        assert result2.is_duplicate is True
        assert result2.scan_id == result1.scan_id

        assert len(uow.scans.rows) == 1
        assert len(dispatcher.dispatched) == 1

    def test_different_keys_create_separate_scans(self):
        uow = _make_uow(["AAPL"])
        dispatcher = FakeTaskDispatcher()
        uc = CreateScanUseCase(dispatcher=dispatcher)

        result1 = uc.execute(uow, _make_command(idempotency_key="key-1"))
        uow.scans.update_status(result1.scan_id, "completed")
        result2 = uc.execute(uow, _make_command(idempotency_key="key-2"))

        assert result1.scan_id != result2.scan_id
        assert len(uow.scans.rows) == 2
        assert len(dispatcher.dispatched) == 2

    def test_no_key_always_creates_new_scan(self):
        uow = _make_uow(["AAPL"])
        dispatcher = FakeTaskDispatcher()
        uc = CreateScanUseCase(dispatcher=dispatcher)

        result1 = uc.execute(uow, _make_command(idempotency_key=None))
        uow.scans.update_status(result1.scan_id, "completed")
        result2 = uc.execute(uow, _make_command(idempotency_key=None))

        assert result1.scan_id != result2.scan_id
        assert len(uow.scans.rows) == 2


class TestFreshnessChecker:
    """Staleness gate runs inside the use case, after idempotency + symbol resolution."""

    _STALE_DETAIL = {
        "code": "market_data_stale",
        "message": "stale test",
        "stale_markets": [{"market": "US", "uncovered_symbols": 1}],
    }

    def test_stale_universe_raises_stale_market_data_error(self):
        uow = _make_uow(["AAPL", "MSFT"])
        dispatcher = FakeTaskDispatcher()
        uc = CreateScanUseCase(
            dispatcher=dispatcher,
            freshness_checker=lambda _symbols: self._STALE_DETAIL,
        )

        with pytest.raises(StaleMarketDataError) as exc_info:
            uc.execute(uow, _make_command())

        assert exc_info.value.to_dict()["code"] == "market_data_stale"
        assert len(dispatcher.dispatched) == 0
        assert len(uow.scans.rows) == 0

    def test_fresh_universe_proceeds_normally(self):
        uow = _make_uow(["AAPL"])
        dispatcher = FakeTaskDispatcher()
        uc = CreateScanUseCase(
            dispatcher=dispatcher,
            freshness_checker=lambda _symbols: None,
        )

        result = uc.execute(uow, _make_command())

        assert result.status == "queued"
        assert len(dispatcher.dispatched) == 1

    def test_idempotent_retry_bypasses_staleness_gate(self):
        """Thread 3 (Codex P2): a retried request with a known idempotency_key
        must return the existing scan even if market data is now stale.
        """
        uow = _make_uow(["AAPL"])
        dispatcher = FakeTaskDispatcher()
        calls: list[list[str]] = []

        def checker(symbols):
            calls.append(list(symbols))
            return self._STALE_DETAIL

        # First call: freshness_checker returns None so the scan is created.
        fresh_uc = CreateScanUseCase(
            dispatcher=dispatcher,
            freshness_checker=lambda _s: None,
        )
        result1 = fresh_uc.execute(uow, _make_command(idempotency_key="retry-key"))

        # Second call: checker would return stale, but idempotency short-circuits first.
        stale_uc = CreateScanUseCase(
            dispatcher=dispatcher,
            freshness_checker=checker,
        )
        result2 = stale_uc.execute(uow, _make_command(idempotency_key="retry-key"))

        assert result2.is_duplicate is True
        assert result2.scan_id == result1.scan_id
        assert calls == [], "freshness checker must not run for idempotent retries"

    def test_checker_receives_resolved_symbols_not_universe_def(self):
        """Thread 1 (Codex P1): the checker gets the symbol list the scan will
        actually process, not the whole market's active set.
        """
        uow = _make_uow(["AAPL", "MSFT", "GOOGL"])
        dispatcher = FakeTaskDispatcher()
        captured: list[list[str]] = []

        def checker(symbols):
            captured.append(list(symbols))
            return None

        uc = CreateScanUseCase(dispatcher=dispatcher, freshness_checker=checker)
        uc.execute(uow, _make_command())

        assert captured == [["AAPL", "MSFT", "GOOGL"]]

    def test_freshness_checker_exception_fails_closed(self):
        """Round 6 CodeRabbit: if the checker itself raises (transient DB
        error, etc.) the exception must propagate — we do not silently skip
        the gate. Fail-closed is intentional for a safety check.
        """
        uow = _make_uow(["AAPL"])
        dispatcher = FakeTaskDispatcher()

        def checker(_symbols):
            raise RuntimeError("transient DB error")

        uc = CreateScanUseCase(dispatcher=dispatcher, freshness_checker=checker)

        with pytest.raises(RuntimeError, match="transient DB error"):
            uc.execute(uow, _make_command())

        assert len(dispatcher.dispatched) == 0

    def test_no_checker_skips_freshness_gate_entirely(self):
        """Round 4 Codex P1: internal callers (e.g., bootstrap scans) opt out
        of the gate by omitting the checker. They must proceed even when the
        data they're about to create doesn't exist yet.
        """
        uow = _make_uow(["AAPL"])
        dispatcher = FakeTaskDispatcher()
        uc = CreateScanUseCase(dispatcher=dispatcher, freshness_checker=None)

        result = uc.execute(uow, _make_command())

        assert result.status == "queued"
        assert len(dispatcher.dispatched) == 1


class TestFeatureRunBinding:
    """Scan binds only when an exact published feature run matches."""

    def test_exact_match_completes_instantly_without_dispatch(self):
        """Exact ALL-universe match reuses a published snapshot."""
        from datetime import date
        from app.domain.feature_store.models import RunStats, RunType
        from app.domain.scanning.signature import (
            build_scan_signature_payload,
            hash_scan_signature,
            hash_universe_symbols,
        )

        feature_runs = FakeFeatureRunRepository()
        symbols = ["AAPL"]
        signature = build_scan_signature_payload(
            universe_type="all",
            screeners=["minervini"],
            composite_method="weighted_average",
            criteria=None,
        )
        run = feature_runs.start_run(
            as_of_date=date(2026, 2, 18),
            run_type=RunType.DAILY_SNAPSHOT,
            universe_hash=hash_universe_symbols(symbols),
            input_hash=hash_scan_signature(signature),
        )
        feature_runs.mark_completed(run.id, stats=RunStats(
            total_symbols=100, processed_symbols=100,
            failed_symbols=0, duration_seconds=1.0, passed_symbols=42,
        ))
        feature_runs.publish_atomically(run.id)

        uow = FakeUnitOfWork(
            universe=FakeUniverseRepository(symbols),
            feature_runs=feature_runs,
        )
        dispatcher = FakeTaskDispatcher()
        uc = CreateScanUseCase(dispatcher=dispatcher)

        result = uc.execute(uow, _make_command())

        assert result.status == "completed"
        assert result.feature_run_id == run.id
        scan = uow.scans.rows[0]
        assert scan.status == "completed"
        assert scan.feature_run_id == run.id
        assert scan.task_id is None
        assert scan.passed_stocks == 42
        assert len(dispatcher.dispatched) == 0

    def test_non_matching_published_run_leaves_feature_run_id_none(self):
        from datetime import date
        from app.domain.feature_store.models import RunStats, RunType

        feature_runs = FakeFeatureRunRepository()
        run = feature_runs.start_run(
            as_of_date=date(2026, 2, 18),
            run_type=RunType.DAILY_SNAPSHOT,
            universe_hash="different-universe",
            input_hash="different-input",
        )
        feature_runs.mark_completed(run.id, stats=RunStats(
            total_symbols=100, processed_symbols=100,
            failed_symbols=0, duration_seconds=1.0, passed_symbols=10,
        ))
        feature_runs.publish_atomically(run.id)

        uow = FakeUnitOfWork(
            universe=FakeUniverseRepository(["AAPL"]),
            feature_runs=feature_runs,
        )
        dispatcher = FakeTaskDispatcher()
        uc = CreateScanUseCase(dispatcher=dispatcher)

        result = uc.execute(uow, _make_command())

        assert result.feature_run_id is None
        scan = uow.scans.rows[0]
        assert scan.feature_run_id is None
        assert result.status == "queued"
        assert len(dispatcher.dispatched) == 1

    def test_market_universe_exact_match_completes_instantly_without_dispatch(self):
        from datetime import date
        from app.domain.feature_store.models import RunStats, RunType
        from app.domain.scanning.signature import (
            build_scan_signature_payload,
            hash_scan_signature,
            hash_universe_symbols,
        )

        symbols = ["0700.HK"]
        signature = build_scan_signature_payload(
            universe_type="market",
            screeners=["minervini"],
            composite_method="weighted_average",
            criteria=None,
        )
        feature_runs = FakeFeatureRunRepository()
        run = feature_runs.start_run(
            as_of_date=date(2026, 2, 18),
            run_type=RunType.DAILY_SNAPSHOT,
            universe_hash=hash_universe_symbols(symbols),
            input_hash=hash_scan_signature(signature),
        )
        feature_runs.mark_completed(run.id, stats=RunStats(
            total_symbols=1, processed_symbols=1,
            failed_symbols=0, duration_seconds=1.0, passed_symbols=1,
        ))
        feature_runs.publish_atomically(run.id)

        uow = FakeUnitOfWork(
            universe=FakeUniverseRepository(symbols),
            feature_runs=feature_runs,
        )
        dispatcher = FakeTaskDispatcher()
        uc = CreateScanUseCase(dispatcher=dispatcher)

        result = uc.execute(
            uow,
            _make_command(
                universe_label="Hong Kong Market",
                universe_key="market:HK",
                universe_type="market",
                universe_market="HK",
            ),
        )

        assert result.status == "completed"
        assert result.feature_run_id == run.id
        assert len(dispatcher.dispatched) == 0

    def test_non_all_universe_stays_async_even_if_exact_run_exists(self):
        from datetime import date
        from app.domain.feature_store.models import RunStats, RunType
        from app.domain.scanning.signature import (
            build_scan_signature_payload,
            hash_scan_signature,
            hash_universe_symbols,
        )

        symbols = ["AAPL"]
        signature = build_scan_signature_payload(
            universe_type="exchange",
            screeners=["minervini"],
            composite_method="weighted_average",
            criteria=None,
        )
        feature_runs = FakeFeatureRunRepository()
        run = feature_runs.start_run(
            as_of_date=date(2026, 2, 18),
            run_type=RunType.DAILY_SNAPSHOT,
            universe_hash=hash_universe_symbols(symbols),
            input_hash=hash_scan_signature(signature),
        )
        feature_runs.mark_completed(run.id, stats=RunStats(
            total_symbols=1, processed_symbols=1,
            failed_symbols=0, duration_seconds=1.0, passed_symbols=1,
        ))
        feature_runs.publish_atomically(run.id)

        uow = FakeUnitOfWork(
            universe=FakeUniverseRepository(symbols),
            feature_runs=feature_runs,
        )
        dispatcher = FakeTaskDispatcher()
        uc = CreateScanUseCase(dispatcher=dispatcher)

        result = uc.execute(
            uow,
            _make_command(
                universe_label="NYSE",
                universe_key="exchange:NYSE",
                universe_type="exchange",
                universe_exchange="NYSE",
            ),
        )

        assert result.status == "queued"
        assert result.feature_run_id is None
        assert len(dispatcher.dispatched) == 1

    def test_feature_run_lookup_failure_does_not_block_scan(self):
        """If feature run lookup raises, scan still creates successfully."""

        class BrokenFeatureRunRepo(FakeFeatureRunRepository):
            def find_latest_published_exact(
                self,
                *,
                input_hash: str,
                universe_hash: str,
                as_of_date=None,
            ):
                raise RuntimeError("feature_runs table missing")

        uow = FakeUnitOfWork(
            universe=FakeUniverseRepository(["AAPL"]),
            feature_runs=BrokenFeatureRunRepo(),
        )
        dispatcher = FakeTaskDispatcher()
        uc = CreateScanUseCase(dispatcher=dispatcher)

        result = uc.execute(uow, _make_command())

        assert result.feature_run_id is None
        assert result.is_duplicate is False
        assert result.total_stocks == 1
        assert len(dispatcher.dispatched) == 1
