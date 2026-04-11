"""Unit tests for CreateScanUseCase — pure in-memory, no infrastructure."""

import pytest

from app.domain.common.errors import ValidationError
from app.use_cases.scanning.create_scan import (
    CreateScanCommand,
    CreateScanResult,
    CreateScanUseCase,
)

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

    def test_dispatch_failure_marks_scan_failed(self):
        uow = _make_uow(["AAPL"])
        dispatcher = FakeTaskDispatcher(should_fail=True)
        uc = CreateScanUseCase(dispatcher=dispatcher)

        with pytest.raises(RuntimeError, match="Celery is down"):
            uc.execute(uow, _make_command())

        scan = uow.scans.rows[0]
        assert scan.status == "failed"

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
        result2 = uc.execute(uow, _make_command(idempotency_key="key-2"))

        assert result1.scan_id != result2.scan_id
        assert len(uow.scans.rows) == 2
        assert len(dispatcher.dispatched) == 2

    def test_no_key_always_creates_new_scan(self):
        uow = _make_uow(["AAPL"])
        dispatcher = FakeTaskDispatcher()
        uc = CreateScanUseCase(dispatcher=dispatcher)

        result1 = uc.execute(uow, _make_command(idempotency_key=None))
        result2 = uc.execute(uow, _make_command(idempotency_key=None))

        assert result1.scan_id != result2.scan_id
        assert len(uow.scans.rows) == 2


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
