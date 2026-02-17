"""Unit tests for CreateScanUseCase — pure in-memory, no infrastructure."""

import pytest

from app.domain.common.errors import ValidationError
from app.domain.common.uow import UnitOfWork
from app.domain.scanning.ports import (
    ScanRepository,
    TaskDispatcher,
    UniverseRepository,
)
from app.use_cases.scanning.create_scan import (
    CreateScanCommand,
    CreateScanResult,
    CreateScanUseCase,
)

from tests.unit.scanning_fakes import (
    _ScanRecord,
    FakeScanResultRepository,
    FakeUnitOfWork as _BaseFakeUnitOfWork,
)


# ── Specialised fakes ───────────────────────────────────────────────────


class IdempotentScanRepository(ScanRepository):
    """Scan repository that supports idempotency key lookups."""

    def __init__(self):
        self.rows: list[_ScanRecord] = []

    def create(self, *, scan_id: str, **fields) -> _ScanRecord:
        rec = _ScanRecord(scan_id=scan_id, **fields)
        self.rows.append(rec)
        return rec

    def get_by_scan_id(self, scan_id: str) -> _ScanRecord | None:
        return next((r for r in self.rows if r.scan_id == scan_id), None)

    def get_by_idempotency_key(self, key: str) -> _ScanRecord | None:
        return next(
            (r for r in self.rows if getattr(r, "idempotency_key", None) == key),
            None,
        )

    def update_status(self, scan_id: str, status: str, **fields) -> None:
        rec = self.get_by_scan_id(scan_id)
        if rec is not None:
            rec.status = status


class SymbolUniverseRepository(UniverseRepository):
    """Universe repository that returns a configurable symbol list."""

    def __init__(self, symbols: list[str] | None = None):
        self._symbols = symbols or []

    def resolve_symbols(self, universe_def: object) -> list[str]:
        return self._symbols


class FakeTaskDispatcher(TaskDispatcher):
    def __init__(self, task_id: str = "fake-task-123"):
        self._task_id = task_id
        self.dispatched: list[tuple[str, list[str], dict]] = []

    def dispatch_scan(self, scan_id: str, symbols: list[str], criteria: dict) -> str:
        self.dispatched.append((scan_id, symbols, criteria))
        return self._task_id


class FailingTaskDispatcher(TaskDispatcher):
    def dispatch_scan(self, scan_id: str, symbols: list[str], criteria: dict) -> str:
        raise RuntimeError("Celery is down")


class FakeUnitOfWork(UnitOfWork):
    """UoW with configurable symbols for universe resolution."""

    def __init__(self, symbols: list[str] | None = None):
        self.scans = IdempotentScanRepository()
        self.scan_results = FakeScanResultRepository()
        self.universe = SymbolUniverseRepository(symbols)
        self.committed = 0
        self.rolled_back = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.rollback()

    def commit(self):
        self.committed += 1

    def rollback(self):
        self.rolled_back += 1


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


# ── Tests ────────────────────────────────────────────────────────────────


class TestCreateScanUseCase:
    """Core business logic for scan creation."""

    def test_creates_scan_and_dispatches_task(self):
        uow = FakeUnitOfWork(symbols=["AAPL", "MSFT", "GOOGL"])
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
        uow = FakeUnitOfWork(symbols=["AAPL"])
        dispatcher = FakeTaskDispatcher()
        uc = CreateScanUseCase(dispatcher=dispatcher)

        uc.execute(uow, _make_command())

        assert len(uow.scans.rows) == 1
        scan = uow.scans.rows[0]
        assert scan.status == "queued"
        assert scan.total_stocks == 1
        assert scan.task_id == "fake-task-123"

    def test_stores_universe_metadata(self):
        uow = FakeUnitOfWork(symbols=["AAPL"])
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

    def test_empty_universe_raises_validation_error(self):
        uow = FakeUnitOfWork(symbols=[])  # no symbols
        dispatcher = FakeTaskDispatcher()
        uc = CreateScanUseCase(dispatcher=dispatcher)

        with pytest.raises(ValidationError, match="No symbols found"):
            uc.execute(uow, _make_command())

        # Should not have dispatched
        assert len(dispatcher.dispatched) == 0

    def test_dispatch_failure_marks_scan_failed(self):
        uow = FakeUnitOfWork(symbols=["AAPL"])
        dispatcher = FailingTaskDispatcher()
        uc = CreateScanUseCase(dispatcher=dispatcher)

        with pytest.raises(RuntimeError, match="Celery is down"):
            uc.execute(uow, _make_command())

        scan = uow.scans.rows[0]
        assert scan.status == "failed"

    def test_commits_at_least_twice_on_success(self):
        """First commit persists scan, second stores task_id."""
        uow = FakeUnitOfWork(symbols=["AAPL"])
        dispatcher = FakeTaskDispatcher()
        uc = CreateScanUseCase(dispatcher=dispatcher)

        uc.execute(uow, _make_command())

        assert uow.committed >= 2


class TestIdempotency:
    """Idempotency key prevents duplicate scans."""

    def test_duplicate_key_returns_existing_scan(self):
        uow = FakeUnitOfWork(symbols=["AAPL"])
        dispatcher = FakeTaskDispatcher()
        uc = CreateScanUseCase(dispatcher=dispatcher)

        # First call creates
        cmd = _make_command(idempotency_key="abc-123")
        result1 = uc.execute(uow, cmd)
        assert result1.is_duplicate is False

        # Second call with same key returns existing
        result2 = uc.execute(uow, cmd)
        assert result2.is_duplicate is True
        assert result2.scan_id == result1.scan_id

        # Only one scan created, only one dispatch
        assert len(uow.scans.rows) == 1
        assert len(dispatcher.dispatched) == 1

    def test_different_keys_create_separate_scans(self):
        uow = FakeUnitOfWork(symbols=["AAPL"])
        dispatcher = FakeTaskDispatcher()
        uc = CreateScanUseCase(dispatcher=dispatcher)

        result1 = uc.execute(uow, _make_command(idempotency_key="key-1"))
        result2 = uc.execute(uow, _make_command(idempotency_key="key-2"))

        assert result1.scan_id != result2.scan_id
        assert len(uow.scans.rows) == 2
        assert len(dispatcher.dispatched) == 2

    def test_no_key_always_creates_new_scan(self):
        uow = FakeUnitOfWork(symbols=["AAPL"])
        dispatcher = FakeTaskDispatcher()
        uc = CreateScanUseCase(dispatcher=dispatcher)

        result1 = uc.execute(uow, _make_command(idempotency_key=None))
        result2 = uc.execute(uow, _make_command(idempotency_key=None))

        assert result1.scan_id != result2.scan_id
        assert len(uow.scans.rows) == 2
