"""Unit tests for CreateScanUseCase — pure in-memory, no infrastructure."""

import pytest

from app.domain.common.errors import ValidationError
from app.domain.scanning.errors import SingleActiveScanViolation
from app.domain.scanning.models import (
    FreshnessDecision,
    FreshnessOmissionWarning,
)
from app.use_cases.scanning.create_scan import (
    ActiveScanConflictError,
    CreateScanCommand,
    CreateScanUseCase,
    StaleMarketDataError,
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


def _fresh_decision(symbols, *, policy=None):
    return FreshnessDecision(symbols_to_scan=tuple(symbols))


def _stale_detail_decision(detail):
    def _evaluator(symbols, *, policy=None):
        return FreshnessDecision(
            symbols_to_scan=tuple(symbols),
            blocking_detail=detail,
        )

    return _evaluator


def _omission_warning():
    return FreshnessOmissionWarning(
        code="market_data_stale_tail_omitted",
        message="Omitted 1 stale symbol from this scan (66.67% fresh).",
        markets=("US",),
        omitted_symbols=("LHSW",),
        omitted_count=1,
        total_symbols=3,
        fresh_count=2,
        freshness_rate=0.666667,
        expected_dates={"US": "2026-06-18"},
        oldest_last_cached_dates={"US": "2026-05-13"},
    )


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
        assert scan.criteria == {}
        assert scan.metadata_json == {
            "materialization_versions": {"opportunity_state": 1}
        }

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

    def test_custom_universe_with_no_active_symbol_says_so(self):
        uow = FakeUnitOfWork(universe=FakeUniverseRepository([], dropped_symbols=("ZZZZ", "QQQQ")))
        uc = CreateScanUseCase(dispatcher=FakeTaskDispatcher())

        with pytest.raises(ValidationError) as exc_info:
            uc.execute(uow, _make_command(
                universe_def={"type": "custom", "symbols": ["ZZZZ", "QQQQ"]},
                universe_label="Custom (2 symbols)",
                universe_key="custom:abc",
                universe_type="custom",
            ))

        assert str(exc_info.value) == "None of the requested symbols are in the active stock universe."

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
            freshness_evaluator=_stale_detail_decision(self._STALE_DETAIL),
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
            freshness_evaluator=_fresh_decision,
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

        def checker(symbols, *, policy=None):
            calls.append(list(symbols))
            return FreshnessDecision(
                symbols_to_scan=tuple(symbols),
                blocking_detail=self._STALE_DETAIL,
            )

        # First call: freshness_evaluator allows the scan to be created.
        fresh_uc = CreateScanUseCase(
            dispatcher=dispatcher,
            freshness_evaluator=_fresh_decision,
        )
        result1 = fresh_uc.execute(uow, _make_command(idempotency_key="retry-key"))

        # Second call: checker would return stale, but idempotency short-circuits first.
        stale_uc = CreateScanUseCase(
            dispatcher=dispatcher,
            freshness_evaluator=checker,
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

        def checker(symbols, *, policy=None):
            captured.append(list(symbols))
            return FreshnessDecision(symbols_to_scan=tuple(symbols))

        uc = CreateScanUseCase(dispatcher=dispatcher, freshness_evaluator=checker)
        uc.execute(uow, _make_command())

        assert captured == [["AAPL", "MSFT", "GOOGL"]]

    def test_freshness_evaluator_exception_fails_closed(self):
        """Round 6 CodeRabbit: if the checker itself raises (transient DB
        error, etc.) the exception must propagate — we do not silently skip
        the gate. Fail-closed is intentional for a safety check.
        """
        uow = _make_uow(["AAPL"])
        dispatcher = FakeTaskDispatcher()

        def checker(_symbols, *, policy=None):
            raise RuntimeError("transient DB error")

        uc = CreateScanUseCase(dispatcher=dispatcher, freshness_evaluator=checker)

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
        uc = CreateScanUseCase(dispatcher=dispatcher, freshness_evaluator=None)

        result = uc.execute(uow, _make_command())

        assert result.status == "queued"
        assert len(dispatcher.dispatched) == 1

    def test_broad_universe_omits_stale_tail_before_dispatch(self):
        uow = _make_uow(["AAPL", "LHSW", "MSFT"])
        dispatcher = FakeTaskDispatcher()

        def evaluator(symbols, *, policy=None):
            assert list(symbols) == ["AAPL", "LHSW", "MSFT"]
            assert policy.allow_stale_tail is True
            return FreshnessDecision(
                symbols_to_scan=("AAPL", "MSFT"),
                warnings=(_omission_warning(),),
            )

        uc = CreateScanUseCase(dispatcher=dispatcher, freshness_evaluator=evaluator)

        result = uc.execute(
            uow,
            _make_command(
                universe_type="market",
                universe_market="US",
                universe_key="market:US",
            ),
        )

        assert dispatcher.dispatched[0][1] == ["AAPL", "MSFT"]
        assert uow.scans.rows[0].total_stocks == 2
        assert uow.scans.rows[0].warnings[0]["omitted_symbols"] == ["LHSW"]
        assert result.warnings[0].omitted_count == 1

    def test_custom_universe_omits_a_stale_symbol_from_a_small_list(self):
        from datetime import date
        from types import SimpleNamespace
        from unittest.mock import patch

        from app.services import market_data_freshness

        fresh = [f"FRESH{i:03d}" for i in range(9)]
        symbols = [*fresh, "HALTED"]
        rows = [
            SimpleNamespace(symbol=symbol, market="US", last_date=date(2026, 6, 18))
            for symbol in fresh
        ] + [SimpleNamespace(symbol="HALTED", market="US", last_date=date(2026, 5, 13))]
        session = SimpleNamespace(close=lambda: None)
        for method in ("query", "outerjoin", "filter", "group_by"):
            setattr(session, method, lambda *args, **kwargs: session)
        session.all = lambda: rows
        uow = _make_uow(symbols)
        dispatcher = FakeTaskDispatcher()
        uc = CreateScanUseCase(
            dispatcher=dispatcher,
            freshness_evaluator=market_data_freshness.evaluate_symbol_freshness,
        )

        with (
            patch.object(market_data_freshness, "SessionLocal", return_value=session),
            patch.object(
                market_data_freshness,
                "get_market_calendar_service",
                return_value=SimpleNamespace(
                    last_completed_trading_day=lambda market, now=None: date(2026, 6, 18)
                ),
            ),
            patch.object(
                market_data_freshness,
                "get_market_refresh_state",
                return_value={
                    "market": "US",
                    "status": "completed",
                    "last_refreshed_trading_day": "2026-06-18",
                },
            ),
            patch.object(
                market_data_freshness,
                "_latest_benchmark_bar",
                return_value=(("SPY",), date.max),
            ),
        ):
            result = uc.execute(
                uow,
                _make_command(
                    universe_type="custom",
                    universe_key="custom:watchlist",
                    universe_symbols=symbols,
                ),
            )

        assert dispatcher.dispatched[0][1] == fresh
        assert [warning.code for warning in result.warnings] == [
            "market_data_stale_tail_omitted"
        ]
        assert result.warnings[0].omitted_symbols == ("HALTED",)
        assert uow.scans.rows[0].warnings[0]["omitted_count"] == 1

    def test_test_universe_still_blocks_on_any_stale_symbol(self):
        uow = _make_uow(["AAPL", "LHSW"])
        dispatcher = FakeTaskDispatcher()
        captured: list[bool] = []

        def evaluator(symbols, *, policy=None):
            captured.append(policy.allow_stale_tail)
            return FreshnessDecision(symbols_to_scan=tuple(symbols))

        uc = CreateScanUseCase(dispatcher=dispatcher, freshness_evaluator=evaluator)
        uc.execute(
            uow,
            _make_command(
                universe_type="test",
                universe_key="test:AAPL,LHSW",
                universe_symbols=["AAPL", "LHSW"],
            ),
        )

        assert captured == [False]

    def test_market_universe_allows_stale_tail(self):
        uow = _make_uow(["AAPL"])
        dispatcher = FakeTaskDispatcher()
        captured: list[bool] = []

        def evaluator(symbols, *, policy=None):
            captured.append(policy.allow_stale_tail)
            return FreshnessDecision(symbols_to_scan=tuple(symbols))

        uc = CreateScanUseCase(dispatcher=dispatcher, freshness_evaluator=evaluator)
        uc.execute(
            uow,
            _make_command(
                universe_type="market",
                universe_market="US",
                universe_key="market:US",
            ),
        )

        assert captured == [True]

    def test_idempotent_duplicate_returns_existing_warnings(self):
        uow = _make_uow(["AAPL", "LHSW", "MSFT"])
        dispatcher = FakeTaskDispatcher()
        calls = 0

        def evaluator(symbols, *, policy=None):
            nonlocal calls
            calls += 1
            return FreshnessDecision(
                symbols_to_scan=("AAPL", "MSFT"),
                warnings=(_omission_warning(),),
            )

        uc = CreateScanUseCase(dispatcher=dispatcher, freshness_evaluator=evaluator)
        result1 = uc.execute(
            uow,
            _make_command(
                idempotency_key="retry-key",
                universe_type="market",
                universe_market="US",
                universe_key="market:US",
            ),
        )
        uow.scans.rows[0].warnings.append({
            "code": "future_warning",
            "message": "This warning type is not known by this server version.",
        })
        result2 = uc.execute(
            uow,
            _make_command(
                idempotency_key="retry-key",
                universe_type="market",
                universe_market="US",
                universe_key="market:US",
            ),
        )

        assert result1.is_duplicate is False
        assert result2.is_duplicate is True
        assert result2.warnings[0].omitted_symbols == ("LHSW",)
        assert calls == 1

    def test_idempotent_duplicate_returns_stored_dropped_symbols_warning(self):
        uow = FakeUnitOfWork(
            universe=FakeUniverseRepository(["NVDA"], dropped_symbols=("ZZZZ",))
        )
        uc = CreateScanUseCase(dispatcher=FakeTaskDispatcher())
        command = _make_command(
            idempotency_key="custom-retry",
            universe_type="custom",
            universe_key="custom:abc",
        )

        uc.execute(uow, command)
        retried = uc.execute(uow, command)

        assert retried.is_duplicate is True
        assert [warning.to_dict() for warning in retried.warnings] == [
            {
                "code": "symbols_not_in_universe",
                "message": "Skipped 1 symbol not in the active stock universe: ZZZZ",
                "dropped_symbols": ["ZZZZ"],
                "dropped_count": 1,
            }
        ]

    def test_dropped_symbols_warning_stores_a_bounded_sample(self):
        dropped = tuple(f"GONE{i:03d}" for i in range(250))
        uow = FakeUnitOfWork(
            universe=FakeUniverseRepository(["NVDA"], dropped_symbols=dropped)
        )
        uc = CreateScanUseCase(dispatcher=FakeTaskDispatcher())

        result = uc.execute(
            uow, _make_command(universe_type="custom", universe_key="custom:big")
        )

        stored = uow.scans.rows[0].warnings[0]
        assert stored["dropped_count"] == 250
        assert stored["dropped_symbols"] == [f"GONE{i:03d}" for i in range(200)]
        assert result.warnings[0].dropped_count == 250

    def test_feature_run_hash_uses_filtered_symbols(self):
        from datetime import date

        from app.domain.feature_store.models import RunStats, RunType
        from app.domain.scanning.signature import (
            build_scan_signature_payload,
            hash_scan_signature,
            hash_universe_symbols,
        )

        feature_runs = FakeFeatureRunRepository()
        filtered_symbols = ["AAPL", "MSFT"]
        signature = build_scan_signature_payload(
            universe_type="market",
            screeners=["minervini"],
            composite_method="weighted_average",
            criteria=None,
        )
        run = feature_runs.start_run(
            as_of_date=date(2026, 2, 18),
            run_type=RunType.DAILY_SNAPSHOT,
            universe_hash=hash_universe_symbols(filtered_symbols),
            input_hash=hash_scan_signature(signature),
        )
        feature_runs.mark_completed(run.id, stats=RunStats(
            total_symbols=2, processed_symbols=2,
            failed_symbols=0, duration_seconds=1.0, passed_symbols=2,
        ))
        feature_runs.publish_atomically(run.id)
        uow = FakeUnitOfWork(
            universe=FakeUniverseRepository(["AAPL", "LHSW", "MSFT"]),
            feature_runs=feature_runs,
        )
        dispatcher = FakeTaskDispatcher()

        def evaluator(symbols, *, policy=None):
            return FreshnessDecision(
                symbols_to_scan=tuple(filtered_symbols),
                warnings=(_omission_warning(),),
            )

        uc = CreateScanUseCase(dispatcher=dispatcher, freshness_evaluator=evaluator)

        result = uc.execute(
            uow,
            _make_command(
                universe_type="market",
                universe_market="US",
                universe_key="market:US",
            ),
        )

        assert result.status == "completed"
        assert result.feature_run_id == run.id
        assert result.total_stocks == 2
        assert len(dispatcher.dispatched) == 0


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
