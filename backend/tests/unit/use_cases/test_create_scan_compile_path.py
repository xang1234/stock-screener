"""Unit tests for the feature-store-first compile path in CreateScanUseCase.

These tests cover the new behavior where custom-only scans whose criteria
compile cleanly into queryable feature-store fields are answered by a
single SQL query against a published feature run, instead of being
dispatched to the async chunked-compute path.
"""

from __future__ import annotations

from datetime import date, datetime

from app.domain.feature_store.models import (
    FeatureRowWrite,
    RunStats,
    RunType,
)
from app.use_cases.scanning.create_scan import (
    CreateScanCommand,
    CreateScanUseCase,
)

from tests.unit.use_cases.conftest import (
    FakeFeatureRunRepository,
    FakeFeatureStoreRepository,
    FakeTaskDispatcher,
    FakeUnitOfWork,
    FakeUniverseRepository,
)


# ── Helpers ──────────────────────────────────────────────────────────────


def _custom_command(
    *,
    universe_market: str | None = None,
    criteria: dict | None = None,
    screeners: list[str] | None = None,
) -> CreateScanCommand:
    return CreateScanCommand(
        universe_def="all",
        universe_label="All Stocks",
        universe_key="all",
        universe_type="all",
        universe_market=universe_market,
        screeners=screeners or ["custom"],
        composite_method="weighted_average",
        criteria=criteria or {
            "custom_filters": {"price_min": 20, "ma_alignment": True},
            "min_score": 70,
        },
    )


def _publish_run(
    feature_runs: FakeFeatureRunRepository,
    feature_store: FakeFeatureStoreRepository,
    *,
    symbols: list[str],
    rows: list[FeatureRowWrite],
    as_of: date | None = None,
) -> int:
    """Create + populate + publish a feature run on the fake repos."""
    run = feature_runs.start_run(
        as_of_date=as_of or date(2026, 4, 24),
        run_type=RunType.DAILY_SNAPSHOT,
        universe_hash="universe-hash-irrelevant",
        input_hash="input-hash-irrelevant",
    )
    feature_runs.set_run_universe(run.id, symbols)
    feature_runs.mark_completed(
        run.id,
        stats=RunStats(
            total_symbols=len(symbols),
            processed_symbols=len(symbols),
            failed_symbols=0,
            duration_seconds=1.0,
            passed_symbols=len([r for r in rows if r.composite_score and r.composite_score >= 70]),
        ),
    )
    feature_runs.publish_atomically(run.id)
    feature_store.save_run_universe_symbols(run.id, symbols)
    feature_store.upsert_snapshot_rows(run.id, rows)
    return run.id


def _row(symbol: str, *, custom_score: float, **details) -> FeatureRowWrite:
    """Build a FeatureRowWrite that mimics what the daily snapshot stores."""
    payload = {
        "custom_score": custom_score,
        "current_price": details.pop("current_price", 100.0),
        "ma_alignment": details.pop("ma_alignment", True),
        "rs_rating": details.pop("rs_rating", 80),
        "rating": "Buy" if custom_score >= 70 else "Watch",
        "screeners_run": ["custom"],
        "screeners_passed": 1 if custom_score >= 70 else 0,
        "screeners_total": 1,
        **details,
    }
    return FeatureRowWrite(
        symbol=symbol,
        as_of_date=date(2026, 4, 24),
        composite_score=custom_score,
        overall_rating=4 if custom_score >= 80 else (3 if custom_score >= 70 else 2),
        passes_count=1 if custom_score >= 70 else 0,
        details=payload,
    )


# ── Tests ────────────────────────────────────────────────────────────────


class TestCompilePathHappyPath:
    """Custom-only scans matching a covering published run complete instantly."""

    def test_compile_path_skips_dispatch_and_persists_passing_rows(self):
        feature_runs = FakeFeatureRunRepository()
        feature_store = FakeFeatureStoreRepository()
        symbols = ["AAPL", "MSFT", "GOOGL"]
        rows = [
            # Both per-field thresholds (price>=20, ma_alignment=True) pass.
            _row("AAPL", custom_score=85, current_price=150, ma_alignment=True),
            _row("MSFT", custom_score=72, current_price=300, ma_alignment=True),
            # MA misaligned — must fail the boolean filter and be excluded.
            _row("GOOGL", custom_score=85, current_price=120, ma_alignment=False),
        ]
        run_id = _publish_run(
            feature_runs, feature_store, symbols=symbols, rows=rows
        )

        uow = FakeUnitOfWork(
            universe=FakeUniverseRepository(symbols),
            feature_runs=feature_runs,
            feature_store=feature_store,
        )
        dispatcher = FakeTaskDispatcher()
        uc = CreateScanUseCase(dispatcher=dispatcher)

        result = uc.execute(uow, _custom_command())

        assert result.status == "completed"
        assert result.total_stocks == 3
        # Per-field threshold pass test (no stale-criteria custom_score gate):
        # GOOGL excluded because ma_alignment=False; AAPL + MSFT persisted.
        assert len(dispatcher.dispatched) == 0
        persisted = uow.scan_results._persisted_results
        persisted_symbols = sorted(sym for _, sym, _ in persisted)
        assert persisted_symbols == ["AAPL", "MSFT"]
        scan = uow.scans.rows[0]
        assert scan.status == "completed"
        assert scan.passed_stocks == 2
        # feature_run_id is intentionally None — read path uses scan_results.
        assert scan.feature_run_id is None
        assert run_id is not None

    def test_compile_path_does_not_apply_stale_custom_score_gate(self):
        """A covering run reached this path because its (input_hash) differs
        from the request, so its stored custom_score was computed under a
        different criteria set. The compile path must not use it as a pass
        gate — only per-field thresholds gate passing.
        """
        feature_runs = FakeFeatureRunRepository()
        feature_store = FakeFeatureStoreRepository()
        symbols = ["AAPL"]
        # custom_score=10 was computed under different criteria; the user's
        # actual per-field thresholds (price>=20, ma_alignment=True) all
        # pass, so the row must be persisted regardless of the stored score.
        rows = [
            _row(
                "AAPL", custom_score=10, current_price=150, ma_alignment=True
            ),
        ]
        _publish_run(
            feature_runs, feature_store, symbols=symbols, rows=rows
        )

        uow = FakeUnitOfWork(
            universe=FakeUniverseRepository(symbols),
            feature_runs=feature_runs,
            feature_store=feature_store,
        )
        dispatcher = FakeTaskDispatcher()
        uc = CreateScanUseCase(dispatcher=dispatcher)

        result = uc.execute(
            uow,
            _custom_command(
                criteria={
                    "custom_filters": {"price_min": 20, "ma_alignment": True},
                    "min_score": 70,
                },
            ),
        )

        assert result.status == "completed"
        assert len(dispatcher.dispatched) == 0
        persisted = sorted(
            sym for _, sym, _ in uow.scan_results._persisted_results
        )
        assert persisted == ["AAPL"]

    def test_compile_path_applies_price_range(self):
        feature_runs = FakeFeatureRunRepository()
        feature_store = FakeFeatureStoreRepository()
        symbols = ["A", "B", "C"]
        rows = [
            _row("A", custom_score=85, current_price=10),  # below price_min
            _row("B", custom_score=85, current_price=50),  # in range
            _row("C", custom_score=85, current_price=600),  # above price_max
        ]
        _publish_run(feature_runs, feature_store, symbols=symbols, rows=rows)

        uow = FakeUnitOfWork(
            universe=FakeUniverseRepository(symbols),
            feature_runs=feature_runs,
            feature_store=feature_store,
        )
        uc = CreateScanUseCase(dispatcher=FakeTaskDispatcher())

        result = uc.execute(
            uow,
            _custom_command(
                criteria={
                    "custom_filters": {"price_min": 20, "price_max": 500},
                    "min_score": 70,
                },
            ),
        )

        assert result.status == "completed"
        persisted = sorted(sym for _, sym, _ in uow.scan_results._persisted_results)
        assert persisted == ["B"]


class TestCompilePathFallsBack:
    """Non-applicable scans must defer to the async path."""

    def test_unrepresentable_criteria_dispatches_async(self):
        feature_runs = FakeFeatureRunRepository()
        feature_store = FakeFeatureStoreRepository()
        symbols = ["AAPL"]
        _publish_run(
            feature_runs,
            feature_store,
            symbols=symbols,
            rows=[_row("AAPL", custom_score=85)],
        )

        uow = FakeUnitOfWork(
            universe=FakeUniverseRepository(symbols),
            feature_runs=feature_runs,
            feature_store=feature_store,
        )
        dispatcher = FakeTaskDispatcher()
        uc = CreateScanUseCase(dispatcher=dispatcher)

        # debt_to_equity_max is not in the feature store JSON map yet.
        result = uc.execute(
            uow,
            _custom_command(
                criteria={
                    "custom_filters": {
                        "price_min": 20,
                        "debt_to_equity_max": 0.5,
                    },
                    "min_score": 70,
                },
            ),
        )

        assert result.status == "queued"
        assert len(dispatcher.dispatched) == 1
        assert uow.scans.rows[0].feature_run_id is None

    def test_no_covering_run_dispatches_async(self):
        """Run universe doesn't include all requested symbols."""
        feature_runs = FakeFeatureRunRepository()
        feature_store = FakeFeatureStoreRepository()
        # Run only covers AAPL + MSFT; user asks for GOOGL too.
        _publish_run(
            feature_runs,
            feature_store,
            symbols=["AAPL", "MSFT"],
            rows=[
                _row("AAPL", custom_score=85),
                _row("MSFT", custom_score=85),
            ],
        )

        uow = FakeUnitOfWork(
            universe=FakeUniverseRepository(["AAPL", "MSFT", "GOOGL"]),
            feature_runs=feature_runs,
            feature_store=feature_store,
        )
        dispatcher = FakeTaskDispatcher()
        uc = CreateScanUseCase(dispatcher=dispatcher)

        result = uc.execute(uow, _custom_command())

        assert result.status == "queued"
        assert len(dispatcher.dispatched) == 1

    def test_no_published_run_dispatches_async(self):
        """No feature run has been published yet."""
        uow = FakeUnitOfWork(
            universe=FakeUniverseRepository(["AAPL"]),
        )
        dispatcher = FakeTaskDispatcher()
        uc = CreateScanUseCase(dispatcher=dispatcher)

        result = uc.execute(uow, _custom_command())

        assert result.status == "queued"
        assert len(dispatcher.dispatched) == 1

    def test_multi_screener_dispatches_async(self):
        """Multi-screener composites are out of scope for the compile path."""
        feature_runs = FakeFeatureRunRepository()
        feature_store = FakeFeatureStoreRepository()
        symbols = ["AAPL"]
        _publish_run(
            feature_runs,
            feature_store,
            symbols=symbols,
            rows=[_row("AAPL", custom_score=85)],
        )

        uow = FakeUnitOfWork(
            universe=FakeUniverseRepository(symbols),
            feature_runs=feature_runs,
            feature_store=feature_store,
        )
        dispatcher = FakeTaskDispatcher()
        uc = CreateScanUseCase(dispatcher=dispatcher)

        result = uc.execute(
            uow,
            _custom_command(screeners=["custom", "minervini"]),
        )

        assert result.status == "queued"
        assert len(dispatcher.dispatched) == 1

    def test_index_universe_dispatches_async(self):
        """INDEX universes pass universe_market=None even when symbols are
        single-market, which would mismatch async unit semantics. Defer.
        """
        feature_runs = FakeFeatureRunRepository()
        feature_store = FakeFeatureStoreRepository()
        symbols = ["AAPL", "MSFT"]
        _publish_run(
            feature_runs,
            feature_store,
            symbols=symbols,
            rows=[
                _row("AAPL", custom_score=85),
                _row("MSFT", custom_score=85),
            ],
        )

        uow = FakeUnitOfWork(
            universe=FakeUniverseRepository(symbols),
            feature_runs=feature_runs,
            feature_store=feature_store,
        )
        dispatcher = FakeTaskDispatcher()
        uc = CreateScanUseCase(dispatcher=dispatcher)

        result = uc.execute(
            uow,
            CreateScanCommand(
                universe_def="index:SP500",
                universe_label="S&P 500",
                universe_key="index:SP500",
                universe_type="index",
                universe_index="SP500",
                screeners=["custom"],
                criteria={
                    "custom_filters": {"price_min": 20, "ma_alignment": True},
                    "min_score": 70,
                },
            ),
        )

        assert result.status == "queued"
        assert len(dispatcher.dispatched) == 1

    def test_custom_universe_dispatches_async(self):
        """Symbol-list (CUSTOM) universes also defer to async — same reason."""
        feature_runs = FakeFeatureRunRepository()
        feature_store = FakeFeatureStoreRepository()
        symbols = ["AAPL", "MSFT"]
        _publish_run(
            feature_runs,
            feature_store,
            symbols=symbols,
            rows=[
                _row("AAPL", custom_score=85),
                _row("MSFT", custom_score=85),
            ],
        )

        uow = FakeUnitOfWork(
            universe=FakeUniverseRepository(symbols),
            feature_runs=feature_runs,
            feature_store=feature_store,
        )
        dispatcher = FakeTaskDispatcher()
        uc = CreateScanUseCase(dispatcher=dispatcher)

        result = uc.execute(
            uow,
            CreateScanCommand(
                universe_def="custom",
                universe_label="My Symbols",
                universe_key="custom",
                universe_type="custom",
                universe_symbols=symbols,
                screeners=["custom"],
                criteria={
                    "custom_filters": {"price_min": 20, "ma_alignment": True},
                    "min_score": 70,
                },
            ),
        )

        assert result.status == "queued"
        assert len(dispatcher.dispatched) == 1

    def test_non_us_single_market_with_volume_filter_dispatches_async(self):
        """USD-normalised columns can't safely answer HK volume thresholds."""
        feature_runs = FakeFeatureRunRepository()
        feature_store = FakeFeatureStoreRepository()
        symbols = ["0700.HK"]
        _publish_run(
            feature_runs,
            feature_store,
            symbols=symbols,
            rows=[_row("0700.HK", custom_score=85)],
        )

        uow = FakeUnitOfWork(
            universe=FakeUniverseRepository(symbols),
            feature_runs=feature_runs,
            feature_store=feature_store,
        )
        dispatcher = FakeTaskDispatcher()
        uc = CreateScanUseCase(dispatcher=dispatcher)

        result = uc.execute(
            uow,
            CreateScanCommand(
                universe_def="market:HK",
                universe_label="Hong Kong",
                universe_key="market:HK",
                universe_type="market",
                universe_market="HK",
                screeners=["custom"],
                criteria={
                    "custom_filters": {"volume_min": 1_000_000, "price_min": 20},
                    "min_score": 70,
                },
            ),
        )

        assert result.status == "queued"
        assert len(dispatcher.dispatched) == 1


class TestCompilePathErrorHandling:
    """Compile-path failures must never block the async fallback."""

    def test_lookup_exception_falls_back_to_async(self):
        class BrokenFeatureRunRepo(FakeFeatureRunRepository):
            def find_latest_published_covering(self, *, symbols, market=None):
                raise RuntimeError("feature_runs table unreachable")

        uow = FakeUnitOfWork(
            universe=FakeUniverseRepository(["AAPL"]),
            feature_runs=BrokenFeatureRunRepo(),
        )
        dispatcher = FakeTaskDispatcher()
        uc = CreateScanUseCase(dispatcher=dispatcher)

        result = uc.execute(uow, _custom_command())

        assert result.status == "queued"
        assert len(dispatcher.dispatched) == 1

    def test_query_exception_falls_back_to_async(self):
        feature_runs = FakeFeatureRunRepository()
        feature_store = FakeFeatureStoreRepository()
        symbols = ["AAPL"]
        _publish_run(
            feature_runs,
            feature_store,
            symbols=symbols,
            rows=[_row("AAPL", custom_score=85)],
        )

        class BrokenFeatureStore(FakeFeatureStoreRepository):
            def query_run_details(self, run_id, filters=None, *, symbols=None):
                raise RuntimeError("query failed")

        # Replace store but copy data over so coverage check passes.
        broken = BrokenFeatureStore()
        broken._rows = feature_store._rows
        broken._universe = feature_store._universe

        uow = FakeUnitOfWork(
            universe=FakeUniverseRepository(symbols),
            feature_runs=feature_runs,
            feature_store=broken,
        )
        dispatcher = FakeTaskDispatcher()
        uc = CreateScanUseCase(dispatcher=dispatcher)

        result = uc.execute(uow, _custom_command())

        assert result.status == "queued"
        assert len(dispatcher.dispatched) == 1


class TestCompilePathCoexistsWithExactMatch:
    """The exact-signature path should still take precedence when applicable."""

    def test_exact_match_wins_over_compile_path(self):
        from app.domain.scanning.signature import (
            build_scan_signature_payload,
            hash_scan_signature,
            hash_universe_symbols,
        )

        symbols = ["AAPL"]
        criteria = {
            "custom_filters": {"price_min": 20, "ma_alignment": True},
            "min_score": 70,
        }
        signature = build_scan_signature_payload(
            universe_type="all",
            screeners=["custom"],
            composite_method="weighted_average",
            criteria=criteria,
        )

        feature_runs = FakeFeatureRunRepository()
        feature_store = FakeFeatureStoreRepository()
        run = feature_runs.start_run(
            as_of_date=date(2026, 4, 24),
            run_type=RunType.DAILY_SNAPSHOT,
            universe_hash=hash_universe_symbols(symbols),
            input_hash=hash_scan_signature(signature),
        )
        feature_runs.set_run_universe(run.id, symbols)
        feature_runs.mark_completed(
            run.id,
            stats=RunStats(
                total_symbols=1, processed_symbols=1,
                failed_symbols=0, duration_seconds=1.0, passed_symbols=1,
            ),
        )
        feature_runs.publish_atomically(run.id)
        feature_store.save_run_universe_symbols(run.id, symbols)
        feature_store.upsert_snapshot_rows(
            run.id, [_row("AAPL", custom_score=85)]
        )

        uow = FakeUnitOfWork(
            universe=FakeUniverseRepository(symbols),
            feature_runs=feature_runs,
            feature_store=feature_store,
        )
        dispatcher = FakeTaskDispatcher()
        uc = CreateScanUseCase(dispatcher=dispatcher)

        result = uc.execute(
            uow,
            _custom_command(criteria=criteria),
        )

        # Exact match path sets feature_run_id (run IS the result).
        assert result.status == "completed"
        assert result.feature_run_id == run.id
        # No compile-path persistence happened.
        assert uow.scan_results._persisted_results == []
        assert len(dispatcher.dispatched) == 0
