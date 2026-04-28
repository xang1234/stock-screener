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
    composite_method: str = "weighted_average",
) -> CreateScanCommand:
    return CreateScanCommand(
        universe_def="all",
        universe_label="All Stocks",
        universe_key="all",
        universe_type="all",
        universe_market=universe_market,
        screeners=screeners or ["custom"],
        composite_method=composite_method,
        criteria=criteria or {
            # Use only filters the compiler can express today: price_min
            # maps to an indexed JSON field and is hard-gate equivalent to
            # CustomScanner when it is the only active filter. Avoid
            # ma_alignment in the default — the stored field has stricter
            # Minervini semantics so the compiler marks it unrepresentable.
            "custom_filters": {"price_min": 20},
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
    pointer_key: str = "latest_published",
) -> int:
    """Create + populate + publish a feature run on the fake repos.

    Coverage in the fake is keyed off the symbols actually persisted as
    rows (``[r.symbol for r in rows]``), not the requested universe — this
    matches the production coverage check, which counts
    ``stock_feature_daily`` rows so partial-publish runs are correctly
    rejected. Pass a ``rows`` list shorter than ``symbols`` to model a
    partial run.
    """
    run = feature_runs.start_run(
        as_of_date=as_of or date(2026, 4, 24),
        run_type=RunType.DAILY_SNAPSHOT,
        universe_hash="universe-hash-irrelevant",
        input_hash="input-hash-irrelevant",
    )
    feature_runs.set_run_covered_symbols(run.id, [r.symbol for r in rows])
    feature_runs.mark_completed(
        run.id,
        stats=RunStats(
            total_symbols=len(symbols),
            processed_symbols=len(rows),
            failed_symbols=max(0, len(symbols) - len(rows)),
            duration_seconds=1.0,
            passed_symbols=len([r for r in rows if r.composite_score and r.composite_score >= 70]),
        ),
    )
    feature_runs.publish_atomically(run.id, pointer_key=pointer_key)
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
            _row("AAPL", custom_score=85, current_price=150),
            _row("MSFT", custom_score=72, current_price=300),
            # price below threshold — must be excluded.
            _row("GOOGL", custom_score=85, current_price=10),
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
        # GOOGL excluded because current_price=10 < 20; AAPL + MSFT persisted.
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
        # hard-gate-equivalent price threshold passes, so the row must be
        # persisted regardless of the stored score.
        rows = [
            _row(
                "AAPL", custom_score=10, current_price=150, rs_rating=85
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
                    "custom_filters": {"price_min": 20},
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

    def test_compile_path_normalises_stale_score_fields(self):
        """Stored custom_score/composite_score/rating/passes_template come
        from a different criteria set (otherwise the exact-match path would
        have caught it). They must not leak into the persisted scan_result
        row; sorting by score in the UI would otherwise misrank matches.
        """
        feature_runs = FakeFeatureRunRepository()
        feature_store = FakeFeatureStoreRepository()
        symbols = ["AAPL"]
        # Stored row carries stale-criteria metadata that must not survive,
        # plus factual snapshot fields that must.
        stale_row = FeatureRowWrite(
            symbol="AAPL",
            as_of_date=date(2026, 4, 24),
            composite_score=32.0,
            overall_rating=2,
            passes_count=0,
            details={
                "custom_score": 32.0,
                "current_price": 150.0,
                "ma_alignment": True,
                "rating": "Watch",
                "passes_template": False,
                "minervini_score": 88.0,
                "canslim_score": 70.0,
                "screeners_run": ["custom", "minervini", "canslim"],
                "screeners_passed": 2,
                "screeners_total": 3,
                "ipo_score": 62.0,
                "ipo_bonus": 8.0,
                "rs_rating": 91,
                "gics_sector": "Technology",
            },
        )
        _publish_run(
            feature_runs,
            feature_store,
            symbols=symbols,
            rows=[stale_row],
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
                composite_method="maximum",
                criteria={
                    "custom_filters": {"price_min": 20},
                    "min_score": 70,
                },
            ),
        )

        assert result.status == "completed"
        assert len(uow.scan_results._persisted_results) == 1
        _, _, persisted = uow.scan_results._persisted_results[0]

        # Stale score/rating/pass fields are normalised to compile-path
        # semantics ("the row passed every user-specified filter").
        assert persisted["custom_score"] == 100.0
        assert persisted["composite_score"] == 100.0
        assert persisted["rating"] == "Strong Buy"
        assert persisted["passes_template"] is True
        assert persisted["screeners_run"] == ["custom"]
        assert persisted["screeners_passed"] == 1
        assert persisted["screeners_total"] == 1
        assert persisted["composite_method"] == "maximum"

        # Other-screener scores from the snapshot are dropped — the user's
        # scan was custom-only, so they have no meaning in this context.
        assert "minervini_score" not in persisted
        assert "canslim_score" not in persisted
        assert "ipo_score" not in persisted
        assert "ipo_bonus" not in persisted

        # Per-symbol facts (not derived from custom criteria) survive.
        assert persisted["current_price"] == 150.0
        assert persisted["rs_rating"] == 91
        assert persisted["gics_sector"] == "Technology"

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

    def test_multi_filter_partial_score_case_dispatches_async(self):
        """Four binary filters are not equivalent to SQL hard gates.

        CustomScanner would give this row 30/40 points because it passes
        price, market cap, and sector, but fails industry exclusion. At the
        default min_score=70, 75 points passes async. A SQL WHERE chain would
        drop it, so the compile path must defer.
        """
        feature_runs = FakeFeatureRunRepository()
        feature_store = FakeFeatureStoreRepository()
        symbols = ["AAPL"]
        _publish_run(
            feature_runs,
            feature_store,
            symbols=symbols,
            rows=[
                _row(
                    "AAPL",
                    custom_score=85,
                    current_price=150,
                    market_cap_usd=2_000_000_000,
                    gics_sector="Technology",
                    gics_industry="Tobacco",
                )
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
            _custom_command(
                criteria={
                    "custom_filters": {
                        "price_min": 20,
                        "market_cap_min": 1_000_000_000,
                        "sectors": ["Technology"],
                        "exclude_industries": ["Tobacco"],
                    },
                    "min_score": 70,
                },
            ),
        )

        assert result.status == "queued"
        assert len(dispatcher.dispatched) == 1

    def test_scaled_score_filter_dispatches_async(self):
        """RS rating threshold has partial score semantics in CustomScanner.

        A row exactly at rs_rating_min satisfies the hard SQL gate, but gets
        10/15 points in CustomScanner and fails the default min_score=70.
        """
        feature_runs = FakeFeatureRunRepository()
        feature_store = FakeFeatureStoreRepository()
        symbols = ["AAPL"]
        _publish_run(
            feature_runs,
            feature_store,
            symbols=symbols,
            rows=[_row("AAPL", custom_score=85, rs_rating=80)],
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
                    "custom_filters": {"rs_rating_min": 80},
                    "min_score": 70,
                },
            ),
        )

        assert result.status == "queued"
        assert len(dispatcher.dispatched) == 1

    def test_non_positive_volume_min_with_hard_gate_dispatches_async(self):
        """``volume_min<=0`` contributes full async points without adding a
        SQL predicate. With ``min_score=50``, a row can fail price but pass via
        the volume filter, so strict SQL price filtering would false-negative.
        """
        feature_runs = FakeFeatureRunRepository()
        feature_store = FakeFeatureStoreRepository()
        symbols = ["AAPL"]
        _publish_run(
            feature_runs,
            feature_store,
            symbols=symbols,
            rows=[_row("AAPL", custom_score=85, current_price=10)],
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
                    "custom_filters": {"price_min": 20, "volume_min": 0},
                    "min_score": 50,
                },
            ),
        )

        assert result.status == "queued"
        assert len(dispatcher.dispatched) == 1

    def test_empty_exclude_industries_with_hard_gate_dispatches_async(self):
        """``exclude_industries=[]`` contributes full async points without a
        SQL predicate. With ``min_score=50``, a row can fail price but pass via
        the empty exclusion filter, so compile path must defer.
        """
        feature_runs = FakeFeatureRunRepository()
        feature_store = FakeFeatureStoreRepository()
        symbols = ["AAPL"]
        _publish_run(
            feature_runs,
            feature_store,
            symbols=symbols,
            rows=[_row("AAPL", custom_score=85, current_price=10)],
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
                    "custom_filters": {
                        "price_min": 20,
                        "exclude_industries": [],
                    },
                    "min_score": 50,
                },
            ),
        )

        assert result.status == "queued"
        assert len(dispatcher.dispatched) == 1


class TestCompilePathFallsBack:
    """Non-applicable scans must defer to the async path."""

    def test_ma_alignment_filter_dispatches_async(self):
        """Stored ``ma_alignment`` has Minervini's stricter semantics, so
        the compiler marks it unrepresentable and the scan goes async.
        """
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
            _custom_command(
                criteria={
                    "custom_filters": {"price_min": 20, "ma_alignment": True},
                    "min_score": 70,
                },
            ),
        )

        assert result.status == "queued"
        assert len(dispatcher.dispatched) == 1

    def test_empty_sectors_filter_dispatches_async(self):
        """``sectors=[]`` makes every async stock fail; we can't express
        "match nothing" cleanly in SQL, so defer to async.
        """
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
            _custom_command(
                criteria={
                    "custom_filters": {"price_min": 20, "sectors": []},
                    "min_score": 70,
                },
            ),
        )

        assert result.status == "queued"
        assert len(dispatcher.dispatched) == 1

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

    def test_partial_publish_run_dispatches_async(self):
        """Default DQ thresholds permit publishing a run with up to 10% of
        universe symbols missing feature rows. The compile path must check
        actual ``stock_feature_daily`` rows, not just universe membership,
        otherwise it would silently omit those symbols from results.
        """
        feature_runs = FakeFeatureRunRepository()
        feature_store = FakeFeatureStoreRepository()
        # Universe lists AAPL + MSFT, but only AAPL has a row (partial run).
        _publish_run(
            feature_runs,
            feature_store,
            symbols=["AAPL", "MSFT"],
            rows=[_row("AAPL", custom_score=85)],
        )

        uow = FakeUnitOfWork(
            universe=FakeUniverseRepository(["AAPL", "MSFT"]),
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
                    "custom_filters": {"price_min": 20, "rs_rating_min": 70},
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
                    "custom_filters": {"price_min": 20, "rs_rating_min": 70},
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


class TestMarketAwareCoveringRunSelection:
    """The fake mirrors the production market-pointer fallback so a
    regression that picks the wrong run for a market-scoped scan would
    still be caught here.
    """

    def test_market_pointer_takes_precedence_over_global(self):
        feature_runs = FakeFeatureRunRepository()
        feature_store = FakeFeatureStoreRepository()
        symbols = ["AAPL"]

        # Older global run + newer US-specific run; both cover the symbol.
        global_run_id = _publish_run(
            feature_runs, feature_store, symbols=symbols,
            rows=[_row("AAPL", custom_score=85, current_price=100, rs_rating=80)],
            as_of=date(2026, 4, 23),
            pointer_key="latest_published",
        )
        us_run_id = _publish_run(
            feature_runs, feature_store, symbols=symbols,
            rows=[_row("AAPL", custom_score=85, current_price=200, rs_rating=80)],
            as_of=date(2026, 4, 24),
            pointer_key="latest_published_market:US",
        )
        assert us_run_id != global_run_id

        run = feature_runs.find_latest_published_covering(
            symbols=symbols, market="US"
        )

        assert run is not None
        assert run.id == us_run_id

    def test_falls_back_to_global_when_market_pointer_missing(self):
        feature_runs = FakeFeatureRunRepository()
        feature_store = FakeFeatureStoreRepository()
        symbols = ["AAPL"]
        global_run_id = _publish_run(
            feature_runs, feature_store, symbols=symbols,
            rows=[_row("AAPL", custom_score=85)],
            pointer_key="latest_published",
        )

        run = feature_runs.find_latest_published_covering(
            symbols=symbols, market="HK"
        )

        assert run is not None
        assert run.id == global_run_id

    def test_returns_none_when_no_covering_pointer(self):
        feature_runs = FakeFeatureRunRepository()
        feature_store = FakeFeatureStoreRepository()
        # Run published only under HK pointer, but query asks for US — and
        # no global pointer exists either.
        _publish_run(
            feature_runs, feature_store, symbols=["0700.HK"],
            rows=[_row("0700.HK", custom_score=85)],
            pointer_key="latest_published_market:HK",
        )

        run = feature_runs.find_latest_published_covering(
            symbols=["0700.HK"], market="US"
        )

        assert run is None


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
            "custom_filters": {"price_min": 20, "rs_rating_min": 70},
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
        feature_runs.set_run_covered_symbols(run.id, symbols)
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
