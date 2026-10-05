"""Pinned per-Market snapshot reuse and last-published mode (#492)."""

from __future__ import annotations

from datetime import date

import pytest

from app.domain.feature_store.models import FeatureRowWrite, RunStats, RunType
from app.domain.relative_strength.calculator import (
    BALANCED_RS_FORMULA_VERSION,
    LEGACY_RS_FORMULA_VERSION,
)
from app.domain.scanning.signature import build_scan_signature_payload, hash_scan_signature
from app.use_cases.scanning.create_scan import (
    ActiveScanConflictError,
    CreateScanCommand,
    CreateScanUseCase,
    SnapshotUnavailableError,
)
from tests.unit.use_cases.conftest import (
    FakeFeatureRunRepository,
    FakeFeatureStoreRepository,
    FakeTaskDispatcher,
    FakeUnitOfWork,
    FakeUniverseRepository,
)

SESSION = date(2026, 10, 2)
PROFILE = dict(screeners=["minervini", "canslim"], composite_method="weighted_average", criteria=None)


def _row(symbol: str, *, passes: bool, price: float = 100.0) -> FeatureRowWrite:
    return FeatureRowWrite(
        symbol=symbol,
        as_of_date=SESSION,
        composite_score=85.0 if passes else 40.0,
        overall_rating=4 if passes else 2,
        passes_count=1 if passes else 0,
        details={
            "composite_score": 85.0 if passes else 40.0,
            "custom_score": 90.0,
            "rating": "Buy" if passes else "Pass",
            "passes_template": passes,
            "current_price": price,
            "rs_rating": 80,
        },
    )


def _publish(
    runs: FakeFeatureRunRepository,
    store: FakeFeatureStoreRepository,
    rows: list[FeatureRowWrite],
    *,
    market: str = "US",
    as_of: date = SESSION,
    rs_formula: str = BALANCED_RS_FORMULA_VERSION,
    pointer_key: str | None = None,
    universe_hash: str = "whole-market",
) -> int:
    signature = build_scan_signature_payload(universe_type="market", **PROFILE)
    run = runs.start_run(
        as_of_date=as_of,
        run_type=RunType.DAILY_SNAPSHOT,
        universe_hash=universe_hash,
        input_hash=hash_scan_signature(signature),
        config_json={
            **signature,
            "signature": signature,
            "market": market,
            "rs_formula_version": rs_formula,
            "market_rs_run_id": 7 if rs_formula != LEGACY_RS_FORMULA_VERSION else None,
        },
    )
    runs.set_run_covered_symbols(run.id, [r.symbol for r in rows])
    runs.mark_completed(
        run.id,
        stats=RunStats(
            total_symbols=len(rows), processed_symbols=len(rows), failed_symbols=0,
            duration_seconds=1.0, passed_symbols=0,
        ),
    )
    runs.publish_atomically(run.id, pointer_key=pointer_key or f"latest_published_market:{market}")
    store.upsert_snapshot_rows(run.id, rows)
    return run.id


def _world(symbols, *, markets=None, rows=None, **publish_kwargs):
    runs, store = FakeFeatureRunRepository(), FakeFeatureStoreRepository()
    rows = rows if rows is not None else [
        _row("AAPL", passes=True),
        _row("MSFT", passes=False),
        _row("NVDA", passes=True, price=10.0),
        _row("TSLA", passes=True),  # outside every requested subset below
    ]
    run_id = _publish(runs, store, rows, **publish_kwargs)
    uow = FakeUnitOfWork(
        universe=FakeUniverseRepository(symbols, markets=markets),
        feature_runs=runs,
        feature_store=store,
    )
    return uow, run_id


def _cmd(**overrides) -> CreateScanCommand:
    values = dict(
        universe_def="index",
        universe_label="S&P 500",
        universe_key="index:SP500",
        universe_type="index",
        **PROFILE,
    )
    values.update(overrides)
    return CreateScanCommand(**values)


def _use_case(dispatcher=None, *, session=SESSION, freshness=None, rs_run_id=7):
    return CreateScanUseCase(
        dispatcher or FakeTaskDispatcher(),
        freshness_evaluator=freshness,
        expected_session=(lambda market: session) if session else None,
        current_rs_run_id=lambda market: rs_run_id,
    )


def _persisted(uow, scan_id):
    return {sym: res for sid, sym, res in uow.scan_results._persisted_results if sid == scan_id}


class TestSameProfileSubset:
    def test_index_reuses_scores_for_exactly_the_resolved_subset(self):
        uow, run_id = _world(["AAPL", "MSFT", "NVDA"])
        dispatcher = FakeTaskDispatcher()

        result = _use_case(dispatcher).execute(uow, _cmd())

        assert result.status == "completed"
        assert dispatcher.dispatched == []
        rows = _persisted(uow, result.scan_id)
        assert set(rows) == {"AAPL", "MSFT", "NVDA"}  # TSLA never leaks in
        assert rows["MSFT"]["passes_template"] is False  # source scores kept
        scan = uow.scans.get_by_scan_id(result.scan_id)
        assert scan.passed_stocks == 2
        assert scan.feature_run_id is None  # rows are materialized, not bound
        source = result.published_source
        assert source["match"] == "profile"
        assert source["feature_run_id"] == run_id
        assert source["as_of_date"] == SESSION.isoformat()
        assert source["is_current"] is True
        assert source["data_mode"] == "current"
        assert source["membership_count"] == 3
        assert scan.metadata_json["published_source"] == source

    def test_custom_universe_with_changed_criteria_uses_compiled_gate(self):
        uow, _ = _world(["AAPL", "NVDA"])
        criteria = {"custom_filters": {"price_min": 20}, "min_score": 70}

        result = _use_case().execute(
            uow,
            _cmd(universe_type="custom", universe_def="custom", screeners=["custom"], criteria=criteria),
        )

        assert result.status == "completed"
        rows = _persisted(uow, result.scan_id)
        assert set(rows) == {"AAPL"}  # NVDA at $10 fails the price gate
        # Source scores came from other criteria, so they are normalised.
        assert rows["AAPL"]["custom_score"] == 100.0
        assert rows["AAPL"]["screeners_run"] == ["custom"]
        assert result.published_source["match"] == "compiled"

    def test_zero_compiled_matches_still_completes(self):
        uow, _ = _world(["NVDA"])
        criteria = {"custom_filters": {"price_min": 20}, "min_score": 70}

        result = _use_case().execute(
            uow, _cmd(universe_type="custom", screeners=["custom"], criteria=criteria)
        )

        assert result.status == "completed"
        assert _persisted(uow, result.scan_id) == {}
        assert uow.scans.get_by_scan_id(result.scan_id).passed_stocks == 0


class TestCurrentModeFallsBackToCompute:
    @pytest.mark.parametrize(
        "world_kwargs, symbols, markets",
        [
            ({"as_of": date(2026, 10, 1)}, ["AAPL"], None),  # older than last session
            ({"rs_formula": LEGACY_RS_FORMULA_VERSION}, ["AAPL"], None),
            ({}, ["AAPL", "ZZZZ"], None),  # ZZZZ has no feature row
            ({}, ["AAPL", "0700.HK"], {"AAPL": "US", "0700.HK": "HK"}),
            ({}, ["AAPL", "GONE"], {"AAPL": "US"}),  # unknown listing
        ],
    )
    def test_ineligible_requests_dispatch_async(self, world_kwargs, symbols, markets):
        uow, _ = _world(symbols, markets=markets, **world_kwargs)
        dispatcher = FakeTaskDispatcher()

        result = _use_case(dispatcher).execute(uow, _cmd())

        assert result.status == "queued"
        assert len(dispatcher.dispatched) == 1

    def test_unknown_session_never_counts_as_current(self):
        uow, _ = _world(["AAPL"])
        dispatcher = FakeTaskDispatcher()

        result = _use_case(dispatcher, session=None).execute(uow, _cmd())

        assert result.status == "queued"

    def test_unpinned_market_cap_fact_is_not_served_from_snapshot(self):
        uow, _ = _world(["AAPL"])
        criteria = {"custom_filters": {"market_cap_min": 1e9}, "min_score": 70}

        result = _use_case().execute(
            uow, _cmd(universe_type="custom", screeners=["custom"], criteria=criteria)
        )

        assert result.status == "queued"

    def test_global_pointer_run_is_not_a_market_publication(self):
        uow, _ = _world(["AAPL"], pointer_key="latest_published")

        result = _use_case().execute(uow, _cmd())

        assert result.status == "queued"

    def test_active_scan_still_conflicts_in_current_mode(self):
        uow, _ = _world(["AAPL"])
        uow.scans.create(scan_id="running", status="running", total_stocks=5)

        with pytest.raises(ActiveScanConflictError):
            _use_case().execute(uow, _cmd())


class TestLastPublishedMode:
    def test_serves_an_older_publication_and_discloses_it(self):
        uow, _ = _world(["AAPL", "MSFT"], as_of=date(2026, 9, 30))

        result = _use_case().execute(uow, _cmd(data_mode="last_published"))

        assert result.status == "completed"
        assert result.published_source["as_of_date"] == "2026-09-30"
        assert result.published_source["is_current"] is False
        assert result.published_source["data_mode"] == "last_published"

    def test_succeeds_while_another_scan_is_active(self):
        uow, _ = _world(["AAPL"])
        uow.scans.create(scan_id="running", status="running", total_stocks=5)
        dispatcher = FakeTaskDispatcher()

        result = _use_case(dispatcher).execute(uow, _cmd(data_mode="last_published"))

        assert result.status == "completed"
        assert dispatcher.dispatched == []
        assert uow.scans.get_by_scan_id(result.scan_id).status == "completed"

    def test_skips_current_price_freshness(self):
        def stale(*_args, **_kwargs):
            raise AssertionError("last_published must not consult price freshness")

        uow, _ = _world(["AAPL"])

        result = _use_case(freshness=stale).execute(uow, _cmd(data_mode="last_published"))

        assert result.status == "completed"

    @pytest.mark.parametrize(
        "symbols, markets, world_kwargs, reason",
        [
            (["AAPL", "0700.HK"], {"AAPL": "US", "0700.HK": "HK"}, {}, "mixed_market_universe"),
            (["AAPL", "ZZZZ"], None, {}, "incomplete_coverage"),
            (["AAPL"], None, {"rs_formula": LEGACY_RS_FORMULA_VERSION}, "rs_not_canonical"),
            (["AAPL", "GONE"], {"AAPL": "US"}, {}, "unresolved_symbols"),
            (["0700.HK"], {"0700.HK": "HK"}, {}, "no_published_run"),
        ],
    )
    def test_ineligible_returns_reason_without_compute(self, symbols, markets, world_kwargs, reason):
        uow, _ = _world(symbols, markets=markets, **world_kwargs)
        dispatcher = FakeTaskDispatcher()

        with pytest.raises(SnapshotUnavailableError) as exc:
            _use_case(dispatcher).execute(uow, _cmd(data_mode="last_published"))

        assert exc.value.to_dict()["code"] == "snapshot_unavailable"
        assert exc.value.to_dict()["reason"] == reason
        assert dispatcher.dispatched == []
        assert uow.scans.rows == []

    def test_native_currency_cap_filter_is_not_equivalent(self):
        uow, _ = _world(["0700.HK"], markets={"0700.HK": "HK"}, market="HK",
                        rows=[_row("0700.HK", passes=True)])
        criteria = {"custom_filters": {"market_cap_min": 1e9}, "min_score": 70}

        with pytest.raises(SnapshotUnavailableError) as exc:
            _use_case().execute(
                uow,
                _cmd(universe_type="custom", screeners=["custom"], criteria=criteria,
                     data_mode="last_published"),
            )

        assert exc.value.to_dict()["reason"] == "criteria_not_equivalent"

    def test_mixed_market_all_universe_is_unavailable(self):
        uow, _ = _world(["AAPL", "0700.HK"], markets={"AAPL": "US", "0700.HK": "HK"})

        with pytest.raises(SnapshotUnavailableError) as exc:
            _use_case().execute(
                uow, _cmd(universe_type="all", universe_def="all", data_mode="last_published")
            )

        assert exc.value.to_dict()["reason"] == "mixed_market_universe"

    def test_idempotent_retry_returns_the_recorded_source(self):
        uow, _ = _world(["AAPL"])
        cmd = _cmd(data_mode="last_published", idempotency_key="k1")
        first = _use_case().execute(uow, cmd)

        again = _use_case().execute(uow, cmd)

        assert again.is_duplicate is True
        assert again.scan_id == first.scan_id
        assert again.published_source == first.published_source

    def test_rejects_unknown_data_mode(self):
        from app.domain.common.errors import ValidationError

        uow, _ = _world(["AAPL"])
        with pytest.raises(ValidationError):
            _use_case().execute(uow, _cmd(data_mode="whenever"))



class TestReviewFindings:
    def test_compiled_gate_never_passes_insufficient_history_rows(self):
        listing_only = _row("NEWCO", passes=False, price=50.0)
        listing_only.details.update(
            result_status="insufficient_history", scan_mode="listing_only", rating="Insufficient Data"
        )
        uow, _ = _world(["AAPL", "NEWCO"], rows=[_row("AAPL", passes=True), listing_only])
        criteria = {"custom_filters": {"price_min": 20}, "min_score": 70}

        result = _use_case().execute(
            uow, _cmd(universe_type="custom", screeners=["custom"], criteria=criteria)
        )

        assert set(_persisted(uow, result.scan_id)) == {"AAPL"}
        assert uow.scans.get_by_scan_id(result.scan_id).passed_stocks == 1

    def test_current_mode_rejects_a_superseded_market_rs_run(self):
        uow, _ = _world(["AAPL"])
        dispatcher = FakeTaskDispatcher()

        result = _use_case(dispatcher, rs_run_id=8).execute(uow, _cmd())

        assert result.status == "queued"

    def test_last_published_keeps_the_runs_own_rs(self):
        uow, _ = _world(["AAPL"])

        result = _use_case(rs_run_id=8).execute(uow, _cmd(data_mode="last_published"))

        assert result.status == "completed"

    def test_session_is_resolved_once_and_recorded(self):
        calls = []

        def session_for(market):
            calls.append(market)
            # A second call would land after the close and disagree.
            return SESSION if len(calls) == 1 else date(2026, 10, 5)

        uow, _ = _world(["AAPL"])
        use_case = CreateScanUseCase(
            FakeTaskDispatcher(), expected_session=session_for, current_rs_run_id=lambda m: 7
        )

        result = use_case.execute(uow, _cmd())

        assert calls == ["US"]
        assert result.published_source["is_current"] is True
        assert result.published_source["expected_session"] == SESSION.isoformat()

    def test_lookup_failure_falls_back_inside_a_savepoint(self):
        class _Savepoint:
            entered = exited_with = None

            def __enter__(self):
                type(self).entered = True

            def __exit__(self, exc_type, *_):
                type(self).exited_with = exc_type
                return False

        class _Session:
            def begin_nested(self):
                return _Savepoint()

        uow, _ = _world(["AAPL"])
        uow.session = _Session()

        def boom(*_a, **_k):
            raise RuntimeError("statement timeout")

        uow.feature_runs.has_feature_rows_for = boom
        dispatcher = FakeTaskDispatcher()

        result = _use_case(dispatcher).execute(uow, _cmd())

        assert result.status == "queued"
        assert _Savepoint.entered and _Savepoint.exited_with is RuntimeError

    def test_concurrent_duplicate_idempotency_key_returns_the_winner(self):
        from app.domain.scanning.errors import DuplicateIdempotencyKey

        uow, _ = _world(["AAPL"])
        winner = uow.scans.create(scan_id="winner", status="completed", total_stocks=1)
        lookups = iter([None, winner])
        uow.scans.get_by_idempotency_key = lambda key: next(lookups)

        def create(**_fields):
            raise DuplicateIdempotencyKey("taken")

        uow.scans.create = create

        result = _use_case().execute(uow, _cmd(data_mode="last_published", idempotency_key="k"))

        assert result.is_duplicate is True
        assert result.scan_id == "winner"

    def test_existing_market_compile_path_also_drops_insufficient_rows(self):
        listing_only = _row("NEWCO", passes=False, price=50.0)
        listing_only.details.update(result_status="insufficient_history", scan_mode="listing_only")
        uow, _ = _world(
            ["AAPL", "NEWCO"],
            rows=[_row("AAPL", passes=True), listing_only],
            pointer_key="latest_published",
        )
        criteria = {"custom_filters": {"price_min": 20}, "min_score": 70}

        result = _use_case().execute(
            uow,
            _cmd(universe_type="all", universe_def="all", screeners=["custom"], criteria=criteria),
        )

        assert set(_persisted(uow, result.scan_id)) == {"AAPL"}


class TestLastPublishedIgnoresExactShortcut:
    """An exact signature match is not proof of a pinned, complete, canonical run."""

    SYMBOLS = ["AAPL", "MSFT"]

    def _market_cmd(self):
        return _cmd(
            universe_type="market", universe_def="market", universe_market="US",
            data_mode="last_published",
        )

    def _exact_world(self, **exact_kwargs):
        from app.domain.scanning.signature import hash_universe_symbols

        uow, _ = _world(self.SYMBOLS, rows=[_row("AAPL", passes=True), _row("MSFT", passes=True)])
        exact_rows = exact_kwargs.pop("rows", [_row("AAPL", passes=True), _row("MSFT", passes=True)])
        exact_id = _publish(
            uow.feature_runs, uow.feature_store, exact_rows,
            universe_hash=hash_universe_symbols(self.SYMBOLS),
            pointer_key="exact-only",
            **exact_kwargs,
        )
        return uow, exact_id

    def test_superseded_exact_run_is_not_served(self):
        uow, exact_id = self._exact_world(as_of=date(2026, 9, 1))
        pointer_id = uow.feature_runs.get_latest_published("latest_published_market:US").id

        result = _use_case().execute(uow, self._market_cmd())

        assert result.published_source["feature_run_id"] == pointer_id != exact_id

    def test_legacy_rs_exact_run_is_not_served(self):
        uow, _ = self._exact_world(rs_formula=LEGACY_RS_FORMULA_VERSION)
        uow.feature_runs._pointers.pop("latest_published_market:US")

        with pytest.raises(SnapshotUnavailableError) as exc:
            _use_case().execute(uow, self._market_cmd())

        assert exc.value.to_dict()["reason"] == "no_published_run"

    def test_exact_run_missing_a_row_is_not_served(self):
        uow, exact_id = self._exact_world(rows=[_row("AAPL", passes=True)])
        uow.feature_runs.repoint_published(exact_id, "latest_published_market:US")

        with pytest.raises(SnapshotUnavailableError) as exc:
            _use_case().execute(uow, self._market_cmd())

        assert exc.value.to_dict()["reason"] == "incomplete_coverage"

    def test_single_market_all_universe_uses_the_market_publication(self):
        uow, run_id = _world(self.SYMBOLS)

        result = _use_case().execute(
            uow, _cmd(universe_type="all", universe_def="all", data_mode="last_published")
        )

        assert result.published_source["feature_run_id"] == run_id
