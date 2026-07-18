# PR 303 Review Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve the still-valid review findings on PR #303 while preserving cache-only refresh-guard behavior, manual historical provider access, and existing task/API compatibility.

**Architecture:** Extend the existing cache-only price read boundary with optional target-session validation, then pass that requirement only from daily breadth and group-ranking paths. Keep compatibility adapters at public task/service boundaries, reuse the established transient-database classifier, and make coverage/accounting invariants explicit without changing historical execution defaults.

**Tech Stack:** Python 3.11, pandas, SQLAlchemy, Celery, pytest, PyYAML, GitHub Actions, GitHub CLI.

## Global Constraints

- Do not edit historical implementation-plan documents.
- Preserve existing Celery task names, arguments, and non-transient error payloads.
- Preserve `BreadthCalculatorService.backfill_range(..., cache_only=True)`.
- Preserve provider-capable manual historical breadth and group runs by default.
- Preserve cache-only execution after the refresh guard.
- Add a failing regression test before every production behavior change.
- Do not stage or overwrite unrelated `.beads/issues.jsonl` changes.
- Use `PYTHON="${PYTHON:-python3}"` in portable verification commands; callers may override it with a project virtualenv interpreter.

---

### Task 1: Enforce optional target-session coverage in cache-only price reads

**Files:**
- Modify: `backend/app/services/price_cache_service.py:218-291`
- Test: `backend/tests/unit/test_yahoo_batch_ingestion.py`

**Interfaces:**
- Consumes: cached price DataFrames whose index values are convertible by `pandas.Timestamp`.
- Produces: `PriceCacheService.get_cached_only_fresh(symbol, period="2y", *, required_as_of_date: date | None = None)` and `get_many_cached_only_fresh(symbols, period="2y", *, required_as_of_date: date | None = None)`.

- [ ] **Step 1: Write failing single- and bulk-read tests**

Add:

```python
def test_get_cached_only_fresh_requires_requested_session(monkeypatch):
    service = PriceCacheService(
        redis_client=None,
        session_factory=lambda: MagicMock(),
    )
    frame = _price_df(date(2026, 3, 20), 123.0)
    monkeypatch.setattr(
        service,
        "_get_from_database",
        lambda symbol, period: (frame, date(2026, 3, 20)),
    )
    monkeypatch.setattr(service, "_is_data_fresh", lambda _last: True)
    monkeypatch.setattr(
        service,
        "_is_intraday_data_stale",
        lambda _symbol: False,
    )

    assert service.get_cached_only_fresh(
        "AAPL",
        required_as_of_date=date(2026, 3, 19),
    ) is None
    assert service.get_cached_only_fresh(
        "AAPL",
        required_as_of_date=date(2026, 3, 20),
    ) is frame


def test_get_many_cached_only_fresh_requires_requested_session(monkeypatch):
    service = PriceCacheService(
        redis_client=None,
        session_factory=lambda: MagicMock(),
    )
    complete = pd.concat(
        [
            _price_df(date(2026, 3, 19), 122.0),
            _price_df(date(2026, 3, 20), 123.0),
        ]
    )
    missing_target = _price_df(date(2026, 3, 20), 123.0)
    monkeypatch.setattr(
        service,
        "_get_many_from_database",
        lambda symbols, period: {
            "AAPL": (complete, date(2026, 3, 20)),
            "MSFT": (missing_target, date(2026, 3, 20)),
        },
    )
    monkeypatch.setattr(service, "_is_data_fresh", lambda _last: True)
    monkeypatch.setattr(
        service,
        "_is_intraday_data_stale",
        lambda _symbol: False,
    )

    result = service.get_many_cached_only_fresh(
        ["AAPL", "MSFT"],
        required_as_of_date=date(2026, 3, 19),
    )

    assert result["AAPL"] is complete
    assert result["MSFT"] is None
```

- [ ] **Step 2: Run the tests and verify the missing keyword fails**

Run:

```bash
cd backend
"$PYTHON" -m pytest \
  tests/unit/test_yahoo_batch_ingestion.py::test_get_cached_only_fresh_requires_requested_session \
  tests/unit/test_yahoo_batch_ingestion.py::test_get_many_cached_only_fresh_requires_requested_session -q
```

Expected: both tests fail with `TypeError` because `required_as_of_date` is not accepted.

- [ ] **Step 3: Implement the shared exact-session predicate and optional keywords**

Add inside `PriceCacheService`:

```python
    @staticmethod
    def _contains_required_as_of_date(
        data: Optional[pd.DataFrame],
        required_as_of_date: date | None,
    ) -> bool:
        if required_as_of_date is None:
            return True
        if data is None or data.empty:
            return False
        return any(
            pd.Timestamp(index_value).date() == required_as_of_date
            for index_value in data.index
        )
```

Extend both method signatures with the keyword-only argument. In
`get_cached_only_fresh`, return `None` after the freshness/intraday checks when
the predicate is false. In `get_many_cached_only_fresh`, include the predicate
in the existing success condition:

```python
                and self._contains_required_as_of_date(
                    data,
                    required_as_of_date,
                )
```

- [ ] **Step 4: Run the focused and existing cache tests**

Run:

```bash
cd backend
"$PYTHON" -m pytest \
  tests/unit/test_yahoo_batch_ingestion.py::test_get_cached_only_fresh_requires_requested_session \
  tests/unit/test_yahoo_batch_ingestion.py::test_get_many_cached_only_fresh_requires_requested_session \
  tests/unit/test_yahoo_batch_ingestion.py::test_get_many_cached_only_fresh_filters_stale_database_rows -q
```

Expected: three tests pass.

- [ ] **Step 5: Commit the cache boundary**

```bash
git add backend/app/services/price_cache_service.py backend/tests/unit/test_yahoo_batch_ingestion.py
git commit -m "fix: require target session in cache-only reads"
```

### Task 2: Make breadth target-date complete and preserve legacy backfill calls

**Files:**
- Modify: `backend/app/services/breadth_calculator_service.py:72-291,542-567`
- Modify: `backend/app/services/daily_breadth_runner.py:139-178`
- Test: `backend/tests/unit/test_breadth_calculator_service.py`
- Test: `backend/tests/unit/test_daily_breadth_runner.py`

**Interfaces:**
- Consumes: Task 1's `required_as_of_date` cache keyword and existing `DerivedDataExecutionPolicy` values.
- Produces: target-session-aware daily breadth and the compatibility keyword `cache_only: bool | None = None` on `backfill_range`.

- [ ] **Step 1: Write a failing target-session forwarding test**

Add to `test_breadth_calculator_service.py`:

```python
def test_cache_only_daily_breadth_requires_calculation_session(monkeypatch):
    db = MagicMock()
    db.query.return_value.filter.return_value.all.return_value = [
        SimpleNamespace(symbol="AAA")
    ]
    price_cache = MagicMock()
    price_cache.get_many_cached_only_fresh.return_value = {"AAA": None}
    service = BreadthCalculatorService(db, price_cache)
    monkeypatch.setattr(
        service,
        "_calculate_ratios",
        lambda _date: {"ratio_5day": None, "ratio_10day": None},
    )
    calculation_date = date(2026, 3, 20)

    result = service.calculate_daily_breadth(
        calculation_date,
        policy=_policy("refresh_guarded", calculation_date),
    )

    assert result.coverage.cache_miss_stocks == 1
    price_cache.get_many_cached_only_fresh.assert_called_once_with(
        ["AAA"],
        period="2y",
        required_as_of_date=calculation_date,
    )
```

- [ ] **Step 2: Write a failing legacy `cache_only` compatibility test**

Add:

```python
def test_backfill_range_accepts_legacy_cache_only_keyword():
    db = _make_db_session()
    db.add(
        StockUniverse(
            symbol="AAA",
            is_active=True,
            status=UNIVERSE_STATUS_ACTIVE,
        )
    )
    db.commit()
    target = date(2026, 3, 20)
    price_cache = MagicMock()
    price_cache.get_many_cached_only_fresh.return_value = {
        "AAA": _make_price_df(target)
    }
    price_cache.get_historical_data.side_effect = AssertionError(
        "legacy cache-only backfill must not call a provider"
    )

    result = BreadthCalculatorService(db, price_cache).backfill_range(
        target,
        target,
        trading_dates=[target],
        cache_only=True,
    )

    assert result["processed"] == 1
    assert result["cache_miss_stocks"] == 0
    price_cache.get_historical_data.assert_not_called()
```

- [ ] **Step 3: Write a failing zero-usable-stock strict validation test**

Add to `test_daily_breadth_runner.py`:

```python
def test_strict_cache_only_rejects_zero_usable_stocks():
    calculator = MagicMock()
    calculator.calculate_daily_breadth.return_value = _calculation(
        scanned=0,
        skipped=100,
        misses=0,
    )
    dependencies = _dependencies(calculator)

    with pytest.raises(
        IncompleteDailyBreadth,
        match="processed no usable stocks",
    ):
        run_daily_breadth(
            MagicMock(),
            DailyBreadthRequest(
                calculation_date=CALCULATION_DATE,
                market="US",
                policy=_policy("strict_cache_only"),
            ),
            dependencies,
        )

    calculator.store_daily_breadth.assert_not_called()
    dependencies.publish_snapshot.assert_not_called()
```

- [ ] **Step 4: Run the three tests and verify their distinct failures**

Run:

```bash
cd backend
"$PYTHON" -m pytest \
  tests/unit/test_breadth_calculator_service.py::test_cache_only_daily_breadth_requires_calculation_session \
  tests/unit/test_breadth_calculator_service.py::test_backfill_range_accepts_legacy_cache_only_keyword \
  tests/unit/test_daily_breadth_runner.py::test_strict_cache_only_rejects_zero_usable_stocks -q
```

Expected: the first fails on the missing cache-call keyword, the second fails
with `TypeError: unexpected keyword argument 'cache_only'`, and the third fails
because no exception is raised.

- [ ] **Step 5: Implement daily target-date forwarding**

Extend `_load_price_data_for_batch`:

```python
    def _load_price_data_for_batch(
        self,
        batch_symbols: List[str],
        cache_only: bool,
        *,
        required_as_of_date: date | None = None,
    ) -> tuple[Dict[str, Optional[pd.DataFrame]], List[str]]:
        cache_kwargs = {"period": "2y"}
        if required_as_of_date is not None:
            cache_kwargs["required_as_of_date"] = required_as_of_date
        price_data_by_symbol = self.price_cache.get_many_cached_only_fresh(
            batch_symbols,
            **cache_kwargs,
        )
```

Pass `required_as_of_date=calculation_date if policy.cache_only else None` from
`calculate_daily_breadth`. Leave multi-date backfill calls without the
requirement so historical date reuse remains unchanged.

- [ ] **Step 6: Implement the legacy backfill policy adapter**

Import `DerivedDataExecutionMode` and `DerivedDataTargetKind`. Add the optional
keyword to `backfill_range`, then normalize it before any cache reads:

```python
        if cache_only is not None:
            policy = DerivedDataExecutionPolicy(
                mode=(
                    DerivedDataExecutionMode.STRICT_CACHE_ONLY
                    if cache_only
                    else DerivedDataExecutionMode.AUTO
                ),
                target_kind=DerivedDataTargetKind.HISTORICAL,
            )
```

When `cache_only` is omitted, retain the caller-supplied typed policy exactly.

- [ ] **Step 7: Reject strict runs with zero usable stocks**

In `_validate_strict_cache_only_breadth`, add this check after
`total_attempted == 0` and before the miss-ratio calculation:

```python
    if coverage.total_stocks_scanned == 0:
        return "Cache-only breadth run processed no usable stocks"
```

- [ ] **Step 8: Correct the daily breadth return documentation**

Replace the dictionary-shaped `Returns` block with:

```python
        Returns:
            BreadthCalculationResult containing ``indicators`` and the
            authoritative ``coverage`` report. Use ``to_metrics_dict()`` when
            a merged persistence mapping is required. Task responses add
            execution-policy metadata at their serialization boundary.
```

- [ ] **Step 9: Run focused breadth suites**

Run:

```bash
cd backend
"$PYTHON" -m pytest \
  tests/unit/test_breadth_calculator_service.py \
  tests/unit/test_daily_breadth_runner.py \
  tests/unit/test_breadth_tasks.py \
  tests/unit/test_export_static_site_script.py -q
```

Expected: all selected tests pass.

- [ ] **Step 10: Commit breadth fixes**

```bash
git add \
  backend/app/services/breadth_calculator_service.py \
  backend/app/services/daily_breadth_runner.py \
  backend/tests/unit/test_breadth_calculator_service.py \
  backend/tests/unit/test_daily_breadth_runner.py
git commit -m "fix: enforce cache-only breadth completeness"
```

### Task 3: Require target-session group-ranking inputs

**Files:**
- Modify: `backend/app/services/group_rank_input_loader.py:49-145,244-274`
- Modify: `backend/app/services/ibd_group_rank_service.py:115-155,538-551`
- Test: `backend/tests/unit/test_group_rank_input_loader.py`
- Test: `backend/tests/unit/test_group_rank_service.py`

**Interfaces:**
- Consumes: Task 1's target-session cache keyword and daily `calculation_date`.
- Produces: optional `calculation_date: date | None = None` on `GroupRankInputLoader.load` and `IBDGroupRankService._prefetch_all_data`.

- [ ] **Step 1: Write failing loader tests for benchmark and constituent coverage**

Add to `test_group_rank_input_loader.py`:

```python
def test_guarded_load_requires_target_session_for_benchmark(db_session):
    target = date(2026, 3, 20)
    price_cache = Mock()
    price_cache.get_cached_only_fresh.return_value = None
    benchmark_cache = Mock()
    benchmark_cache.get_benchmark_symbol.return_value = "SPY"
    benchmark_cache.get_benchmark_candidates.return_value = ["SPY"]
    loader = _loader(
        price_cache=price_cache,
        benchmark_cache=benchmark_cache,
        groups={"Software": ("AAPL",)},
        active=("AAPL",),
    )

    prefetch = loader.load(
        db_session,
        market="US",
        policy=_policy("refresh_guarded"),
        calculation_date=target,
    )

    assert prefetch.stats.benchmark_available is False
    price_cache.get_cached_only_fresh.assert_called_once_with(
        "SPY",
        period="2y",
        required_as_of_date=target,
    )


def test_guarded_load_requires_target_session_for_constituents(db_session):
    target = date(2026, 3, 20)
    benchmark_prices = _price_frame()
    price_cache = Mock()
    price_cache.get_cached_only_fresh.return_value = benchmark_prices
    price_cache.get_many_cached_only_fresh.return_value = {"AAPL": None}
    benchmark_cache = Mock()
    benchmark_cache.get_benchmark_symbol.return_value = "SPY"
    benchmark_cache.get_benchmark_candidates.return_value = ["SPY"]
    loader = _loader(
        price_cache=price_cache,
        benchmark_cache=benchmark_cache,
        groups={"Software": ("AAPL",)},
        active=("AAPL",),
    )

    prefetch = loader.load(
        db_session,
        market="US",
        policy=_policy("refresh_guarded"),
        calculation_date=target,
    )

    assert prefetch.stats.cache_miss_symbols == 1
    assert prefetch.stats.cache_miss_symbols_sample == ("AAPL",)
    price_cache.get_many_cached_only_fresh.assert_called_once_with(
        ["AAPL"],
        period="2y",
        required_as_of_date=target,
    )
```

- [ ] **Step 2: Run the loader tests and verify the missing keyword fails**

Run:

```bash
cd backend
"$PYTHON" -m pytest \
  tests/unit/test_group_rank_input_loader.py::test_guarded_load_requires_target_session_for_benchmark \
  tests/unit/test_group_rank_input_loader.py::test_guarded_load_requires_target_session_for_constituents -q
```

Expected: both tests fail because `GroupRankInputLoader.load` does not accept
`calculation_date`.

- [ ] **Step 3: Implement optional date forwarding in the loader**

Add `calculation_date: date | None = None` to `load`. Add
`required_as_of_date: date | None = None` to `_get_cached_benchmark`. Build
cache kwargs conditionally so existing historical/mock calls remain unchanged:

```python
        benchmark_cache_kwargs = {"period": "2y"}
        if calculation_date is not None:
            benchmark_cache_kwargs["required_as_of_date"] = calculation_date
```

Pass those kwargs through `_get_cached_benchmark` to
`get_cached_only_fresh`. Apply the same conditional kwargs pattern to
`get_many_cached_only_fresh` for constituent histories.

- [ ] **Step 4: Write a failing service forwarding test**

Add to `test_group_rank_service.py`, using `_make_group_rank_service()` and a
`Mock` replacement for its loader:

```python
def test_prefetch_for_daily_calculation_forwards_target_date():
    service = _make_group_rank_service()
    expected = GroupRankPrefetchData(
        benchmark_prices=_price_frame(),
        prices_by_symbol={},
        active_symbols=frozenset(),
        market_caps={},
        stats=_prefetch_stats(0),
        symbols_by_group={},
        group_names=(),
    )
    service.input_loader.load = Mock(return_value=expected)
    db = Mock()
    target = date(2026, 3, 20)
    policy = _policy("refresh_guarded", target)

    result = service._prefetch_all_data(
        db,
        market="US",
        policy=policy,
        calculation_date=target,
    )

    assert result is expected
    service.input_loader.load.assert_called_once_with(
        db,
        market="US",
        policy=policy,
        calculation_date=target,
    )
```

- [ ] **Step 5: Run the service test and verify it fails on the missing keyword**

Run:

```bash
cd backend
"$PYTHON" -m pytest \
  tests/unit/test_group_rank_service.py::test_prefetch_for_daily_calculation_forwards_target_date -q
```

Expected: failure because `_prefetch_all_data` does not accept
`calculation_date`.

- [ ] **Step 6: Forward the daily calculation date through the facade**

Add the optional parameter to `_prefetch_all_data`, pass it to
`input_loader.load`, and update `calculate_group_rankings`:

```python
        raw_prefetch = self._prefetch_all_data(
            db,
            market=normalized_market,
            policy=policy,
            calculation_date=calculation_date,
        )
```

Do not add the parameter to `GroupRankHistoricalCalculator._load_prefetch`;
multi-date historical execution retains its existing behavior.

- [ ] **Step 7: Run focused group input and facade suites**

Run:

```bash
cd backend
"$PYTHON" -m pytest \
  tests/unit/test_group_rank_input_loader.py \
  tests/unit/test_group_rank_service.py \
  tests/unit/test_daily_group_rank_runner.py \
  tests/unit/test_group_rank_tasks.py -q
```

Expected: all selected tests pass, including the existing single-pass taxonomy
regression.

- [ ] **Step 8: Commit group input completeness**

```bash
git add \
  backend/app/services/group_rank_input_loader.py \
  backend/app/services/ibd_group_rank_service.py \
  backend/tests/unit/test_group_rank_input_loader.py \
  backend/tests/unit/test_group_rank_service.py
git commit -m "fix: require target session for group rankings"
```

### Task 4: Make breadth price coverage accounting internally consistent

**Files:**
- Modify: `backend/app/services/breadth_coverage.py:20-55`
- Test: `backend/tests/unit/test_breadth_coverage.py`

**Interfaces:**
- Consumes: candidate and missing symbol iterables for one batch.
- Produces: atomic `record_batch` validation with disjoint cached/missing sets.

- [ ] **Step 1: Write failing membership and conflict tests**

Add:

```python
def test_price_coverage_rejects_misses_outside_candidate_batch():
    accumulator = BreadthPriceCoverageAccumulator()

    with pytest.raises(ValueError, match="outside candidate batch"):
        accumulator.record_batch(["AAA"], ["BBB"])

    assert accumulator.report().candidate_stocks == 0


def test_price_coverage_rejects_conflicting_repeated_classification():
    accumulator = BreadthPriceCoverageAccumulator()
    accumulator.record_batch(["AAA", "BBB"], ["BBB"])

    with pytest.raises(ValueError, match="conflicting classification"):
        accumulator.record_batch(["BBB"], [])

    report = accumulator.report()
    assert report.candidate_stocks == 2
    assert report.symbols_with_cached_history == 1
    assert report.cache_miss_stocks == 1
    assert report.cache_coverage_ratio == 0.5
```

Add `import pytest` to the test module.

- [ ] **Step 2: Run both tests and verify inconsistent reports are currently accepted**

Run:

```bash
cd backend
"$PYTHON" -m pytest \
  tests/unit/test_breadth_coverage.py::test_price_coverage_rejects_misses_outside_candidate_batch \
  tests/unit/test_breadth_coverage.py::test_price_coverage_rejects_conflicting_repeated_classification -q
```

Expected: both fail because no `ValueError` is raised.

- [ ] **Step 3: Validate before mutating accumulator state**

Replace the body of `record_batch` with:

```python
        candidates = set(candidate_symbols)
        misses = set(cache_miss_symbols)
        outside_candidates = misses - candidates
        if outside_candidates:
            raise ValueError(
                "Cache misses outside candidate batch: "
                f"{sorted(outside_candidates)}"
            )

        cached = candidates - misses
        conflicting = (
            (misses & self._cached_symbols)
            | (cached & self._cache_miss_symbols)
        )
        if conflicting:
            raise ValueError(
                "Repeated symbols have conflicting classification: "
                f"{sorted(conflicting)}"
            )

        self._candidate_symbols.update(candidates)
        self._cache_miss_symbols.update(misses)
        self._cached_symbols.update(cached)
```

- [ ] **Step 4: Run the complete coverage and breadth suites**

Run:

```bash
cd backend
"$PYTHON" -m pytest \
  tests/unit/test_breadth_coverage.py \
  tests/unit/test_breadth_calculator_service.py \
  tests/unit/test_daily_breadth_runner.py \
  tests/unit/test_breadth_tasks.py -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit coverage invariants**

```bash
git add backend/app/services/breadth_coverage.py backend/tests/unit/test_breadth_coverage.py
git commit -m "fix: keep breadth coverage classifications disjoint"
```

### Task 5: Validate group backfill chunks and propagate transient database failures

**Files:**
- Modify: `backend/app/services/group_rank_historical_calculator.py:358-406`
- Modify: `backend/app/tasks/group_rank_backfill_tasks.py:1-255`
- Test: `backend/tests/unit/test_group_rank_historical_calculator.py`
- Test: `backend/tests/unit/test_group_rank_backfill_tasks.py`

**Interfaces:**
- Consumes: `raise_if_transient_database_error(exc)` from `app.tasks.transient_database`.
- Produces: synchronous positive chunk-size validation and retry-visible transient task failures.

- [ ] **Step 1: Write a failing non-positive chunk-size test**

Add to `test_group_rank_historical_calculator.py`:

```python
@pytest.mark.parametrize("chunk_size_days", [0, -1])
def test_backfill_rankings_chunked_rejects_non_positive_size(
    monkeypatch,
    chunk_size_days,
):
    historical, _, _, _, _ = _historical()
    monkeypatch.setattr(
        historical_module,
        "timedelta",
        MagicMock(
            side_effect=AssertionError(
                "chunk loop entered before size validation"
            )
        ),
    )

    with pytest.raises(ValueError, match="chunk_size_days must be positive"):
        historical.backfill_rankings_chunked(
            MagicMock(),
            date(2026, 3, 1),
            date(2026, 3, 20),
            chunk_size_days=chunk_size_days,
        )
```

- [ ] **Step 2: Run the sentinel test and verify validation happens too late**

Add `import pytest` to the module, then run:

```bash
cd backend
"$PYTHON" -m pytest \
  tests/unit/test_group_rank_historical_calculator.py::test_backfill_rankings_chunked_rejects_non_positive_size \
  -q
```

Expected: both cases fail with the sentinel `AssertionError`, proving the loop
is entered before argument validation without allowing the infinite loop to
continue.

- [ ] **Step 3: Add validation before the chunk loop**

Add at the start of `backfill_rankings_chunked`:

```python
        if chunk_size_days < 1:
            raise ValueError("chunk_size_days must be positive")
```

- [ ] **Step 4: Write retry tests for all three administrative tasks**

Add these imports to `test_group_rank_backfill_tasks.py`:

```python
import pytest
from celery.exceptions import Retry
from sqlalchemy.exc import OperationalError
```

Add a helper and parameterized test:

```python
def _transient_database_error():
    return OperationalError(
        "select 1",
        {},
        Exception("database system is not yet accepting connections"),
    )


def _configure_failing_group_backfill(
    monkeypatch,
    task_name,
    exc,
):
    import app.tasks.group_rank_backfill_tasks as module

    fake_db = MagicMock()
    fake_service = MagicMock()
    monkeypatch.setattr(module, "SessionLocal", lambda: fake_db)
    monkeypatch.setattr(
        module,
        "get_group_rank_service",
        lambda: fake_service,
    )
    _patch_serialized_lock(monkeypatch)

    if task_name == "backfill_group_rankings":
        fake_service.backfill_rankings_optimized.side_effect = exc
        args = ("2026-03-20", "2026-03-20")
        kwargs = {}
    elif task_name == "gapfill_group_rankings":
        fake_service.find_missing_dates.return_value = [
            datetime(2026, 3, 20).date()
        ]
        fake_service.fill_gaps_optimized.side_effect = exc
        args = ()
        kwargs = {"max_days": 1}
    else:
        fake_service.backfill_rankings_optimized.side_effect = exc
        _patch_calendar_service(
            monkeypatch,
            module,
            datetime(2026, 3, 20, 17, 40),
        )
        args = ()
        kwargs = {}

    return getattr(module, task_name), args, kwargs, fake_db


@pytest.mark.parametrize(
    "task_name",
    [
        "backfill_group_rankings",
        "gapfill_group_rankings",
        "backfill_group_rankings_1year",
    ],
)
def test_group_backfill_tasks_retry_transient_database_errors(
    monkeypatch,
    task_name,
):
    task, args, kwargs, fake_db = _configure_failing_group_backfill(
        monkeypatch,
        task_name,
        _transient_database_error(),
    )
    retry = MagicMock(side_effect=Retry("retry"))
    monkeypatch.setattr(task, "retry", retry)

    with pytest.raises(Retry):
        task.run(*args, **kwargs)

    retry.assert_called_once()
    fake_db.rollback.assert_called_once()
    fake_db.close.assert_called_once()
```

- [ ] **Step 5: Run the retry tests and verify handlers currently swallow the errors**

Run:

```bash
cd backend
"$PYTHON" -m pytest \
  tests/unit/test_group_rank_backfill_tasks.py::test_group_backfill_tasks_retry_transient_database_errors -q
```

Expected: all parameter cases fail because the task returns an error dictionary
instead of raising `Retry`.

- [ ] **Step 6: Propagate only recognized transient database errors**

Import:

```python
from .transient_database import raise_if_transient_database_error
```

In each of the three broad exception handlers, retain rollback and logging,
then call this before constructing the existing error dictionary:

```python
        raise_if_transient_database_error(exc)
```

Do not re-raise non-transient exceptions.

- [ ] **Step 7: Add a non-transient compatibility regression**

Add:

```python
@pytest.mark.parametrize(
    "task_name",
    [
        "backfill_group_rankings",
        "gapfill_group_rankings",
        "backfill_group_rankings_1year",
    ],
)
def test_group_backfill_tasks_preserve_non_transient_error_payloads(
    monkeypatch,
    task_name,
):
    task, args, kwargs, fake_db = _configure_failing_group_backfill(
        monkeypatch,
        task_name,
        RuntimeError("calculation failed"),
    )
    retry = MagicMock()
    monkeypatch.setattr(task, "retry", retry)

    result = task.run(*args, **kwargs)

    assert result["error"] == "calculation failed"
    retry.assert_not_called()
    fake_db.rollback.assert_called_once()
    fake_db.close.assert_called_once()
```

- [ ] **Step 8: Run historical and task suites**

Run:

```bash
cd backend
"$PYTHON" -m pytest \
  tests/unit/test_group_rank_historical_calculator.py \
  tests/unit/test_group_rank_backfill_tasks.py \
  tests/unit/test_workload_coordination.py \
  tests/unit/test_transient_database.py -q
```

Expected: all selected tests pass.

- [ ] **Step 9: Commit group backfill safety**

```bash
git add \
  backend/app/services/group_rank_historical_calculator.py \
  backend/app/tasks/group_rank_backfill_tasks.py \
  backend/tests/unit/test_group_rank_historical_calculator.py \
  backend/tests/unit/test_group_rank_backfill_tasks.py
git commit -m "fix: preserve retries in group backfills"
```

### Task 6: Restore US calendar compatibility and guard manual task headers

**Files:**
- Modify: `backend/app/use_cases/feature_store/build_daily_snapshot.py:69-71`
- Modify: `backend/app/services/price_refresh_workflow.py:480-494`
- Test: `backend/tests/unit/use_cases/feature_store/test_build_daily_snapshot.py`
- Test: `backend/tests/unit/test_price_refresh_workflow.py`

**Interfaces:**
- Consumes: `MarketCalendarService.is_trading_day(market, day)` and arbitrary Celery request header values.
- Produces: functional `_is_us_trading_day(date) -> bool` and exception-safe manual origin detection.

- [ ] **Step 1: Write a failing real-helper delegation test**

Import `_is_us_trading_day` in the snapshot test module and add:

```python
def test_us_trading_day_compatibility_hook_uses_market_calendar(monkeypatch):
    calls = []

    def is_trading_day(_self, market, day):
        calls.append((market, day))
        return True

    monkeypatch.setattr(
        "app.services.market_calendar_service.MarketCalendarService.is_trading_day",
        is_trading_day,
    )

    assert _is_us_trading_day(AS_OF) is True
    assert calls == [("US", AS_OF)]
```

- [ ] **Step 2: Run the helper test and verify the unconditional validation error**

Run:

```bash
cd backend
"$PYTHON" -m pytest \
  tests/unit/use_cases/feature_store/test_build_daily_snapshot.py::test_us_trading_day_compatibility_hook_uses_market_calendar -q
```

Expected: failure with `ValidationError: No market calendar configured for US`.

- [ ] **Step 3: Delegate the compatibility hook to the canonical calendar**

Replace the helper body with:

```python
    from app.services.market_calendar_service import MarketCalendarService

    return MarketCalendarService().is_trading_day("US", dt)
```

Keep the lazy import so normal injected use-case construction remains free of
infrastructure initialization.

- [ ] **Step 4: Write failing non-mapping and mapping header tests**

Add to `test_price_refresh_workflow.py`:

```python
def test_full_refresh_header_detection_requires_mapping():
    from app.services.price_refresh_planning import PriceRefreshMode
    from app.services.price_refresh_workflow import PriceRefreshWorkflow

    workflow = object.__new__(PriceRefreshWorkflow)
    workflow._deps = SimpleNamespace(
        market_gateway=SimpleNamespace(
            get_eastern_now=lambda: SimpleNamespace(
                weekday=lambda: 0,
                hour=10,
            )
        )
    )

    malformed = SimpleNamespace(
        request=SimpleNamespace(headers=object())
    )
    manual = SimpleNamespace(
        request=SimpleNamespace(headers={"origin": "manual"})
    )

    assert workflow._should_reject_full_refresh(
        PriceRefreshMode.FULL,
        malformed,
        None,
    ) is True
    assert workflow._should_reject_full_refresh(
        PriceRefreshMode.FULL,
        manual,
        None,
    ) is False
```

- [ ] **Step 5: Run the test and verify malformed headers raise `AttributeError`**

Run:

```bash
cd backend
"$PYTHON" -m pytest \
  tests/unit/test_price_refresh_workflow.py::test_full_refresh_header_detection_requires_mapping -q
```

Expected: failure because `object()` has no `.get` method.

- [ ] **Step 6: Guard header access without changing manual semantics**

Import `Mapping` from `collections.abc`, then replace the current expression:

```python
        headers = getattr(
            getattr(task, "request", None),
            "headers",
            None,
        )
        is_manual = (
            isinstance(headers, Mapping)
            and headers.get("origin") == "manual"
        )
```

- [ ] **Step 7: Run snapshot, feature-task, and refresh workflow suites**

Run:

```bash
cd backend
"$PYTHON" -m pytest \
  tests/unit/use_cases/feature_store/test_build_daily_snapshot.py \
  tests/unit/test_feature_store_tasks.py \
  tests/unit/test_price_refresh_workflow.py \
  tests/unit/test_cache_refresh_unification.py -q
```

Expected: all selected tests pass.

- [ ] **Step 8: Commit calendar and header fixes**

```bash
git add \
  backend/app/use_cases/feature_store/build_daily_snapshot.py \
  backend/app/services/price_refresh_workflow.py \
  backend/tests/unit/use_cases/feature_store/test_build_daily_snapshot.py \
  backend/tests/unit/test_price_refresh_workflow.py
git commit -m "fix: restore snapshot calendar guard"
```

### Task 7: Make the sharded CI collection enforce opt-in markers

**Files:**
- Modify: `.github/workflows/ci.yml:80-90`
- Create: `backend/tests/unit/test_ci_workflow.py`

**Interfaces:**
- Consumes: the `backend-unit` GitHub Actions job and registered `live_service`/`load` markers.
- Produces: one marker-filtered sharded collection command with no redundant shard-1 collection step.

- [ ] **Step 1: Write a failing structural workflow test**

Create:

```python
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_backend_unit_shards_exclude_opt_in_markers_during_collection():
    workflow = yaml.safe_load(
        (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text()
    )
    steps = workflow["jobs"]["backend-unit"]["steps"]
    names = [step.get("name") for step in steps]

    assert "Safe test collection" not in names
    shard_script = next(
        step["run"]
        for step in steps
        if str(step.get("name", "")).startswith(
            "Comprehensive backend unit suite"
        )
    )
    assert (
        'python -m pytest tests/unit --collect-only -qq '
        '-m "not live_service and not load"'
    ) in shard_script
```

- [ ] **Step 2: Run the test and verify both assertions fail against current CI**

Run:

```bash
cd backend
"$PYTHON" -m pytest tests/unit/test_ci_workflow.py -q
```

Expected: failure because `Safe test collection` exists and the sharded
collection command lacks the marker filter.

- [ ] **Step 3: Remove the redundant step and filter the authoritative collection**

Delete the three-line `Safe test collection` step. Change the node-id
collection command to:

```yaml
          python -m pytest tests/unit --collect-only -qq -m "not live_service and not load" > "${collection_file}"
```

Do not change sharding arithmetic, execution, timeouts, or publish dependencies.

- [ ] **Step 4: Run the structural test and parse the workflow**

Run:

```bash
cd backend
"$PYTHON" -m pytest tests/unit/test_ci_workflow.py -q
cd ..
"$PYTHON" -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml'))"
```

Expected: both commands exit 0.

- [ ] **Step 5: Commit CI enforcement**

```bash
git add .github/workflows/ci.yml backend/tests/unit/test_ci_workflow.py
git commit -m "ci: filter opt-in tests during sharding"
```

### Task 8: Verify, publish, and close the review loop

**Files:**
- Verify only: all modified production and test files.
- Do not modify: `docs/superpowers/plans/2026-07-17-backend-test-baseline-remediation.md`.

**Interfaces:**
- Consumes: Tasks 1-7 and PR #303 review thread IDs.
- Produces: green local/remote verification, pushed commits, technically specific thread replies, and resolved dispositions.

- [ ] **Step 1: Run the complete focused remediation suite**

Run:

```bash
cd backend
"$PYTHON" -m pytest \
  tests/unit/test_yahoo_batch_ingestion.py \
  tests/unit/test_breadth_calculator_service.py \
  tests/unit/test_breadth_coverage.py \
  tests/unit/test_daily_breadth_runner.py \
  tests/unit/test_breadth_tasks.py \
  tests/unit/test_export_static_site_script.py \
  tests/unit/test_group_rank_input_loader.py \
  tests/unit/test_group_rank_service.py \
  tests/unit/test_daily_group_rank_runner.py \
  tests/unit/test_group_rank_tasks.py \
  tests/unit/test_group_rank_historical_calculator.py \
  tests/unit/test_group_rank_backfill_tasks.py \
  tests/unit/test_workload_coordination.py \
  tests/unit/test_transient_database.py \
  tests/unit/use_cases/feature_store/test_build_daily_snapshot.py \
  tests/unit/test_feature_store_tasks.py \
  tests/unit/test_price_refresh_workflow.py \
  tests/unit/test_cache_refresh_unification.py \
  tests/unit/test_ci_workflow.py -q
```

Expected: all selected tests pass.

- [ ] **Step 2: Run the complete backend unit suite**

Run:

```bash
cd backend
"$PYTHON" -m pytest tests/unit -q -m "not live_service and not load"
```

Expected: zero failures.

- [ ] **Step 3: Run generated-contract and repository quality gates**

Run:

```bash
cd backend
"$PYTHON" ../scripts/generate_scan_filter_contract.py --check
make PYTHON="$PYTHON" gate-identity gate-check gate-1 gate-2 gate-3 gate-4 gate-5
cd ..
git diff --check
```

Expected: every command exits 0.

- [ ] **Step 4: Verify intended scope and unchanged historical plan**

Run:

```bash
git diff origin/main...HEAD --name-only
git diff origin/main...HEAD -- docs/superpowers/plans/2026-07-17-backend-test-baseline-remediation.md
git status --short --branch
```

Expected: the historical plan diff contains only its pre-existing PR changes;
there is no new remediation commit touching it. The only unrelated working-tree
change may be the previously observed `.beads/issues.jsonl`, which must remain
unstaged.

- [ ] **Step 5: Rebase and push the branch**

Run:

```bash
git pull --rebase --autostash
git push origin codex/issue-301-cache-only-derived-data
git rev-list --left-right --count origin/codex/issue-301-cache-only-derived-data...HEAD
```

Expected: push succeeds and the final count is `0  0`.

- [ ] **Step 6: Wait for every required PR check**

Run:

```bash
gh pr checks 303 --watch --interval 30
```

Expected: frontend, compose smoke, backend quality gates, all four backend unit
shards, and review checks pass or skip only where the workflow intentionally
skips publishing/release jobs on pull requests.

- [ ] **Step 7: Reply to current inline review threads**

Use the inline reply endpoint, not a top-level comment, for these comment IDs:

- `3600033151`: target-date breadth cache validation;
- `3600033153`: target-date group cache validation;
- `3600061355`: disjoint breadth coverage accounting;
- `3600061359`: positive chunk-size validation;
- `3600061374`: transient database retry propagation;
- `3600362757`: legacy breadth `cache_only` compatibility;
- `3602325535`: restored US market-calendar guard.

Resolve the remediation commit IDs from their exact subjects, then reply:

```bash
CACHE_COMMIT=$(git log -1 --format=%h --grep='^fix: require target session in cache-only reads$')
BREADTH_COMMIT=$(git log -1 --format=%h --grep='^fix: enforce cache-only breadth completeness$')
GROUP_COMMIT=$(git log -1 --format=%h --grep='^fix: require target session for group rankings$')
COVERAGE_COMMIT=$(git log -1 --format=%h --grep='^fix: keep breadth coverage classifications disjoint$')
BACKFILL_COMMIT=$(git log -1 --format=%h --grep='^fix: preserve retries in group backfills$')
CALENDAR_COMMIT=$(git log -1 --format=%h --grep='^fix: restore snapshot calendar guard$')

gh api repos/xang1234/stock-screener/pulls/303/comments/3600033151/replies \
  -f body="Fixed in ${CACHE_COMMIT}/${BREADTH_COMMIT}. Daily cache-only breadth now requires the requested session and records a missing session as a cache miss. Focused breadth and cache-service suites pass."
gh api repos/xang1234/stock-screener/pulls/303/comments/3600033153/replies \
  -f body="Fixed in ${CACHE_COMMIT}/${GROUP_COMMIT}. Daily cache-only group inputs now require the requested session for benchmark and constituent histories, feeding misses into existing coverage validation. Focused loader, facade, and group-task suites pass."
gh api repos/xang1234/stock-screener/pulls/303/comments/3600061355/replies \
  -f body="Fixed in ${COVERAGE_COMMIT}. Coverage batches reject misses outside their candidates and conflicting repeated classifications before mutating accumulator state. Breadth coverage and calculation suites pass."
gh api repos/xang1234/stock-screener/pulls/303/comments/3600061359/replies \
  -f body="Fixed in ${BACKFILL_COMMIT}. Non-positive chunk_size_days now raises ValueError before the chunk loop. Historical calculator tests cover zero and negative values."
gh api repos/xang1234/stock-screener/pulls/303/comments/3600061374/replies \
  -f body="Fixed in ${BACKFILL_COMMIT}. All three administrative group backfill handlers propagate recognized transient database failures to serialized_market_workload while preserving legacy dictionaries for non-transient errors. Retry and compatibility tests pass."
gh api repos/xang1234/stock-screener/pulls/303/comments/3600362757/replies \
  -f body="Fixed in ${BREADTH_COMMIT}. backfill_range again accepts cache_only=True and maps it to the equivalent historical strict-cache-only policy; the static export path remains provider-free. Breadth and static-export suites pass."
gh api repos/xang1234/stock-screener/pulls/303/comments/3602325535/replies \
  -f body="Fixed in ${CALENDAR_COMMIT}. The US compatibility hook delegates to the canonical market calendar instead of raising unconditionally. Snapshot use-case and feature-task suites pass."
```

- [ ] **Step 8: Record non-code dispositions without editing historical plans**

Reply to historical-plan comment IDs `3600482283`, `3600482285`,
`3600482290`, `3600482292`, and `3600482297`:

```bash
for comment_id in 3600482283 3600482285 3600482290 3600482292 3600482297; do
  gh api \
    "repos/xang1234/stock-screener/pulls/303/comments/${comment_id}/replies" \
    -f body='No code change. This file is a completed historical execution record; current runtime and CI behavior is validated by the implementation and required checks, so the prior plan is intentionally not rewritten.'
done
```

Post the outside-diff disposition:

```bash
gh pr comment 303 --body='Outside-diff review disposition:
- Breadth return documentation now describes BreadthCalculationResult and its typed accessors.
- Strict cache-only breadth rejects zero usable scanned stocks.
- Group taxonomy membership already uses one read per group and retains its regression test.
- Manual refresh header detection now requires a mapping before reading origin.
- Required backend shards now exclude live_service/load markers during their authoritative collection; the redundant collection step is removed.
- The earlier 54-failure warning is obsolete: the remediated full backend unit suite and all four required CI shards are green.
- Historical implementation plans remain unchanged as completed execution records.'
```

- [ ] **Step 9: Resolve threads only after replies and green CI**

Query GraphQL review thread node IDs and resolve only the replied comment IDs.
Leave any new or technically disputed thread unresolved:

```bash
THREAD_IDS=$(gh api graphql -f query='query {
  repository(owner:"xang1234", name:"stock-screener") {
    pullRequest(number:303) {
      reviewThreads(first:100) {
        nodes { id isResolved comments(first:1) { nodes { databaseId } } }
      }
    }
  }
}' | jq -r \
  --argjson ids '[3600033151,3600033153,3600061355,3600061359,3600061374,3600362757,3600482283,3600482285,3600482290,3600482292,3600482297,3602325535]' \
  '.data.repository.pullRequest.reviewThreads.nodes[] as $thread
   | select($ids | index($thread.comments.nodes[0].databaseId))
   | select($thread.isResolved == false)
   | $thread.id')

for thread_id in ${THREAD_IDS}; do
  gh api graphql \
    -f query='mutation($thread:ID!) {
      resolveReviewThread(input:{threadId:$thread}) { thread { isResolved } }
    }' \
    -f thread="${thread_id}"
done
```

- [ ] **Step 10: Request one final automated review and confirm mergeability**

Run:

```bash
gh pr comment 303 --body '@coderabbitai review'
gh pr view 303 --json state,mergeStateStatus,reviewDecision,url
git status --short --branch
```

Expected: PR remains open and mergeable. Report any unrelated unstaged Beads
ledger change explicitly and leave it untouched.
