# Refresh-Guarded Cache-Only Derived Data Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make daily-pipeline breadth and group target-date plus gap-fill calculations provider-free after the price-refresh guard, while allowing usable partial cache coverage and preserving manual and strict behavior.

**Architecture:** The daily pipeline explicitly authorizes a new `refresh_guarded_cache_only` policy on both derived-data wrappers. Each wrapper propagates the policy to its gap-fill service and nested daily task; services continue to use their existing cached-only read paths, while task-level validation distinguishes tolerant guarded runs from strict `force_cache_only` runs. Diagnostics are returned in successful guarded results so tolerated gaps remain observable without using failure-shaped payloads.

**Tech Stack:** Python 3.12, Celery task signatures, SQLAlchemy, pandas, pytest, `unittest.mock`, Beads (`bd`).

## Global Constraints

- The price-refresh guard's 90% acceptance threshold does not change.
- `refresh_guarded_cache_only` defaults to `False` at every task boundary.
- `force_cache_only=True` retains today's strict static/manual completeness behavior and wins if both policies are true.
- Direct manual historical breadth/group runs retain provider fallback unless they explicitly opt into guarded cache-only behavior.
- Daily-pipeline target-date and automatic gap-fill breadth/group work makes zero provider requests after the refresh guard succeeds.
- Missing or insufficient individual histories are skipped and reported; breadth fails only with zero usable stocks or calculation errors, and group ranking fails only with no cached benchmark, zero rankable groups, or actual exceptions.
- Existing breadth formulas, relative-strength formulas, publication dates, snapshot behavior, and the separate orphan/stale snapshot investigation do not change.
- Missing-symbol samples are deterministic, sorted, and bounded to 20 symbols.

---

## File Structure

- `backend/app/tasks/daily_market_pipeline_tasks.py`: authorizes guarded cache-only behavior only inside the post-refresh daily chain.
- `backend/app/services/breadth_calculator_service.py`: performs cache-only breadth reads for target and gap-fill dates and produces coverage diagnostics.
- `backend/app/tasks/breadth_tasks.py`: resolves strict versus guarded breadth policy, validates meaningful output, and propagates the flag through the wrapper.
- `backend/app/services/ibd_group_rank_service.py`: performs cached-only benchmark/constituent prefetch, tolerates partial constituents when no strict requirement is supplied, and exposes prefetch diagnostics without shared mutable state.
- `backend/app/tasks/group_rank_tasks.py`: resolves strict versus guarded group policy and propagates it through optimized gap-fill and the nested daily task.
- `backend/tests/unit/test_daily_market_pipeline_tasks.py`: verifies chain ordering and guarded-policy authorization.
- `backend/tests/unit/test_breadth_calculator_service.py`: verifies provider exclusion, gap-fill propagation, and coverage diagnostics.
- `backend/tests/unit/test_breadth_tasks.py`: verifies guarded success/failure semantics, wrapper propagation, strict precedence, and manual compatibility.
- `backend/tests/unit/test_group_rank_service.py`: verifies provider exclusion, tolerant partial group calculation, and optimized gap-fill diagnostics.
- `backend/tests/unit/test_group_rank_tasks.py`: verifies guarded policy propagation, strict precedence, and manual compatibility.
- `docs/superpowers/specs/2026-07-16-refresh-guarded-cache-only-derived-data-design.md`: records implementation status after verification.
- `.beads/issues.jsonl` and `.beads/interactions.jsonl`: record completion of `stockscreenclaude-duw`.

---

### Task 1: Authorize guarded cache-only work in the daily pipeline

**Files:**
- Modify: `backend/tests/unit/test_daily_market_pipeline_tasks.py:30-83`
- Modify: `backend/app/tasks/daily_market_pipeline_tasks.py:151-205`

**Interfaces:**
- Consumes: the existing immutable Celery chain built by `_build_daily_market_pipeline_signatures(market: str, trading_date: date) -> list`.
- Produces: `refresh_guarded_cache_only=True` in the immutable breadth and group wrapper signature kwargs, ordered after `guard_price_refresh`.

- [ ] **Step 1: Extend the pipeline signature test with the guarded authorization**

Replace the two derived-stage kwargs assertions in `test_daily_market_pipeline_orders_refresh_compute_and_scan` with:

```python
    assert signatures[2].kwargs == {
        "market": "HK",
        "calculation_date": "2026-03-16",
        "refresh_guarded_cache_only": True,
    }
    assert signatures[6].kwargs == {
        "market": "HK",
        "calculation_date": "2026-03-16",
        "refresh_guarded_cache_only": True,
    }
```

The existing ordered task-name assertion remains the proof that both signatures occur after `guard_price_refresh`.

- [ ] **Step 2: Run the pipeline test and verify the new expectation fails**

Run:

```bash
cd backend && /Users/admin/StockScreenClaude/backend/venv/bin/python -m pytest tests/unit/test_daily_market_pipeline_tasks.py::test_daily_market_pipeline_orders_refresh_compute_and_scan -q
```

Expected: FAIL because both task kwargs omit `refresh_guarded_cache_only`.

- [ ] **Step 3: Add the authorization to both derived-stage signatures**

Replace the breadth and group signature construction blocks with:

```python
        calculate_daily_breadth_with_gapfill.si(
            market=market_code,
            calculation_date=as_of_date,
            refresh_guarded_cache_only=True,
        ).set(
            queue=market_jobs_queue_for_market(market_code)
        ),
```

and:

```python
        calculate_daily_group_rankings_with_gapfill.si(
            market=market_code,
            calculation_date=as_of_date,
            refresh_guarded_cache_only=True,
        ).set(
            queue=market_jobs_queue_for_market(market_code)
        ),
```

- [ ] **Step 4: Run the pipeline test and verify it passes**

Run:

```bash
cd backend && /Users/admin/StockScreenClaude/backend/venv/bin/python -m pytest tests/unit/test_daily_market_pipeline_tasks.py::test_daily_market_pipeline_orders_refresh_compute_and_scan -q
```

Expected: PASS.

- [ ] **Step 5: Commit the pipeline authorization**

```bash
git add backend/app/tasks/daily_market_pipeline_tasks.py backend/tests/unit/test_daily_market_pipeline_tasks.py
git commit -m "fix(pipeline): authorize guarded cache-only derived data"
```

---

### Task 2: Make breadth gap-fill cache-only with bounded diagnostics

**Files:**
- Modify: `backend/tests/unit/test_breadth_calculator_service.py:302-417`
- Modify: `backend/app/services/breadth_calculator_service.py:46-302`
- Modify: `backend/app/services/breadth_calculator_service.py:681-721`

**Interfaces:**
- Consumes: `PriceCacheService.get_many_cached_only_fresh(symbols, period="2y")` and the existing defaulted `cache_only: bool = False` switch on `backfill_range`.
- Produces: `BreadthCalculatorService.fill_gaps(missing_dates: List[date], cache_only: bool = False) -> Dict`; guarded diagnostics named `target_symbols`, `symbols_with_cached_history`, `cache_miss_stocks`, `cache_miss_symbols_sample`, `cache_coverage_ratio`, and `insufficient_history_observations`.

- [ ] **Step 1: Add a failing cache-only gap-fill propagation test**

Add this test after `test_fill_gaps_delegates_to_single_backfill_range_call`:

```python
def test_fill_gaps_propagates_cache_only_to_backfill_range(monkeypatch):
    service = BreadthCalculatorService(_make_db_session(), MagicMock())
    expected = {
        "total_dates": 1,
        "processed": 1,
        "errors": 0,
        "error_dates": [],
        "target_symbols": 2,
        "symbols_with_cached_history": 1,
        "cache_miss_stocks": 1,
        "cache_miss_symbols_sample": ["BBB"],
        "cache_coverage_ratio": 0.5,
        "insufficient_history_observations": 0,
    }
    backfill_range = MagicMock(return_value=expected)
    monkeypatch.setattr(service, "backfill_range", backfill_range)

    result = service.fill_gaps([date(2026, 3, 12)], cache_only=True)

    assert result == expected
    backfill_range.assert_called_once_with(
        date(2026, 3, 12),
        date(2026, 3, 12),
        trading_dates=[date(2026, 3, 12)],
        cache_only=True,
    )
```

- [ ] **Step 2: Add a failing cache-only partial-coverage diagnostics test**

Add:

```python
def test_backfill_range_cache_only_reports_gaps_without_provider_fallback():
    db = _make_db_session()
    db.add_all([
        StockUniverse(symbol="AAA", is_active=True, status=UNIVERSE_STATUS_ACTIVE),
        StockUniverse(symbol="BBB", is_active=True, status=UNIVERSE_STATUS_ACTIVE),
        StockUniverse(symbol="NEW", is_active=True, status=UNIVERSE_STATUS_ACTIVE),
    ])
    db.commit()

    trading_date = date(2026, 3, 20)
    price_cache = MagicMock()
    price_cache.get_many_cached_only_fresh.return_value = {
        "AAA": _make_price_df(trading_date),
        "BBB": None,
        "NEW": _flat_price_df(trading_date, periods=20),
    }
    price_cache.get_historical_data.side_effect = AssertionError(
        "guarded breadth gap-fill must not call a provider"
    )
    service = BreadthCalculatorService(db, price_cache)

    result = service.backfill_range(
        trading_date,
        trading_date,
        trading_dates=[trading_date],
        cache_only=True,
    )

    assert result == {
        "total_dates": 1,
        "processed": 1,
        "errors": 0,
        "error_dates": [],
        "target_symbols": 3,
        "symbols_with_cached_history": 2,
        "cache_miss_stocks": 1,
        "cache_miss_symbols_sample": ["BBB"],
        "cache_coverage_ratio": pytest.approx(2 / 3),
        "insufficient_history_observations": 1,
    }
    price_cache.get_historical_data.assert_not_called()
```

Add `import pytest` beside the existing test imports.

- [ ] **Step 3: Run both new tests and verify they fail**

Run:

```bash
cd backend && /Users/admin/StockScreenClaude/backend/venv/bin/python -m pytest \
  tests/unit/test_breadth_calculator_service.py::test_fill_gaps_propagates_cache_only_to_backfill_range \
  tests/unit/test_breadth_calculator_service.py::test_backfill_range_cache_only_reports_gaps_without_provider_fallback -q
```

Expected: FAIL because `fill_gaps` does not accept `cache_only` and `backfill_range` does not return diagnostics.

- [ ] **Step 4: Add bounded diagnostic collection to breadth calculations**

Add near the service logger:

```python
CACHE_MISS_SYMBOL_SAMPLE_LIMIT = 20
```

Add these fields to `_empty_metrics()`:

```python
            'candidate_stocks': 0,
            'symbols_with_cached_history': 0,
            'cache_coverage_ratio': 0.0,
            'cache_miss_symbols_sample': [],
```

In `calculate_daily_breadth`, immediately after updating `cache_miss_stocks`, add:

```python
            remaining_sample_slots = (
                CACHE_MISS_SYMBOL_SAMPLE_LIMIT
                - len(metrics['cache_miss_symbols_sample'])
            )
            if remaining_sample_slots > 0:
                metrics['cache_miss_symbols_sample'].extend(
                    sorted(cache_miss_symbols)[:remaining_sample_slots]
                )
```

Before calculating ratios, add:

```python
        metrics['candidate_stocks'] = total_stocks
        metrics['symbols_with_cached_history'] = total_stocks - metrics['cache_miss_stocks']
        metrics['cache_coverage_ratio'] = (
            metrics['symbols_with_cached_history'] / total_stocks
            if total_stocks > 0
            else 0.0
        )
```

- [ ] **Step 5: Record cache misses and insufficient histories in `backfill_range`**

After `total_stocks = len(active_stocks)`, add:

```python
        cache_miss_symbols: set[str] = set()
        insufficient_history_observations = 0
```

Replace the batch load and per-stock processing block with:

```python
            price_data_by_symbol, batch_cache_miss_symbols = self._load_price_data_for_batch(
                batch_symbols=batch_symbols,
                cache_only=cache_only,
            )
            cache_miss_symbols.update(batch_cache_miss_symbols)

            for stock in batch:
                try:
                    price_history = price_data_by_symbol.get(stock.symbol)
                    if price_history is None or price_history.empty:
                        for calc_date in ordered_dates:
                            metrics_by_date[calc_date]['cache_miss_stocks'] += 1
                            metrics_by_date[calc_date]['skipped_stocks'] += 1
                        continue

                    stock_metrics_by_date = self._calculate_stock_metrics_by_date_from_prices(
                        prices_df=price_history,
                        calculation_dates=ordered_dates,
                    )
                    for calc_date in ordered_dates:
                        daily_metrics = metrics_by_date[calc_date]
                        stock_metrics = stock_metrics_by_date.get(calc_date)
                        if stock_metrics is None:
                            daily_metrics['insufficient_data_stocks'] += 1
                            daily_metrics['skipped_stocks'] += 1
                            insufficient_history_observations += 1
                            continue
                        self._apply_stock_metrics(daily_metrics, stock_metrics)
                        daily_metrics['total_stocks_scanned'] += 1
                except Exception as e:
                    logger.warning("Error processing %s in breadth backfill: %s", stock.symbol, e)
                    for calc_date in ordered_dates:
                        metrics_by_date[calc_date]['error_stocks'] += 1
                        metrics_by_date[calc_date]['skipped_stocks'] += 1
```

Replace the final return with a local result and attach diagnostics only for
cache-only callers, preserving the exact default/manual result contract:

```python
        result = {
            'total_dates': len(ordered_dates),
            'processed': len(processed_dates),
            'errors': len(error_dates),
            'error_dates': error_dates,
        }
        if cache_only:
            result.update({
                'target_symbols': total_stocks,
                'symbols_with_cached_history': total_stocks - len(cache_miss_symbols),
                'cache_miss_stocks': len(cache_miss_symbols),
                'cache_miss_symbols_sample': sorted(cache_miss_symbols)[
                :CACHE_MISS_SYMBOL_SAMPLE_LIMIT
                ],
                'cache_coverage_ratio': (
                    (total_stocks - len(cache_miss_symbols)) / total_stocks
                    if total_stocks > 0
                    else 0.0
                ),
                'insufficient_history_observations': insufficient_history_observations,
            })
        return result
```

Do not add these keys to the early no-date return because no stock read occurs there.

- [ ] **Step 6: Propagate `cache_only` through `fill_gaps` without changing its default**

Change the signature and delegated call to:

```python
    def fill_gaps(self, missing_dates: List[date], cache_only: bool = False) -> Dict:
```

and:

```python
        stats = self.backfill_range(
            ordered_dates[0],
            ordered_dates[-1],
            trading_dates=ordered_dates,
            cache_only=cache_only,
        )
```

- [ ] **Step 7: Update the existing default-path assertion and run the breadth service suite**

Update `test_fill_gaps_delegates_to_single_backfill_range_call` so its expected call includes `cache_only=False`. Then run:

```bash
cd backend && /Users/admin/StockScreenClaude/backend/venv/bin/python -m pytest tests/unit/test_breadth_calculator_service.py -q
```

Expected: all breadth calculator service tests PASS, including the existing test proving historical `cache_only=False` still calls `get_historical_data`.

- [ ] **Step 8: Commit the breadth service behavior**

```bash
git add backend/app/services/breadth_calculator_service.py backend/tests/unit/test_breadth_calculator_service.py
git commit -m "fix(breadth): make guarded gap fill cache-only"
```

---

### Task 3: Apply tolerant guarded policy in breadth tasks

**Files:**
- Modify: `backend/tests/unit/test_breadth_tasks.py:121-469`
- Modify: `backend/app/tasks/breadth_tasks.py:83-123`
- Modify: `backend/app/tasks/breadth_tasks.py:145-350`
- Modify: `backend/app/tasks/breadth_tasks.py:461-478`
- Modify: `backend/app/tasks/breadth_tasks.py:599-755`

**Interfaces:**
- Consumes: `BreadthCalculatorService.calculate_daily_breadth(cache_only=True)` and `fill_gaps(cache_only=True)` from Task 2.
- Produces: optional `refresh_guarded_cache_only: bool = False` on `calculate_daily_breadth` and `calculate_daily_breadth_with_gapfill`; guarded success results include `cache_only=True`, `cache_policy="refresh_guarded"`, and `cache_diagnostics`.

- [ ] **Step 1: Add a guarded target-date test that tolerates cache misses**

Add a helper near the test imports:

```python
def _breadth_metrics(*, scanned: int, skipped: int, misses: int, errors: int = 0) -> dict:
    return {
        "stocks_up_4pct": 1,
        "stocks_down_4pct": 0,
        "ratio_5day": 1.0,
        "ratio_10day": 1.0,
        "stocks_up_25pct_quarter": 1,
        "stocks_down_25pct_quarter": 0,
        "stocks_up_25pct_month": 1,
        "stocks_down_25pct_month": 0,
        "stocks_up_50pct_month": 0,
        "stocks_down_50pct_month": 0,
        "stocks_up_13pct_34days": 1,
        "stocks_down_13pct_34days": 0,
        "total_stocks_scanned": scanned,
        "skipped_stocks": skipped,
        "cache_miss_stocks": misses,
        "insufficient_data_stocks": max(skipped - misses, 0),
        "error_stocks": errors,
        "candidate_stocks": scanned + skipped,
        "symbols_with_cached_history": scanned + skipped - misses,
        "cache_coverage_ratio": (
            (scanned + skipped - misses) / (scanned + skipped)
            if scanned + skipped
            else 0.0
        ),
        "cache_miss_symbols_sample": ["MISS"],
    }
```

Add:

```python
def test_guarded_historical_breadth_tolerates_cache_misses(monkeypatch):
    import app.services.ui_snapshot_service as snapshot_module
    import app.tasks.breadth_tasks as module

    fake_db = MagicMock()
    fake_db.query.return_value.filter.return_value.first.return_value = None
    fake_calculator = MagicMock()
    fake_calculator.price_cache = MagicMock()
    fake_calculator.calculate_daily_breadth.return_value = _breadth_metrics(
        scanned=60,
        skipped=40,
        misses=35,
    )
    monkeypatch.setattr(module, "SessionLocal", lambda: fake_db)
    monkeypatch.setattr(module, "BreadthCalculatorService", lambda *args, **kwargs: fake_calculator)
    monkeypatch.setattr(snapshot_module, "safe_publish_breadth_bootstrap", lambda _market: None)
    _patch_serialized_lock(monkeypatch)
    _patch_calendar_service(monkeypatch, datetime(2026, 3, 20, 17, 35, 0))

    result = module.calculate_daily_breadth.run(
        "2026-03-19",
        refresh_guarded_cache_only=True,
    )

    assert "error" not in result
    assert result["cache_only"] is True
    assert result["cache_policy"] == "refresh_guarded"
    assert result["cache_diagnostics"]["cache_miss_stocks"] == 35
    fake_calculator.calculate_daily_breadth.assert_called_once_with(
        calculation_date=date(2026, 3, 19),
        cache_only=True,
    )
    fake_db.commit.assert_called_once()
```

- [ ] **Step 2: Add zero-output and strict-precedence tests**

Add:

```python
def test_guarded_historical_breadth_fails_when_no_stock_is_usable(monkeypatch):
    import app.tasks.breadth_tasks as module

    fake_db = MagicMock()
    fake_calculator = MagicMock()
    fake_calculator.price_cache = MagicMock()
    fake_calculator.calculate_daily_breadth.return_value = _breadth_metrics(
        scanned=0,
        skipped=100,
        misses=100,
    )
    monkeypatch.setattr(module, "SessionLocal", lambda: fake_db)
    monkeypatch.setattr(module, "BreadthCalculatorService", lambda *args, **kwargs: fake_calculator)
    _patch_serialized_lock(monkeypatch)
    _patch_calendar_service(monkeypatch, datetime(2026, 3, 20, 17, 35, 0))

    result = module.calculate_daily_breadth.run(
        "2026-03-19",
        refresh_guarded_cache_only=True,
    )

    assert result["cache_policy"] == "refresh_guarded"
    assert "processed no usable stocks" in result["error"].lower()
    fake_db.commit.assert_not_called()


def test_force_cache_only_wins_over_guarded_tolerance(monkeypatch):
    import app.tasks.breadth_tasks as module

    fake_db = MagicMock()
    fake_calculator = MagicMock()
    fake_calculator.price_cache = MagicMock()
    fake_calculator.calculate_daily_breadth.return_value = _breadth_metrics(
        scanned=60,
        skipped=40,
        misses=35,
    )
    monkeypatch.setattr(module, "SessionLocal", lambda: fake_db)
    monkeypatch.setattr(module, "BreadthCalculatorService", lambda *args, **kwargs: fake_calculator)
    _patch_serialized_lock(monkeypatch)
    _patch_calendar_service(monkeypatch, datetime(2026, 3, 20, 17, 35, 0))

    result = module.calculate_daily_breadth.run(
        "2026-03-19",
        force_cache_only=True,
        refresh_guarded_cache_only=True,
    )

    assert "exceeds miss tolerance" in result["error"].lower()
    assert result.get("cache_policy") != "refresh_guarded"
    fake_db.commit.assert_not_called()
```

- [ ] **Step 3: Add a wrapper propagation test covering gap-fill and target date**

Add:

```python
def test_guarded_breadth_wrapper_propagates_cache_only_to_gapfill_and_target(monkeypatch):
    import app.tasks.breadth_tasks as module

    fake_db = MagicMock()
    fake_calculator = MagicMock()
    fake_calculator.find_missing_dates.return_value = [date(2026, 3, 18)]
    fake_calculator.fill_gaps.return_value = {
        "total_dates": 1,
        "processed": 0,
        "errors": 1,
        "error_dates": ["2026-03-18"],
        "cache_miss_stocks": 4,
        "cache_miss_symbols_sample": ["MISS"],
    }
    monkeypatch.setattr(module, "SessionLocal", lambda: fake_db)
    monkeypatch.setattr(module, "BreadthCalculatorService", lambda *args, **kwargs: fake_calculator)
    monkeypatch.setattr(module.settings, "breadth_gapfill_enabled", True)
    monkeypatch.setattr(
        "app.services.runtime_preferences_service.is_market_enabled_now",
        lambda _market: True,
    )
    _patch_serialized_lock(monkeypatch)
    _patch_calendar_service(monkeypatch, datetime(2026, 3, 20, 17, 40, 0))
    target_call = MagicMock(return_value={
        "date": "2026-03-19",
        "cache_only": True,
        "cache_policy": "refresh_guarded",
    })
    monkeypatch.setattr(module, "_calculate_daily_breadth_in_process", target_call)

    result = module.calculate_daily_breadth_with_gapfill.run(
        market="US",
        calculation_date="2026-03-19",
        refresh_guarded_cache_only=True,
    )

    fake_calculator.fill_gaps.assert_called_once_with(
        [date(2026, 3, 18)],
        cache_only=True,
    )
    target_call.assert_called_once_with(
        market="US",
        calculation_date="2026-03-19",
        refresh_guarded_cache_only=True,
    )
    assert result["cache_only"] is True
    assert result["cache_policy"] == "refresh_guarded"
    assert "error" not in result
    assert result["gap_fill"]["errors"] == 1
    assert result["gap_fill"]["cache_miss_stocks"] == 4
```

- [ ] **Step 4: Run the new breadth task tests and verify they fail**

Run:

```bash
cd backend && /Users/admin/StockScreenClaude/backend/venv/bin/python -m pytest \
  tests/unit/test_breadth_tasks.py::test_guarded_historical_breadth_tolerates_cache_misses \
  tests/unit/test_breadth_tasks.py::test_guarded_historical_breadth_fails_when_no_stock_is_usable \
  tests/unit/test_breadth_tasks.py::test_force_cache_only_wins_over_guarded_tolerance \
  tests/unit/test_breadth_tasks.py::test_guarded_breadth_wrapper_propagates_cache_only_to_gapfill_and_target -q
```

Expected: FAIL because breadth tasks do not accept or propagate `refresh_guarded_cache_only`.

- [ ] **Step 5: Add guarded breadth validation and diagnostic helpers**

Add below `_validate_same_day_cache_only_breadth_metrics`:

```python
def _breadth_cache_diagnostics(metrics: dict) -> dict:
    total_scanned = int(metrics.get("total_stocks_scanned", 0) or 0)
    skipped = int(metrics.get("skipped_stocks", 0) or 0)
    candidates = int(metrics.get("candidate_stocks", total_scanned + skipped) or 0)
    cache_misses = int(metrics.get("cache_miss_stocks", 0) or 0)
    cached_histories = int(
        metrics.get("symbols_with_cached_history", max(candidates - cache_misses, 0)) or 0
    )
    coverage_ratio = (
        float(metrics.get("cache_coverage_ratio"))
        if metrics.get("cache_coverage_ratio") is not None
        else (cached_histories / candidates if candidates > 0 else 0.0)
    )
    return {
        "candidate_stocks": candidates,
        "total_stocks_scanned": total_scanned,
        "symbols_with_cached_history": cached_histories,
        "skipped_stocks": skipped,
        "cache_miss_stocks": cache_misses,
        "insufficient_data_stocks": int(metrics.get("insufficient_data_stocks", 0) or 0),
        "error_stocks": int(metrics.get("error_stocks", 0) or 0),
        "cache_coverage_ratio": coverage_ratio,
        "cache_miss_symbols_sample": list(metrics.get("cache_miss_symbols_sample", []))[:20],
    }


def _validate_refresh_guarded_breadth_metrics(metrics: dict) -> Optional[str]:
    diagnostics = _breadth_cache_diagnostics(metrics)
    if diagnostics["error_stocks"] > 0:
        return (
            "Refresh-guarded breadth run has calculation errors "
            f"(errors={diagnostics['error_stocks']})"
        )
    if diagnostics["total_stocks_scanned"] == 0:
        return "Refresh-guarded breadth run processed no usable stocks"
    return None
```

- [ ] **Step 6: Resolve strict, guarded, and default breadth policies explicitly**

Add `refresh_guarded_cache_only: bool = False` after `force_cache_only` in `calculate_daily_breadth`. Replace the cache-policy assignment with:

```python
        guarded_cache_only = refresh_guarded_cache_only and not force_cache_only
        cache_only = force_cache_only or refresh_guarded_cache_only or calc_date == today_local
```

Replace the completeness branch with:

```python
        if cache_only:
            if guarded_cache_only:
                completeness_error = _validate_refresh_guarded_breadth_metrics(metrics)
            elif force_cache_only or _ALLOW_SAME_DAY_BREADTH_WARMUP_BYPASS.get():
                logger.info(
                    "Bypassing same-day breadth warmup metadata gate for in-process static export"
                )
                completeness_error = _validate_same_day_cache_only_breadth_metrics(metrics)
            else:
                completeness_error = _validate_same_day_cache_only_breadth(
                    calculator.price_cache,
                    metrics,
                    market=effective_market,
                )
            if completeness_error:
                logger.error("✗ Refusing to publish daily breadth: %s", completeness_error)
                logger.info("=" * 60)
                error_result = {
                    'error': completeness_error,
                    'date': calc_date.strftime('%Y-%m-%d'),
                    'timestamp': datetime.now().isoformat(),
                    'cache_only': True,
                    'metrics': _breadth_cache_diagnostics(metrics),
                }
                if guarded_cache_only:
                    error_result['cache_policy'] = 'refresh_guarded'
                    error_result['cache_diagnostics'] = _breadth_cache_diagnostics(metrics)
                return error_result
```

Build the success payload in a local variable and attach guarded metadata without changing other callers:

```python
        task_result = {
            'date': calc_date.strftime('%Y-%m-%d'),
            'indicators': {
                'stocks_up_4pct': metrics['stocks_up_4pct'],
                'stocks_down_4pct': metrics['stocks_down_4pct'],
                'ratio_5day': metrics['ratio_5day'],
                'ratio_10day': metrics['ratio_10day'],
                'stocks_up_25pct_quarter': metrics['stocks_up_25pct_quarter'],
                'stocks_down_25pct_quarter': metrics['stocks_down_25pct_quarter'],
                'stocks_up_25pct_month': metrics['stocks_up_25pct_month'],
                'stocks_down_25pct_month': metrics['stocks_down_25pct_month'],
                'stocks_up_50pct_month': metrics['stocks_up_50pct_month'],
                'stocks_down_50pct_month': metrics['stocks_down_50pct_month'],
                'stocks_up_13pct_34days': metrics['stocks_up_13pct_34days'],
                'stocks_down_13pct_34days': metrics['stocks_down_13pct_34days'],
            },
            'total_stocks_scanned': metrics['total_stocks_scanned'],
            'calculation_duration_seconds': duration,
            'cache_only': cache_only,
            'timestamp': datetime.now().isoformat(),
        }
        if guarded_cache_only:
            task_result['cache_policy'] = 'refresh_guarded'
            task_result['cache_diagnostics'] = _breadth_cache_diagnostics(metrics)
        return task_result
```

- [ ] **Step 7: Propagate the guarded policy through the in-process helper and wrapper**

Add `refresh_guarded_cache_only: bool = False` to `_calculate_daily_breadth_in_process` and add this conditional beside the existing calculation-date conditional:

```python
    if refresh_guarded_cache_only:
        kwargs["refresh_guarded_cache_only"] = True
```

Add the same defaulted parameter to `calculate_daily_breadth_with_gapfill`. After initializing `result`, add:

```python
    if refresh_guarded_cache_only:
        result['cache_only'] = True
        result['cache_policy'] = 'refresh_guarded'
```

Replace the gap-fill call with:

```python
                gap_kwargs = (
                    {"cache_only": True}
                    if refresh_guarded_cache_only
                    else {}
                )
                gap_stats = calculator.fill_gaps(missing_dates, **gap_kwargs)
```

After building `inner_kwargs`, add:

```python
            if refresh_guarded_cache_only:
                inner_kwargs["refresh_guarded_cache_only"] = True
```

Immediately after assigning `result['today']`, surface nested hard failures:

```python
            if isinstance(today_result, dict) and today_result.get("error"):
                raise RuntimeError(
                    f"Daily breadth calculation failed: {today_result['error']}"
                )
```

Replace the wrapper's generic exception return with a local payload so guarded
failure results retain their policy:

```python
        error_result = {
            'error': str(e),
            'gap_fill': result.get('gap_fill'),
            'today': result.get('today'),
            'timestamp': datetime.now().isoformat(),
        }
        if refresh_guarded_cache_only:
            error_result['cache_only'] = True
            error_result['cache_policy'] = 'refresh_guarded'
        return error_result
```

- [ ] **Step 8: Preserve the manual historical default explicitly**

Add this complete compatibility test:

```python
def test_manual_historical_breadth_keeps_fetch_capable_behavior(monkeypatch):
    import app.services.ui_snapshot_service as snapshot_module
    import app.tasks.breadth_tasks as module

    fake_db = MagicMock()
    fake_db.query.return_value.filter.return_value.first.return_value = None
    fake_calculator = MagicMock()
    fake_calculator.price_cache = MagicMock()
    fake_calculator.calculate_daily_breadth.return_value = _breadth_metrics(
        scanned=100,
        skipped=0,
        misses=0,
    )
    monkeypatch.setattr(module, "SessionLocal", lambda: fake_db)
    monkeypatch.setattr(module, "BreadthCalculatorService", lambda *args, **kwargs: fake_calculator)
    monkeypatch.setattr(snapshot_module, "safe_publish_breadth_bootstrap", lambda _market: None)
    _patch_serialized_lock(monkeypatch)
    _patch_calendar_service(monkeypatch, datetime(2026, 4, 3, 0, 30, 0))

    result = module.calculate_daily_breadth.run("2026-04-02")

    assert result["cache_only"] is False
    assert result.get("cache_policy") is None
    fake_calculator.calculate_daily_breadth.assert_called_once_with(
        calculation_date=date(2026, 4, 2),
        cache_only=False,
    )
    fake_db.commit.assert_called_once()
```

- [ ] **Step 9: Run the complete breadth task and service suites**

Run:

```bash
cd backend && /Users/admin/StockScreenClaude/backend/venv/bin/python -m pytest \
  tests/unit/test_breadth_tasks.py \
  tests/unit/test_breadth_calculator_service.py -q
```

Expected: all tests PASS; existing same-day warmup and strict static tests remain green.

- [ ] **Step 10: Commit the breadth task policy**

```bash
git add backend/app/tasks/breadth_tasks.py backend/tests/unit/test_breadth_tasks.py
git commit -m "fix(breadth): tolerate guarded cache gaps"
```

---

### Task 4: Add cache-only group prefetch diagnostics and tolerant partial calculation

**Files:**
- Modify: `backend/tests/unit/test_group_rank_service.py:331-704`
- Modify: `backend/tests/unit/test_group_rank_service.py:1049-1163`
- Modify: `backend/app/services/ibd_group_rank_service.py:31-74`
- Modify: `backend/app/services/ibd_group_rank_service.py:105-233`
- Modify: `backend/app/services/ibd_group_rank_service.py:925-1086`
- Modify: `backend/app/services/ibd_group_rank_service.py:1725-1853`

**Interfaces:**
- Consumes: cached benchmark candidates through `_get_cached_only_benchmark_data` and cached constituents through `get_many_cached_only_fresh`.
- Produces: optional caller-owned `diagnostics: Optional[Dict[str, Any]] = None` on `calculate_group_rankings`; a defaulted `cache_only: bool = False` on `fill_gaps_optimized`; and `prefetch_stats` in optimized gap-fill results. The diagnostics mapping avoids unsafe state on the singleton group-rank service.

- [ ] **Step 1: Add a failing optimized gap-fill cache-only propagation test**

Add after `test_fill_gaps_optimized_accepts_prefetch_stats_tuple`:

```python
def test_fill_gaps_optimized_propagates_cache_only_and_returns_prefetch_stats(
    db_session,
    monkeypatch,
):
    service = _make_group_rank_service()
    price_data = _price_frame()
    captured: dict = {}
    prefetch = group_rank_module.GroupRankPrefetchData(
        benchmark_prices=price_data,
        prices_by_symbol={"AAA": price_data, "BBB": None},
        active_symbols={"AAA", "BBB"},
        market_caps={"AAA": 1_000_000_000},
        stats={
            "target_symbols": 2,
            "symbols_with_prices": 1,
            "cache_miss_symbols": 1,
            "cache_miss_symbols_sample": ["BBB"],
            "cache_coverage_ratio": 0.5,
            "spy_cached": True,
            "benchmark_cached": True,
            "benchmark_symbol": "SPY",
            "benchmark_role": "primary",
            "market": "US",
            "cache_only": True,
            "skipped_unsupported_symbols": 0,
        },
        symbols_by_group={"Software": ["AAA", "BBB"]},
    )
    monkeypatch.setattr(
        service,
        "_prefetch_all_data",
        lambda db, **kwargs: captured.update(kwargs) or prefetch,
    )
    monkeypatch.setattr(
        "app.services.ibd_group_rank_service.IBDIndustryService.get_all_groups",
        lambda db, **kwargs: ["Software"],
    )

    result = service.fill_gaps_optimized(
        db_session,
        [date(2026, 3, 20)],
        market="US",
        cache_only=True,
    )

    assert captured == {"market": "US", "cache_only": True}
    assert result["prefetch_stats"]["cache_miss_symbols"] == 1
    assert result["prefetch_stats"]["cache_miss_symbols_sample"] == ["BBB"]
```

- [ ] **Step 2: Add a failing tolerant partial-constituent calculation test**

Add:

```python
def test_calculate_group_rankings_tolerates_partial_cache_when_requirement_disabled(
    db_session,
    monkeypatch,
):
    service = _make_group_rank_service()
    price_data = _price_frame()
    symbols = ["AAA", "BBB", "CCC", "MISS"]
    stats = {
        "target_symbols": 4,
        "symbols_with_prices": 3,
        "cache_miss_symbols": 1,
        "cache_miss_symbols_sample": ["MISS"],
        "cache_coverage_ratio": 0.75,
        "spy_cached": True,
        "benchmark_cached": True,
        "benchmark_symbol": "SPY",
        "benchmark_role": "primary",
        "market": "US",
        "cache_only": True,
        "skipped_unsupported_symbols": 0,
    }
    prefetch = group_rank_module.GroupRankPrefetchData(
        benchmark_prices=price_data,
        prices_by_symbol={
            "AAA": price_data,
            "BBB": price_data,
            "CCC": price_data,
            "MISS": None,
        },
        active_symbols=set(symbols),
        market_caps={symbol: 1_000_000_000 for symbol in symbols},
        stats=stats,
        symbols_by_group={"Software": symbols},
    )
    monkeypatch.setattr(
        "app.services.ibd_group_rank_service.IBDIndustryService.get_all_groups",
        lambda db, **kwargs: ["Software"],
    )
    monkeypatch.setattr(service, "_prefetch_all_data", lambda db, **kwargs: prefetch)
    monkeypatch.setattr(
        service,
        "_calculate_rs_by_symbol_for_dates",
        lambda _prefetch, dates: {
            dates[0]: {"AAA": 91.0, "BBB": 85.0, "CCC": 80.0}
        },
    )
    monkeypatch.setattr(service, "_store_rankings", Mock())
    diagnostics: dict = {}

    results = service.calculate_group_rankings(
        db_session,
        date(2026, 3, 20),
        market="US",
        cache_only=True,
        cache_requirement=GroupRankCacheRequirement.disabled(),
        diagnostics=diagnostics,
    )

    assert len(results) == 1
    assert results[0]["industry_group"] == "Software"
    assert results[0]["num_stocks"] == 3
    assert diagnostics["cache_miss_symbols"] == 1
    assert diagnostics["cache_miss_symbols_sample"] == ["MISS"]
```

- [ ] **Step 3: Run both new group service tests and verify they fail**

Run:

```bash
cd backend && /Users/admin/StockScreenClaude/backend/venv/bin/python -m pytest \
  tests/unit/test_group_rank_service.py::test_fill_gaps_optimized_propagates_cache_only_and_returns_prefetch_stats \
  tests/unit/test_group_rank_service.py::test_calculate_group_rankings_tolerates_partial_cache_when_requirement_disabled -q
```

Expected: FAIL because the new parameters and diagnostic outputs do not exist.

- [ ] **Step 4: Enrich prefetch statistics without changing strict validation**

Add near the service constants:

```python
CACHE_MISS_SYMBOL_SAMPLE_LIMIT = 20
```

After fetching all constituent prices, calculate:

```python
        missing_symbols = sorted(
            symbol
            for symbol in symbols_to_fetch
            if all_prices.get(symbol) is None or all_prices[symbol].empty
        )
        valid_count = len(symbols_to_fetch) - len(missing_symbols)
        cache_coverage_ratio = (
            valid_count / len(symbols_to_fetch)
            if symbols_to_fetch
            else 1.0
        )
```

Replace the populated prefetch stats with:

```python
        stats = {
            "target_symbols": len(symbols_to_fetch),
            "symbols_with_prices": valid_count,
            "cache_miss_symbols": len(missing_symbols),
            "cache_miss_symbols_sample": missing_symbols[:CACHE_MISS_SYMBOL_SAMPLE_LIMIT],
            "cache_coverage_ratio": cache_coverage_ratio,
            "spy_cached": True,
            "benchmark_cached": cache_only,
            "benchmark_symbol": benchmark_symbol,
            "benchmark_role": benchmark_role,
            "market": normalized_market,
            "cache_only": cache_only,
            "skipped_unsupported_symbols": len(skipped_unsupported_symbols),
        }
```

Extend the missing-benchmark stats mapping with:

```python
                    "cache_miss_symbols_sample": [],
                    "cache_coverage_ratio": 0.0,
                    "benchmark_role": benchmark_role,
                    "cache_only": cache_only,
                    "skipped_unsupported_symbols": 0,
```

Keep `spy_cached` for compatibility with `IncompleteGroupRankingCacheError` and current strict callers.

- [ ] **Step 5: Return target calculation diagnostics through a caller-owned mapping**

Add this keyword parameter to `calculate_group_rankings`:

```python
        diagnostics: Optional[Dict[str, Any]] = None,
```

Immediately after `prefetch_stats = prefetch.stats`, add:

```python
        if diagnostics is not None:
            diagnostics.clear()
            diagnostics.update(prefetch_stats)
```

The strict `cache_requirement.enabled` block remains unchanged except that it reuses the already-computed `cache_coverage_ratio`:

```python
            coverage_ratio = float(prefetch_stats.get("cache_coverage_ratio", 0.0))
```

No strict coverage check runs when `GroupRankCacheRequirement.disabled()` is supplied, so partial constituents remain eligible for existing per-group minimum-stock filtering.

- [ ] **Step 6: Propagate cache-only mode and diagnostics through optimized gap-fill**

Change the signature to:

```python
    def fill_gaps_optimized(
        self,
        db: Session,
        missing_dates: List[date],
        *,
        market: str = "US",
        cache_only: bool = False,
    ) -> Dict:
```

Change prefetch to:

```python
        prefetch = self._coerce_prefetch_data(
            self._prefetch_all_data(
                db,
                market=normalized_market,
                cache_only=cache_only,
            )
        )
```

Include `prefetch_stats: prefetch.stats` in both the missing-benchmark error return and the normal `stats` mapping:

```python
                'prefetch_stats': prefetch.stats,
```

and:

```python
        stats = {
            'total_dates': len(missing_dates),
            'processed': 0,
            'skipped': 0,
            'errors': 0,
            'prefetch_stats': prefetch.stats,
        }
```

- [ ] **Step 7: Update exact prefetch-stat assertions and run the group service suite**

Update exact dictionaries in the existing cache-only and fetch-capable prefetch tests to include:

```python
        "cache_miss_symbols_sample": [],
        "cache_coverage_ratio": 1.0,
        "benchmark_cached": True,
        "benchmark_symbol": "SPY",
        "benchmark_role": "primary",
        "market": "US",
        "cache_only": True,
```

for cache-only success, and use `benchmark_cached=False`, `cache_only=False` for fetch-capable success. For the stale/missing constituent test, use `cache_miss_symbols_sample=["AAPL"]` and `cache_coverage_ratio=0.0`.

Run:

```bash
cd backend && /Users/admin/StockScreenClaude/backend/venv/bin/python -m pytest tests/unit/test_group_rank_service.py -q
```

Expected: all tests PASS, including existing strict coverage rejection and historical fetch-capable prefetch tests.

- [ ] **Step 8: Commit the group service behavior**

```bash
git add backend/app/services/ibd_group_rank_service.py backend/tests/unit/test_group_rank_service.py
git commit -m "fix(groups): support tolerant cache-only prefetch"
```

---

### Task 5: Apply tolerant guarded policy in group tasks

**Files:**
- Modify: `backend/tests/unit/test_group_rank_tasks.py:153-336`
- Modify: `backend/tests/unit/test_group_rank_tasks.py:767-923`
- Modify: `backend/app/tasks/group_rank_tasks.py:120-368`
- Modify: `backend/app/tasks/group_rank_tasks.py:427-467`
- Modify: `backend/app/tasks/group_rank_tasks.py:477-682`

**Interfaces:**
- Consumes: tolerant `calculate_group_rankings` with `cache_only=True`, a disabled `GroupRankCacheRequirement`, and a caller-owned diagnostics mapping; plus `fill_gaps_optimized(cache_only=True)` from Task 4.
- Produces: optional `refresh_guarded_cache_only: bool = False` on the group daily and wrapper tasks; guarded successful results include `cache_only=True`, `cache_policy="refresh_guarded"`, and `prefetch_stats`.

- [ ] **Step 1: Add a guarded target-date policy test**

Add after the manual/strict task tests:

```python
def test_guarded_historical_group_rankings_use_tolerant_cache_only_policy(monkeypatch):
    import app.services.ui_snapshot_service as snapshot_module
    import app.tasks.group_rank_tasks as module

    fake_db = MagicMock()
    fake_service = MagicMock()
    fake_service.price_cache = MagicMock()

    def calculate(*args, **kwargs):
        kwargs["diagnostics"].update({
            "target_symbols": 100,
            "symbols_with_prices": 70,
            "cache_miss_symbols": 30,
            "cache_miss_symbols_sample": ["MISS"],
            "cache_coverage_ratio": 0.70,
            "benchmark_cached": True,
        })
        return [{
            "industry_group": "Software",
            "avg_rs_rating": 91.0,
            "rank": 1,
            "num_stocks": 12,
        }]

    fake_service.calculate_group_rankings.side_effect = calculate
    monkeypatch.setattr(module, "SessionLocal", lambda: fake_db)
    monkeypatch.setattr(module, "get_group_rank_service", lambda: fake_service)
    monkeypatch.setattr(snapshot_module, "safe_publish_groups_bootstrap", lambda: None)
    monkeypatch.setattr(
        "app.services.runtime_preferences_service.is_market_enabled_now",
        lambda _market: True,
    )
    _patch_serialized_lock(monkeypatch)
    _patch_calendar_service(monkeypatch, datetime(2026, 3, 20, 17, 40, 0))

    result = module.calculate_daily_group_rankings.run(
        "2026-03-19",
        market="US",
        refresh_guarded_cache_only=True,
    )

    assert result["groups_ranked"] == 1
    assert result["cache_only"] is True
    assert result["cache_policy"] == "refresh_guarded"
    assert result["prefetch_stats"]["cache_miss_symbols"] == 30
    call_kwargs = fake_service.calculate_group_rankings.call_args.kwargs
    assert call_kwargs["cache_only"] is True
    assert call_kwargs["cache_requirement"] == GroupRankCacheRequirement.disabled()
```

- [ ] **Step 2: Add strict-precedence and wrapper propagation tests**

Add:

```python
def test_force_cache_only_wins_over_guarded_group_tolerance(monkeypatch):
    import app.services.ui_snapshot_service as snapshot_module
    import app.tasks.group_rank_tasks as module

    fake_db = MagicMock()
    fake_service = MagicMock()
    fake_service.price_cache = MagicMock()
    fake_service.calculate_group_rankings.return_value = [{
        "industry_group": "Software",
        "avg_rs_rating": 91.0,
        "rank": 1,
        "num_stocks": 12,
    }]
    monkeypatch.setattr(module, "SessionLocal", lambda: fake_db)
    monkeypatch.setattr(module, "get_group_rank_service", lambda: fake_service)
    monkeypatch.setattr(snapshot_module, "safe_publish_groups_bootstrap", lambda: None)
    _patch_serialized_lock(monkeypatch)
    _patch_calendar_service(monkeypatch, datetime(2026, 3, 20, 17, 40, 0))

    result = module.calculate_daily_group_rankings.run(
        "2026-03-19",
        force_cache_only=True,
        refresh_guarded_cache_only=True,
    )

    assert result["groups_ranked"] == 1
    fake_service.calculate_group_rankings.assert_called_once_with(
        fake_db,
        date(2026, 3, 19),
        market="US",
        cache_only=True,
        cache_requirement=GroupRankCacheRequirement.strict(),
    )


def test_guarded_group_wrapper_propagates_cache_only_to_gapfill_and_target(monkeypatch):
    import app.tasks.group_rank_tasks as module

    fake_db = MagicMock()
    fake_service = MagicMock()
    fake_service.find_missing_dates.return_value = [date(2026, 3, 18)]
    fake_service.fill_gaps_optimized.return_value = {
        "total_dates": 1,
        "processed": 0,
        "errors": 1,
        "prefetch_stats": {"cache_miss_symbols": 4},
    }
    monkeypatch.setattr(module, "SessionLocal", lambda: fake_db)
    monkeypatch.setattr(module, "get_group_rank_service", lambda: fake_service)
    monkeypatch.setattr(module.settings, "group_rank_gapfill_enabled", True)
    monkeypatch.setattr(
        "app.services.ibd_industry_service.IBDIndustryService.get_all_groups",
        lambda db, market: ["Software"],
    )
    monkeypatch.setattr(
        "app.services.runtime_preferences_service.is_market_enabled_now",
        lambda _market: True,
    )
    _patch_serialized_lock(monkeypatch)
    _patch_calendar_service(monkeypatch, datetime(2026, 3, 20, 17, 40, 0))
    target_call = MagicMock(return_value={
        "date": "2026-03-19",
        "groups_ranked": 1,
        "cache_only": True,
        "cache_policy": "refresh_guarded",
    })
    monkeypatch.setattr(module, "_calculate_daily_group_rankings_in_process", target_call)

    result = module.calculate_daily_group_rankings_with_gapfill.run(
        market="US",
        calculation_date="2026-03-19",
        refresh_guarded_cache_only=True,
    )

    fake_service.fill_gaps_optimized.assert_called_once_with(
        fake_db,
        [date(2026, 3, 18)],
        market="US",
        cache_only=True,
    )
    target_call.assert_called_once_with(
        market="US",
        activity_lifecycle="daily_refresh",
        calculation_date="2026-03-19",
        refresh_guarded_cache_only=True,
    )
    assert result["cache_only"] is True
    assert result["cache_policy"] == "refresh_guarded"
    assert "error" not in result
    assert result["gap_fill"]["errors"] == 1
```

- [ ] **Step 3: Run the new group task tests and verify they fail**

Run:

```bash
cd backend && /Users/admin/StockScreenClaude/backend/venv/bin/python -m pytest \
  tests/unit/test_group_rank_tasks.py::test_guarded_historical_group_rankings_use_tolerant_cache_only_policy \
  tests/unit/test_group_rank_tasks.py::test_force_cache_only_wins_over_guarded_group_tolerance \
  tests/unit/test_group_rank_tasks.py::test_guarded_group_wrapper_propagates_cache_only_to_gapfill_and_target -q
```

Expected: FAIL because group tasks do not accept or propagate `refresh_guarded_cache_only`.

- [ ] **Step 4: Resolve strict, guarded, and default group policies explicitly**

Add `refresh_guarded_cache_only: bool = False` after `force_cache_only` in `calculate_daily_group_rankings`. Replace the policy setup with:

```python
        guarded_cache_only = refresh_guarded_cache_only and not force_cache_only
        cache_only = force_cache_only or refresh_guarded_cache_only or calc_date == today_local
        cache_requirement = (
            GroupRankCacheRequirement.strict()
            if cache_only and not guarded_cache_only
            else GroupRankCacheRequirement.disabled()
        )
```

Replace `if same_day_cache_only:` with:

```python
        if cache_only and not guarded_cache_only:
```

Retain the current force-cache bypass and same-day warmup evaluation inside that branch. Build ranking kwargs with:

```python
        ranking_kwargs = {
            "market": effective_market,
            "cache_only": cache_only,
            "cache_requirement": cache_requirement,
        }
        prefetch_stats: dict = {}
        if guarded_cache_only:
            ranking_kwargs["diagnostics"] = prefetch_stats
```

Replace uses of `same_day_cache_only` in result/error payloads with `cache_only`. Extend the success result through a local variable:

```python
        task_result = {
            'date': calc_date.strftime('%Y-%m-%d'),
            'groups_ranked': len(results),
            'top_group': results[0]['industry_group'] if results else None,
            'top_avg_rs': results[0]['avg_rs_rating'] if results else None,
            'calculation_duration_seconds': round(duration, 2),
            'cache_only': cache_only,
            'metadata_repair': repair_stats,
            'timestamp': datetime.now().isoformat(),
        }
        if guarded_cache_only:
            task_result['cache_policy'] = 'refresh_guarded'
            task_result['prefetch_stats'] = prefetch_stats
        return task_result
```

The existing no-results branch remains a hard error. Replace its return mapping
with this complete payload construction:

```python
            no_groups_result = {
                'date': calc_date.strftime('%Y-%m-%d'),
                'groups_ranked': 0,
                'warning': 'No groups could be ranked',
                'error': no_groups_message,
                'reason_code': GroupRankReasonCode.NO_GROUPS_RANKED,
                'calculation_duration_seconds': round(duration, 2),
                'timestamp': datetime.now().isoformat(),
            }
            if guarded_cache_only:
                no_groups_result['cache_only'] = True
                no_groups_result['cache_policy'] = 'refresh_guarded'
                no_groups_result['prefetch_stats'] = prefetch_stats
            return no_groups_result
```

- [ ] **Step 5: Propagate the policy through the group helper and wrapper**

Add `refresh_guarded_cache_only: bool = False` to `_calculate_daily_group_rankings_in_process` and conditionally add:

```python
    if refresh_guarded_cache_only:
        kwargs["refresh_guarded_cache_only"] = True
```

Add the same parameter to `calculate_daily_group_rankings_with_gapfill`. After initializing `result`, add:

```python
    if refresh_guarded_cache_only:
        result['cache_only'] = True
        result['cache_policy'] = 'refresh_guarded'
```

Replace the optimized gap-fill call with:

```python
                gapfill_kwargs = (
                    {"cache_only": True}
                    if refresh_guarded_cache_only
                    else {}
                )
                gap_stats = service.fill_gaps_optimized(
                    db,
                    missing_dates,
                    market=effective_market,
                    **gapfill_kwargs,
                )
```

After building `inner_kwargs`, add:

```python
            if refresh_guarded_cache_only:
                inner_kwargs["refresh_guarded_cache_only"] = True
```

The wrapper's existing nested-error promotion remains the hard-failure mechanism for missing benchmark and zero rankable groups.

Replace the wrapper's generic exception return with:

```python
        error_result = {
            'error': str(e),
            'gap_fill': result.get('gap_fill'),
            'today': result.get('today'),
            'market': effective_market,
            'timestamp': datetime.now().isoformat(),
        }
        if refresh_guarded_cache_only:
            error_result['cache_only'] = True
            error_result['cache_policy'] = 'refresh_guarded'
        return error_result
```

- [ ] **Step 6: Confirm direct manual historical behavior remains fetch-capable**

Keep `test_manual_group_rankings_keep_fetch_capable_behavior` unchanged. It must continue to assert:

```python
    assert result["cache_only"] is False
    fake_service.calculate_group_rankings.assert_called_once_with(
        fake_db,
        datetime(2026, 3, 19).date(),
        market="US",
        cache_only=False,
        cache_requirement=GroupRankCacheRequirement.disabled(),
    )
```

Keep `test_manual_group_rankings_can_force_cache_only_for_static_exports` unchanged to prove strict behavior is also preserved.

- [ ] **Step 7: Update existing wrapper assertions only where the new service default is explicit**

Existing unguarded wrapper tests must continue to expect calls without a `cache_only` kwarg because the wrapper only supplies the kwarg for guarded runs. Existing helper mocks with the old signature therefore remain valid.

Run:

```bash
cd backend && /Users/admin/StockScreenClaude/backend/venv/bin/python -m pytest \
  tests/unit/test_group_rank_tasks.py \
  tests/unit/test_group_rank_service.py -q
```

Expected: all tests PASS, including warmup failure, strict static, default manual historical, non-US taxonomy skip, retry, and gap-fill memory-release coverage.

- [ ] **Step 8: Commit the group task policy**

```bash
git add backend/app/tasks/group_rank_tasks.py backend/tests/unit/test_group_rank_tasks.py
git commit -m "fix(groups): tolerate guarded cache gaps"
```

---

### Task 6: Verify the integrated provider-free pipeline and close the tracked work

**Files:**
- Modify: `docs/superpowers/specs/2026-07-16-refresh-guarded-cache-only-derived-data-design.md:3`
- Modify: `.beads/issues.jsonl`
- Modify: `.beads/interactions.jsonl`

**Interfaces:**
- Consumes: all task and service boundaries implemented in Tasks 1-5.
- Produces: verified focused/full backend test results, implemented design status, and closed Beads issue `stockscreenclaude-duw`.

- [ ] **Step 1: Run the focused integration regression set**

Run:

```bash
cd backend && /Users/admin/StockScreenClaude/backend/venv/bin/python -m pytest \
  tests/unit/test_daily_market_pipeline_tasks.py \
  tests/unit/test_breadth_tasks.py \
  tests/unit/test_breadth_calculator_service.py \
  tests/unit/test_group_rank_tasks.py \
  tests/unit/test_group_rank_service.py -q
```

Expected: all focused tests PASS. The guarded tests must prove provider-capable methods are not called, partial usable coverage succeeds, and strict/manual compatibility remains intact.

- [ ] **Step 2: Run the backend quality gate**

Run:

```bash
cd backend && /Users/admin/StockScreenClaude/backend/venv/bin/python -m pytest
```

Expected: PASS. If a known pre-existing collection failure occurs, capture the exact command/output and file a P1/P2 Beads issue with the failing test name before continuing; do not misreport it as caused by this change without reproducing on the branch parent.

- [ ] **Step 3: Run static diff and syntax checks**

Run:

```bash
git diff --check
cd backend && /Users/admin/StockScreenClaude/backend/venv/bin/python -m compileall -q app/tasks app/services
```

Expected: both commands exit 0 with no output.

- [ ] **Step 4: Mark the design implemented and close the Beads issue**

Change the design document status to:

```markdown
**Status:** Implemented and verified
```

Then run:

```bash
bd close stockscreenclaude-duw --reason="Daily-pipeline breadth/group target and gap-fill runs are cache-only after the price guard, tolerate and report individual cache gaps, preserve strict/manual behavior, and pass focused plus backend quality gates."
bd export -o .beads/issues.jsonl
```

Expected: the issue is closed and the canonical `.beads/issues.jsonl` contains the updated state. The installed Beads version does not provide the legacy `bd sync` command, so `bd export` is the supported ledger-persistence step.

- [ ] **Step 5: Commit documentation and issue state**

Because this repository has known Beads hook issue `stockscreenclaude-7jr`, verify that no root-level `issues.jsonl` is staged. Then run:

```bash
git add docs/superpowers/specs/2026-07-16-refresh-guarded-cache-only-derived-data-design.md .beads/issues.jsonl .beads/interactions.jsonl
git diff --cached --check
git commit --no-verify -m "docs: record guarded cache-only completion"
```

Expected: the commit contains only the design status and canonical `.beads` state; no root-level `issues.jsonl` is added.

- [ ] **Step 6: Rebase, export Beads state, and push the completed branch**

Run:

```bash
git pull --rebase
bd export -o .beads/issues.jsonl
git status --short
git push
git status
```

Expected: export produces no uncommitted ledger delta, push succeeds, and final status says the branch is up to date with `origin/codex/issue-301-cache-only-derived-data` with a clean working tree.
