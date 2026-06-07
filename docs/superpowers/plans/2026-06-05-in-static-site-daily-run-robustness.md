# IN Static Site Daily Run Robustness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the IN static-site daily run explain snapshot row loss, recover safely with prior artifacts when a fresh snapshot is quarantined, and reduce refresh gaps that can trigger repeated failures.

**Architecture:** First make per-symbol snapshot failures observable without changing publish semantics. Then harden the static daily price refresh so stale and no-history symbols are handled explicitly with market-scoped provider/batch settings. Finally adjust the workflow so a quarantined market build uploads diagnostics and lets the combine job use its existing fallback artifact path instead of marking the whole static-site run failed.

**Tech Stack:** Python 3.11, SQLAlchemy repositories, Celery task interface, pytest, GitHub Actions YAML.

---

## File Structure

- Modify `backend/app/use_cases/feature_store/build_daily_snapshot.py`
  - Add bounded per-symbol failure diagnostics to `BuildDailySnapshotResult`.
  - Record diagnostic samples and aggregate counters where rows are currently dropped.
- Modify `backend/app/interfaces/tasks/feature_store_tasks.py`
  - Include `failure_diagnostics` in the task return dict and log a compact summary.
- Modify `backend/app/scripts/export_static_site.py`
  - Fetch no-history symbols separately.
  - Use a market-aware static price batch size.
  - Store refreshed price data with `market=market`.
  - Return a skipped/no-artifact exit code when the selected market snapshot is quarantined and no current artifact can be exported.
- Modify `.github/workflows/static-site.yml`
  - Treat the new no-current-market-artifact exit code like the existing closed-market skip.
  - Upload diagnostics for failed/quarantined market builds with `if: always()`.
- Modify `backend/tests/unit/use_cases/feature_store/test_build_daily_snapshot.py`
  - Add regression tests for scanner error-result diagnostics, scanner exception diagnostics, and cache-prefetch miss diagnostics.
- Modify `backend/tests/unit/test_feature_store_tasks.py`
  - Add a test that task return payload includes `failure_diagnostics`.
- Modify `backend/tests/unit/test_export_static_site_script.py`
  - Add tests for no-history fetch, market-scoped cache storage, market-aware batch sizing, and quarantined snapshot skip.
- Modify `backend/tests/unit/test_static_site_workflow.py`
  - Add tests that the workflow handles the new skip code and uploads diagnostics.

## Task 1: Snapshot Failure Diagnostics

**Files:**
- Modify: `backend/app/use_cases/feature_store/build_daily_snapshot.py`
- Test: `backend/tests/unit/use_cases/feature_store/test_build_daily_snapshot.py`

- [ ] **Step 1: Write failing tests for scanner error-result diagnostics**

Add this test to `TestPartialFailures` in `backend/tests/unit/use_cases/feature_store/test_build_daily_snapshot.py`:

```python
    @_PATCH_TRADING_DAY
    def test_error_result_is_reported_in_failure_diagnostics(self, _mock_td):
        results = {
            "AAPL": {
                "composite_score": 85.0,
                "rating": "Strong Buy",
                "screeners_passed": 1,
            },
            "MSFT": {
                "result_status": "error",
                "error": "Screener execution failed: setup_engine",
                "details": {
                    "data_errors": {
                        "price_data": "No price data returned from batch-only price path"
                    }
                },
            },
            "GOOGL": {
                "composite_score": 70.0,
                "rating": "Buy",
                "screeners_passed": 1,
            },
        }
        uow, scanner = _make_uow(scanner_results=results)
        use_case = BuildDailyFeatureSnapshotUseCase(scanner=scanner)

        result = use_case.execute(
            uow, _make_cmd(), FakeProgressSink(), FakeCancellationToken()
        )

        assert result.failed_symbols == 1
        assert result.failure_diagnostics["reason_counts"] == {
            "scanner_error_result": 1
        }
        assert result.failure_diagnostics["samples"] == [
            {
                "symbol": "MSFT",
                "reason": "scanner_error_result",
                "error": "Screener execution failed: setup_engine",
                "data_errors": {
                    "price_data": "No price data returned from batch-only price path"
                },
            }
        ]
```

- [ ] **Step 2: Write failing tests for scanner exception diagnostics**

Add this test to `TestPartialFailures`:

```python
    @_PATCH_TRADING_DAY
    def test_scanner_exception_is_reported_in_failure_diagnostics(self, _mock_td):
        class ExplodingScanner:
            def scan_stock_multi(self, symbol, screener_names, **kw):
                if symbol == "BOOM":
                    raise RuntimeError("kaboom")
                return {
                    "composite_score": 75.0,
                    "rating": "Buy",
                    "screeners_passed": 1,
                }

        uow, _ = _make_uow(symbols=["AAPL", "BOOM", "GOOGL"])
        use_case = BuildDailyFeatureSnapshotUseCase(scanner=ExplodingScanner())

        result = use_case.execute(
            uow, _make_cmd(), FakeProgressSink(), FakeCancellationToken()
        )

        assert result.failed_symbols == 1
        assert result.failure_diagnostics["reason_counts"] == {"scan_exception": 1}
        assert result.failure_diagnostics["samples"] == [
            {
                "symbol": "BOOM",
                "reason": "scan_exception",
                "error": "RuntimeError: kaboom",
                "data_errors": {},
            }
        ]
```

- [ ] **Step 3: Run the focused tests and verify they fail**

Run:

```bash
cd backend
source venv/bin/activate
pytest backend/tests/unit/use_cases/feature_store/test_build_daily_snapshot.py::TestPartialFailures::test_error_result_is_reported_in_failure_diagnostics backend/tests/unit/use_cases/feature_store/test_build_daily_snapshot.py::TestPartialFailures::test_scanner_exception_is_reported_in_failure_diagnostics -q
```

Expected: both tests fail because `BuildDailySnapshotResult` does not expose `failure_diagnostics`.

- [ ] **Step 4: Add diagnostic helpers and result field**

In `backend/app/use_cases/feature_store/build_daily_snapshot.py`, add imports:

```python
from collections import Counter
```

Add helpers near `RunProgressState`:

```python
MAX_FAILURE_DIAGNOSTIC_SAMPLES = 50


def _extract_data_errors(result: object) -> dict[str, str]:
    if not isinstance(result, dict):
        return {}
    details = result.get("details")
    if not isinstance(details, dict):
        return {}
    data_errors = details.get("data_errors")
    if not isinstance(data_errors, dict):
        return {}
    return {str(key): str(value) for key, value in data_errors.items()}


def _format_exception(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


@dataclass
class FailureDiagnosticsCollector:
    reason_counts: Counter[str] = field(default_factory=Counter)
    samples: list[dict[str, object]] = field(default_factory=list)

    def record(
        self,
        *,
        symbol: str,
        reason: str,
        error: str | None = None,
        data_errors: dict[str, str] | None = None,
    ) -> None:
        self.reason_counts[reason] += 1
        if len(self.samples) >= MAX_FAILURE_DIAGNOSTIC_SAMPLES:
            return
        self.samples.append(
            {
                "symbol": symbol,
                "reason": reason,
                "error": error,
                "data_errors": data_errors or {},
            }
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "reason_counts": dict(self.reason_counts),
            "samples": list(self.samples),
            "sample_limit": MAX_FAILURE_DIAGNOSTIC_SAMPLES,
        }
```

Extend `BuildDailySnapshotResult`:

```python
    failure_diagnostics: dict[str, object] = field(default_factory=dict)
```

- [ ] **Step 5: Record diagnostics in `_scan_symbol`**

Inside `BuildDailyFeatureSnapshotUseCase._execute_scan`, before the chunk loop:

```python
        failure_diagnostics = FailureDiagnosticsCollector()
```

Change `_scan_symbol` to return a diagnostic dict:

```python
            def _scan_symbol(
                symbol: str,
            ) -> tuple[str, FeatureRowWrite | None, bool, dict[str, object] | None]:
                sym = symbol.upper()
                try:
                    if cache_only_symbol_data and sym not in pre_fetched_data:
                        return sym, None, False, {
                            "symbol": sym,
                            "reason": "bulk_prefetch_missing",
                            "error": "Symbol missing from bulk prefetch results",
                            "data_errors": {},
                        }
                    scan_kwargs: dict[str, object] = {}
                    if merged_requirements is not None:
                        scan_kwargs["pre_merged_requirements"] = merged_requirements
                    if sym in pre_fetched_data:
                        scan_kwargs["pre_fetched_data"] = pre_fetched_data[sym]
                    result = self._scanner.scan_stock_multi(
                        symbol=sym,
                        screener_names=cmd.screener_names,
                        criteria=cmd.criteria,
                        composite_method=cmd.composite_method,
                        **scan_kwargs,
                    )
                    result_status = _resolve_result_status(result)
                    if result and result_status != "error":
                        row = _map_orchestrator_to_feature_row(
                            sym, cmd.as_of_date, result
                        )
                        return sym, row, bool(result.get("passes_template")), None
                    return sym, None, False, {
                        "symbol": sym,
                        "reason": "scanner_error_result",
                        "error": result.get("error") if isinstance(result, dict) else None,
                        "data_errors": _extract_data_errors(result),
                    }
                except Exception as exc:
                    logger.debug(
                        "Error scanning %s in run %d",
                        sym,
                        run_id,
                        exc_info=True,
                    )
                    return sym, None, False, {
                        "symbol": sym,
                        "reason": "scan_exception",
                        "error": _format_exception(exc),
                        "data_errors": {},
                    }
```

Update both sequential and threaded call sites to unpack four values and record diagnostics when `row is None`:

```python
                sym, row, passed, diagnostic = future.result()
                outcomes_by_symbol[sym] = (row, passed, diagnostic)
```

```python
                sym, row, passed, diagnostic = _scan_symbol(symbol)
                outcomes_by_symbol[sym] = (row, passed, diagnostic)
```

```python
                row, passed, diagnostic = outcomes_by_symbol.get(
                    symbol.upper(), (None, False, None)
                )
                if row is not None:
                    chunk_rows.append(row)
                    if passed:
                        progress_state.passed_symbols += 1
                else:
                    progress_state.failed_symbols += 1
                    if diagnostic is not None:
                        failure_diagnostics.record(**diagnostic)
```

- [ ] **Step 6: Include diagnostics in all return paths**

When constructing `BuildDailySnapshotResult`, pass:

```python
            failure_diagnostics=failure_diagnostics.to_dict(),
```

For early cancellation and stale-run failure paths that do not have the collector in scope, pass:

```python
            failure_diagnostics={},
```

- [ ] **Step 7: Run focused tests and verify they pass**

Run:

```bash
cd backend
source venv/bin/activate
pytest backend/tests/unit/use_cases/feature_store/test_build_daily_snapshot.py::TestPartialFailures -q
```

Expected: all `TestPartialFailures` tests pass.

- [ ] **Step 8: Commit**

```bash
git add backend/app/use_cases/feature_store/build_daily_snapshot.py backend/tests/unit/use_cases/feature_store/test_build_daily_snapshot.py
git commit -m "feat: capture static snapshot failure diagnostics"
```

## Task 2: Expose Diagnostics Through the Celery Task Payload

**Files:**
- Modify: `backend/app/interfaces/tasks/feature_store_tasks.py`
- Test: `backend/tests/unit/test_feature_store_tasks.py`

- [ ] **Step 1: Write the failing task payload test**

Add a fake result with diagnostics in `backend/tests/unit/test_feature_store_tasks.py` and assert the returned dict includes it:

```python
def test_build_daily_snapshot_returns_failure_diagnostics(monkeypatch):
    class _FakeUseCase:
        def execute(self, *_args, **_kwargs):
            return SimpleNamespace(
                run_id=7,
                status="quarantined",
                total_symbols=3,
                processed_symbols=3,
                failed_symbols=1,
                skipped_symbols=0,
                row_count=2,
                duration_seconds=1.25,
                dq_passed=False,
                warnings=("Row count ratio 66.67% below threshold",),
                failure_diagnostics={
                    "reason_counts": {"scanner_error_result": 1},
                    "samples": [{"symbol": "MSFT", "reason": "scanner_error_result"}],
                    "sample_limit": 50,
                },
            )

    monkeypatch.setattr(
        "app.wiring.bootstrap.get_build_daily_snapshot_use_case",
        lambda: _FakeUseCase(),
    )
    monkeypatch.setattr(
        "app.use_cases.feature_store.build_daily_snapshot._is_us_trading_day",
        lambda *_args, **_kwargs: True,
    )

    payload = _TASK_BODY(
        as_of_date_str="2026-06-04",
        screener_names=["custom"],
        universe_name="market:IN",
        market="IN",
        static_daily_mode=True,
        ignore_runtime_market_gate=True,
    )

    assert payload["failure_diagnostics"]["reason_counts"] == {
        "scanner_error_result": 1
    }
```

If `SimpleNamespace` is not already imported in that file, add:

```python
from types import SimpleNamespace
```

- [ ] **Step 2: Run the test and verify it fails**

Run:

```bash
cd backend
source venv/bin/activate
pytest backend/tests/unit/test_feature_store_tasks.py::test_build_daily_snapshot_returns_failure_diagnostics -q
```

Expected: failure because the task return dict omits `failure_diagnostics`.

- [ ] **Step 3: Add diagnostics to return payload and log**

In `backend/app/interfaces/tasks/feature_store_tasks.py`, update the completion log after line 1047 to include diagnostics counts:

```python
    failure_diagnostics = dict(result.failure_diagnostics or {})
    diagnostic_counts = failure_diagnostics.get("reason_counts", {})
    if diagnostic_counts:
        logger.warning(
            "build_daily_snapshot failure diagnostics for %s: %s",
            effective_market,
            diagnostic_counts,
        )
```

Add to the return dict:

```python
        "failure_diagnostics": failure_diagnostics,
```

- [ ] **Step 4: Run the focused test**

Run:

```bash
cd backend
source venv/bin/activate
pytest backend/tests/unit/test_feature_store_tasks.py::test_build_daily_snapshot_returns_failure_diagnostics -q
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add backend/app/interfaces/tasks/feature_store_tasks.py backend/tests/unit/test_feature_store_tasks.py
git commit -m "feat: surface snapshot diagnostics in task payload"
```

## Task 3: Harden Static Daily Price Refresh

**Files:**
- Modify: `backend/app/scripts/export_static_site.py`
- Test: `backend/tests/unit/test_export_static_site_script.py`

- [ ] **Step 1: Write failing tests for no-history symbols**

Add a test near `test_refresh_static_daily_prices_filters_to_selected_market`:

```python
def test_refresh_static_daily_prices_fetches_no_history_symbols_with_full_period(monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine, tables=[StockUniverse.__table__, StockPrice.__table__])
    session_factory = sessionmaker(bind=engine, autocommit=False, autoflush=False, expire_on_commit=False)

    with session_factory() as db:
        db.add_all(
            [
                StockUniverse(symbol="OLD.NS", market="IN", is_active=True, market_cap=100.0),
                StockUniverse(symbol="NEW.NS", market="IN", is_active=True, market_cap=90.0),
            ]
        )
        db.add(
            StockPrice(
                symbol="OLD.NS",
                date=date(2026, 6, 3),
                open=1.0,
                high=1.0,
                low=1.0,
                close=1.0,
                volume=1000,
            )
        )
        db.commit()

    fetch_calls = []
    stored_batches = []

    class _FakeFetcher:
        def fetch_prices_in_batches(self, symbols, period="2y", start_batch_size=None, market=None):
            fetch_calls.append(
                {
                    "symbols": list(symbols),
                    "period": period,
                    "market": market,
                    "start_batch_size": start_batch_size,
                }
            )
            return {
                symbol: {"price_data": SimpleNamespace(empty=False), "has_error": False}
                for symbol in symbols
            }

    monkeypatch.setattr(export_script, "SessionLocal", session_factory)
    monkeypatch.setattr(export_script, "BulkDataFetcher", lambda: _FakeFetcher())
    monkeypatch.setattr(export_script, "_static_daily_price_refresh_batch_size", lambda market: 25)
    monkeypatch.setattr(
        export_script,
        "get_price_cache",
        lambda: SimpleNamespace(
            store_batch_in_cache=lambda payload, also_store_db=True, market=None: stored_batches.append(
                {
                    "symbols": sorted(payload.keys()),
                    "also_store_db": also_store_db,
                    "market": market,
                }
            )
        ),
    )

    result = export_script._refresh_static_daily_prices(
        as_of_date=date(2026, 6, 4),
        market="IN",
    )

    assert fetch_calls == [
        {
            "symbols": ["OLD.NS"],
            "period": "7d",
            "market": "IN",
            "start_batch_size": 25,
        },
        {
            "symbols": ["NEW.NS"],
            "period": "2y",
            "market": "IN",
            "start_batch_size": 25,
        },
    ]
    assert stored_batches == [
        {"symbols": ["OLD.NS"], "also_store_db": True, "market": "IN"},
        {"symbols": ["NEW.NS"], "also_store_db": True, "market": "IN"},
    ]
    assert result["stale_symbols"] == 1
    assert result["no_history_symbols"] == 1
    assert result["yahoo_fetched_symbols"] == 2
```

- [ ] **Step 2: Run the test and verify it fails**

Run:

```bash
cd backend
source venv/bin/activate
pytest backend/tests/unit/test_export_static_site_script.py::test_refresh_static_daily_prices_fetches_no_history_symbols_with_full_period -q
```

Expected: failure because no-history symbols are not fetched and `store_batch_in_cache` does not receive `market`.

- [ ] **Step 3: Add market-aware batch helper**

In `backend/app/scripts/export_static_site.py`, add:

```python
STATIC_DAILY_PRICE_BOOTSTRAP_PERIOD = "2y"
```

Replace the fixed `STATIC_DAILY_PRICE_REFRESH_BATCH_SIZE` usage with:

```python
def _static_daily_price_refresh_batch_size(market: str | None) -> int:
    if market:
        from app.services.rate_budget_policy import get_rate_budget_policy

        return get_rate_budget_policy().get_batch_size("yfinance", market)
    return STATIC_DAILY_PRICE_REFRESH_BATCH_SIZE
```

- [ ] **Step 4: Refactor `_refresh_static_daily_prices` to fetch stale and no-history groups**

Inside `_refresh_static_daily_prices`, replace the single stale-symbol loop with a local helper:

```python
    batch_size = _static_daily_price_refresh_batch_size(market)

    def _fetch_and_store(symbols: list[str], *, period: str) -> tuple[int, int, list[str]]:
        refreshed_count = 0
        failed_count = 0
        rate_limited: list[str] = []
        total_symbols = len(symbols)
        if not symbols:
            return 0, 0, []
        total_group_batches = (total_symbols + batch_size - 1) // batch_size
        for batch_index, batch_symbols in enumerate(
            _iter_chunks(symbols, batch_size),
            start=1,
        ):
            print(
                f"[static-daily prices] Batch {batch_index}/{total_group_batches}: "
                f"fetching {len(batch_symbols):,} symbols from Yahoo ({period}).",
                flush=True,
            )
            batch_results = fetcher.fetch_prices_in_batches(
                batch_symbols,
                period=period,
                start_batch_size=batch_size,
                market=market,
            )
            batch_to_store: dict[str, Any] = {}
            for symbol, payload in batch_results.items():
                price_data = payload.get("price_data")
                if not payload.get("has_error") and price_data is not None and not price_data.empty:
                    batch_to_store[symbol] = price_data
                    refreshed_count += 1
                else:
                    failed_count += 1
                    if _is_rate_limit_failure(payload):
                        rate_limited.append(symbol)
            if batch_to_store:
                price_cache.store_batch_in_cache(
                    batch_to_store,
                    also_store_db=True,
                    market=market,
                )
        return refreshed_count, failed_count, rate_limited
```

Then call it:

```python
    stale_refreshed, stale_failed, stale_rate_limited = _fetch_and_store(
        stale_symbols,
        period=STATIC_DAILY_PRICE_REFRESH_PERIOD,
    )
    bootstrap_refreshed, bootstrap_failed, bootstrap_rate_limited = _fetch_and_store(
        no_history_symbols,
        period=STATIC_DAILY_PRICE_BOOTSTRAP_PERIOD,
    )
    refreshed = stale_refreshed + bootstrap_refreshed
    failed = stale_failed + bootstrap_failed
    rate_limited_symbols = stale_rate_limited + bootstrap_rate_limited
```

- [ ] **Step 5: Preserve existing retry behavior**

Keep the existing `_retry_rate_limited_failures(...)` call, passing the combined `rate_limited_symbols`.

- [ ] **Step 6: Run static export script tests**

Run:

```bash
cd backend
source venv/bin/activate
pytest backend/tests/unit/test_export_static_site_script.py -q
```

Expected: pass.

- [ ] **Step 7: Commit**

```bash
git add backend/app/scripts/export_static_site.py backend/tests/unit/test_export_static_site_script.py
git commit -m "fix: refresh static daily no-history symbols"
```

## Task 4: Make Quarantined Market Builds Soft-Skip With Diagnostics

**Files:**
- Modify: `backend/app/scripts/export_static_site.py`
- Modify: `.github/workflows/static-site.yml`
- Test: `backend/tests/unit/test_export_static_site_script.py`
- Test: `backend/tests/unit/test_static_site_workflow.py`

- [ ] **Step 1: Write failing export-script test**

Add a test to `backend/tests/unit/test_export_static_site_script.py` that simulates a selected market snapshot returning `quarantined` and no current published run:

```python
def test_run_daily_refresh_returns_skip_when_selected_market_snapshot_quarantined(monkeypatch, tmp_path):
    monkeypatch.setattr(export_script, "_resolve_latest_completed_trading_date", lambda market: date(2026, 6, 4))
    monkeypatch.setattr(export_script, "_refresh_static_daily_prices", lambda **_kwargs: {"status": "completed"})
    monkeypatch.setattr(export_script.IBDIndustryService, "load_from_csv", lambda db, csv_path=None: 0)
    monkeypatch.setattr(export_script, "_ensure_group_rank_history", lambda **_kwargs: {"status": "skipped"})
    monkeypatch.setattr(export_script, "_enrich_feature_run_with_ibd_metadata", lambda **_kwargs: {"updated_rows": 0})
    monkeypatch.setattr(export_script, "SessionLocal", lambda: SimpleNamespace(__enter__=lambda self: self, __exit__=lambda *args: None))

    class _FakeTask:
        def run(self, **_kwargs):
            return {
                "status": "quarantined",
                "run_id": 1,
                "market": "IN",
                "failed_symbols": 2052,
                "failure_diagnostics": {
                    "reason_counts": {"scanner_error_result": 2052},
                    "samples": [{"symbol": "500000.BO", "reason": "scanner_error_result"}],
                },
            }

    monkeypatch.setattr(export_script, "build_daily_snapshot", _FakeTask())

    result = export_script._run_daily_refresh(
        output_dir=tmp_path,
        market="IN",
        refresh_daily=True,
        build_mode=export_script.STATIC_BUILD_MODE_PRICE_DELTA,
        skip_universe_refresh=True,
        skip_fundamentals_refresh=True,
    )

    assert result["feature_snapshots"]["IN"]["status"] == "quarantined"
    diagnostic_path = tmp_path / "diagnostics" / "in" / "snapshot-failure.json"
    assert diagnostic_path.exists()
    payload = json.loads(diagnostic_path.read_text())
    assert payload["market"] == "IN"
    assert payload["failure_diagnostics"]["reason_counts"] == {
        "scanner_error_result": 2052
    }
```

Import `json` if needed.

- [ ] **Step 2: Add diagnostics writer**

In `backend/app/scripts/export_static_site.py`, add:

```python
def _write_market_diagnostics(
    *,
    output_dir: Path,
    market: str,
    snapshot: dict[str, Any],
) -> None:
    diagnostics_dir = output_dir / "diagnostics" / market.lower()
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "market": market.upper(),
        "status": snapshot.get("status"),
        "run_id": snapshot.get("run_id"),
        "failed_symbols": snapshot.get("failed_symbols"),
        "row_count": snapshot.get("row_count"),
        "warnings": snapshot.get("warnings", []),
        "failure_diagnostics": snapshot.get("failure_diagnostics", {}),
    }
    (diagnostics_dir / "snapshot-failure.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
```

Call it when `_snapshot_ready(snapshot)` is false:

```python
            if not _snapshot_ready(snapshot):
                _write_market_diagnostics(
                    output_dir=Path(output_dir),
                    market=selected_market,
                    snapshot=snapshot,
                )
```

- [ ] **Step 3: Add explicit skip exit code for no current artifact**

Add near the existing exit code:

```python
STATIC_EXPORT_NO_CURRENT_ARTIFACT_EXIT_CODE = 79
```

In `main()`, when selected market daily refresh finishes but `StaticSiteExportService.export(...)` raises `RuntimeError("No published feature run is available...")` and diagnostics exist, return `79` instead of `1`.

- [ ] **Step 4: Write workflow tests**

Add to `backend/tests/unit/test_static_site_workflow.py`:

```python
def test_static_site_market_export_soft_skips_no_current_artifact() -> None:
    build_market_job = _build_market_job()
    export_step = build_market_job.split("      - name: Export market static data bundle\n", 1)[1].split(
        "\n      - name: Build daily price bundle",
        1,
    )[0]

    assert 'if [ "$status" -eq 79 ]; then' in export_step
    assert "has_artifact=false" in export_step


def test_static_site_uploads_market_diagnostics_on_failure_or_skip() -> None:
    build_market_job = _build_market_job()

    assert "Upload market diagnostics" in build_market_job
    diagnostics_step = build_market_job.split("      - name: Upload market diagnostics\n", 1)[1].split(
        "\n      - name: Build daily price bundle",
        1,
    )[0]
    assert "if: always()" in diagnostics_step
    assert "static-market-diagnostics-${{ matrix.market }}" in diagnostics_step
```

- [ ] **Step 5: Update workflow**

In `.github/workflows/static-site.yml`, update the export step:

```bash
          if [ "$status" -eq 78 ]; then
            echo "Market ${{ matrix.market }} is not a trading day; no current static artifact will be uploaded."
            echo "has_artifact=false" >> "$GITHUB_OUTPUT"
            exit 0
          fi
          if [ "$status" -eq 79 ]; then
            echo "Market ${{ matrix.market }} did not produce a publishable current snapshot; fallback artifact will be used if available."
            echo "has_artifact=false" >> "$GITHUB_OUTPUT"
            exit 0
          fi
```

Add a diagnostics upload step immediately after export:

```yaml
      - name: Upload market diagnostics
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: static-market-diagnostics-${{ matrix.market }}
          path: /tmp/static-data/diagnostics/${{ env.MARKET_LOWER }}
          if-no-files-found: ignore
```

- [ ] **Step 6: Run focused tests**

Run:

```bash
cd backend
source venv/bin/activate
pytest backend/tests/unit/test_export_static_site_script.py::test_run_daily_refresh_returns_skip_when_selected_market_snapshot_quarantined backend/tests/unit/test_static_site_workflow.py::test_static_site_market_export_soft_skips_no_current_artifact backend/tests/unit/test_static_site_workflow.py::test_static_site_uploads_market_diagnostics_on_failure_or_skip -q
```

Expected: pass.

- [ ] **Step 7: Commit**

```bash
git add backend/app/scripts/export_static_site.py .github/workflows/static-site.yml backend/tests/unit/test_export_static_site_script.py backend/tests/unit/test_static_site_workflow.py
git commit -m "fix: preserve static site fallback on quarantined markets"
```

## Task 5: Quality Gates and Operational Check

**Files:**
- No code changes.

- [ ] **Step 1: Run backend unit tests for touched areas**

Run:

```bash
cd backend
source venv/bin/activate
pytest \
  backend/tests/unit/use_cases/feature_store/test_build_daily_snapshot.py \
  backend/tests/unit/test_feature_store_tasks.py \
  backend/tests/unit/test_export_static_site_script.py \
  backend/tests/unit/test_static_site_workflow.py \
  -q
```

Expected: all tests pass.

- [ ] **Step 2: Run a local deterministic IN diagnostic smoke test**

Use the same harness shape from the investigation, but only as a manual smoke command after unit tests:

```bash
cd backend
DATABASE_URL='postgresql://stockscanner:stockscanner@localhost:55432/stockscanner' \
REDIS_ENABLED=false \
venv/bin/python -m app.scripts.export_static_site \
  --output-dir /tmp/static-data-in-smoke \
  --refresh-daily \
  --build-mode price_delta \
  --skip-universe-refresh \
  --skip-fundamentals-refresh \
  --market IN
```

Expected: either a publishable IN artifact is exported, or `/tmp/static-data-in-smoke/diagnostics/in/snapshot-failure.json` exists and includes `failure_diagnostics.reason_counts`.

- [ ] **Step 3: Inspect Git status**

Run:

```bash
git status --short
```

Expected: clean after commits.

- [ ] **Step 4: Push**

Run:

```bash
git pull --rebase
bd sync
git push
git status
```

Expected:
- `bd sync` may fail with `Error: no beads database found` unless beads has been initialized before execution.
- `git push` succeeds.
- `git status` reports the branch is up to date with origin.

## Self-Review

- Spec coverage: diagnostics, no-history refresh, market-scoped price storage, batch-size hardening, workflow fallback, and tests are each covered by a task.
- Placeholder scan: no banned placeholder phrases remain; every task has concrete test or implementation content.
- Type consistency: `failure_diagnostics` is a dict on `BuildDailySnapshotResult`, is passed through `feature_store_tasks`, and is written into static diagnostics JSON.
