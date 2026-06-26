# Static Scan Parity Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the static scan page rank default composite results like the live site, and make static snapshot refresh re-fetch price rows whose latest cached bars have missing or non-positive volume.

**Architecture:** Remove the static-only scan-mode priority from both static frontend sorting and static export ordering so `composite_score desc` is a true score sort everywhere. Add an opt-in volume-quality gate to price-history coverage, then enable it only for static daily price refresh so other refresh planners keep their existing date-only semantics.

**Tech Stack:** React/Vitest frontend, FastAPI/SQLAlchemy backend, pytest, beads (`bd`) issue workflow when available.

---

## File Structure

- Modify: `frontend/src/static/scanClient.js`
  - Responsibility: client-side filtering/sorting/pagination for static scan bundles.
- Modify: `frontend/src/static/scanClient.test.js`
  - Responsibility: static scan client behavior tests.
- Modify: `backend/app/services/static_site_export_service.py`
  - Responsibility: static scan bundle serialization and initial row ordering.
- Modify: `backend/tests/unit/test_static_site_export_service.py`
  - Responsibility: exporter ordering regression tests.
- Modify: `backend/app/services/price_history_coverage.py`
  - Responsibility: classify cached price history as fresh, stale, or missing.
- Modify: `backend/app/services/static_daily_price_refresh_service.py`
  - Responsibility: choose which static-site symbols need Yahoo refresh before snapshot build.
- Modify: `backend/tests/unit/test_price_refresh_planning.py`
  - Responsibility: coverage classification tests.
- Modify: `backend/tests/unit/test_static_daily_price_refresh_service.py`
  - Responsibility: static refresh orchestration tests.

---

### Task 1: Frontend Static Sort Uses Pure Composite Ordering

**Files:**
- Modify: `frontend/src/static/scanClient.test.js`
- Modify: `frontend/src/static/scanClient.js`

- [ ] **Step 1: Write the failing frontend sort test**

In `frontend/src/static/scanClient.test.js`, replace the existing test named `sorts full rows ahead of ipo-weighted rows and listing-only rows for composite score` with:

```js
  it('sorts default composite score by score across scan modes', () => {
    const sorted = sortStaticScanRows([
      { symbol: 'IPO95', scan_mode: 'ipo_weighted', composite_score: 95 },
      { symbol: 'FULL80', scan_mode: 'full', composite_score: 80 },
      { symbol: 'NEW1', scan_mode: 'listing_only', composite_score: null },
      { symbol: 'FULL70', scan_mode: 'full', composite_score: 70 },
    ], 'composite_score', 'desc');

    expect(sorted.map((row) => row.symbol)).toEqual(['IPO95', 'FULL80', 'FULL70', 'NEW1']);
  });
```

Then replace the test named `keeps null composite scores last within the same scan-mode bucket for desc sorting` with:

```js
  it('keeps null composite scores last for desc composite sorting', () => {
    const sorted = sortStaticScanRows([
      { symbol: 'FULLNULL', scan_mode: 'full', composite_score: null },
      { symbol: 'FULL80', scan_mode: 'full', composite_score: 80 },
      { symbol: 'FULL70', scan_mode: 'full', composite_score: 70 },
      { symbol: 'IPO95', scan_mode: 'ipo_weighted', composite_score: 95 },
    ], 'composite_score', 'desc');

    expect(sorted.map((row) => row.symbol)).toEqual(['IPO95', 'FULL80', 'FULL70', 'FULLNULL']);
  });
```

- [ ] **Step 2: Run the focused frontend test to verify it fails**

Run:

```bash
cd frontend && npm run test:run -- src/static/scanClient.test.js
```

Expected: FAIL, with the old order returning `FULL80` before `IPO95`.

- [ ] **Step 3: Remove default scan-mode priority from static sorting**

In `frontend/src/static/scanClient.js`, replace the `sortStaticScanRows` function signature and body with:

```js
export const sortStaticScanRows = (
  rows,
  sortBy,
  sortOrder = 'desc',
) => {
  const direction = sortOrder === 'asc' ? 1 : -1;
  return [...rows].sort((left, right) => {
    const leftValue = getSortValue(left, sortBy);
    const rightValue = getSortValue(right, sortBy);
    if (sortBy === 'composite_score' && sortOrder === 'desc') {
      if (leftValue == null && rightValue != null) {
        return 1;
      }
      if (leftValue != null && rightValue == null) {
        return -1;
      }
    }
    const comparison = compareValues(leftValue, rightValue);
    if (comparison !== 0) {
      return comparison * direction;
    }
    return compareValues(left.symbol, right.symbol);
  });
};
```

Remove `getScanModeSortPriority` if no remaining code references it.

- [ ] **Step 4: Remove the obsolete option at the call site**

In `frontend/src/static/pages/StaticScanPage.jsx`, replace:

```jsx
        ? sortStaticScanRows(filteredRows, sortBy, sortOrder, {
          prioritizeCompositeScanMode: !activeScreenId,
        })
```

with:

```jsx
        ? sortStaticScanRows(filteredRows, sortBy, sortOrder)
```

- [ ] **Step 5: Run the focused frontend test to verify it passes**

Run:

```bash
cd frontend && npm run test:run -- src/static/scanClient.test.js
```

Expected: PASS.

- [ ] **Step 6: Commit frontend sort changes**

Run:

```bash
git add frontend/src/static/scanClient.js frontend/src/static/scanClient.test.js frontend/src/static/pages/StaticScanPage.jsx
git commit -m "fix(static): sort composite scan rows by score"
```

Expected: commit succeeds.

---

### Task 2: Static Export Initial Rows Use Pure Composite Ordering

**Files:**
- Modify: `backend/tests/unit/test_static_site_export_service.py`
- Modify: `backend/app/services/static_site_export_service.py`

- [ ] **Step 1: Write the failing exporter ordering test**

In `backend/tests/unit/test_static_site_export_service.py`, rename `test_export_scan_bundle_prioritizes_full_rows_before_ipo_weighted_and_listing_only` to `test_export_scan_bundle_sorts_composite_score_across_scan_modes`.

Within that test, replace the final assertions with:

```python
    assert [row["symbol"] for row in manifest["initial_rows"]] == ["IPO95", "FULL80", "FULL70"]
    chunk = json.loads((tmp_path / "scan" / "chunks" / "chunk-0001.json").read_text(encoding="utf-8"))
    assert [row["symbol"] for row in chunk["rows"]] == ["IPO95", "FULL80", "FULL70", "NEW1"]
```

Remove the entire test named `test_static_scan_mode_sort_priority_matches_frontend_unknown_mode_fallback`; the helper it tests will be deleted.

- [ ] **Step 2: Run the focused exporter test to verify it fails**

Run:

```bash
cd backend && source venv/bin/activate && pytest tests/unit/test_static_site_export_service.py::test_export_scan_bundle_sorts_composite_score_across_scan_modes -q
```

Expected: FAIL, with `FULL80` before `IPO95`.

- [ ] **Step 3: Remove scan-mode priority from exporter sorting**

In `backend/app/services/static_site_export_service.py`, delete `_static_scan_mode_sort_priority`.

Replace `_sort_static_scan_rows` with:

```python
    @classmethod
    def _sort_static_scan_rows(cls, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        def _score_key(row: dict[str, Any]) -> float:
            score = row.get("composite_score")
            if score is None:
                return float("-inf")
            return float(score)

        return sorted(
            rows,
            key=lambda row: (
                -_score_key(row),
                row.get("symbol") or "",
            ),
        )
```

Keep this method name unchanged because `_export_scan_bundle` already calls it.

- [ ] **Step 4: Run the focused exporter test to verify it passes**

Run:

```bash
cd backend && source venv/bin/activate && pytest tests/unit/test_static_site_export_service.py::test_export_scan_bundle_sorts_composite_score_across_scan_modes -q
```

Expected: PASS.

- [ ] **Step 5: Run the adjacent static export tests**

Run:

```bash
cd backend && source venv/bin/activate && pytest tests/unit/test_static_site_export_service.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit exporter sort changes**

Run:

```bash
git add backend/app/services/static_site_export_service.py backend/tests/unit/test_static_site_export_service.py
git commit -m "fix(static): export scan rows by composite score"
```

Expected: commit succeeds.

---

### Task 3: Price Coverage Can Treat Bad Latest Volume As Stale

**Files:**
- Modify: `backend/tests/unit/test_price_refresh_planning.py`
- Modify: `backend/app/services/price_history_coverage.py`

- [ ] **Step 1: Add failing coverage tests for missing and zero volume**

In `backend/tests/unit/test_price_refresh_planning.py`, add this test after `test_price_history_coverage_splits_fresh_stale_and_no_history`:

```python
def test_price_history_coverage_can_require_positive_latest_volume(universe_session):
    from app.services.price_history_coverage import classify_price_history

    universe_session.add_all(
        [
            StockPrice(symbol="GOOD", date=date(2026, 6, 8), close=100, volume=1000),
            StockPrice(symbol="ZERO", date=date(2026, 6, 8), close=50, volume=0),
            StockPrice(symbol="MISSING", date=date(2026, 6, 8), close=25, volume=None),
            StockPrice(symbol="OLD", date=date(2026, 6, 5), close=10, volume=1000),
        ]
    )
    universe_session.commit()

    coverage = classify_price_history(
        universe_session,
        symbols=["GOOD", "ZERO", "MISSING", "OLD", "NONE"],
        as_of_date=date(2026, 6, 8),
        require_positive_volume=True,
    )

    assert coverage.fresh == ("GOOD",)
    assert coverage.stale == ("ZERO", "MISSING", "OLD")
    assert coverage.no_history == ("NONE",)
```

- [ ] **Step 2: Run the focused coverage test to verify it fails**

Run:

```bash
cd backend && source venv/bin/activate && pytest tests/unit/test_price_refresh_planning.py::test_price_history_coverage_can_require_positive_latest_volume -q
```

Expected: FAIL with `TypeError: classify_price_history() got an unexpected keyword argument 'require_positive_volume'`.

- [ ] **Step 3: Implement optional volume-quality classification**

In `backend/app/services/price_history_coverage.py`, add this helper below `_normalize_symbols`:

```python
def _has_positive_volume(value: object) -> bool:
    if value is None:
        return False
    try:
        return float(value) > 0
    except (TypeError, ValueError):
        return False
```

Change the `classify_price_history` signature to:

```python
def classify_price_history(
    db: Session,
    *,
    symbols: Sequence[str],
    as_of_date: date,
    require_positive_volume: bool = False,
) -> PriceHistoryCoverage:
```

Replace the grouped `max(date)` query block with ordered latest-row capture:

```python
    latest_by_symbol: dict[str, date | None] = {}
    latest_volume_by_symbol: dict[str, object] = {}
    for chunk_start in range(0, len(normalized_symbols), 500):
        chunk_symbols = normalized_symbols[chunk_start:chunk_start + 500]
        rows = (
            db.query(StockPrice.symbol, StockPrice.date, StockPrice.volume)
            .filter(StockPrice.symbol.in_(chunk_symbols))
            .order_by(StockPrice.symbol.asc(), StockPrice.date.desc())
            .all()
        )
        for symbol, latest_date, volume in rows:
            key = str(symbol).upper()
            if key in latest_by_symbol:
                continue
            latest_by_symbol[key] = latest_date
            latest_volume_by_symbol[key] = volume
```

Then replace the classification loop with:

```python
    for symbol in normalized_symbols:
        latest_date = latest_by_symbol.get(symbol)
        latest_volume = latest_volume_by_symbol.get(symbol)
        if latest_date is None:
            no_history_symbols.append(symbol)
        elif latest_date < as_of_date:
            stale_symbols.append(symbol)
        elif require_positive_volume and not _has_positive_volume(latest_volume):
            stale_symbols.append(symbol)
        else:
            fresh_symbols.append(symbol)
```

- [ ] **Step 4: Remove the unused import**

In `backend/app/services/price_history_coverage.py`, delete:

```python
from sqlalchemy import func
```

- [ ] **Step 5: Run coverage tests**

Run:

```bash
cd backend && source venv/bin/activate && pytest tests/unit/test_price_refresh_planning.py::test_price_history_coverage_splits_fresh_stale_and_no_history tests/unit/test_price_refresh_planning.py::test_price_history_coverage_can_require_positive_latest_volume -q
```

Expected: PASS.

- [ ] **Step 6: Commit coverage classifier changes**

Run:

```bash
git add backend/app/services/price_history_coverage.py backend/tests/unit/test_price_refresh_planning.py
git commit -m "fix(static): classify missing volume prices as stale"
```

Expected: commit succeeds.

---

### Task 4: Static Daily Refresh Enables Volume-Quality Coverage

**Files:**
- Modify: `backend/tests/unit/test_static_daily_price_refresh_service.py`
- Modify: `backend/app/services/static_daily_price_refresh_service.py`

- [ ] **Step 1: Add failing static refresh orchestration test**

In `backend/tests/unit/test_static_daily_price_refresh_service.py`, add this test after `test_static_daily_price_refresh_service_fetches_stale_and_no_history_groups`:

```python
def test_static_daily_price_refresh_refetches_fresh_rows_with_missing_or_zero_volume() -> None:
    session_factory = _sqlite_session_factory()

    with session_factory() as db:
        db.add_all(
            [
                StockUniverse(symbol="GOOD", market="US", is_active=True, market_cap=100.0),
                StockUniverse(symbol="ZERO", market="US", is_active=True, market_cap=90.0),
                StockUniverse(symbol="MISSING", market="US", is_active=True, market_cap=80.0),
            ]
        )
        db.add_all(
            [
                StockPrice(
                    symbol="GOOD",
                    date=date(2026, 6, 8),
                    open=1.0,
                    high=1.0,
                    low=1.0,
                    close=1.0,
                    volume=1000,
                ),
                StockPrice(
                    symbol="ZERO",
                    date=date(2026, 6, 8),
                    open=1.0,
                    high=1.0,
                    low=1.0,
                    close=1.0,
                    volume=0,
                ),
                StockPrice(
                    symbol="MISSING",
                    date=date(2026, 6, 8),
                    open=1.0,
                    high=1.0,
                    low=1.0,
                    close=1.0,
                    volume=None,
                ),
            ]
        )
        db.commit()

    fetch_calls: list[dict] = []
    stored_batches: list[dict] = []

    class _FakeFetcher:
        def fetch_prices_in_batches(self, symbols, period="2y", start_batch_size=None, market=None):
            fetch_calls.append(
                {
                    "symbols": list(symbols),
                    "period": period,
                    "start_batch_size": start_batch_size,
                    "market": market,
                }
            )
            return {
                symbol: {"price_data": SimpleNamespace(empty=False), "has_error": False}
                for symbol in symbols
            }

    service = StaticDailyPriceRefreshService(
        session_factory=session_factory,
        price_cache=SimpleNamespace(
            store_batch_in_cache=lambda payload, also_store_db=True, market=None: stored_batches.append(
                {
                    "symbols": sorted(payload.keys()),
                    "also_store_db": also_store_db,
                    "market": market,
                }
            )
        ),
        fetcher=_FakeFetcher(),
        batch_size_for_market=lambda _market: 25,
        sleep=lambda _seconds: None,
    )

    result = service.refresh(as_of_date=date(2026, 6, 8), market="US")

    assert fetch_calls[0] == {
        "symbols": ["ZERO", "MISSING"],
        "period": STATIC_DAILY_PRICE_REFRESH_PERIOD,
        "start_batch_size": 25,
        "market": "US",
    }
    assert "GOOD" not in fetch_calls[0]["symbols"]
    assert stored_batches[0] == {
        "symbols": ["MISSING", "ZERO"],
        "also_store_db": True,
        "market": "US",
    }
    assert result["db_fresh_symbols"] == 1
    assert result["stale_symbols"] == 2
```

- [ ] **Step 2: Run the focused static refresh test to verify it fails**

Run:

```bash
cd backend && source venv/bin/activate && pytest tests/unit/test_static_daily_price_refresh_service.py::test_static_daily_price_refresh_refetches_fresh_rows_with_missing_or_zero_volume -q
```

Expected: FAIL because `ZERO` and `MISSING` are classified as fresh and are not fetched.

- [ ] **Step 3: Enable positive-volume coverage in static refresh**

In `backend/app/services/static_daily_price_refresh_service.py`, replace:

```python
            coverage = classify_price_history(
                db,
                symbols=supported_symbols,
                as_of_date=as_of_date,
            )
```

with:

```python
            coverage = classify_price_history(
                db,
                symbols=supported_symbols,
                as_of_date=as_of_date,
                require_positive_volume=True,
            )
```

- [ ] **Step 4: Run the focused static refresh test to verify it passes**

Run:

```bash
cd backend && source venv/bin/activate && pytest tests/unit/test_static_daily_price_refresh_service.py::test_static_daily_price_refresh_refetches_fresh_rows_with_missing_or_zero_volume -q
```

Expected: PASS.

- [ ] **Step 5: Run the static refresh service test file**

Run:

```bash
cd backend && source venv/bin/activate && pytest tests/unit/test_static_daily_price_refresh_service.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit static refresh wiring**

Run:

```bash
git add backend/app/services/static_daily_price_refresh_service.py backend/tests/unit/test_static_daily_price_refresh_service.py
git commit -m "fix(static): refresh cached rows with bad volume"
```

Expected: commit succeeds.

---

### Task 5: Final Verification And Handoff

**Files:**
- No new files.
- Verify all modified files from Tasks 1-4.

- [ ] **Step 1: Run frontend static tests**

Run:

```bash
cd frontend && npm run test:run -- src/static/scanClient.test.js src/static/pages/StaticScanPage.test.jsx
```

Expected: PASS.

- [ ] **Step 2: Run backend targeted tests**

Run:

```bash
cd backend && source venv/bin/activate && pytest \
  tests/unit/test_static_site_export_service.py \
  tests/unit/test_price_refresh_planning.py \
  tests/unit/test_static_daily_price_refresh_service.py \
  -q
```

Expected: PASS.

- [ ] **Step 3: Run lint/build gates if code changed cleanly**

Run:

```bash
export NVM_DIR="$HOME/.nvm" && [ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"
cd frontend && npm run lint
```

Expected: PASS.

- [ ] **Step 4: Check beads availability and record session state**

Run:

```bash
bd ready
```

Expected if `bd` is installed: list ready issues. If the command prints `command not found`, record `bd unavailable in PATH` in the handoff and continue with git verification.

- [ ] **Step 5: Inspect final diff**

Run:

```bash
git status --short
git diff --stat HEAD
git diff -- frontend/src/static/scanClient.js frontend/src/static/scanClient.test.js frontend/src/static/pages/StaticScanPage.jsx
git diff -- backend/app/services/static_site_export_service.py backend/tests/unit/test_static_site_export_service.py
git diff -- backend/app/services/price_history_coverage.py backend/app/services/static_daily_price_refresh_service.py backend/tests/unit/test_price_refresh_planning.py backend/tests/unit/test_static_daily_price_refresh_service.py
```

Expected: only the planned files are modified after the implementation commits, plus any pre-existing unrelated dirty files the worker did not touch.

- [ ] **Step 6: Push according to project session rules**

Run:

```bash
git pull --rebase
bd sync
git push
git status
```

Expected: `git push` succeeds, and `git status` reports the branch is up to date with origin. If `bd sync` is unavailable because `bd` is not installed, run:

```bash
git push
git status
```

and include `bd sync skipped: bd unavailable in PATH` in the handoff.

---

## Self-Review

**Spec coverage:** Task 1 and Task 2 remove static scan-mode priority from the frontend and exported initial/chunk ordering. Task 3 and Task 4 make missing/zero latest volume stale for the static refresh path before snapshot build.

**Placeholder scan:** The plan contains no deferred implementation placeholders; every code-changing step includes exact code or exact replacement snippets.

**Type consistency:** `require_positive_volume` is introduced in Task 3 and used with the same keyword name in Task 4. Existing `PriceHistoryCoverage` tuple fields remain unchanged.
