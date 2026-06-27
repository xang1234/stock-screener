# RS Line New-High Signal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add first-class scanner, filter, result-table, and chart support for the O'Neil/Minervini RS line new-high and blue-dot leadership signal.

**Architecture:** Reuse the existing benchmark-aligned stock/benchmark ratio logic in `backend/app/analysis/patterns/rs_line.py`, compute one leadership snapshot per symbol in `ScanOrchestrator`, and persist flattened fields on both `scan_results` and `stock_feature_daily` while also keeping the values in each row's details JSON. The frontend exposes one marquee visible filter/column for recent blue dots, keeps the existing setup-engine `se_*` filters for compatibility, and continues to draw chart blue dots from the existing `/rs-line` payload.

**Tech Stack:** Python, pandas, SQLAlchemy, Alembic, FastAPI/Pydantic, React, MUI, Vitest, pytest.

---

## Definitions

Use these names consistently across backend, API, static export, and frontend state:

- `rs_line_new_high`: latest RS line value is at its trailing `lookback` high.
- `rs_line_new_high_before_price`: latest RS line is at a trailing high while latest price is not.
- `rs_line_blue_dot_recent`: `rs_line_new_high_before_price` happened at least once in the trailing `recent_days` trading sessions.
- `rs_line_new_high_date`: latest date where `rs_line_new_high` was true, serialized as `YYYY-MM-DD`, or `None`.

Defaults:

- `lookback = 252` trading sessions.
- `recent_days = 5` trading sessions.

Setup-engine compatibility:

- Keep `se_rs_line_new_high` and `se_rs_line_blue_dot`.
- Preserve `se_rs_line_blue_dot` as the current-bar predicate. It should map to `rs_line_new_high_before_price`, not to the recent-days predicate.

## File Structure

- Modify `backend/app/analysis/patterns/rs_line.py`: add the reusable snapshot calculator.
- Modify `backend/app/analysis/patterns/readiness.py`: reuse the snapshot calculator for setup-engine RS flags.
- Modify `backend/app/scanners/base_screener.py`: extend `PrecomputedScanContext` with the new RS leadership fields.
- Modify `backend/app/scanners/scan_orchestrator.py`: compute and promote the new fields into every result dict that has benchmark data.
- Modify `backend/app/models/scan_result.py`: add first-class columns and indexes for legacy `scan_results`.
- Modify `backend/app/infra/db/models/feature_store.py`: add first-class columns and indexes for `stock_feature_daily`.
- Create `backend/alembic/versions/20260627_0023_add_rs_line_leadership_fields.py`: database migration for both tables.
- Modify `backend/app/infra/db/repositories/scan_result_repo.py`: persist and read the new scan-result columns.
- Modify `backend/app/infra/db/repositories/feature_store_repo.py`: persist and read the new feature-store columns.
- Modify `backend/app/infra/query/scan_result_query.py` and `backend/app/infra/query/feature_store_query.py`: make the new fields filterable/sortable through column maps.
- Modify `backend/app/schemas/scanning.py` and `backend/app/api/v1/scan_filter_params.py`: expose response fields and query params.
- Modify `backend/app/services/preset_screens.py`: switch the Blue Dot Leaders preset to the first-class recent-blue-dot field.
- Modify `frontend/src/features/scan/defaultFilters.js`: add `rsLineBlueDotRecent`.
- Modify `frontend/src/utils/filterUtils.js`: send `rs_line_blue_dot_recent`.
- Modify `frontend/src/static/scanClient.js`: make static scans honor the new boolean filter.
- Modify `frontend/src/features/scan/components/filterPanel/constants.js`, `frontend/src/features/scan/components/filterPanel/utils.js`, and `frontend/src/features/scan/components/filterPanel/TechnicalFiltersSection.jsx`: expose the visible filter.
- Modify `frontend/src/components/Scan/ResultsTable.jsx`: add one first-class `BD5` column near RS fields.
- Modify `frontend/src/test/fixtures/setupEngineFixtures.js`, `frontend/src/components/Scan/ResultsTable.test.jsx`, and `frontend/src/components/Scan/FilterPanel.test.jsx`: cover the new column and filter.
- Create or modify `frontend/src/components/Charts/rsMarkers.js` and `frontend/src/components/Charts/rsMarkers.test.js` only if the inline marker builder in `CandlestickChart.jsx` needs a pure test seam.

---

### Task 1: Backend RS Leadership Calculator

**Files:**
- Modify: `backend/app/analysis/patterns/rs_line.py`
- Test: `backend/tests/unit/test_rs_line.py`

- [ ] **Step 1: Write failing unit tests for current and recent signals**

Add tests that cover all four public fields:

```python
def test_rs_line_leadership_snapshot_detects_current_blue_dot():
    dates = pd.date_range("2026-01-01", periods=6, freq="D")
    stock = pd.Series([10, 11, 12, 13, 12.5, 12.8], index=dates)
    benchmark = pd.Series([10, 10, 10, 10, 9, 8], index=dates)

    snapshot = rs_line_leadership_snapshot(stock, benchmark, lookback=6, recent_days=5)

    assert snapshot == {
        "rs_line_new_high": True,
        "rs_line_new_high_before_price": True,
        "rs_line_blue_dot_recent": True,
        "rs_line_new_high_date": "2026-01-06",
    }


def test_rs_line_leadership_snapshot_distinguishes_price_new_high():
    dates = pd.date_range("2026-01-01", periods=6, freq="D")
    stock = pd.Series([10, 11, 12, 13, 14, 15], index=dates)
    benchmark = pd.Series([10, 10, 10, 10, 10, 9], index=dates)

    snapshot = rs_line_leadership_snapshot(stock, benchmark, lookback=6, recent_days=5)

    assert snapshot["rs_line_new_high"] is True
    assert snapshot["rs_line_new_high_before_price"] is False
    assert snapshot["rs_line_blue_dot_recent"] is False
    assert snapshot["rs_line_new_high_date"] == "2026-01-06"


def test_rs_line_leadership_snapshot_keeps_recent_blue_dot_after_current_flag_fades():
    dates = pd.date_range("2026-01-01", periods=8, freq="D")
    stock = pd.Series([10, 11, 12, 13, 12.5, 12.3, 12.2, 12.1], index=dates)
    benchmark = pd.Series([10, 10, 10, 10, 8, 8.2, 8.4, 8.6], index=dates)

    snapshot = rs_line_leadership_snapshot(stock, benchmark, lookback=8, recent_days=5)

    assert snapshot["rs_line_new_high"] is False
    assert snapshot["rs_line_new_high_before_price"] is False
    assert snapshot["rs_line_blue_dot_recent"] is True
    assert snapshot["rs_line_new_high_date"] == "2026-01-05"


def test_rs_line_leadership_snapshot_empty_when_benchmark_missing():
    stock = pd.Series([10, 11, 12])
    benchmark = pd.Series([], dtype=float)

    snapshot = rs_line_leadership_snapshot(stock, benchmark)

    assert snapshot == {
        "rs_line_new_high": False,
        "rs_line_new_high_before_price": False,
        "rs_line_blue_dot_recent": False,
        "rs_line_new_high_date": None,
    }
```

- [ ] **Step 2: Run the test and confirm it fails**

Run:

```bash
cd backend && source venv/bin/activate && pytest tests/unit/test_rs_line.py -q
```

Expected: fail because `rs_line_leadership_snapshot` is not defined.

- [ ] **Step 3: Implement the calculator**

Add this public function in `backend/app/analysis/patterns/rs_line.py` below `blue_dot_series`:

```python
def _empty_leadership_snapshot() -> dict[str, bool | str | None]:
    return {
        "rs_line_new_high": False,
        "rs_line_new_high_before_price": False,
        "rs_line_blue_dot_recent": False,
        "rs_line_new_high_date": None,
    }


def _date_string(value: object) -> str:
    if hasattr(value, "strftime"):
        return value.strftime("%Y-%m-%d")
    return str(value)[:10]


def rs_line_leadership_snapshot(
    stock_close: pd.Series,
    benchmark_close: pd.Series,
    *,
    lookback: int = DEFAULT_LOOKBACK,
    recent_days: int = 5,
) -> dict[str, bool | str | None]:
    """Latest RS leadership flags for scanner/filter rows."""
    if lookback < 1:
        raise ValueError("lookback must be >= 1")
    if recent_days < 1:
        raise ValueError("recent_days must be >= 1")

    rs = _aligned_ratio(stock_close, benchmark_close)
    frame = pd.DataFrame({"rs": rs, "price": stock_close.astype(float)}).dropna()
    if frame.empty:
        return _empty_leadership_snapshot()

    rs_new_high = rolling_at_new_high(frame["rs"], window=lookback)
    price_new_high = rolling_at_new_high(frame["price"], window=lookback)
    blue_dot = rs_new_high & (~price_new_high)

    new_high_dates = frame.index[rs_new_high]
    latest_new_high_date = (
        _date_string(new_high_dates[-1]) if len(new_high_dates) > 0 else None
    )

    return {
        "rs_line_new_high": bool(rs_new_high.iloc[-1]),
        "rs_line_new_high_before_price": bool(blue_dot.iloc[-1]),
        "rs_line_blue_dot_recent": bool(blue_dot.tail(recent_days).any()),
        "rs_line_new_high_date": latest_new_high_date,
    }
```

- [ ] **Step 4: Run the RS line tests**

Run:

```bash
cd backend && source venv/bin/activate && pytest tests/unit/test_rs_line.py -q
```

Expected: pass.

- [ ] **Step 5: Commit the calculator**

```bash
git add backend/app/analysis/patterns/rs_line.py backend/tests/unit/test_rs_line.py
git commit -m "feat: add rs line leadership snapshot"
```

---

### Task 2: Reuse the Calculator in Setup Engine Readiness

**Files:**
- Modify: `backend/app/analysis/patterns/readiness.py`
- Test: `backend/tests/unit/test_setup_engine_readiness.py`

- [ ] **Step 1: Add a regression assertion for current semantics**

In `test_compute_breakout_readiness_blue_dot_when_rs_leads_price`, keep the existing assertions and add:

```python
assert computed.rs_line_blue_dot is True
```

In the monotonic price-new-high case, keep:

```python
assert computed.rs_line_new_high is True
assert computed.rs_line_blue_dot is False
```

- [ ] **Step 2: Replace the duplicate readiness logic**

In `backend/app/analysis/patterns/readiness.py`, import the helper:

```python
from app.analysis.patterns.rs_line import rs_line_leadership_snapshot
```

Replace the manual `rs_line_new_high` / `rs_line_blue_dot` block with:

```python
        aligned = pd.DataFrame({"rs": rs_series, "price": close}).dropna()
        rs_tail = aligned["rs"].tail(rs_lookback)
        if not rs_tail.empty:
            snapshot = rs_line_leadership_snapshot(
                aligned["price"],
                aligned["price"] / aligned["rs"],
                lookback=rs_lookback,
                recent_days=1,
            )
            rs_line_new_high = bool(snapshot["rs_line_new_high"])
            rs_line_blue_dot = bool(snapshot["rs_line_new_high_before_price"])
            rs_252_max = float(rs_tail.max())
```

This preserves setup-engine `rs_line_blue_dot` as current-bar only.

- [ ] **Step 3: Run setup-engine readiness tests**

Run:

```bash
cd backend && source venv/bin/activate && pytest tests/unit/test_setup_engine_readiness.py -q
```

Expected: pass.

- [ ] **Step 4: Commit readiness reuse**

```bash
git add backend/app/analysis/patterns/readiness.py backend/tests/unit/test_setup_engine_readiness.py
git commit -m "refactor: share rs line new-high logic"
```

---

### Task 3: Compute First-Class Fields in ScanOrchestrator

**Files:**
- Modify: `backend/app/scanners/base_screener.py`
- Modify: `backend/app/scanners/scan_orchestrator.py`
- Test: `backend/tests/unit/test_scan_orchestrator.py`
- Test: `backend/tests/unit/test_precomputed_scan_context.py`

- [ ] **Step 1: Extend precomputed context**

Add these fields to `PrecomputedScanContext` in `backend/app/scanners/base_screener.py`:

```python
    rs_line_new_high: bool = False
    rs_line_new_high_before_price: bool = False
    rs_line_blue_dot_recent: bool = False
    rs_line_new_high_date: Optional[str] = None
```

- [ ] **Step 2: Write an orchestrator unit test**

Add a test that scans a stock with benchmark data where RS leads price, then assert the combined result includes:

```python
assert result["rs_line_new_high"] is True
assert result["rs_line_new_high_before_price"] is True
assert result["rs_line_blue_dot_recent"] is True
assert result["rs_line_new_high_date"] == "2026-01-06"
```

- [ ] **Step 3: Implement shared metric extraction**

In `backend/app/scanners/scan_orchestrator.py`, import:

```python
from app.analysis.patterns.rs_line import rs_line_leadership_snapshot
```

Add constants near the scanner minimums:

```python
RS_LINE_LOOKBACK = 252
RS_LINE_BLUE_DOT_RECENT_DAYS = 5
```

Add helper:

```python
def _empty_rs_line_leadership_metrics() -> dict[str, object]:
    return {
        "rs_line_new_high": False,
        "rs_line_new_high_before_price": False,
        "rs_line_blue_dot_recent": False,
        "rs_line_new_high_date": None,
    }
```

Inside `_build_precomputed_scan_context`, compute:

```python
    rs_leadership = _empty_rs_line_leadership_metrics()
    if benchmark_close_chrono is not None and not benchmark_close_chrono.empty:
        rs_leadership = rs_line_leadership_snapshot(
            close_chrono,
            benchmark_close_chrono,
            lookback=RS_LINE_LOOKBACK,
            recent_days=RS_LINE_BLUE_DOT_RECENT_DAYS,
        )
```

Pass these into `PrecomputedScanContext`:

```python
        rs_line_new_high=bool(rs_leadership["rs_line_new_high"]),
        rs_line_new_high_before_price=bool(rs_leadership["rs_line_new_high_before_price"]),
        rs_line_blue_dot_recent=bool(rs_leadership["rs_line_blue_dot_recent"]),
        rs_line_new_high_date=rs_leadership["rs_line_new_high_date"],
```

- [ ] **Step 4: Promote metrics into every result dict**

In `_partial_history_metrics`, initialize the four fields with `_empty_rs_line_leadership_metrics()` and populate them from precomputed context when available.

In `_combine_results`, add these top-level keys before `details`:

```python
            "rs_line_new_high": bool(
                stock_data.precomputed_scan_context.rs_line_new_high
                if stock_data.precomputed_scan_context is not None
                else False
            ),
            "rs_line_new_high_before_price": bool(
                stock_data.precomputed_scan_context.rs_line_new_high_before_price
                if stock_data.precomputed_scan_context is not None
                else False
            ),
            "rs_line_blue_dot_recent": bool(
                stock_data.precomputed_scan_context.rs_line_blue_dot_recent
                if stock_data.precomputed_scan_context is not None
                else False
            ),
            "rs_line_new_high_date": (
                stock_data.precomputed_scan_context.rs_line_new_high_date
                if stock_data.precomputed_scan_context is not None
                else None
            ),
```

- [ ] **Step 5: Run orchestrator tests**

Run:

```bash
cd backend && source venv/bin/activate && pytest tests/unit/test_scan_orchestrator.py tests/unit/test_precomputed_scan_context.py -q
```

Expected: pass.

- [ ] **Step 6: Commit orchestrator computation**

```bash
git add backend/app/scanners/base_screener.py backend/app/scanners/scan_orchestrator.py backend/tests/unit/test_scan_orchestrator.py backend/tests/unit/test_precomputed_scan_context.py
git commit -m "feat: compute rs line leadership in scans"
```

---

### Task 4: Persist Columns on Scan Results and Feature Store

**Files:**
- Modify: `backend/app/models/scan_result.py`
- Modify: `backend/app/infra/db/models/feature_store.py`
- Create: `backend/alembic/versions/20260627_0023_add_rs_line_leadership_fields.py`
- Modify: `backend/app/infra/db/repositories/scan_result_repo.py`
- Modify: `backend/app/infra/db/repositories/feature_store_repo.py`
- Test: `backend/tests/unit/test_setup_engine_persistence.py`
- Test: `backend/tests/unit/repositories/test_feature_store_repo.py`

- [ ] **Step 1: Add ORM columns**

In `ScanResult`, add:

```python
    rs_line_new_high = Column(Boolean, nullable=True, index=True)
    rs_line_new_high_before_price = Column(Boolean, nullable=True, index=True)
    rs_line_blue_dot_recent = Column(Boolean, nullable=True, index=True)
    rs_line_new_high_date = Column(String(10), nullable=True, index=True)
```

In `StockFeatureDaily`, add:

```python
    rs_line_new_high = Column(Boolean, nullable=True, index=True)
    rs_line_new_high_before_price = Column(Boolean, nullable=True, index=True)
    rs_line_blue_dot_recent = Column(Boolean, nullable=True, index=True)
    rs_line_new_high_date = Column(Text, nullable=True, index=True)
```

Import `Boolean` in both ORM modules.

- [ ] **Step 2: Add migration**

Create the Alembic revision with:

```python
"""Add RS line leadership fields."""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "20260627_0023"
down_revision = "20260618_0022"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("scan_results", sa.Column("rs_line_new_high", sa.Boolean(), nullable=True))
    op.add_column("scan_results", sa.Column("rs_line_new_high_before_price", sa.Boolean(), nullable=True))
    op.add_column("scan_results", sa.Column("rs_line_blue_dot_recent", sa.Boolean(), nullable=True))
    op.add_column("scan_results", sa.Column("rs_line_new_high_date", sa.String(length=10), nullable=True))
    op.create_index("idx_scan_rs_line_new_high", "scan_results", ["scan_id", "rs_line_new_high"])
    op.create_index("idx_scan_rs_line_new_high_before_price", "scan_results", ["scan_id", "rs_line_new_high_before_price"])
    op.create_index("idx_scan_rs_line_blue_dot_recent", "scan_results", ["scan_id", "rs_line_blue_dot_recent"])
    op.create_index("idx_scan_rs_line_new_high_date", "scan_results", ["scan_id", "rs_line_new_high_date"])

    op.add_column("stock_feature_daily", sa.Column("rs_line_new_high", sa.Boolean(), nullable=True))
    op.add_column("stock_feature_daily", sa.Column("rs_line_new_high_before_price", sa.Boolean(), nullable=True))
    op.add_column("stock_feature_daily", sa.Column("rs_line_blue_dot_recent", sa.Boolean(), nullable=True))
    op.add_column("stock_feature_daily", sa.Column("rs_line_new_high_date", sa.Text(), nullable=True))
    op.create_index("ix_sfd_run_rs_line_new_high", "stock_feature_daily", ["run_id", "rs_line_new_high"])
    op.create_index("ix_sfd_run_rs_line_new_high_before_price", "stock_feature_daily", ["run_id", "rs_line_new_high_before_price"])
    op.create_index("ix_sfd_run_rs_line_blue_dot_recent", "stock_feature_daily", ["run_id", "rs_line_blue_dot_recent"])
    op.create_index("ix_sfd_run_rs_line_new_high_date", "stock_feature_daily", ["run_id", "rs_line_new_high_date"])


def downgrade() -> None:
    op.drop_index("ix_sfd_run_rs_line_new_high_date", table_name="stock_feature_daily")
    op.drop_index("ix_sfd_run_rs_line_blue_dot_recent", table_name="stock_feature_daily")
    op.drop_index("ix_sfd_run_rs_line_new_high_before_price", table_name="stock_feature_daily")
    op.drop_index("ix_sfd_run_rs_line_new_high", table_name="stock_feature_daily")
    op.drop_column("stock_feature_daily", "rs_line_new_high_date")
    op.drop_column("stock_feature_daily", "rs_line_blue_dot_recent")
    op.drop_column("stock_feature_daily", "rs_line_new_high_before_price")
    op.drop_column("stock_feature_daily", "rs_line_new_high")

    op.drop_index("idx_scan_rs_line_new_high_date", table_name="scan_results")
    op.drop_index("idx_scan_rs_line_blue_dot_recent", table_name="scan_results")
    op.drop_index("idx_scan_rs_line_new_high_before_price", table_name="scan_results")
    op.drop_index("idx_scan_rs_line_new_high", table_name="scan_results")
    op.drop_column("scan_results", "rs_line_new_high_date")
    op.drop_column("scan_results", "rs_line_blue_dot_recent")
    op.drop_column("scan_results", "rs_line_new_high_before_price")
    op.drop_column("scan_results", "rs_line_new_high")
```

- [ ] **Step 3: Persist legacy scan rows**

In `_map_orchestrator_result`, add:

```python
    r["rs_line_new_high"] = raw.get("rs_line_new_high")
    r["rs_line_new_high_before_price"] = raw.get("rs_line_new_high_before_price")
    r["rs_line_blue_dot_recent"] = raw.get("rs_line_blue_dot_recent")
    r["rs_line_new_high_date"] = raw.get("rs_line_new_high_date")
```

In `_to_scan_result_item_domain`, add these fields to `extended`.

- [ ] **Step 4: Persist feature-store rows**

In `upsert_snapshot_rows`, compute `details = convert_numpy_types(row.details) or {}` once per row, then set:

```python
                    "rs_line_new_high": details.get("rs_line_new_high"),
                    "rs_line_new_high_before_price": details.get("rs_line_new_high_before_price"),
                    "rs_line_blue_dot_recent": details.get("rs_line_blue_dot_recent"),
                    "rs_line_new_high_date": details.get("rs_line_new_high_date"),
```

In `_upsert_stmt`, add each field to `set_`.

In `_feature_row_to_scan_result_item`, add these fields to `extended` from the ORM columns first, falling back to `details_json`.

- [ ] **Step 5: Run persistence tests**

Run:

```bash
cd backend && source venv/bin/activate && pytest tests/unit/test_setup_engine_persistence.py tests/unit/repositories/test_feature_store_repo.py -q
```

Expected: pass.

- [ ] **Step 6: Commit persistence**

```bash
git add backend/app/models/scan_result.py backend/app/infra/db/models/feature_store.py backend/alembic/versions/20260627_0023_add_rs_line_leadership_fields.py backend/app/infra/db/repositories/scan_result_repo.py backend/app/infra/db/repositories/feature_store_repo.py backend/tests/unit/test_setup_engine_persistence.py backend/tests/unit/repositories/test_feature_store_repo.py
git commit -m "feat: persist rs line leadership fields"
```

---

### Task 5: API Filters, Query Maps, and Response Shape

**Files:**
- Modify: `backend/app/infra/query/scan_result_query.py`
- Modify: `backend/app/infra/query/feature_store_query.py`
- Modify: `backend/app/schemas/scanning.py`
- Modify: `backend/app/api/v1/scan_filter_params.py`
- Test: `backend/tests/unit/test_scan_result_query_builder.py`
- Test: `backend/tests/unit/test_feature_store_query_builder.py`
- Test: `backend/tests/unit/test_scan_filter_params.py`
- Test: `backend/tests/unit/test_scan_results_endpoints.py`

- [ ] **Step 1: Add query map coverage**

Add the four fields to both `_COLUMN_MAP` dictionaries:

```python
    "rs_line_new_high": ScanResult.rs_line_new_high,
    "rs_line_new_high_before_price": ScanResult.rs_line_new_high_before_price,
    "rs_line_blue_dot_recent": ScanResult.rs_line_blue_dot_recent,
    "rs_line_new_high_date": ScanResult.rs_line_new_high_date,
```

Use `StockFeatureDaily` instead of `ScanResult` in `feature_store_query.py`.

- [ ] **Step 2: Expose response fields**

In `ScanResultItem`, add:

```python
    rs_line_new_high: Optional[bool] = None
    rs_line_new_high_before_price: Optional[bool] = None
    rs_line_blue_dot_recent: Optional[bool] = None
    rs_line_new_high_date: Optional[str] = None
```

In `from_domain`, map:

```python
            rs_line_new_high=ef.get("rs_line_new_high"),
            rs_line_new_high_before_price=ef.get("rs_line_new_high_before_price"),
            rs_line_blue_dot_recent=ef.get("rs_line_blue_dot_recent"),
            rs_line_new_high_date=ef.get("rs_line_new_high_date"),
```

- [ ] **Step 3: Add FastAPI filter params**

In `parse_scan_filters`, add query params:

```python
    rs_line_new_high: Optional[bool] = Query(None, description="RS line at new trailing high"),
    rs_line_new_high_before_price: Optional[bool] = Query(None, description="RS line new high while price is not"),
    rs_line_blue_dot_recent: Optional[bool] = Query(None, description="RS blue dot in the last 5 trading days"),
```

Add boolean filters:

```python
    if rs_line_new_high is not None:
        f.add_boolean("rs_line_new_high", rs_line_new_high)
    if rs_line_new_high_before_price is not None:
        f.add_boolean("rs_line_new_high_before_price", rs_line_new_high_before_price)
    if rs_line_blue_dot_recent is not None:
        f.add_boolean("rs_line_blue_dot_recent", rs_line_blue_dot_recent)
```

- [ ] **Step 4: Run API/query tests**

Run:

```bash
cd backend && source venv/bin/activate && pytest tests/unit/test_scan_result_query_builder.py tests/unit/test_feature_store_query_builder.py tests/unit/test_scan_filter_params.py tests/unit/test_scan_results_endpoints.py -q
```

Expected: pass.

- [ ] **Step 5: Commit API surface**

```bash
git add backend/app/infra/query/scan_result_query.py backend/app/infra/query/feature_store_query.py backend/app/schemas/scanning.py backend/app/api/v1/scan_filter_params.py backend/tests/unit/test_scan_result_query_builder.py backend/tests/unit/test_feature_store_query_builder.py backend/tests/unit/test_scan_filter_params.py backend/tests/unit/test_scan_results_endpoints.py
git commit -m "feat: expose rs line leadership filters"
```

---

### Task 6: Frontend Filter, Table Column, Static Scan Parity, and Preset

**Files:**
- Modify: `frontend/src/features/scan/defaultFilters.js`
- Modify: `frontend/src/utils/filterUtils.js`
- Modify: `frontend/src/static/scanClient.js`
- Modify: `frontend/src/features/scan/components/filterPanel/constants.js`
- Modify: `frontend/src/features/scan/components/filterPanel/utils.js`
- Modify: `frontend/src/features/scan/components/filterPanel/TechnicalFiltersSection.jsx`
- Modify: `frontend/src/components/Scan/ResultsTable.jsx`
- Modify: `frontend/src/test/fixtures/setupEngineFixtures.js`
- Modify: `backend/app/services/preset_screens.py`
- Test: `frontend/src/components/Scan/FilterPanel.test.jsx`
- Test: `frontend/src/components/Scan/ResultsTable.test.jsx`
- Test: `frontend/src/static/scanClient.test.js`

- [ ] **Step 1: Add frontend state and param mapping**

Add `rsLineBlueDotRecent: null` to `buildDefaultScanFilters`.

In `buildFilterParams`, add:

```javascript
  if (filters.rsLineBlueDotRecent != null) {
    params.rs_line_blue_dot_recent = filters.rsLineBlueDotRecent;
  }
```

In static scan mappings, add:

```javascript
  rsLineBlueDotRecent: 'rs_line_blue_dot_recent',
```

- [ ] **Step 2: Add one visible filter**

Add `rsLineBlueDotRecent` to `TECHNICAL_KEYS` and `BOOLEAN_RESET_KEYS`.

In `buildActiveFilters`, add:

```javascript
  if (filters.rsLineBlueDotRecent != null) {
    active.push({
      key: 'rsLineBlueDotRecent',
      label: `Blue Dot <=5d: ${filters.rsLineBlueDotRecent ? 'Yes' : 'No'}`,
    });
  }
```

In `TechnicalFiltersSection.jsx`, place this checkbox near RS Rating:

```jsx
        <Grid item xs={6} sm={3} md={1}>
          <CompactCheckbox
            label="Blue Dot"
            value={filters.rsLineBlueDotRecent}
            onChange={(value) => updateFilter('rsLineBlueDotRecent', value)}
          />
        </Grid>
```

- [ ] **Step 3: Add one visible table column**

In `ResultsTable.jsx`, add a column near RS fields:

```javascript
  { id: 'rs_line_blue_dot_recent', label: 'BD5', sortable: true, width: 40 },
```

Render it as the existing blue circle when true:

```jsx
      <TableCell align="center" sx={{ width: 40, minWidth: 40 }}>
        {row.rs_line_blue_dot_recent ? (
          <Box
            sx={{ width: 10, height: 10, borderRadius: '50%', bgcolor: '#2196f3', display: 'inline-block' }}
            title={row.rs_line_new_high_date ? `RS line new high: ${row.rs_line_new_high_date}` : 'RS line blue dot within 5 trading days'}
          />
        ) : (
          <Box component="span" sx={{ color: 'text.disabled' }}>-</Box>
        )}
      </TableCell>
```

- [ ] **Step 4: Update Blue Dot Leaders preset**

In `backend/app/services/preset_screens.py`, update `blue_dot_leaders` to:

```python
        "filters": {
            "stage": 2,
            "rsLineBlueDotRecent": True,
            "rsRating": {"min": 80, "max": None},
        },
```

Keep `seRsLineBlueDot` in boolean mappings so old saved presets still work.

- [ ] **Step 5: Run frontend/static tests**

Run:

```bash
export NVM_DIR="$HOME/.nvm" && [ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"
cd frontend && npm run test:run -- src/components/Scan/FilterPanel.test.jsx src/components/Scan/ResultsTable.test.jsx src/static/scanClient.test.js
```

Expected: pass.

- [ ] **Step 6: Commit frontend surface**

```bash
git add frontend/src/features/scan/defaultFilters.js frontend/src/utils/filterUtils.js frontend/src/static/scanClient.js frontend/src/features/scan/components/filterPanel/constants.js frontend/src/features/scan/components/filterPanel/utils.js frontend/src/features/scan/components/filterPanel/TechnicalFiltersSection.jsx frontend/src/components/Scan/ResultsTable.jsx frontend/src/test/fixtures/setupEngineFixtures.js frontend/src/components/Scan/FilterPanel.test.jsx frontend/src/components/Scan/ResultsTable.test.jsx frontend/src/static/scanClient.test.js backend/app/services/preset_screens.py
git commit -m "feat: surface rs blue dot scan filter"
```

---

### Task 7: Chart Annotation Contract

**Files:**
- Modify: `backend/app/api/v1/stocks_rs_line.py`
- Modify: `frontend/src/components/Charts/CandlestickChart.jsx`
- Test: `backend/tests/unit/test_rs_line.py`
- Test: `frontend/src/components/Charts/rsMarkers.test.js` if `rsMarkers.js` is created

- [ ] **Step 1: Keep chart data source unchanged**

Confirm `/api/v1/stocks/{symbol}/rs-line` still computes `blue_dots` over the full cached window and trims markers to the displayed period. No new database dependency is needed for chart markers.

- [ ] **Step 2: Add a pure marker-builder test if chart marker behavior changes**

If the inline marker mapping in `CandlestickChart.jsx` is edited, extract it to `frontend/src/components/Charts/rsMarkers.js`:

```javascript
export const buildRsBlueDotMarkers = (rsLinePoints = [], blueDots = []) => {
  const timesInSeries = new Set(rsLinePoints.map((point) => point.time));
  return blueDots
    .filter((time) => timesInSeries.has(time))
    .map((time) => ({
      time,
      position: 'inBar',
      color: '#2196f3',
      shape: 'circle',
    }));
};
```

Then replace the inline marker map with:

```javascript
    const markerList = buildRsBlueDotMarkers(points, rsData.blue_dots || []);
```

- [ ] **Step 3: Run chart-adjacent tests**

Run:

```bash
cd backend && source venv/bin/activate && pytest tests/unit/test_rs_line.py -q
export NVM_DIR="$HOME/.nvm" && [ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"
cd frontend && npm run test:run -- src/components/Charts/rsMarkers.test.js
```

Expected: backend passes; frontend marker test passes when the helper exists.

- [ ] **Step 4: Commit chart contract**

```bash
git add backend/app/api/v1/stocks_rs_line.py frontend/src/components/Charts/CandlestickChart.jsx frontend/src/components/Charts/rsMarkers.js frontend/src/components/Charts/rsMarkers.test.js backend/tests/unit/test_rs_line.py
git commit -m "test: cover rs blue dot chart markers"
```

---

### Task 8: End-to-End Verification and Session Close

**Files:**
- No planned source edits after this task.

- [ ] **Step 1: Run focused backend tests**

```bash
cd backend && source venv/bin/activate && pytest \
  tests/unit/test_rs_line.py \
  tests/unit/test_setup_engine_readiness.py \
  tests/unit/test_scan_orchestrator.py \
  tests/unit/test_precomputed_scan_context.py \
  tests/unit/test_scan_result_query_builder.py \
  tests/unit/test_feature_store_query_builder.py \
  tests/unit/test_scan_filter_params.py \
  tests/unit/test_scan_results_endpoints.py \
  tests/unit/repositories/test_feature_store_repo.py \
  -q
```

Expected: pass.

- [ ] **Step 2: Run focused frontend tests**

```bash
export NVM_DIR="$HOME/.nvm" && [ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"
cd frontend && npm run test:run -- \
  src/components/Scan/FilterPanel.test.jsx \
  src/components/Scan/ResultsTable.test.jsx \
  src/static/scanClient.test.js
```

Expected: pass.

- [ ] **Step 3: Run broader quality gates**

```bash
cd backend && source venv/bin/activate && pytest
export NVM_DIR="$HOME/.nvm" && [ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"
cd frontend && npm run test:run && npm run lint
```

Expected: pass.

- [ ] **Step 4: Sync Beads and push**

```bash
git status
bd sync
git pull --rebase
bd sync
git push
git status
```

Expected: `git status` reports the branch is up to date with origin.

## Self-Review

- Spec coverage: The plan covers derived RS flags, current-vs-price distinction, recent blue-dot behavior, persistence on both scan tables, backend filters, frontend table/filter, preset update, and existing chart markers.
- Placeholder scan: The plan intentionally uses concrete field names, files, commands, and expected outcomes.
- Type consistency: Field names are consistent across persistence, API, filter params, static scan filtering, frontend state, and visible UI.
