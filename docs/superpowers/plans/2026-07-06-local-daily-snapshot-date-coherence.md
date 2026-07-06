# Local Daily Snapshot Date Coherence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the local/live Daily Snapshot use one explicit market-session date across scan, breadth, exposure, groups, RRG-capable group data, and key-market cards.

**Architecture:** The latest completed scan remains the anchor because candidates and leaders are already scan-derived. The service extracts a `snapshot_as_of_date` from that scan, pins every reader that can leak newer data to that date, exposes the market timezone and section dates in the freshness contract, and changes the daily pipeline so producers receive the same date instead of recomputing "today" inside later tasks.

**Tech Stack:** Python, FastAPI, SQLAlchemy, Celery, Pydantic, pytest, React/Vitest.

---

## File Structure

- Modify: `backend/app/services/daily_snapshot_service.py`
  - Owns the local/live Daily Snapshot assembly.
  - Add the anchor-date helper and exact-date section readers.
  - Increment the snapshot cache schema version.

- Modify: `backend/app/services/key_market_history.py`
  - Add an optional `as_of_date` cap so key-market cards cannot show prices after the snapshot date.

- Modify: `backend/app/schemas/market_scan.py`
  - Extend `DailySnapshotFreshness` with `snapshot_as_of_date`, `market_timezone`, `exposure_latest_date`, `key_markets_latest_date`, and `date_coherence_status`.

- Modify: `backend/app/tasks/breadth_tasks.py`
  - Add `calculation_date` to `calculate_daily_breadth_with_gapfill` and pass it to the inner daily breadth calculation.

- Modify: `backend/app/tasks/group_rank_tasks.py`
  - Add `calculation_date` to `calculate_daily_group_rankings_with_gapfill` and pass it to the inner daily group ranking calculation.

- Modify: `backend/app/tasks/daily_market_pipeline_tasks.py`
  - Pass the pipeline `trading_date` into breadth and group gapfill tasks.

- Modify: `backend/app/api/v1/groups.py`
  - Add optional `as_of_date` support to live group ranking and RRG readers so local RRG can be pinned when the frontend asks for a snapshot date.

- Modify: `frontend/src/components/MarketScan/DailyMarketSnapshotTab.jsx`
  - Display the single snapshot date and market timezone, then show section dates only when they differ or are missing.

- Tests:
  - Modify: `backend/tests/unit/test_daily_snapshot_service.py`
  - Modify: `backend/tests/unit/test_daily_market_pipeline_tasks.py`
  - Modify: `backend/tests/unit/test_breadth_tasks.py`
  - Modify: `backend/tests/unit/test_group_rank_tasks.py`
  - Modify: `backend/tests/unit/test_rrg_service.py`
  - Modify: `frontend/src/components/MarketScan/DailyMarketSnapshotTab.test.jsx`

## Date Contract

- `snapshot_as_of_date` is a date-only market session date.
- `market_timezone` comes from `MarketCatalogEntry.display_timezone`.
- Scan rows, breadth, exposure, and top groups must be exact-date matches for `snapshot_as_of_date`.
- Key-market cards must use latest rows on or before `snapshot_as_of_date`; some instruments may lag due to separate calendars, but none may show future rows relative to the snapshot.
- When an exact-date section is unavailable, the section returns empty or `None`; it must not fall forward to a newer date.
- `generated_at` remains a UTC timestamp.

## Task 1: Pin Key-Market Cards To An Anchor Date

**Files:**
- Modify: `backend/app/services/key_market_history.py`
- Test: `backend/tests/unit/test_daily_snapshot_service.py`

- [ ] **Step 1: Write the failing key-market cap test**

Add this test under `class TestKeyMarketEntries` in `backend/tests/unit/test_daily_snapshot_service.py`:

```python
    def test_key_market_entries_do_not_include_rows_after_as_of_date(self, monkeypatch):
        captured_filters = []

        class FakeQuery:
            def filter(self, *args):
                captured_filters.extend(args)
                return self

            def order_by(self, *_args):
                return self

            def all(self):
                return [
                    ("SPY", date(2026, 6, 10), 500.0),
                    ("SPY", date(2026, 6, 11), 505.0),
                ]

        class FakeDb:
            def query(self, *_args):
                return FakeQuery()

        monkeypatch.setattr(
            key_market_history,
            "key_market_instruments",
            lambda _market: [
                SimpleNamespace(
                    data_symbol="SPY",
                    display_symbol="SPY",
                    display_name="S&P 500 ETF",
                    currency="USD",
                )
            ],
        )

        entries = key_market_history.build_key_market_entries(
            FakeDb(),
            "US",
            as_of_date=date(2026, 6, 11),
        )

        assert entries[0]["latest_date"] == "2026-06-11"
        assert any(getattr(expr.right, "value", None) == date(2026, 6, 11) for expr in captured_filters)
```

- [ ] **Step 2: Run the focused failing test**

Run:

```bash
cd backend && pytest tests/unit/test_daily_snapshot_service.py::TestKeyMarketEntries::test_key_market_entries_do_not_include_rows_after_as_of_date -q
```

Expected: FAIL with `TypeError: build_key_market_entries() got an unexpected keyword argument 'as_of_date'`.

- [ ] **Step 3: Add the `as_of_date` cap**

In `backend/app/services/key_market_history.py`, replace the function signature and cutoff/filter block with this code:

```python
def build_key_market_entries(
    db: Session,
    market: str,
    *,
    points: int = KEY_MARKET_HISTORY_POINTS,
    as_of_date: date | None = None,
) -> list[dict[str, Any]]:
    instruments = key_market_instruments(market)
    if not instruments:
        return []
    data_symbols = [instrument.data_symbol for instrument in instruments]
    end_date = as_of_date or date.today()
    cutoff = end_date - timedelta(days=KEY_MARKET_HISTORY_CALENDAR_DAYS)
    rows = (
        db.query(StockPrice.symbol, StockPrice.date, StockPrice.close)
        .filter(
            StockPrice.symbol.in_(data_symbols),
            StockPrice.date >= cutoff,
            StockPrice.date <= end_date,
        )
        .order_by(StockPrice.symbol.asc(), StockPrice.date.asc())
        .all()
    )
```

Leave the remainder of the function unchanged.

- [ ] **Step 4: Run the focused key-market test**

Run:

```bash
cd backend && pytest tests/unit/test_daily_snapshot_service.py::TestKeyMarketEntries::test_key_market_entries_do_not_include_rows_after_as_of_date -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/key_market_history.py backend/tests/unit/test_daily_snapshot_service.py
git commit -m "fix: cap key market cards at snapshot date"
```

## Task 2: Anchor Daily Snapshot Readers To The Scan Date

**Files:**
- Modify: `backend/app/services/daily_snapshot_service.py`
- Modify: `backend/app/schemas/market_scan.py`
- Test: `backend/tests/unit/test_daily_snapshot_service.py`

- [ ] **Step 1: Write the failing Daily Snapshot service test**

Add this test class to `backend/tests/unit/test_daily_snapshot_service.py` after `TestScanFreshness`:

```python
class TestDailySnapshotDateCoherence:
    def test_payload_pins_latest_sections_to_scan_as_of_date(self, monkeypatch):
        class FakeBreadthQuery:
            def __init__(self):
                self.filters = []

            def filter(self, *args):
                self.filters.extend(args)
                return self

            def order_by(self, *_args):
                return self

            def first(self):
                return (date(2026, 6, 11),)

        breadth_query = FakeBreadthQuery()

        class FakeDb:
            def query(self, *_args):
                return breadth_query

        class FakeGroupService:
            def get_current_rankings(self, db, limit=10, calculation_date=None, market="US"):
                assert calculation_date == date(2026, 6, 11)
                assert market == "US"
                return [
                    {
                        "industry_group": "Software",
                        "rank": 1,
                        "rank_change_1w": 2,
                        "rank_change_1m": 3,
                        "top_symbol": "APP",
                        "top_symbol_name": "AppLovin",
                        "top_rs_rating": 98,
                        "date": "2026-06-11",
                    }
                ]

        scan = SimpleNamespace(
            scan_id="scan-abc",
            feature_run=SimpleNamespace(
                as_of_date=date(2026, 6, 11),
                published_at=datetime(2026, 6, 11, 23, 0, tzinfo=timezone.utc),
            ),
            completed_at=datetime(2026, 6, 12, 1, 0, tzinfo=timezone.utc),
        )

        exposure_calls = []
        monkeypatch.setattr(
            daily_snapshot_service,
            "build_exposure_payload",
            lambda db, market, as_of_date=None: exposure_calls.append(as_of_date)
            or {"date": "2026-06-11", "history": []},
        )
        monkeypatch.setattr(
            daily_snapshot_service,
            "build_key_market_entries",
            lambda db, market, as_of_date=None: [
                {"symbol": "SPY", "latest_date": as_of_date.isoformat(), "history": []}
            ],
        )
        monkeypatch.setattr(
            "app.wiring.bootstrap.get_group_rank_service",
            lambda: FakeGroupService(),
        )

        class FakeUseCase:
            def execute(self, *_args, **_kwargs):
                return SimpleNamespace(page=SimpleNamespace(items=[]))

        payload = daily_snapshot_service.build_daily_snapshot_payload(
            FakeDb(),
            market="US",
            market_display_name="United States",
            scan=scan,
            uow=object(),
            scan_results_use_case=FakeUseCase(),
        )

        assert payload["freshness"]["snapshot_as_of_date"] == "2026-06-11"
        assert payload["freshness"]["market_timezone"] == "America/New_York"
        assert payload["freshness"]["breadth_latest_date"] == "2026-06-11"
        assert payload["freshness"]["groups_latest_date"] == "2026-06-11"
        assert payload["freshness"]["exposure_latest_date"] == "2026-06-11"
        assert payload["freshness"]["key_markets_latest_date"] == "2026-06-11"
        assert payload["freshness"]["date_coherence_status"] == "coherent"
        assert exposure_calls == [date(2026, 6, 11)]
```

- [ ] **Step 2: Run the focused failing test**

Run:

```bash
cd backend && pytest tests/unit/test_daily_snapshot_service.py::TestDailySnapshotDateCoherence::test_payload_pins_latest_sections_to_scan_as_of_date -q
```

Expected: FAIL because the new freshness fields do not exist and `build_exposure_payload` is called without `as_of_date`.

- [ ] **Step 3: Extend the schema**

In `backend/app/schemas/market_scan.py`, replace `DailySnapshotFreshness` with:

```python
class DailySnapshotFreshness(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scan_id: Optional[str] = None
    scan_as_of_date: Optional[str] = None
    scan_published_at: Optional[str] = None
    snapshot_as_of_date: Optional[str] = None
    market_timezone: Optional[str] = None
    breadth_latest_date: Optional[str] = None
    groups_latest_date: Optional[str] = None
    exposure_latest_date: Optional[str] = None
    key_markets_latest_date: Optional[str] = None
    date_coherence_status: str = "unknown"
```

- [ ] **Step 4: Add anchor and coherence helpers**

In `backend/app/services/daily_snapshot_service.py`, change the import:

```python
from datetime import date, datetime, timezone
```

Then add these helpers below `_scan_freshness`:

```python
def _snapshot_anchor_date(scan: Scan | None) -> date | None:
    value = _scan_freshness(scan).get("scan_as_of_date")
    if not value:
        return None
    return date.fromisoformat(value)


def _iso_or_none(value: date | None) -> str | None:
    return value.isoformat() if value is not None else None


def _latest_key_market_date(entries: list[dict[str, Any]]) -> str | None:
    dates = [entry.get("latest_date") for entry in entries if entry.get("latest_date")]
    return max(dates) if dates else None


def _coherence_status(
    *,
    anchor: date | None,
    breadth_date: str | None,
    groups_date: str | None,
    exposure_date: str | None,
) -> str:
    if anchor is None:
        return "unanchored"
    expected = anchor.isoformat()
    section_dates = {
        "breadth": breadth_date,
        "groups": groups_date,
        "exposure": exposure_date,
    }
    if all(value == expected for value in section_dates.values()):
        return "coherent"
    if any(value and value > expected for value in section_dates.values()):
        return "future_section_data"
    return "partial"
```

- [ ] **Step 5: Pin breadth and groups**

Replace `_build_top_groups` and `_latest_breadth_date` in `backend/app/services/daily_snapshot_service.py` with:

```python
def _build_top_groups(
    db: Session,
    market: str,
    *,
    as_of_date: date | None,
) -> tuple[list[dict[str, Any]], str | None]:
    from app.wiring.bootstrap import get_group_rank_service

    if not get_market_catalog().get(market).capabilities.group_rankings:
        return [], None
    rankings = get_group_rank_service().get_current_rankings(
        db,
        limit=TOP_GROUPS_LIMIT,
        market=market,
        calculation_date=as_of_date,
    )
    if not rankings:
        return [], None
    groups_date = rankings[0].get("date")
    keep = (
        "industry_group",
        "rank",
        "rank_change_1w",
        "rank_change_1m",
        "top_symbol",
        "top_symbol_name",
        "top_rs_rating",
    )
    return [{key: row.get(key) for key in keep} for row in rankings], groups_date


def _latest_breadth_date(
    db: Session,
    market: str,
    *,
    as_of_date: date | None,
) -> str | None:
    query = db.query(MarketBreadth.date).filter(MarketBreadth.market == market)
    if as_of_date is not None:
        query = query.filter(MarketBreadth.date == as_of_date)
    latest = query.order_by(MarketBreadth.date.desc()).first()
    if latest is None or latest[0] is None:
        return None
    value = latest[0]
    return value.isoformat() if hasattr(value, "isoformat") else str(value)
```

- [ ] **Step 6: Pin exposure and key markets in the payload**

In `build_daily_snapshot_payload`, replace the `top_groups`, `freshness`, and return preparation block with:

```python
    anchor_date = _snapshot_anchor_date(scan)
    anchor_iso = _iso_or_none(anchor_date)
    top_groups, groups_date = _build_top_groups(db, normalized, as_of_date=anchor_date)
    breadth_date = _latest_breadth_date(db, normalized, as_of_date=anchor_date)
    key_markets = build_key_market_entries(db, normalized, as_of_date=anchor_date)
    market_health_exposure = build_exposure_payload(db, normalized, as_of_date=anchor_date)
    exposure_date = (
        market_health_exposure.get("date")
        if isinstance(market_health_exposure, dict)
        else None
    )
    freshness = _scan_freshness(scan)
    freshness["snapshot_as_of_date"] = anchor_iso
    freshness["market_timezone"] = get_market_catalog().get(normalized).display_timezone
    freshness["breadth_latest_date"] = breadth_date
    freshness["groups_latest_date"] = groups_date
    freshness["exposure_latest_date"] = exposure_date
    freshness["key_markets_latest_date"] = _latest_key_market_date(key_markets)
    freshness["date_coherence_status"] = _coherence_status(
        anchor=anchor_date,
        breadth_date=breadth_date,
        groups_date=groups_date,
        exposure_date=exposure_date,
    )
```

Then update the return body to use the local variables:

```python
        "key_markets": key_markets,
        "market_health_exposure": market_health_exposure,
```

- [ ] **Step 7: Bump the cache schema version**

In `backend/app/services/daily_snapshot_service.py`, change:

```python
DAILY_SNAPSHOT_SCHEMA_VERSION = 3
```

- [ ] **Step 8: Update schema payload tests**

In `TestDailySnapshotResponseSchema._payload`, add these fields inside `freshness`:

```python
                "snapshot_as_of_date": "2026-06-11",
                "market_timezone": "America/New_York",
                "exposure_latest_date": "2026-06-11",
                "key_markets_latest_date": "2026-06-11",
                "date_coherence_status": "coherent",
```

- [ ] **Step 9: Run the Daily Snapshot tests**

Run:

```bash
cd backend && pytest tests/unit/test_daily_snapshot_service.py -q
```

Expected: PASS.

- [ ] **Step 10: Commit**

```bash
git add backend/app/services/daily_snapshot_service.py backend/app/schemas/market_scan.py backend/tests/unit/test_daily_snapshot_service.py
git commit -m "fix: anchor live daily snapshot dates"
```

## Task 3: Pass One Trading Date Through The Daily Pipeline Producers

**Files:**
- Modify: `backend/app/tasks/breadth_tasks.py`
- Modify: `backend/app/tasks/group_rank_tasks.py`
- Modify: `backend/app/tasks/daily_market_pipeline_tasks.py`
- Test: `backend/tests/unit/test_daily_market_pipeline_tasks.py`
- Test: `backend/tests/unit/test_breadth_tasks.py`
- Test: `backend/tests/unit/test_group_rank_tasks.py`

- [ ] **Step 1: Update the pipeline signature test first**

In `backend/tests/unit/test_daily_market_pipeline_tasks.py`, add these assertions at the end of `test_daily_market_pipeline_orders_refresh_compute_and_scan`:

```python
    assert signatures[2].kwargs == {
        "market": "HK",
        "calculation_date": "2026-03-16",
    }
    assert signatures[6].kwargs == {
        "market": "HK",
        "calculation_date": "2026-03-16",
    }
```

- [ ] **Step 2: Run the failing pipeline test**

Run:

```bash
cd backend && pytest tests/unit/test_daily_market_pipeline_tasks.py::test_daily_market_pipeline_orders_refresh_compute_and_scan -q
```

Expected: FAIL because breadth and group gapfill signatures do not include `calculation_date`.

- [ ] **Step 3: Pass the date from the pipeline**

In `backend/app/tasks/daily_market_pipeline_tasks.py`, replace the breadth and group signatures with:

```python
        calculate_daily_breadth_with_gapfill.si(
            market=market_code,
            calculation_date=as_of_date,
        ).set(
            queue=market_jobs_queue_for_market(market_code)
        ),
```

and:

```python
        calculate_daily_group_rankings_with_gapfill.si(
            market=market_code,
            calculation_date=as_of_date,
        ).set(
            queue=market_jobs_queue_for_market(market_code)
        ),
```

- [ ] **Step 4: Add breadth gapfill date test**

Add this test to `backend/tests/unit/test_breadth_tasks.py`:

```python
def test_breadth_gapfill_uses_requested_calculation_date_for_daily_calc(monkeypatch):
    import app.tasks.breadth_tasks as module

    fake_db = MagicMock()
    fake_calculator = MagicMock()
    fake_calculator.find_missing_dates.return_value = []

    monkeypatch.setattr(module, "SessionLocal", lambda: fake_db)
    _patch_serialized_lock(monkeypatch)
    monkeypatch.setattr(module.settings, "breadth_gapfill_enabled", False)
    monkeypatch.setattr(module, "BreadthCalculatorService", lambda *a, **kw: fake_calculator)
    _patch_calendar_service(monkeypatch, datetime(2026, 3, 17, 12, 0, 0))

    captured = []

    def fake_inner(calculation_date=None, market=None):
        captured.append((calculation_date, market))
        return {"date": calculation_date, "market": market}

    monkeypatch.setattr(module, "_calculate_daily_breadth_in_process", fake_inner)

    result = module.calculate_daily_breadth_with_gapfill.run(
        market="HK",
        calculation_date="2026-03-16",
    )

    assert result["today"]["date"] == "2026-03-16"
    assert captured == [("2026-03-16", "HK")]
```

- [ ] **Step 5: Add the breadth `calculation_date` parameter**

In `backend/app/tasks/breadth_tasks.py`, change the wrapper signature to:

```python
def calculate_daily_breadth_with_gapfill(
    self,
    max_gap_days: int | None = None,
    market: str | None = None,
    activity_lifecycle: str | None = None,
    calculation_date: str | None = None,
):
```

Then replace the same-day calculation block with:

```python
        target_date = None
        if calculation_date:
            target_date = datetime.strptime(calculation_date, "%Y-%m-%d").date()
        else:
            target_date = calendar_service.market_now(effective_market).date()

        if calendar_service.is_trading_day(effective_market, target_date):
            logger.info(
                "Calculating breadth for %s (%s)...",
                effective_market,
                target_date,
            )
            inner_kwargs = {"market": market}
            if calculation_date:
                inner_kwargs["calculation_date"] = target_date.isoformat()
            today_result = _calculate_daily_breadth_in_process(**inner_kwargs)
            result['today'] = today_result
```

Update `_calculate_daily_breadth_in_process` in the same file to accept and forward the date:

```python
def _calculate_daily_breadth_in_process(
    *,
    calculation_date: str | None = None,
    market: str | None = None,
) -> dict:
    """Run breadth logic without reacquiring the market workload lease."""
    from .workload_coordination import disable_serialized_market_workload

    task = calculate_daily_breadth
    kwargs = {"market": market}
    if calculation_date is not None:
        kwargs["calculation_date"] = calculation_date
    if str(getattr(task, "__module__", "")).startswith("unittest.mock"):
        return task(**kwargs)
    with disable_serialized_market_workload():
        if hasattr(task, "request") and callable(getattr(task, "run", None)):
            return task.run(**kwargs)
        return task(**kwargs)
```

- [ ] **Step 6: Run the breadth focused test**

Run:

```bash
cd backend && pytest tests/unit/test_breadth_tasks.py::test_breadth_gapfill_uses_requested_calculation_date_for_daily_calc -q
```

Expected: PASS.

- [ ] **Step 7: Add group gapfill date test**

Add this test to `backend/tests/unit/test_group_rank_tasks.py`:

```python
def test_group_gapfill_uses_requested_calculation_date_for_daily_calc(monkeypatch):
    import app.tasks.group_rank_tasks as module

    fake_db = MagicMock()
    fake_service = MagicMock()
    fake_service.find_missing_dates.return_value = []

    monkeypatch.setattr(module, "SessionLocal", lambda: fake_db)
    _patch_serialized_lock(monkeypatch)
    monkeypatch.setattr(module.settings, "group_rank_gapfill_enabled", False)
    monkeypatch.setattr(module, "get_group_rank_service", lambda: fake_service)
    _patch_calendar_service(monkeypatch, datetime(2026, 3, 17, 12, 0, 0))

    captured = []

    def fake_inner(calculation_date=None, market=None, activity_lifecycle=None):
        captured.append((calculation_date, market, activity_lifecycle))
        return {"date": calculation_date, "market": market}

    monkeypatch.setattr(module, "_calculate_daily_group_rankings_in_process", fake_inner)

    result = module.calculate_daily_group_rankings_with_gapfill.run(
        market="HK",
        calculation_date="2026-03-16",
        activity_lifecycle="daily_refresh",
    )

    assert result["today"]["date"] == "2026-03-16"
    assert captured == [("2026-03-16", "HK", "daily_refresh")]
```

- [ ] **Step 8: Add the group `calculation_date` parameter**

In `backend/app/tasks/group_rank_tasks.py`, change the wrapper signature to:

```python
def calculate_daily_group_rankings_with_gapfill(
    self,
    max_gap_days: int | None = None,
    market: str | None = None,
    activity_lifecycle: str | None = None,
    calculation_date: str | None = None,
):
```

Then replace the same-day calculation block with:

```python
        target_date = None
        if calculation_date:
            target_date = datetime.strptime(calculation_date, "%Y-%m-%d").date()
        else:
            target_date = calendar_service.market_now(effective_market).date()

        if calendar_service.is_trading_day(effective_market, target_date):
            logger.info(
                "Calculating group rankings for %s (%s)...",
                effective_market,
                target_date,
            )
            inner_kwargs = {
                "market": market,
                "activity_lifecycle": activity_lifecycle,
            }
            if calculation_date:
                inner_kwargs["calculation_date"] = target_date.isoformat()
            today_result = _calculate_daily_group_rankings_in_process(**inner_kwargs)
            result['today'] = today_result
```

Update `_calculate_daily_group_rankings_in_process` to accept and forward the date:

```python
def _calculate_daily_group_rankings_in_process(
    *,
    calculation_date: str | None = None,
    market: str | None = None,
    activity_lifecycle: str | None = None,
) -> dict:
    """Run the daily ranking task body without re-acquiring the market workload lease."""
    from .workload_coordination import disable_serialized_market_workload

    task = calculate_daily_group_rankings
    kwargs = {
        "market": market,
        "activity_lifecycle": activity_lifecycle,
    }
    if calculation_date is not None:
        kwargs["calculation_date"] = calculation_date
    transient_token = _PROPAGATE_IN_PROCESS_TRANSIENT_ERRORS.set(True)
    try:
        if str(getattr(task, "__module__", "")).startswith("unittest.mock"):
            return task(**kwargs)
        with disable_serialized_market_workload():
            if hasattr(task, "request") and callable(getattr(task, "run", None)):
                return task.run(**kwargs)
            return task(**kwargs)
    finally:
        _PROPAGATE_IN_PROCESS_TRANSIENT_ERRORS.reset(transient_token)
```

- [ ] **Step 9: Run pipeline, breadth, and group task tests**

Run:

```bash
cd backend && pytest \
  tests/unit/test_daily_market_pipeline_tasks.py \
  tests/unit/test_breadth_tasks.py::test_breadth_gapfill_uses_requested_calculation_date_for_daily_calc \
  tests/unit/test_group_rank_tasks.py::test_group_gapfill_uses_requested_calculation_date_for_daily_calc \
  -q
```

Expected: PASS.

- [ ] **Step 10: Commit**

```bash
git add backend/app/tasks/daily_market_pipeline_tasks.py backend/app/tasks/breadth_tasks.py backend/app/tasks/group_rank_tasks.py backend/tests/unit/test_daily_market_pipeline_tasks.py backend/tests/unit/test_breadth_tasks.py backend/tests/unit/test_group_rank_tasks.py
git commit -m "fix: propagate daily pipeline trading date"
```

## Task 4: Add Live Group And RRG Date Pinning

**Files:**
- Modify: `backend/app/api/v1/groups.py`
- Test: `backend/tests/unit/test_rrg_service.py`

- [ ] **Step 1: Add an RRG service regression for pinned input**

Add this test to `backend/tests/unit/test_rrg_service.py` after `test_get_rrg_groups_scope_returns_quadrants_and_tails`:

```python
def test_get_rrg_passes_as_of_date_to_history_provider():
    session = _session()
    expected_date = date(2026, 6, 11)

    class _FakeHistoryProvider:
        def get_all_groups_history(self, db, *, market, days, as_of_date=None):
            assert market == "US"
            assert as_of_date == expected_date
            return (
                expected_date.isoformat(),
                {
                    "Software": {
                        "industry_group": "Software",
                        "date": expected_date.isoformat(),
                        "rank": 1,
                        "num_stocks": 10,
                        "avg_rs_rating": 80.0,
                    }
                },
                {"Software": [(expected_date - timedelta(weeks=i), 80.0 - i, 10) for i in range(40)]},
            )

    payload = RRGService(history_provider=_FakeHistoryProvider()).get_rrg(
        session,
        market="US",
        as_of_date=expected_date,
    )

    assert payload["date"] == "2026-06-11"
```

- [ ] **Step 2: Run the RRG service regression**

Run:

```bash
cd backend && pytest tests/unit/test_rrg_service.py::test_get_rrg_passes_as_of_date_to_history_provider -q
```

Expected: PASS, confirming the lower service already supports the contract.

- [ ] **Step 3: Thread `as_of_date` through group API caches**

In `backend/app/api/v1/groups.py`, change the import:

```python
from datetime import date, datetime, timedelta
```

Change `_fetch_rankings_cached` to:

```python
def _fetch_rankings_cached(
    db: Session,
    *,
    market: str,
    limit: int,
    as_of_date: date | None = None,
) -> list:
    date_param = as_of_date.isoformat() if as_of_date else "latest"
    return cached_group_payload(
        market=market,
        name="rankings",
        params=f"limit={limit}:as_of={date_param}",
        compute=lambda: _get_group_rank_service().get_current_rankings(
            db,
            limit=limit,
            market=market,
            calculation_date=as_of_date,
        ),
    )
```

Change `_fetch_rrg_scopes_cached` to:

```python
def _fetch_rrg_scopes_cached(
    db: Session,
    *,
    market: str,
    scopes: list,
    tail_weeks: int,
    as_of_date: date | None = None,
) -> dict:
    date_param = as_of_date.isoformat() if as_of_date else "latest"
    return cached_group_payload(
        market=market,
        name="rrg_scopes",
        params=f"tail={tail_weeks}:scopes={','.join(scopes)}:as_of={date_param}",
        compute=lambda: _get_rrg_service().get_rrg_scopes(
            db,
            market=market,
            scopes=scopes,
            tail_weeks=tail_weeks,
            as_of_date=as_of_date,
        ),
        should_cache=lambda v: any(_rrg_scope_has_data(v.get(s)) for s in scopes),
    )
```

- [ ] **Step 4: Add `as_of_date` query parameters**

In `get_current_rankings`, add:

```python
    as_of_date: date | None = Query(None, description="Optional YYYY-MM-DD snapshot date"),
```

and pass it:

```python
    rankings = _fetch_rankings_cached(
        db,
        market=normalized_market,
        limit=limit,
        as_of_date=as_of_date,
    )
```

In `get_rrg`, add:

```python
    as_of_date: date | None = Query(None, description="Optional YYYY-MM-DD snapshot date"),
```

and pass it:

```python
    payload = service.get_rrg(
        db,
        market=normalized_market,
        scope=scope,
        tail_weeks=tail_weeks,
        as_of_date=as_of_date,
    )
```

In `get_rrg_scopes`, add:

```python
    as_of_date: date | None = Query(None, description="Optional YYYY-MM-DD snapshot date"),
```

and pass it:

```python
    scopes = _fetch_rrg_scopes_cached(
        db,
        market=normalized_market,
        scopes=requested_scopes,
        tail_weeks=tail_weeks,
        as_of_date=as_of_date,
    )
```

- [ ] **Step 5: Run group and RRG tests**

Run:

```bash
cd backend && pytest tests/unit/test_rrg_service.py tests/unit/test_group_ranking_payloads.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add backend/app/api/v1/groups.py backend/tests/unit/test_rrg_service.py
git commit -m "feat: allow live groups rrg date pinning"
```

## Task 5: Make The Daily Snapshot Header Show The Coherent Contract

**Files:**
- Modify: `frontend/src/components/MarketScan/DailyMarketSnapshotTab.jsx`
- Test: `frontend/src/components/MarketScan/DailyMarketSnapshotTab.test.jsx`

- [ ] **Step 1: Add a frontend assertion for the unified date label**

In `frontend/src/components/MarketScan/DailyMarketSnapshotTab.test.jsx`, add a fixture freshness block containing:

```javascript
freshness: {
  scan_id: 'scan-abc',
  scan_as_of_date: '2026-06-11',
  snapshot_as_of_date: '2026-06-11',
  market_timezone: 'America/New_York',
  breadth_latest_date: '2026-06-11',
  groups_latest_date: '2026-06-11',
  exposure_latest_date: '2026-06-11',
  key_markets_latest_date: '2026-06-11',
  date_coherence_status: 'coherent',
}
```

Add this assertion to the render test that covers the snapshot header:

```javascript
expect(screen.getByText(/As of 2026-06-11/)).toBeInTheDocument()
expect(screen.getByText(/America\/New_York/)).toBeInTheDocument()
expect(screen.queryByText(/Breadth 2026-06-11/)).not.toBeInTheDocument()
```

- [ ] **Step 2: Run the failing frontend test**

Run:

```bash
cd frontend && npm run test:run -- DailyMarketSnapshotTab.test.jsx
```

Expected: FAIL because the header still says `Snapshot ... · Breadth ... · Groups ...`.

- [ ] **Step 3: Update the header copy**

In `frontend/src/components/MarketScan/DailyMarketSnapshotTab.jsx`, replace the current freshness label construction with:

```javascript
const snapshotDate = freshness?.snapshot_as_of_date || freshness?.scan_as_of_date
const sectionDates = [
  ['Breadth', freshness?.breadth_latest_date],
  ['Groups', freshness?.groups_latest_date],
  ['Exposure', freshness?.exposure_latest_date],
  ['Key markets', freshness?.key_markets_latest_date],
].filter(([, value]) => value && value !== snapshotDate)
const freshnessLabel = snapshotDate
  ? [
      `As of ${snapshotDate}`,
      freshness?.market_timezone,
      freshness?.date_coherence_status && freshness.date_coherence_status !== 'coherent'
        ? freshness.date_coherence_status
        : null,
      ...sectionDates.map(([label, value]) => `${label} ${value}`),
    ].filter(Boolean).join(' · ')
  : 'Snapshot date unavailable'
```

Use `freshnessLabel` in the existing metadata text node.

- [ ] **Step 4: Run the frontend focused test**

Run:

```bash
cd frontend && npm run test:run -- DailyMarketSnapshotTab.test.jsx
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/MarketScan/DailyMarketSnapshotTab.jsx frontend/src/components/MarketScan/DailyMarketSnapshotTab.test.jsx
git commit -m "feat: show daily snapshot date contract"
```

## Task 6: End-To-End Verification

**Files:**
- No source edits in this task unless a verification failure points to a bug.

- [ ] **Step 1: Run focused backend tests**

Run:

```bash
cd backend && pytest \
  tests/unit/test_daily_snapshot_service.py \
  tests/unit/test_daily_market_pipeline_tasks.py \
  tests/unit/test_breadth_tasks.py::test_breadth_gapfill_uses_requested_calculation_date_for_daily_calc \
  tests/unit/test_group_rank_tasks.py::test_group_gapfill_uses_requested_calculation_date_for_daily_calc \
  tests/unit/test_rrg_service.py \
  -q
```

Expected: PASS.

- [ ] **Step 2: Run frontend focused tests**

Run:

```bash
cd frontend && npm run test:run -- DailyMarketSnapshotTab.test.jsx MarketHealthExposure.test.jsx
```

Expected: PASS.

- [ ] **Step 3: Run configured quality gates**

Run:

```bash
cd backend && source venv/bin/activate && pytest tests/unit/test_daily_snapshot_service.py tests/unit/test_daily_market_pipeline_tasks.py tests/unit/test_breadth_tasks.py tests/unit/test_group_rank_tasks.py tests/unit/test_rrg_service.py
```

Expected: PASS.

Run:

```bash
export NVM_DIR="$HOME/.nvm" && [ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"
cd frontend && npm run test:run -- DailyMarketSnapshotTab.test.jsx MarketHealthExposure.test.jsx
cd frontend && npm run lint
```

Expected: PASS.

- [ ] **Step 4: Manually inspect one local payload**

Run the backend and frontend the usual way for this repo, then request:

```bash
curl -s "http://localhost:8000/api/v1/market-scan/daily-snapshot?market=US" | jq '.freshness, .market_health_exposure.date, [.key_markets[].latest_date] | unique'
```

Expected:

```json
{
  "snapshot_as_of_date": "YYYY-MM-DD",
  "market_timezone": "America/New_York",
  "breadth_latest_date": "YYYY-MM-DD",
  "groups_latest_date": "YYYY-MM-DD",
  "exposure_latest_date": "YYYY-MM-DD",
  "date_coherence_status": "coherent"
}
```

The exposure date equals `snapshot_as_of_date`. Key-market dates are less than or equal to `snapshot_as_of_date`.

- [ ] **Step 5: Commit verification fixes if any**

If a verification failure required a source edit, commit it:

```bash
git add backend frontend
git commit -m "fix: stabilize snapshot date coherence verification"
```

- [ ] **Step 6: Sync beads and push when implementation is complete**

Follow the repo session protocol:

```bash
git status
bd sync
git pull --rebase
bd sync
git push
git status
```

Expected final status includes `Your branch is up to date with 'origin/<branch>'`.

## Self-Review

- Spec coverage: The plan pins local Daily Snapshot scan, breadth, exposure, groups, key markets, and live RRG-capable group routes to an explicit market-session date.
- Placeholder scan: No step depends on an unspecified function or unnamed test.
- Type consistency: `as_of_date` is a Python `date | None` inside services and FastAPI handlers, and `calculation_date` is a `YYYY-MM-DD` string for Celery tasks.
- Remaining intentional behavior: `generated_at` stays UTC because it is a payload creation timestamp, not a market session date.
