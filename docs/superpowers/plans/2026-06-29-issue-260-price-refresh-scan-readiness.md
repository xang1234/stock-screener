# Issue 260 Price Refresh Scan Readiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix GitHub issue #260 so a market refresh that completes with a tiny failed-symbol tail no longer blocks broad cache-only scans, while stale symbols are never scanned silently.

**Architecture:** Add a structured freshness decision that can either block a scan or return a filtered fresh-symbol universe with omission warnings. Keep the existing strict `check_symbol_freshness()` wrapper for callers/tests that need all-or-nothing behavior. Wire broad universe scans (`all`, `market`, `exchange`, `index`) to omit a small stale tail only when market refresh state is completed and the resolved universe is at least 95% fresh; keep `custom` and `test` scans strict.

**Tech Stack:** FastAPI, SQLAlchemy, Pydantic, pytest, React, TanStack Query, MUI, Vitest.

---

## File Structure

- Modify `backend/app/domain/scanning/models.py`
  - Add immutable value objects for freshness omissions and scan freshness decisions.
- Modify `backend/app/services/market_data_freshness.py`
  - Add `evaluate_symbol_freshness()` and keep `check_symbol_freshness()` as a strict compatibility wrapper.
- Modify `backend/app/use_cases/scanning/create_scan.py`
  - Consume structured freshness decisions, filter broad-universe symbols before hashing/dispatch, and return warnings.
- Modify `backend/app/wiring/bootstrap.py`
  - Inject `evaluate_symbol_freshness` into the HTTP-bound scan use case.
- Modify `backend/app/schemas/scanning.py`
  - Add `warnings` to `ScanCreateResponse`.
- Modify `backend/app/api/v1/scans.py`
  - Include warnings in the create-scan response.
- Modify `frontend/src/features/scan/pages/ScanPageContainer.jsx`
  - Store warnings from successful scan creation and pass them to the control bar.
- Modify `frontend/src/features/scan/components/ScanControlBar.jsx`
  - Render a warning alert for omitted stale symbols.
- Modify tests:
  - `backend/tests/unit/services/test_market_data_freshness.py`
  - `backend/tests/unit/use_cases/test_create_scan.py`
  - `backend/tests/unit/test_scan_create_endpoint.py`
  - `frontend/src/pages/ScanPage.test.jsx`

Policy locked by this plan:

- Broad scans may omit stale symbols only when all of these are true:
  - `allow_degraded_omissions=True`
  - market refresh state exists
  - refresh state status is `completed`
  - refresh state trading day is at least the market expected trading day
  - the currently resolved symbol set is at least `PRICE_REFRESH_COMPLETED_SUCCESS_RATE` fresh
  - refresh state `success_rate`, when present, is also at least `PRICE_REFRESH_COMPLETED_SUCCESS_RATE`
- Custom/test scans never omit requested stale symbols.
- Calendar failures, unresolved symbols, missing refresh state, stale refresh state, and freshness below the threshold still return `409 market_data_stale`.
- A symbol omitted by this path is not included in `total_stocks`, feature-run hashing, result persistence, or Celery dispatch.

---

### Task 1: Add Failing Freshness Evaluator Tests

**Files:**
- Modify: `backend/tests/unit/services/test_market_data_freshness.py`
- Test: `backend/tests/unit/services/test_market_data_freshness.py`

- [ ] **Step 1: Extend the fake refresh-state helper**

Replace the existing `_patch_refresh_state` helper with this version:

```python
def _patch_refresh_state(
    last_refreshed_by_market,
    *,
    status_by_market=None,
    success_rate_by_market=None,
):
    status_by_market = status_by_market or {}
    success_rate_by_market = success_rate_by_market or {}

    def _state(_session, market):
        value = last_refreshed_by_market.get(market)
        if value is None:
            return None
        return {
            "market": market,
            "status": status_by_market.get(market, "completed"),
            "last_refreshed_trading_day": value.isoformat(),
            "success_rate": success_rate_by_market.get(market, 1.0),
        }

    return patch(
        "app.services.market_data_freshness.get_market_refresh_state",
        side_effect=_state,
    )
```

- [ ] **Step 2: Add a failing test for degraded broad-scan omission**

Append this test near the existing freshness tests:

```python
def test_degraded_completed_market_omits_small_stale_tail_for_broad_scan():
    from app.services.market_data_freshness import evaluate_symbol_freshness

    expected = date(2026, 6, 18)
    rows = [
        *[
            _Row(symbol=f"SYM{i:03d}", market="US", last_date=expected)
            for i in range(99)
        ],
        _Row(symbol="LHSW", market="US", last_date=date(2026, 5, 13)),
    ]
    symbols = [row.symbol for row in rows]

    with (
        _patch_session(rows),
        _patch_calendar({"US": expected}),
        _patch_refresh_state(
            {"US": expected},
            success_rate_by_market={"US": 0.99},
        ),
    ):
        decision = evaluate_symbol_freshness(
            symbols,
            allow_degraded_omissions=True,
        )

    assert decision.blocking_detail is None
    assert "LHSW" not in decision.fresh_symbols
    assert len(decision.fresh_symbols) == 99
    assert len(decision.warnings) == 1
    warning = decision.warnings[0]
    assert warning.code == "market_data_omitted_stale_symbols"
    assert warning.market == "US"
    assert warning.omitted_symbols == ("LHSW",)
    assert warning.omitted_count == 1
    assert warning.total_symbols == 100
    assert warning.oldest_last_cached_date == "2026-05-13"
    assert warning.expected_date == "2026-06-18"
```

- [ ] **Step 3: Add a failing test that degraded omission still blocks below threshold**

Append this test:

```python
def test_degraded_completed_market_blocks_when_fresh_coverage_below_threshold():
    from app.services.market_data_freshness import evaluate_symbol_freshness

    expected = date(2026, 6, 18)
    rows = [
        *[
            _Row(symbol=f"FRESH{i:03d}", market="US", last_date=expected)
            for i in range(94)
        ],
        *[
            _Row(symbol=f"STALE{i:03d}", market="US", last_date=date(2026, 5, 13))
            for i in range(6)
        ],
    ]
    symbols = [row.symbol for row in rows]

    with (
        _patch_session(rows),
        _patch_calendar({"US": expected}),
        _patch_refresh_state(
            {"US": expected},
            success_rate_by_market={"US": 0.94},
        ),
    ):
        decision = evaluate_symbol_freshness(
            symbols,
            allow_degraded_omissions=True,
        )

    assert decision.blocking_detail is not None
    assert decision.blocking_detail["code"] == "market_data_stale"
    assert decision.blocking_detail["stale_markets"][0]["market"] == "US"
    assert decision.blocking_detail["stale_markets"][0]["oldest_last_cached_date"] == "2026-05-13"
```

- [ ] **Step 4: Add a failing test that the strict wrapper still blocks**

Append this test:

```python
def test_strict_freshness_wrapper_blocks_small_stale_tail():
    from app.services.market_data_freshness import check_symbol_freshness

    expected = date(2026, 6, 18)
    rows = [
        *[
            _Row(symbol=f"SYM{i:03d}", market="US", last_date=expected)
            for i in range(99)
        ],
        _Row(symbol="LHSW", market="US", last_date=date(2026, 5, 13)),
    ]
    symbols = [row.symbol for row in rows]

    with (
        _patch_session(rows),
        _patch_calendar({"US": expected}),
        _patch_refresh_state(
            {"US": expected},
            success_rate_by_market={"US": 0.99},
        ),
    ):
        detail = check_symbol_freshness(symbols)

    assert detail is not None
    assert detail["code"] == "market_data_stale"
    assert detail["stale_markets"][0]["oldest_last_cached_date"] == "2026-05-13"
```

- [ ] **Step 5: Run evaluator tests and verify they fail**

Run:

```bash
backend/venv/bin/pytest backend/tests/unit/services/test_market_data_freshness.py -q
```

Expected: FAIL with `ImportError` or `AttributeError` for `evaluate_symbol_freshness`, because implementation has not been added yet.

- [ ] **Step 6: Commit failing tests**

```bash
git add backend/tests/unit/services/test_market_data_freshness.py
git commit -m "test: capture degraded price freshness policy"
```

---

### Task 2: Implement Freshness Decision Models And Evaluator

**Files:**
- Modify: `backend/app/domain/scanning/models.py`
- Modify: `backend/app/services/market_data_freshness.py`
- Test: `backend/tests/unit/services/test_market_data_freshness.py`

- [ ] **Step 1: Add freshness value objects**

In `backend/app/domain/scanning/models.py`, insert this block after `UniverseSpec`:

```python
@dataclass(frozen=True)
class FreshnessOmissionWarning:
    """Symbols omitted from a broad cache-only scan because their prices are stale."""

    code: str
    market: str
    omitted_symbols: tuple[str, ...]
    omitted_count: int
    total_symbols: int
    expected_date: str | None
    oldest_last_cached_date: str | None
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "market": self.market,
            "omitted_symbols": list(self.omitted_symbols),
            "omitted_count": self.omitted_count,
            "total_symbols": self.total_symbols,
            "expected_date": self.expected_date,
            "oldest_last_cached_date": self.oldest_last_cached_date,
            "message": self.message,
        }


@dataclass(frozen=True)
class FreshnessDecision:
    """Result of checking resolved scan symbols against cached price freshness."""

    fresh_symbols: tuple[str, ...]
    blocking_detail: dict[str, Any] | None = None
    warnings: tuple[FreshnessOmissionWarning, ...] = ()

    @property
    def is_blocked(self) -> bool:
        return self.blocking_detail is not None
```

- [ ] **Step 2: Add evaluator imports and helper constants**

In `backend/app/services/market_data_freshness.py`, add these imports:

```python
from ..domain.scanning.models import FreshnessDecision, FreshnessOmissionWarning
from ..services.price_refresh_accounting import PRICE_REFRESH_COMPLETED_SUCCESS_RATE
```

Then add these helpers above `check_symbol_freshness`:

```python
def _state_success_rate(state: dict | None) -> float | None:
    if not state or state.get("success_rate") is None:
        return None
    try:
        return float(state["success_rate"])
    except (TypeError, ValueError):
        return None


def _freshness_detail(
    *,
    stale_markets: list[dict],
    unresolved_symbols: list[str],
) -> dict | None:
    if not stale_markets and not unresolved_symbols:
        return None

    def _describe(market_entry: dict) -> str:
        if market_entry.get("reason") == "calendar_unavailable":
            return f"{market_entry['market']} (calendar unavailable - could not verify freshness)"
        return (
            f"{market_entry['market']} "
            f"(oldest: {market_entry['oldest_last_cached_date'] or 'never'}, "
            f"expected: {market_entry['expected_date']})"
        )

    fragments: list[str] = []
    if stale_markets:
        fragments.append(", ".join(_describe(m) for m in stale_markets))
    if unresolved_symbols:
        preview = ", ".join(unresolved_symbols[:5])
        more = "" if len(unresolved_symbols) <= 5 else f" (+{len(unresolved_symbols) - 5} more)"
        fragments.append(f"unknown symbols: {preview}{more}")

    return {
        "code": "market_data_stale",
        "message": (
            f"Price data is stale for: {'; '.join(fragments)}. "
            "Wait for the next scheduled refresh before starting a scan."
        ),
        "stale_markets": stale_markets,
        "unresolved_symbols": unresolved_symbols,
    }
```

- [ ] **Step 3: Replace `check_symbol_freshness` with evaluator plus strict wrapper**

Replace the body of `check_symbol_freshness` with this wrapper:

```python
def check_symbol_freshness(symbols: Iterable[str]) -> Optional[dict]:
    """Strict compatibility wrapper: return 409 detail when any symbol is stale."""
    decision = evaluate_symbol_freshness(
        symbols,
        allow_degraded_omissions=False,
    )
    return decision.blocking_detail
```

Add this new function immediately above the wrapper:

```python
def evaluate_symbol_freshness(
    symbols: Iterable[str],
    allow_degraded_omissions: bool = False,
) -> FreshnessDecision:
    """Return either a blocking detail or a fresh-symbol subset plus warnings."""
    normalized = tuple(sorted({s.upper() for s in symbols if s}))
    if not normalized:
        return FreshnessDecision(fresh_symbols=())

    session = SessionLocal()
    stale_markets: list[dict] = []
    warnings: list[FreshnessOmissionWarning] = []
    omitted_symbols: set[str] = set()
    unresolved_symbols: list[str] = []
    try:
        rows = (
            session.query(
                StockUniverse.symbol.label("symbol"),
                StockUniverse.market.label("market"),
                func.max(StockPrice.date).label("last_date"),
            )
            .outerjoin(StockPrice, StockPrice.symbol == StockUniverse.symbol)
            .filter(StockUniverse.symbol.in_(normalized))
            .group_by(StockUniverse.symbol, StockUniverse.market)
            .all()
        )

        resolved_symbols = {row.symbol for row in rows}
        unresolved_symbols = sorted(set(normalized) - resolved_symbols)

        per_market: dict[str, list] = {}
        for row in rows:
            per_market.setdefault(row.market, []).append(row)

        calendar = get_market_calendar_service()
        for market, market_rows in sorted(per_market.items()):
            try:
                expected_date = calendar.last_completed_trading_day(market)
            except Exception:
                logger.warning(
                    "Could not resolve last completed trading day for market=%s; treating as stale",
                    market,
                    exc_info=True,
                )
                stale_markets.append({
                    "market": market,
                    "total_symbols": len(market_rows),
                    "covered_symbols": sum(1 for r in market_rows if r.last_date is not None),
                    "uncovered_symbols": sum(1 for r in market_rows if r.last_date is None),
                    "oldest_last_cached_date": None,
                    "expected_date": None,
                    "reason": "calendar_unavailable",
                })
                continue

            state = get_market_refresh_state(session, market)
            observed_date = _parse_state_date(
                state.get("last_refreshed_trading_day") if state else None
            )
            comparison_date = expected_date
            if state is not None and (state.get("status") != "completed" or observed_date is None):
                stale_markets.append({
                    "market": market,
                    "total_symbols": len(market_rows),
                    "covered_symbols": sum(1 for r in market_rows if r.last_date is not None),
                    "uncovered_symbols": sum(1 for r in market_rows if r.last_date is None),
                    "oldest_last_cached_date": None,
                    "expected_date": str(expected_date),
                    "reason": "refresh_state_missing",
                })
                continue
            if observed_date is not None and observed_date < expected_date:
                stale_markets.append({
                    "market": market,
                    "total_symbols": len(market_rows),
                    "covered_symbols": sum(1 for r in market_rows if r.last_date is not None),
                    "uncovered_symbols": sum(1 for r in market_rows if r.last_date is None),
                    "oldest_last_cached_date": str(observed_date),
                    "expected_date": str(expected_date),
                    "reason": "refresh_state_stale",
                })
                continue
            if observed_date is not None:
                comparison_date = observed_date

            covered_dates = [r.last_date for r in market_rows if r.last_date is not None]
            stale_rows = [
                r for r in market_rows
                if r.last_date is None or r.last_date < comparison_date
            ]
            stale_symbols = tuple(sorted(r.symbol for r in stale_rows))
            oldest = min(covered_dates) if covered_dates else None
            fresh_count = len(market_rows) - len(stale_rows)
            resolved_success_rate = fresh_count / len(market_rows) if market_rows else 0.0
            state_success_rate = _state_success_rate(state)
            state_rate_ok = (
                state_success_rate is None
                or state_success_rate >= PRICE_REFRESH_COMPLETED_SUCCESS_RATE
            )

            can_omit = (
                allow_degraded_omissions
                and state is not None
                and state.get("status") == "completed"
                and observed_date is not None
                and observed_date >= expected_date
                and resolved_success_rate >= PRICE_REFRESH_COMPLETED_SUCCESS_RATE
                and state_rate_ok
                and stale_symbols
            )
            if can_omit:
                omitted_symbols.update(stale_symbols)
                warnings.append(
                    FreshnessOmissionWarning(
                        code="market_data_omitted_stale_symbols",
                        market=market,
                        omitted_symbols=stale_symbols,
                        omitted_count=len(stale_symbols),
                        total_symbols=len(market_rows),
                        expected_date=str(comparison_date),
                        oldest_last_cached_date=str(oldest) if oldest is not None else None,
                        message=(
                            f"Omitted {len(stale_symbols)} {market} symbols with stale price data "
                            f"from this scan."
                        ),
                    )
                )
                continue

            if stale_rows:
                stale_markets.append({
                    "market": market,
                    "total_symbols": len(market_rows),
                    "covered_symbols": len(covered_dates),
                    "uncovered_symbols": sum(1 for r in market_rows if r.last_date is None),
                    "oldest_last_cached_date": str(oldest) if oldest is not None else None,
                    "expected_date": str(comparison_date),
                })
    finally:
        session.close()

    detail = _freshness_detail(
        stale_markets=stale_markets,
        unresolved_symbols=unresolved_symbols,
    )
    fresh_symbols = tuple(symbol for symbol in normalized if symbol not in omitted_symbols)
    return FreshnessDecision(
        fresh_symbols=fresh_symbols,
        blocking_detail=detail,
        warnings=tuple(warnings),
    )
```

- [ ] **Step 4: Remove the duplicate detail-building block from the old function**

After replacing `check_symbol_freshness`, ensure `backend/app/services/market_data_freshness.py` has only one message-building helper: `_freshness_detail`.

- [ ] **Step 5: Run evaluator tests and verify they pass**

Run:

```bash
backend/venv/bin/pytest backend/tests/unit/services/test_market_data_freshness.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit evaluator implementation**

```bash
git add backend/app/domain/scanning/models.py backend/app/services/market_data_freshness.py backend/tests/unit/services/test_market_data_freshness.py
git commit -m "fix: return degraded price freshness decisions"
```

---

### Task 3: Wire Freshness Decisions Into Scan Creation

**Files:**
- Modify: `backend/app/use_cases/scanning/create_scan.py`
- Modify: `backend/app/wiring/bootstrap.py`
- Modify: `backend/tests/unit/use_cases/test_create_scan.py`
- Test: `backend/tests/unit/use_cases/test_create_scan.py`

- [ ] **Step 1: Add failing use-case tests**

In `backend/tests/unit/use_cases/test_create_scan.py`, add this import:

```python
from app.domain.scanning.models import FreshnessDecision, FreshnessOmissionWarning
```

Append these tests to `TestFreshnessChecker`:

```python
    def test_broad_universe_omits_degraded_stale_symbols_before_dispatch(self):
        uow = _make_uow(["AAPL", "LHSW", "MSFT"])
        dispatcher = FakeTaskDispatcher()
        warning = FreshnessOmissionWarning(
            code="market_data_omitted_stale_symbols",
            market="US",
            omitted_symbols=("LHSW",),
            omitted_count=1,
            total_symbols=3,
            expected_date="2026-06-18",
            oldest_last_cached_date="2026-05-13",
            message="Omitted 1 US symbols with stale price data from this scan.",
        )

        def checker(symbols, allow_degraded_omissions):
            assert list(symbols) == ["AAPL", "LHSW", "MSFT"]
            assert allow_degraded_omissions is True
            return FreshnessDecision(
                fresh_symbols=("AAPL", "MSFT"),
                warnings=(warning,),
            )

        uc = CreateScanUseCase(dispatcher=dispatcher, freshness_checker=checker)
        result = uc.execute(uow, _make_command(universe_type="market", universe_market="US"))

        assert result.status == "queued"
        assert result.total_stocks == 2
        assert result.warnings == [warning.to_dict()]
        assert dispatcher.dispatched[0][1] == ["AAPL", "MSFT"]
        assert uow.scans.rows[0].total_stocks == 2

    def test_custom_universe_does_not_enable_degraded_omission(self):
        uow = _make_uow(["LHSW"])
        dispatcher = FakeTaskDispatcher()
        calls: list[bool] = []

        def checker(_symbols, allow_degraded_omissions):
            calls.append(allow_degraded_omissions)
            return self._STALE_DETAIL

        uc = CreateScanUseCase(dispatcher=dispatcher, freshness_checker=checker)

        with pytest.raises(StaleMarketDataError):
            uc.execute(
                uow,
                _make_command(
                    universe_type="custom",
                    universe_symbols=["LHSW"],
                ),
            )

        assert calls == [False]
        assert len(dispatcher.dispatched) == 0
```

- [ ] **Step 2: Update existing freshness-checker test lambdas**

In the same file, replace freshness checker lambdas that accept one argument with two-argument lambdas:

```python
freshness_checker=lambda _symbols, _allow_degraded: self._STALE_DETAIL
```

and:

```python
freshness_checker=lambda _symbols, _allow_degraded: None
```

For the named `checker(symbols)` helper in `test_idempotent_retry_bypasses_staleness_gate`, change it to:

```python
        def checker(symbols, _allow_degraded):
            calls.append(list(symbols))
            return self._STALE_DETAIL
```

- [ ] **Step 3: Run use-case tests and verify they fail**

Run:

```bash
backend/venv/bin/pytest backend/tests/unit/use_cases/test_create_scan.py -q
```

Expected: FAIL because `CreateScanResult` has no `warnings` field and `CreateScanUseCase` still expects the old checker contract.

- [ ] **Step 4: Update `CreateScanResult` and freshness checker contract**

In `backend/app/use_cases/scanning/create_scan.py`, add the import:

```python
from app.domain.scanning.models import FreshnessDecision
```

Change the type alias to:

```python
FreshnessChecker = Callable[[Iterable[str], bool], Optional[dict] | FreshnessDecision]
```

Add this constant near `_COMPILE_PATH_DROP_KEYS`:

```python
_DEGRADED_OMISSION_UNIVERSE_TYPES: frozenset[str] = frozenset({
    UniverseType.ALL.value,
    UniverseType.MARKET.value,
    UniverseType.EXCHANGE.value,
    UniverseType.INDEX.value,
})
```

Update `CreateScanResult`:

```python
@dataclass(frozen=True)
class CreateScanResult:
    """What the use case returns to the caller."""

    scan_id: str
    status: str
    total_stocks: int
    is_duplicate: bool
    feature_run_id: int | None = None
    warnings: list[dict] = field(default_factory=list)
```

- [ ] **Step 5: Filter symbols after freshness evaluation**

In `CreateScanUseCase.execute`, immediately before the staleness gate comment, add:

```python
            freshness_warnings: list[dict] = []
```

Replace the current staleness gate with:

```python
            if self._freshness_checker is not None:
                allow_degraded_omissions = (
                    cmd.universe_type in _DEGRADED_OMISSION_UNIVERSE_TYPES
                )
                freshness_result = self._freshness_checker(
                    symbols,
                    allow_degraded_omissions,
                )
                if isinstance(freshness_result, FreshnessDecision):
                    if freshness_result.is_blocked:
                        raise StaleMarketDataError(freshness_result.blocking_detail or {})
                    symbols = list(freshness_result.fresh_symbols)
                    freshness_warnings = [
                        warning.to_dict() for warning in freshness_result.warnings
                    ]
                    if not symbols:
                        raise ValidationError(
                            f"No fresh symbols found for universe '{cmd.universe_label}'. "
                            "Refresh market data before starting a scan."
                        )
                elif freshness_result is not None:
                    raise StaleMarketDataError(freshness_result)
```

Then pass `warnings=freshness_warnings` in every non-duplicate `CreateScanResult(...)` returned after the gate. The duplicate idempotency return remains `warnings=[]`.

- [ ] **Step 6: Inject the evaluator**

In `backend/app/wiring/bootstrap.py`, update `get_create_scan_use_case`:

```python
    from app.services.market_data_freshness import evaluate_symbol_freshness

    return CreateScanUseCase(
        dispatcher=get_task_dispatcher(),
        freshness_checker=evaluate_symbol_freshness,
    )
```

- [ ] **Step 7: Run use-case tests and verify they pass**

Run:

```bash
backend/venv/bin/pytest backend/tests/unit/use_cases/test_create_scan.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit use-case wiring**

```bash
git add backend/app/use_cases/scanning/create_scan.py backend/app/wiring/bootstrap.py backend/tests/unit/use_cases/test_create_scan.py
git commit -m "fix: omit stale tail from broad scan creation"
```

---

### Task 4: Surface Omission Warnings Through API And UI

**Files:**
- Modify: `backend/app/schemas/scanning.py`
- Modify: `backend/app/api/v1/scans.py`
- Modify: `backend/tests/unit/test_scan_create_endpoint.py`
- Modify: `frontend/src/features/scan/pages/ScanPageContainer.jsx`
- Modify: `frontend/src/features/scan/components/ScanControlBar.jsx`
- Modify: `frontend/src/pages/ScanPage.test.jsx`
- Test: backend endpoint and frontend scan-page tests.

- [ ] **Step 1: Add backend endpoint test**

In `backend/tests/unit/test_scan_create_endpoint.py`, append this test near the other create-scan response tests:

```python
@pytest.mark.asyncio
async def test_create_scan_returns_degraded_freshness_warnings(client):
    fake_uow = _FakeUoW()
    warning = {
        "code": "market_data_omitted_stale_symbols",
        "market": "US",
        "omitted_symbols": ["LHSW"],
        "omitted_count": 1,
        "total_symbols": 100,
        "expected_date": "2026-06-18",
        "oldest_last_cached_date": "2026-05-13",
        "message": "Omitted 1 US symbols with stale price data from this scan.",
    }
    fake_use_case = _FakeCreateScanUseCase(
        CreateScanResult(
            scan_id="scan-warning",
            status="queued",
            total_stocks=99,
            is_duplicate=False,
            feature_run_id=None,
            warnings=[warning],
        )
    )

    app.dependency_overrides[get_uow] = lambda: fake_uow
    app.dependency_overrides[get_create_scan_use_case] = lambda: fake_use_case
    try:
        response = await client.post(
            "/api/v1/scans",
            json={"universe_def": {"type": "market", "market": "US"}},
        )
    finally:
        app.dependency_overrides.pop(get_uow, None)
        app.dependency_overrides.pop(get_create_scan_use_case, None)

    assert response.status_code == 200
    payload = response.json()
    assert payload["scan_id"] == "scan-warning"
    assert payload["total_stocks"] == 99
    assert payload["warnings"] == [warning]
```

- [ ] **Step 2: Add warnings to the backend response schema and route**

In `backend/app/schemas/scanning.py`, update `ScanCreateResponse`:

```python
class ScanCreateResponse(BaseModel):
    """Response model for scan creation."""

    scan_id: str
    status: str
    total_stocks: int
    message: str
    feature_run_id: Optional[int] = None
    universe_def: UniverseDefinition
    warnings: List[Dict[str, Any]] = Field(default_factory=list)
```

In `backend/app/api/v1/scans.py`, add this to `ScanCreateResponse(...)`:

```python
        warnings=result.warnings,
```

- [ ] **Step 3: Run endpoint test and verify it passes**

Run:

```bash
backend/venv/bin/pytest backend/tests/unit/test_scan_create_endpoint.py::test_create_scan_returns_degraded_freshness_warnings -q
```

Expected: PASS.

- [ ] **Step 4: Store scan warnings in the container**

In `frontend/src/features/scan/pages/ScanPageContainer.jsx`, add state near the other scan state:

```jsx
  const [scanWarnings, setScanWarnings] = useState([]);
```

In `handleLoadScan`, clear warnings when loading or clearing scan history:

```jsx
      setScanWarnings([]);
```

In the `createScanMutation` `onSuccess`, add:

```jsx
      setScanWarnings(Array.isArray(data.warnings) ? data.warnings : []);
```

Pass warnings to `ScanControlBar`:

```jsx
        scanWarnings={scanWarnings}
```

- [ ] **Step 5: Render scan warnings in the control bar**

In `frontend/src/features/scan/components/ScanControlBar.jsx`, add `scanWarnings = []` to the props list.

Add this helper near `staleRefreshMarket`:

```jsx
function formatScanWarning(warning) {
  if (warning?.message) {
    return warning.message;
  }
  if (warning?.code === 'market_data_omitted_stale_symbols') {
    const count = warning.omitted_count ?? warning.omitted_symbols?.length ?? 0;
    const market = warning.market || 'market';
    return `Omitted ${count} ${market} symbols with stale price data from this scan.`;
  }
  return null;
}
```

Render warnings after the create-scan error alert and before refresh/cancel errors:

```jsx
      {scanWarnings.map((warning, index) => {
        const message = formatScanWarning(warning);
        if (!message) {
          return null;
        }
        return (
          <Alert key={`${warning.code || 'scan-warning'}-${index}`} severity="warning" sx={{ mt: 1 }}>
            {message}
          </Alert>
        );
      })}
```

- [ ] **Step 6: Add frontend test for warning display**

In `frontend/src/pages/ScanPage.test.jsx`, append this test near the scan creation tests:

```jsx
  it('shows omitted stale symbol warnings after a degraded scan starts', async () => {
    scanApi.createScan.mockResolvedValueOnce({
      scan_id: 'scan-warning',
      status: 'queued',
      total_stocks: 99,
      message: 'Scan queued for 99 stocks',
      universe_def: { type: 'market', market: 'US' },
      warnings: [
        {
          code: 'market_data_omitted_stale_symbols',
          market: 'US',
          omitted_symbols: ['LHSW'],
          omitted_count: 1,
          total_symbols: 100,
          expected_date: '2026-06-18',
          oldest_last_cached_date: '2026-05-13',
          message: 'Omitted 1 US symbols with stale price data from this scan.',
        },
      ],
    });

    render(<ScanPage />);

    const scanButton = await screen.findByRole('button', { name: /scan/i });
    await userEvent.click(scanButton);

    expect(
      await screen.findByText('Omitted 1 US symbols with stale price data from this scan.')
    ).toBeInTheDocument();
  });
```

- [ ] **Step 7: Run frontend warning test**

Run:

```bash
cd frontend && npm run test:run -- src/pages/ScanPage.test.jsx
```

Expected: PASS.

- [ ] **Step 8: Commit API and UI warning surface**

```bash
git add backend/app/schemas/scanning.py backend/app/api/v1/scans.py backend/tests/unit/test_scan_create_endpoint.py frontend/src/features/scan/pages/ScanPageContainer.jsx frontend/src/features/scan/components/ScanControlBar.jsx frontend/src/pages/ScanPage.test.jsx
git commit -m "feat: surface degraded scan freshness warnings"
```

---

### Task 5: Regression Verification And Session Close

**Files:**
- No source edits unless a verification failure identifies a concrete bug.

- [ ] **Step 1: Run targeted backend regression tests**

```bash
backend/venv/bin/pytest \
  backend/tests/unit/services/test_market_data_freshness.py \
  backend/tests/unit/use_cases/test_create_scan.py \
  backend/tests/unit/test_scan_create_endpoint.py \
  backend/tests/unit/test_price_refresh_accounting.py \
  backend/tests/unit/test_market_refresh_state.py \
  -q
```

Expected: PASS.

- [ ] **Step 2: Run targeted frontend regression tests**

```bash
cd frontend && npm run test:run -- src/pages/ScanPage.test.jsx
```

Expected: PASS.

- [ ] **Step 3: Run backend quality gate**

```bash
cd backend && source venv/bin/activate && pytest tests/unit/services/test_market_data_freshness.py tests/unit/use_cases/test_create_scan.py tests/unit/test_scan_create_endpoint.py tests/unit/test_price_refresh_accounting.py tests/unit/test_market_refresh_state.py -q
```

Expected: PASS.

- [ ] **Step 4: Run frontend lint**

```bash
cd frontend && npm run lint
```

Expected: PASS.

- [ ] **Step 5: Update Beads issue**

```bash
bd update stockscreenclaude-14y --status in_progress
bd close stockscreenclaude-14y --reason="Implemented degraded price freshness filtering for issue #260"
```

Expected: issue closes successfully.

- [ ] **Step 6: Push all committed work**

```bash
git pull --rebase
bd dolt push
git push
git status
```

Expected:

```text
Your branch is up to date with 'origin/<branch-name>'.
nothing to commit, working tree clean
```

If `bd dolt push` reports no Dolt remote configured, record that in the final handoff and continue with `git push`.

---

## Self-Review

**Spec coverage:** The plan covers the issue #260 root cause by aligning broad scan creation with the refresh completion threshold, filtering omitted stale symbols out of cache-only scans, preserving strict behavior for custom/test scans, and showing users a warning when symbols are omitted.

**Placeholder scan:** The plan contains no `TBD`, no deferred implementation notes, and no unspecific edge-case instructions.

**Type consistency:** `FreshnessDecision`, `FreshnessOmissionWarning`, `CreateScanResult.warnings`, and `ScanCreateResponse.warnings` use consistent names across service, use case, API, and UI tasks.
