# Issue 260 High-Threshold Stale Tail Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix GitHub issue #260 by allowing broad scan universes to proceed when only a very small stale tail remains after a completed refresh, while keeping custom/test scans strict. The accepted policy is: omit at most 100 stale symbols, require at least 99% freshness across the resolved scan universe, persist a warning that names what was omitted, and never dispatch omitted symbols.

**Architecture:** Introduce a richer freshness decision object next to the existing freshness checker. `CreateScanUseCase` applies the decision after universe resolution and before signature hashing, scan persistence, feature-run matching, or task dispatch. Omission metadata is stored on the `scans` row and returned by create/status/list/bootstrap APIs so the UI can surface it after reloads.

**Tech Stack:** FastAPI, SQLAlchemy, Alembic, Pydantic, pytest, React, React Query, MUI, Vitest.

---

## Problem Summary

The failing system is a user's database, not this local database. That does not change the root cause.

The policy mismatch does not create stale data. The stale rows are created by ordinary refresh failures: a symbol fails during refresh, so its last successful `stock_prices.date` remains old. In issue #260 the refresh finished with roughly 9,973 refreshed symbols and 10 failures. The refresh layer treated the run as complete, but scan creation still required every active resolved symbol to have prices through the expected trading day. One stale active symbol was enough to block the whole scan.

The fix is not to loosen custom symbol scans. If a user asks for `["AAPL", "LHSW"]`, silently dropping `LHSW` is surprising. The fix is to degrade broad universes only: market, exchange, index, and all-universe scans may omit a small tail when the rest of the scan is fresh.

## Freshness Policy

Broad scans may continue with omissions only when all conditions hold:

- The universe type is `all`, `market`, `exchange`, or `index`.
- Every involved market has refresh state `status == "completed"`.
- Every involved market's `last_refreshed_trading_day` is at least its expected last completed trading day.
- There are no unresolved symbols missing from `stock_universe`.
- The scan-level omitted count is `<= 100`.
- The scan-level fresh rate is `>= 0.99`.
- Calendar lookup succeeds for every involved market.

Strict scans continue to block on any stale, uncovered, unresolved, or unverifiable symbol:

- `custom`
- `test`
- any explicit user-provided symbol list

For a degraded broad scan:

- `total_stocks` means symbols actually scanned after omissions.
- the warning records the original resolved count, omitted count, omitted symbols, freshness rate, expected dates, and oldest cached dates.
- omitted symbols are excluded before `hash_universe_symbols`, feature-run lookup, compile-path lookup, scan persistence totals, and background dispatch.

## Task 1: Add Freshness Decision Tests

Create failing tests first in `backend/tests/unit/services/test_market_data_freshness.py`.

- [ ] Add a helper that creates many `_Row` objects without making each test unreadable:

```python
def _rows(symbols, *, market="US", last_date=date(2026, 6, 18)):
    return [_Row(symbol=s, market=market, last_date=last_date) for s in symbols]
```

- [ ] Add `test_degraded_broad_scan_omits_small_tail_at_99_percent`.

Use 100 resolved symbols where 99 have `last_date == expected_date` and 1 has `last_date == date(2026, 5, 13)`. Call the new evaluator with `allow_stale_tail=True`.

Expected:

```python
assert decision.blocking_detail is None
assert decision.symbols_to_scan == tuple(fresh_symbols_in_original_order)
warning = decision.warnings[0].to_dict()
assert warning["code"] == "market_data_stale_tail_omitted"
assert warning["omitted_count"] == 1
assert warning["total_symbols"] == 100
assert warning["freshness_rate"] == 0.99
assert warning["omitted_symbols"] == ["STALE001"]
```

- [ ] Add `test_degraded_broad_scan_caps_omissions_at_100`.

Use 10,101 resolved symbols: 10,000 fresh and 101 stale. Call with `allow_stale_tail=True`.

Expected:

```python
assert decision.blocking_detail is not None
assert decision.blocking_detail["code"] == "market_data_stale"
assert decision.blocking_detail["stale_markets"][0]["stale_symbol_count"] == 101
assert decision.blocking_detail["stale_markets"][0]["sample_stale_symbols"][:3] == [
    "STALE001",
    "STALE002",
    "STALE003",
]
```

- [ ] Add `test_degraded_broad_scan_requires_99_percent_freshness`.

Use 99 resolved symbols: 98 fresh and 1 stale. This is below 99% freshness and must block even though omitted count is small.

- [ ] Add `test_degraded_decision_preserves_resolver_order_after_omission`.

Use symbols where the stale item is in the middle, for example:

```python
symbols = ["MSFT", "STALE001", "AAPL", *[f"FRESH{i:03d}" for i in range(97)]]
```

Expected:

```python
assert decision.symbols_to_scan[:3] == ("MSFT", "AAPL", "FRESH000")
```

Do not sort alphabetically.

- [ ] Add `test_strict_checker_still_blocks_stale_tail`.

Call the backward-compatible `check_symbol_freshness(symbols)` with the same 99 fresh / 1 stale setup. Expected: a `market_data_stale` detail, not an omission decision.

- [ ] Add `test_degraded_policy_requires_completed_refresh_state`.

Patch refresh state to missing or `status == "running"` while symbol dates include a stale tail. Expected: block with `reason == "refresh_state_missing"` or `reason == "refresh_state_stale"` and no omissions.

- [ ] Run only these tests and confirm they fail before implementation:

```bash
cd backend
source venv/bin/activate
pytest tests/unit/services/test_market_data_freshness.py -q
```

## Task 2: Implement Freshness Decision Model

Edit `backend/app/domain/scanning/models.py`.

- [ ] Add immutable value objects near the other scan domain models:

```python
@dataclass(frozen=True)
class FreshnessOmissionWarning:
    code: str
    message: str
    markets: tuple[str, ...]
    omitted_symbols: tuple[str, ...]
    omitted_count: int
    total_symbols: int
    fresh_count: int
    freshness_rate: float
    expected_dates: dict[str, str | None]
    oldest_last_cached_dates: dict[str, str | None]

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "markets": list(self.markets),
            "omitted_symbols": list(self.omitted_symbols),
            "omitted_count": self.omitted_count,
            "total_symbols": self.total_symbols,
            "fresh_count": self.fresh_count,
            "freshness_rate": self.freshness_rate,
            "expected_dates": self.expected_dates,
            "oldest_last_cached_dates": self.oldest_last_cached_dates,
        }


@dataclass(frozen=True)
class FreshnessDecision:
    symbols_to_scan: tuple[str, ...]
    warnings: tuple[FreshnessOmissionWarning, ...] = ()
    blocking_detail: dict[str, Any] | None = None
```

- [ ] Keep this model infrastructure-free. It should not import SQLAlchemy, FastAPI, or service modules.

## Task 3: Implement High-Threshold Stale-Tail Evaluation

Edit `backend/app/services/market_data_freshness.py`.

- [ ] Add explicit scan-gate constants near the top:

```python
SCAN_DEGRADED_MIN_FRESH_RATE = 0.99
SCAN_DEGRADED_MAX_OMITTED_SYMBOLS = 100
STALE_SYMBOL_SAMPLE_LIMIT = 20
```

Do not reuse the refresh completion threshold. Refresh completion answers "did a refresh finish acceptably?" This scan gate answers "is this particular scan universe fresh enough after omitting a tiny tail?"

- [ ] Preserve resolver order while still deduplicating:

```python
def _ordered_unique_symbols(symbols: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(s).upper() for s in symbols if s))
```

Replace the current `sorted({ ... })` normalization for evaluator paths. Sorting here is a behavioral regression because upstream universe resolution may already be market-cap ordered.

- [ ] Add the richer evaluator. This public signature is what callers use; the body performs the row grouping, refresh-state checks, stale-symbol accounting, and warning construction described below:

```python
def evaluate_symbol_freshness(
    symbols: Iterable[str],
    *,
    allow_stale_tail: bool = False,
) -> FreshnessDecision:
    ordered_symbols = _ordered_unique_symbols(symbols)
    if not ordered_symbols:
        return FreshnessDecision(symbols_to_scan=())
```

- [ ] Keep the existing strict adapter:

```python
def check_symbol_freshness(symbols: Iterable[str]) -> Optional[dict]:
    decision = evaluate_symbol_freshness(symbols, allow_stale_tail=False)
    return decision.blocking_detail
```

- [ ] Build stale/uncovered symbol sets per market, then build the final omitted list from the original ordered symbol tuple:

```python
stale_symbol_set = {row.symbol for row in stale_rows}
omitted_symbols = tuple(symbol for symbol in ordered_symbols if symbol in stale_symbol_set)
symbols_to_scan = tuple(symbol for symbol in ordered_symbols if symbol not in stale_symbol_set)
```

- [ ] Include diagnostics in blocking details:

```python
"stale_symbol_count": len(stale_symbols),
"sample_stale_symbols": stale_symbols[:STALE_SYMBOL_SAMPLE_LIMIT],
"freshness_rate": round(fresh_count / total_symbols, 6) if total_symbols else 1.0,
"omission_thresholds": {
    "min_freshness_rate": SCAN_DEGRADED_MIN_FRESH_RATE,
    "max_omitted_symbols": SCAN_DEGRADED_MAX_OMITTED_SYMBOLS,
},
```

- [ ] Allow stale-tail omission only after all stale market entries have been built and the scan-level totals pass:

```python
can_omit_tail = (
    allow_stale_tail
    and not unresolved_symbols
    and not hard_block_reasons
    and omitted_count > 0
    and omitted_count <= SCAN_DEGRADED_MAX_OMITTED_SYMBOLS
    and freshness_rate >= SCAN_DEGRADED_MIN_FRESH_RATE
)
```

`hard_block_reasons` includes calendar failure, missing/running refresh state, stale refresh state, and any other unverifiable condition.

- [ ] Produce one durable warning for the scan when `can_omit_tail` is true:

```python
FreshnessOmissionWarning(
    code="market_data_stale_tail_omitted",
    message=(
        f"Omitted {omitted_count} stale symbols from this broad scan "
        f"({freshness_rate:.2%} fresh)."
    ),
    ...
)
```

- [ ] Preserve current fallback behavior for missing refresh state only when every requested symbol date is already fresh. Missing state plus stale rows must block, because the scan cannot prove refresh completion.

- [ ] Run:

```bash
cd backend
source venv/bin/activate
pytest tests/unit/services/test_market_data_freshness.py -q
```

## Task 4: Apply Decision in `CreateScanUseCase`

Edit `backend/app/use_cases/scanning/create_scan.py`.

- [ ] Replace the old checker type and constructor argument with a protocol that supports the richer evaluator. Rename the instance attribute from `_freshness_checker` to `_freshness_evaluator` at the same time:

```python
from typing import Protocol
from app.domain.scanning.models import FreshnessDecision


class FreshnessEvaluator(Protocol):
    def __call__(
        self,
        symbols: Iterable[str],
        *,
        allow_stale_tail: bool,
    ) -> FreshnessDecision:
        raise NotImplementedError
```

Constructor target:

```python
def __init__(
    self,
    dispatcher: TaskDispatcher,
    *,
    freshness_evaluator: FreshnessEvaluator | None = None,
) -> None:
    self._dispatcher = dispatcher
    self._freshness_evaluator = freshness_evaluator
```

- [ ] Add broad universe policy:

```python
BROAD_STALE_TAIL_UNIVERSE_TYPES = {
    UniverseType.ALL.value,
    UniverseType.MARKET.value,
    UniverseType.EXCHANGE.value,
    UniverseType.INDEX.value,
}
```

- [ ] Extend `CreateScanResult`:

```python
warnings: tuple[dict[str, object], ...] = ()
```

- [ ] Run the evaluator after symbol resolution and before feature-run hashing:

```python
freshness_warnings: tuple[dict[str, object], ...] = ()
if self._freshness_evaluator is not None:
    decision = self._freshness_evaluator(
        symbols,
        allow_stale_tail=cmd.universe_type in BROAD_STALE_TAIL_UNIVERSE_TYPES,
    )
    if decision.blocking_detail is not None:
        raise StaleMarketDataError(decision.blocking_detail)
    symbols = list(decision.symbols_to_scan)
    freshness_warnings = tuple(w.to_dict() for w in decision.warnings)
```

- [ ] Validate that omission cannot produce an empty scan:

```python
if not symbols:
    raise ValidationError(
        f"No fresh symbols remain for universe '{cmd.universe_label}'. "
        "Refresh market data before starting a scan."
    )
```

- [ ] Persist warnings on scan creation:

```python
warnings=list(freshness_warnings),
```

- [ ] Use filtered `symbols` everywhere below the evaluator:

  - `hash_universe_symbols(symbols)`
  - `uow.feature_runs.find_latest_published_covering(symbols=symbols, ...)`
  - `total_stocks=len(symbols)`
  - `uow.scans.update_status(... total_stocks=len(symbols), ...)`
  - `self._dispatcher.dispatch_scan(scan_id, symbols, ...)`

- [ ] For idempotent duplicate returns, include existing warnings:

```python
warnings=tuple(getattr(existing, "warnings", None) or ()),
```

- [ ] For instant exact-match and compile-path returns, include `freshness_warnings` in `CreateScanResult`.

- [ ] Update `backend/app/wiring/bootstrap.py` to inject `evaluate_symbol_freshness` into HTTP-bound scan creation:

```python
from app.services.market_data_freshness import evaluate_symbol_freshness

return CreateScanUseCase(
    dispatcher=get_task_dispatcher(),
    freshness_evaluator=evaluate_symbol_freshness,
)
```

Keep `get_create_scan_use_case_without_freshness_gate()` using `freshness_evaluator=None`.

## Task 5: Persist Scan Warnings

Add a migration and model support.

- [ ] Create `backend/alembic/versions/20260701_0024_add_scan_warnings.py`:

```python
"""add scan warnings

Revision ID: 20260701_0024
Revises: 20260627_0023
Create Date: 2026-07-01
"""

from alembic import op
import sqlalchemy as sa


revision = "20260701_0024"
down_revision = "20260627_0023"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "scans",
        sa.Column(
            "warnings",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'[]'::json"),
        ),
    )


def downgrade() -> None:
    op.drop_column("scans", "warnings")
```

- [ ] Edit `backend/app/models/scan_result.py`:

```python
warnings = Column(JSON, nullable=False, default=list, server_default=text("'[]'::json"))
```

- [ ] Edit `backend/tests/unit/use_cases/conftest.py`:

```python
warnings: list[dict[str, object]] | None = None
```

Add it to `FakeScan` with a default factory if converting the dataclass is acceptable; otherwise default to `None` and normalize in tests with `getattr(..., "warnings", [])`.

- [ ] Ensure existing scan creation callers that do not pass warnings still work:

```bash
cd backend
source venv/bin/activate
pytest tests/unit/use_cases/test_create_scan.py tests/unit/test_scan_crud_endpoints.py -q
```

## Task 6: Add Use Case Tests

Edit `backend/tests/unit/use_cases/test_create_scan.py`.

- [ ] Add `test_broad_universe_omits_stale_tail_before_dispatch`.

Use a fake evaluator returning:

```python
FreshnessDecision(
    symbols_to_scan=("AAPL", "MSFT"),
    warnings=(
        FreshnessOmissionWarning(
            code="market_data_stale_tail_omitted",
            message="Omitted 1 stale symbol from this broad scan (66.67% fresh).",
            markets=("US",),
            omitted_symbols=("LHSW",),
            omitted_count=1,
            total_symbols=3,
            fresh_count=2,
            freshness_rate=0.666667,
            expected_dates={"US": "2026-06-18"},
            oldest_last_cached_dates={"US": "2026-05-13"},
        ),
    ),
)
```

Command uses `universe_type="market"`. Expected:

```python
assert dispatcher.dispatched[0][1] == ["AAPL", "MSFT"]
assert uow.scans.rows[0].total_stocks == 2
assert uow.scans.rows[0].warnings[0]["omitted_symbols"] == ["LHSW"]
assert result.warnings[0]["omitted_count"] == 1
```

- [ ] Add `test_custom_universe_does_not_allow_stale_tail`.

The fake evaluator should capture `allow_stale_tail`. Command uses `universe_type="custom"`. Expected `allow_stale_tail is False`.

- [ ] Add `test_market_universe_allows_stale_tail`.

Command uses `universe_type="market"`. Expected `allow_stale_tail is True`.

- [ ] Add `test_idempotent_duplicate_returns_existing_warnings`.

Create a scan with a warning, then retry the same idempotency key. Expected duplicate result includes the warning and the evaluator is not called.

- [ ] Add `test_feature_run_hash_uses_filtered_symbols`.

Set up a published feature run whose `universe_hash` matches `["AAPL", "MSFT"]`, not `["AAPL", "LHSW", "MSFT"]`. Use a degraded decision that omits `LHSW`. Expected exact match completes instantly and dispatch is not called.

## Task 7: Return Warnings Through APIs and Snapshots

Edit `backend/app/schemas/scanning.py`.

- [ ] Add warnings fields:

```python
warnings: List[dict[str, Any]] = Field(default_factory=list)
```

to:

- `ScanCreateResponse`
- `ScanStatusResponse`
- `ScanListItem`

Edit `backend/app/api/v1/scans.py`.

- [ ] Include warnings in `list_scans`:

```python
warnings=getattr(scan, "warnings", None) or [],
```

- [ ] Include warnings in `create_scan` response:

```python
warnings=list(result.warnings),
```

- [ ] Include warnings in `get_scan_status`:

```python
warnings=getattr(scan, "warnings", None) or [],
```

Edit `backend/app/services/ui_snapshot_service.py`.

- [ ] Include warnings on every `ScanListItem` and `ScanStatusResponse` created there.

- [ ] Add/update tests:

```bash
cd backend
source venv/bin/activate
pytest \
  tests/unit/test_scan_create_endpoint.py \
  tests/unit/test_scan_crud_endpoints.py \
  tests/unit/test_ui_snapshot_service.py \
  -q
```

Expected API shape:

```json
{
  "scan_id": "scan-123",
  "status": "queued",
  "total_stocks": 9873,
  "warnings": [
    {
      "code": "market_data_stale_tail_omitted",
      "omitted_count": 100,
      "freshness_rate": 0.99,
      "omitted_symbols": ["LHSW"]
    }
  ]
}
```

## Task 8: Surface Durable Warning in the Scan UI

Edit `frontend/src/features/scan/pages/ScanPageContainer.jsx`.

- [ ] Add state:

```javascript
const [scanWarnings, setScanWarnings] = useState([]);
```

- [ ] In `applyScanBootstrapSnapshot`, set warnings from `payload.selected_scan_status?.warnings`, falling back to `payload.selected_scan?.warnings`:

```javascript
setScanWarnings(
  payload.selected_scan_status?.warnings
  ?? payload.selected_scan?.warnings
  ?? []
);
```

- [ ] In `handleLoadScan('')`, clear warnings.

- [ ] When loading from history, seed warnings from the matching history item.

- [ ] In `createScanMutation.onSuccess`, set warnings from `data.warnings ?? []`.

- [ ] In the `statusData` effect, set warnings from `statusData.warnings ?? scanWarnings`.

- [ ] Pass `scanWarnings` into `ScanControlBar`.

Edit `frontend/src/features/scan/components/ScanControlBar.jsx`.

- [ ] Accept `scanWarnings = []`.

- [ ] Render a warning alert below the toolbar for `market_data_stale_tail_omitted` warnings:

```jsx
{scanWarnings.map((warning) => (
  <Alert key={`${warning.code}-${warning.omitted_count}`} severity="warning" sx={{ mt: 1 }}>
    {warning.message}
  </Alert>
))}
```

Keep the stale-data error alert unchanged for strict blocks.

- [ ] Add/adjust tests in `frontend/src/pages/ScanPage.test.jsx` or the closest existing scan page test:

  - create response with warnings renders the warning
  - bootstrap/status warnings render after reload
  - selecting "New Scan" clears the warning

- [ ] Run:

```bash
cd frontend
npm run test:run -- ScanPage
```

## Task 9: Full Verification

- [ ] Backend targeted tests:

```bash
cd backend
source venv/bin/activate
pytest \
  tests/unit/services/test_market_data_freshness.py \
  tests/unit/use_cases/test_create_scan.py \
  tests/unit/test_scan_create_endpoint.py \
  tests/unit/test_scan_crud_endpoints.py \
  tests/unit/test_ui_snapshot_service.py \
  -q
```

- [ ] Backend broader scan tests:

```bash
cd backend
source venv/bin/activate
pytest tests/unit/test_scan_enqueue.py tests/unit/test_scan_execution_service.py tests/unit/use_cases/test_run_bulk_scan.py -q
```

- [ ] Frontend tests:

```bash
cd frontend
npm run test:run -- ScanPage
npm run lint
```

- [ ] If a local database is available, run:

```bash
cd backend
source venv/bin/activate
alembic upgrade head
```

- [ ] Manual acceptance check:

  - Simulate a broad US universe with 9,900 fresh and 100 stale active symbols.
  - POST `/api/v1/scans`.
  - Expected: HTTP 200, queued/completed scan, `total_stocks == 9900`, warning says 100 omitted.
  - Confirm the background dispatch receives 9,900 symbols.
  - Confirm one of the omitted symbols has no `scan_results` row for that scan.

- [ ] Strict acceptance check:

  - POST a custom scan containing one stale symbol.
  - Expected: HTTP 409 `market_data_stale`.
  - Response detail includes `sample_stale_symbols`.

## Rollout Notes

- This change does not make stale DB rows fresh. It prevents a tiny failed-refresh tail from blocking broad scans.
- It intentionally keeps strict user-selected symbol scans strict.
- It intentionally keeps refresh completion and scan freshness as separate policies.
- Omitted symbols are not hidden: they are excluded from execution and recorded on the scan row, status API, list API, bootstrap snapshot, and UI.

## Completion Checklist

- [ ] All tests in Task 9 pass.
- [ ] `bd show stockscreenclaude-14y` reflects current status.
- [ ] Commit the implementation and beads updates.
- [ ] Push the branch.
- [ ] Close `stockscreenclaude-14y` only after the implementation, tests, and push are complete.
