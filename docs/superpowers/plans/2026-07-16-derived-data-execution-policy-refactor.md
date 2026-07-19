# Derived-Data Execution Policy Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace breadth/group cache-mode flag combinations and ad-hoc diagnostics with typed execution policies and result contracts while preserving every existing Celery call and issue-301 behavior.

**Architecture:** A pure policy resolver converts serialized task inputs and legacy booleans into one immutable in-process policy. Breadth calculations use a service-owned coverage accumulator/report, while group calculations return immutable typed prefetch statistics together with rankings and delegate input loading to a focused loader. Celery task dictionaries remain backward-compatible serialization boundaries.

**Tech Stack:** Python 3.12, dataclasses, string enums, Celery, SQLAlchemy, pandas, pytest, `unittest.mock`, Beads (`bd`).

## Global Constraints

- The price-refresh guard's 90% acceptance threshold does not change.
- Existing Celery keywords `force_cache_only` and
  `refresh_guarded_cache_only` remain accepted.
- `force_cache_only=True` wins over every other compatibility input.
- New internal orchestration passes
  `execution_policy="refresh_guarded"` rather than the legacy guarded boolean.
- `auto` historical requests retain provider fallback.
- `auto` same-day requests retain cache-only warmup validation.
- `strict_cache_only` retains static/manual strict completeness behavior.
- `refresh_guarded` remains provider-free and tolerates individual missing or
  insufficient histories when useful output exists.
- External task result keys remain compatible.
- Existing Celery task names remain unchanged.
- Cache-miss samples are sorted and limited to 20 symbols.
- Production implementation uses no mutable diagnostics output parameter.
- Production and tests are changed test-first in red-green-refactor cycles.
- Commits use `--no-verify` until Bead `stockscreenclaude-7jr` repairs the
  existing hook export-path failure.

---

## File Structure

### New production modules

- `backend/app/services/derived_data_execution_policy.py`
  - serialized mode enum, immutable resolved policy, legacy-compatible resolver.
- `backend/app/services/breadth_coverage.py`
  - breadth coverage accumulator, immutable report, typed daily calculation
    result.
- `backend/app/services/group_rank_models.py`
  - immutable group prefetch statistics, prefetch payload, calculation result.
- `backend/app/services/group_rank_input_loader.py`
  - group benchmark, universe, constituent, market-cap, and price prefetching.
- `backend/app/tasks/group_rank_backfill_tasks.py`
  - manual range, gap-fill, and one-year group-ranking Celery tasks.

### Modified production modules

- `backend/app/services/breadth_calculator_service.py`
  - consumes resolved policy, owns coverage recording, returns typed daily
    result.
- `backend/app/tasks/breadth_tasks.py`
  - resolves compatibility inputs once, validates typed coverage, serializes
    external result dictionaries.
- `backend/app/services/ibd_group_rank_service.py`
  - consumes resolved policy, delegates prefetch, returns typed calculation
    result.
- `backend/app/tasks/group_rank_tasks.py`
  - resolves compatibility inputs once, consumes typed group result, re-exports
    extracted manual tasks.
- `backend/app/tasks/daily_market_pipeline_tasks.py`
  - emits the new serialized policy.

### New or split tests

- `backend/tests/unit/test_derived_data_execution_policy.py`
- `backend/tests/unit/test_breadth_coverage.py`
- `backend/tests/unit/test_group_rank_models.py`
- `backend/tests/unit/test_group_rank_input_loader.py`
- `backend/tests/unit/test_group_rank_execution_policy.py`
- `backend/tests/unit/test_group_rank_backfill_tasks.py`

Existing breadth/group task and service tests retain formula, persistence,
activity, retry, and API-contract cases that belong to those modules.

---

### Task 1: Add the typed execution policy and legacy resolver

**Files:**
- Create: `backend/app/services/derived_data_execution_policy.py`
- Create: `backend/tests/unit/test_derived_data_execution_policy.py`

**Interfaces:**
- Consumes: serialized `execution_policy: str | None`, legacy booleans, target
  date, current market date, and warmup-bypass state.
- Produces:
  `resolve_derived_data_execution_policy(...) -> DerivedDataExecutionPolicy`.

- [ ] **Step 1: Write the policy-matrix tests**

Create `backend/tests/unit/test_derived_data_execution_policy.py`:

```python
from datetime import date

import pytest

from app.services.derived_data_execution_policy import (
    DerivedDataExecutionMode,
    resolve_derived_data_execution_policy,
)


TODAY = date(2026, 7, 16)
HISTORICAL = date(2026, 7, 15)


@pytest.mark.parametrize(
    ("requested", "target", "cache_only", "strict", "warmup", "partial"),
    [
        (None, HISTORICAL, False, False, False, False),
        ("auto", HISTORICAL, False, False, False, False),
        ("auto", TODAY, True, True, True, False),
        ("strict_cache_only", HISTORICAL, True, True, False, False),
        ("strict_cache_only", TODAY, True, True, False, False),
        ("refresh_guarded", HISTORICAL, True, False, False, True),
        ("refresh_guarded", TODAY, True, False, False, True),
    ],
)
def test_policy_matrix(
    requested,
    target,
    cache_only,
    strict,
    warmup,
    partial,
):
    policy = resolve_derived_data_execution_policy(
        execution_policy=requested,
        target_date=target,
        current_date=TODAY,
    )

    assert policy.cache_only is cache_only
    assert policy.strict_completeness is strict
    assert policy.requires_warmup_metadata is warmup
    assert policy.tolerates_partial_coverage is partial


def test_force_cache_only_has_legacy_precedence():
    policy = resolve_derived_data_execution_policy(
        execution_policy="refresh_guarded",
        force_cache_only=True,
        refresh_guarded_cache_only=True,
        target_date=HISTORICAL,
        current_date=TODAY,
    )

    assert policy.mode is DerivedDataExecutionMode.STRICT_CACHE_ONLY


def test_legacy_guarded_flag_maps_to_guarded_mode():
    policy = resolve_derived_data_execution_policy(
        refresh_guarded_cache_only=True,
        target_date=HISTORICAL,
        current_date=TODAY,
    )

    assert policy.mode is DerivedDataExecutionMode.REFRESH_GUARDED


def test_same_day_auto_bypass_removes_warmup_requirement():
    policy = resolve_derived_data_execution_policy(
        target_date=TODAY,
        current_date=TODAY,
        allow_same_day_warmup_bypass=True,
    )

    assert policy.mode is DerivedDataExecutionMode.AUTO
    assert policy.cache_only is True
    assert policy.strict_completeness is True
    assert policy.requires_warmup_metadata is False


def test_invalid_serialized_policy_is_rejected():
    with pytest.raises(ValueError, match="Unknown derived-data execution policy"):
        resolve_derived_data_execution_policy(
            execution_policy="provider_if_maybe",
            target_date=HISTORICAL,
            current_date=TODAY,
        )
```

- [ ] **Step 2: Run the new tests and verify the missing module failure**

Run:

```bash
cd backend
/Users/admin/StockScreenClaude/backend/venv/bin/python -m pytest \
  tests/unit/test_derived_data_execution_policy.py -q
```

Expected: collection fails with
`ModuleNotFoundError: No module named 'app.services.derived_data_execution_policy'`.

- [ ] **Step 3: Implement the immutable policy and resolver**

Create `backend/app/services/derived_data_execution_policy.py`:

```python
"""Execution policy shared by breadth and group derived-data calculations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum


class DerivedDataExecutionMode(str, Enum):
    AUTO = "auto"
    STRICT_CACHE_ONLY = "strict_cache_only"
    REFRESH_GUARDED = "refresh_guarded"


@dataclass(frozen=True)
class DerivedDataExecutionPolicy:
    mode: DerivedDataExecutionMode
    cache_only: bool
    strict_completeness: bool
    requires_warmup_metadata: bool
    tolerates_partial_coverage: bool

    @classmethod
    def provider_allowed(cls) -> "DerivedDataExecutionPolicy":
        return cls(
            mode=DerivedDataExecutionMode.AUTO,
            cache_only=False,
            strict_completeness=False,
            requires_warmup_metadata=False,
            tolerates_partial_coverage=False,
        )


def resolve_derived_data_execution_policy(
    *,
    target_date: date,
    current_date: date,
    execution_policy: str | DerivedDataExecutionMode | None = None,
    force_cache_only: bool = False,
    refresh_guarded_cache_only: bool = False,
    allow_same_day_warmup_bypass: bool = False,
) -> DerivedDataExecutionPolicy:
    if force_cache_only:
        mode = DerivedDataExecutionMode.STRICT_CACHE_ONLY
    elif refresh_guarded_cache_only:
        mode = DerivedDataExecutionMode.REFRESH_GUARDED
    elif execution_policy is None:
        mode = DerivedDataExecutionMode.AUTO
    else:
        try:
            mode = DerivedDataExecutionMode(execution_policy)
        except ValueError as exc:
            raise ValueError(
                f"Unknown derived-data execution policy: {execution_policy}"
            ) from exc

    if mode is DerivedDataExecutionMode.STRICT_CACHE_ONLY:
        return DerivedDataExecutionPolicy(
            mode=mode,
            cache_only=True,
            strict_completeness=True,
            requires_warmup_metadata=False,
            tolerates_partial_coverage=False,
        )

    if mode is DerivedDataExecutionMode.REFRESH_GUARDED:
        return DerivedDataExecutionPolicy(
            mode=mode,
            cache_only=True,
            strict_completeness=False,
            requires_warmup_metadata=False,
            tolerates_partial_coverage=True,
        )

    same_day = target_date == current_date
    return DerivedDataExecutionPolicy(
        mode=mode,
        cache_only=same_day,
        strict_completeness=same_day,
        requires_warmup_metadata=(
            same_day and not allow_same_day_warmup_bypass
        ),
        tolerates_partial_coverage=False,
    )
```

- [ ] **Step 4: Run the policy tests**

Run the command from Step 2.

Expected: `11 passed`.

- [ ] **Step 5: Commit the policy contract**

```bash
git add \
  backend/app/services/derived_data_execution_policy.py \
  backend/tests/unit/test_derived_data_execution_policy.py
git commit --no-verify -m "refactor: add derived-data execution policy"
```

---

### Task 2: Add authoritative breadth coverage types

**Files:**
- Create: `backend/app/services/breadth_coverage.py`
- Create: `backend/tests/unit/test_breadth_coverage.py`

**Interfaces:**
- Consumes: candidate/cache-miss symbol batches and per-observation outcomes.
- Produces:
  `BreadthCoverageAccumulator.report() -> BreadthCoverageReport` and
  `BreadthCalculationResult.to_metrics_dict() -> dict[str, Any]`.

- [ ] **Step 1: Write coverage identity, determinism, and serializer tests**

Create `backend/tests/unit/test_breadth_coverage.py`:

```python
from app.services.breadth_coverage import (
    BreadthCalculationResult,
    BreadthCoverageAccumulator,
)


def _build_report(batch_order):
    coverage = BreadthCoverageAccumulator()
    for candidates, misses in batch_order:
        coverage.record_price_batch(candidates, misses)
    coverage.record_scanned()
    coverage.record_insufficient()
    coverage.record_error()
    return coverage.report()


def test_report_derives_counts_from_symbol_identity():
    report = _build_report([
        (["AAA", "MISS2"], ["MISS2"]),
        (["BBB", "MISS1"], ["MISS1"]),
    ])

    assert report.candidate_stocks == 4
    assert report.symbols_with_cached_history == 2
    assert report.cache_miss_stocks == 2
    assert report.cache_coverage_ratio == 0.5
    assert report.total_stocks_scanned == 1
    assert report.skipped_stocks == 2
    assert report.insufficient_data_stocks == 1
    assert report.error_stocks == 1
    assert report.insufficient_history_observations == 1


def test_cache_miss_sample_is_deterministic_across_batch_order():
    forward = _build_report([
        (["ZZZ", "AAA"], ["ZZZ"]),
        (["MMM", "BBB"], ["MMM"]),
    ])
    reverse = _build_report([
        (["MMM", "BBB"], ["MMM"]),
        (["ZZZ", "AAA"], ["ZZZ"]),
    ])

    assert forward.cache_miss_symbols_sample == ("MMM", "ZZZ")
    assert reverse.cache_miss_symbols_sample == forward.cache_miss_symbols_sample


def test_daily_and_backfill_serializers_share_one_report():
    report = _build_report([
        (["AAA", "BBB"], ["BBB"]),
    ])

    assert report.to_daily_dict() == {
        "candidate_stocks": 2,
        "total_stocks_scanned": 1,
        "symbols_with_cached_history": 1,
        "skipped_stocks": 2,
        "cache_miss_stocks": 1,
        "insufficient_data_stocks": 1,
        "error_stocks": 1,
        "cache_coverage_ratio": 0.5,
        "cache_miss_symbols_sample": ["BBB"],
    }
    assert report.to_backfill_dict() == {
        "target_symbols": 2,
        "symbols_with_cached_history": 1,
        "cache_miss_stocks": 1,
        "cache_miss_symbols_sample": ["BBB"],
        "cache_coverage_ratio": 0.5,
        "insufficient_history_observations": 1,
    }


def test_calculation_result_serializes_indicators_and_coverage():
    coverage = BreadthCoverageAccumulator()
    coverage.record_price_batch(["AAA"], [])
    coverage.record_scanned()
    result = BreadthCalculationResult(
        indicators={"stocks_up_4pct": 1, "ratio_5day": None},
        coverage=coverage.report(),
    )

    assert result.to_metrics_dict() == {
        "stocks_up_4pct": 1,
        "ratio_5day": None,
        "candidate_stocks": 1,
        "total_stocks_scanned": 1,
        "symbols_with_cached_history": 1,
        "skipped_stocks": 0,
        "cache_miss_stocks": 0,
        "insufficient_data_stocks": 0,
        "error_stocks": 0,
        "cache_coverage_ratio": 1.0,
        "cache_miss_symbols_sample": [],
    }
```

- [ ] **Step 2: Run the new tests and verify the missing module failure**

```bash
cd backend
/Users/admin/StockScreenClaude/backend/venv/bin/python -m pytest \
  tests/unit/test_breadth_coverage.py -q
```

Expected: collection fails with
`ModuleNotFoundError: No module named 'app.services.breadth_coverage'`.

- [ ] **Step 3: Implement the accumulator, immutable report, and result**

Create `backend/app/services/breadth_coverage.py`:

```python
"""Authoritative cache-coverage accounting for breadth calculations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping


CACHE_MISS_SYMBOL_SAMPLE_LIMIT = 20


@dataclass(frozen=True)
class BreadthCoverageReport:
    candidate_stocks: int
    symbols_with_cached_history: int
    cache_miss_stocks: int
    cache_miss_symbols_sample: tuple[str, ...]
    cache_coverage_ratio: float
    total_stocks_scanned: int
    skipped_stocks: int
    insufficient_data_stocks: int
    error_stocks: int
    insufficient_history_observations: int

    def to_daily_dict(self) -> dict[str, Any]:
        return {
            "candidate_stocks": self.candidate_stocks,
            "total_stocks_scanned": self.total_stocks_scanned,
            "symbols_with_cached_history": self.symbols_with_cached_history,
            "skipped_stocks": self.skipped_stocks,
            "cache_miss_stocks": self.cache_miss_stocks,
            "insufficient_data_stocks": self.insufficient_data_stocks,
            "error_stocks": self.error_stocks,
            "cache_coverage_ratio": self.cache_coverage_ratio,
            "cache_miss_symbols_sample": list(self.cache_miss_symbols_sample),
        }

    def to_backfill_dict(self) -> dict[str, Any]:
        return {
            "target_symbols": self.candidate_stocks,
            "symbols_with_cached_history": self.symbols_with_cached_history,
            "cache_miss_stocks": self.cache_miss_stocks,
            "cache_miss_symbols_sample": list(self.cache_miss_symbols_sample),
            "cache_coverage_ratio": self.cache_coverage_ratio,
            "insufficient_history_observations": (
                self.insufficient_history_observations
            ),
        }


@dataclass
class BreadthCoverageAccumulator:
    _candidate_symbols: set[str] = field(default_factory=set)
    _cached_symbols: set[str] = field(default_factory=set)
    _cache_miss_symbols: set[str] = field(default_factory=set)
    _scanned: int = 0
    _skipped: int = 0
    _insufficient: int = 0
    _errors: int = 0
    _insufficient_observations: int = 0

    def record_price_batch(
        self,
        candidate_symbols: Iterable[str],
        cache_miss_symbols: Iterable[str],
    ) -> None:
        candidates = set(candidate_symbols)
        misses = set(cache_miss_symbols)
        self._candidate_symbols.update(candidates)
        self._cache_miss_symbols.update(misses)
        self._cached_symbols.update(candidates - misses)

    def record_scanned(self) -> None:
        self._scanned += 1

    def record_cache_miss(self) -> None:
        self._skipped += 1

    def record_insufficient(self) -> None:
        self._insufficient += 1
        self._insufficient_observations += 1
        self._skipped += 1

    def record_error(self) -> None:
        self._errors += 1
        self._skipped += 1

    def report(self) -> BreadthCoverageReport:
        candidate_count = len(self._candidate_symbols)
        cached_count = len(self._cached_symbols)
        return BreadthCoverageReport(
            candidate_stocks=candidate_count,
            symbols_with_cached_history=cached_count,
            cache_miss_stocks=len(self._cache_miss_symbols),
            cache_miss_symbols_sample=tuple(
                sorted(self._cache_miss_symbols)[
                    :CACHE_MISS_SYMBOL_SAMPLE_LIMIT
                ]
            ),
            cache_coverage_ratio=(
                cached_count / candidate_count if candidate_count else 0.0
            ),
            total_stocks_scanned=self._scanned,
            skipped_stocks=self._skipped,
            insufficient_data_stocks=self._insufficient,
            error_stocks=self._errors,
            insufficient_history_observations=self._insufficient_observations,
        )


@dataclass(frozen=True)
class BreadthCalculationResult:
    indicators: Mapping[str, Any]
    coverage: BreadthCoverageReport

    def to_metrics_dict(self) -> dict[str, Any]:
        return {
            **dict(self.indicators),
            **self.coverage.to_daily_dict(),
        }
```

- [ ] **Step 4: Run the coverage tests**

Run the command from Step 2.

Expected: `4 passed`.

- [ ] **Step 5: Commit the breadth coverage contract**

```bash
git add \
  backend/app/services/breadth_coverage.py \
  backend/tests/unit/test_breadth_coverage.py
git commit --no-verify -m "refactor: centralize breadth coverage"
```

---

### Task 3: Integrate typed coverage into breadth daily and backfill services

**Files:**
- Modify: `backend/app/services/breadth_calculator_service.py`
- Modify: `backend/tests/unit/test_breadth_calculator.py`
- Modify: `backend/tests/unit/test_breadth_calculator_service.py`

**Interfaces:**
- Consumes:
  `policy: DerivedDataExecutionPolicy =
  DerivedDataExecutionPolicy.provider_allowed()`.
- Produces:
  `calculate_daily_breadth(...) -> BreadthCalculationResult`; existing
  backfill/gap-fill dictionaries enriched by
  `BreadthCoverageReport.to_backfill_dict()`.

- [ ] **Step 1: Change daily service tests to require a typed result**

In `backend/tests/unit/test_breadth_calculator_service.py`, update the three
daily calculation tests to use:

```python
from app.services.derived_data_execution_policy import (
    DerivedDataExecutionMode,
    resolve_derived_data_execution_policy,
)


def _policy(mode: str, target: date):
    return resolve_derived_data_execution_policy(
        execution_policy=mode,
        target_date=target,
        current_date=date(2026, 3, 20),
    )
```

Replace direct `cache_only=True` calls with:

```python
result = calculator.calculate_daily_breadth(
    date(2026, 3, 20),
    policy=_policy("refresh_guarded", date(2026, 3, 20)),
)
metrics = result.to_metrics_dict()
```

Replace the historical `cache_only=False` call with:

```python
result = calculator.calculate_daily_breadth(
    date(2026, 3, 19),
    policy=_policy("auto", date(2026, 3, 19)),
)
metrics = result.to_metrics_dict()
```

Add this assertion to the cache-miss test:

```python
assert result.coverage.cache_miss_symbols_sample == ("MISS",)
```

Update `backend/tests/unit/test_breadth_calculator.py` to serialize the result:

```python
result = calculator.calculate_daily_breadth()
metrics = result.to_metrics_dict()
```

- [ ] **Step 2: Add a backfill sample-order regression**

Add to `backend/tests/unit/test_breadth_calculator_service.py`:

```python
def test_backfill_range_cache_sample_is_sorted_across_batches(monkeypatch):
    db = _make_db_session()
    active_stocks = [
        StockUniverse(
            symbol=f"S{index:03d}",
            market="US",
            is_active=True,
            status=UNIVERSE_STATUS_ACTIVE,
        )
        for index in range(501)
    ]
    db.add_all(active_stocks)
    db.commit()
    service = BreadthCalculatorService(db, MagicMock())
    full_history = _make_price_df(date(2026, 3, 20))

    responses = [
        (
            {
                stock.symbol: (
                    None if stock.symbol == "S499" else full_history
                )
                for stock in active_stocks[:500]
            },
            {"S499"},
        ),
        (
            {"S500": None},
            {"S500"},
        ),
    ]
    monkeypatch.setattr(
        service,
        "_load_price_data_for_batch",
        MagicMock(side_effect=responses),
    )
    monkeypatch.setattr(service, "_store_breadth_records", MagicMock())

    result = service.backfill_range(
        date(2026, 3, 20),
        date(2026, 3, 20),
        trading_dates=[date(2026, 3, 20)],
        policy=_policy("refresh_guarded", date(2026, 3, 20)),
    )

    assert result["cache_miss_symbols_sample"] == ["S499", "S500"]
```

- [ ] **Step 3: Run the affected breadth service tests and verify signature failures**

```bash
cd backend
/Users/admin/StockScreenClaude/backend/venv/bin/python -m pytest \
  tests/unit/test_breadth_calculator.py \
  tests/unit/test_breadth_calculator_service.py -q
```

Expected: failures because `policy` is not accepted and the daily method still
returns a dictionary.

- [ ] **Step 4: Replace breadth metric-owned coverage with accumulators**

In `backend/app/services/breadth_calculator_service.py`:

```python
from .breadth_coverage import (
    BreadthCalculationResult,
    BreadthCoverageAccumulator,
)
from .derived_data_execution_policy import DerivedDataExecutionPolicy
```

Remove `CACHE_MISS_SYMBOL_SAMPLE_LIMIT` and remove all coverage keys from
`_empty_metrics()`, leaving indicator keys plus `ratio_5day` and `ratio_10day`.

Change the daily signature:

```python
def calculate_daily_breadth(
    self,
    calculation_date: date = None,
    *,
    policy: DerivedDataExecutionPolicy = (
        DerivedDataExecutionPolicy.provider_allowed()
    ),
) -> BreadthCalculationResult:
```

Initialize `coverage = BreadthCoverageAccumulator()`. For each loaded batch:

```python
coverage.record_price_batch(batch_symbols, cache_miss_symbols)
```

Use `policy.cache_only` in `_load_price_data_for_batch`. Record outcomes:

```python
if price_history is None or price_history.empty:
    coverage.record_cache_miss()
    continue

if stock_metrics is None:
    coverage.record_insufficient()
    continue

self._apply_stock_metrics(metrics, stock_metrics)
coverage.record_scanned()
```

and in the exception branch:

```python
coverage.record_error()
```

Return:

```python
return BreadthCalculationResult(
    indicators=metrics,
    coverage=coverage.report(),
)
```

- [ ] **Step 5: Use per-date and overall accumulators in backfill**

Change `backfill_range` and `fill_gaps` to accept the same keyword-only policy:

```python
policy: DerivedDataExecutionPolicy = (
    DerivedDataExecutionPolicy.provider_allowed()
)
```

Create:

```python
coverage_by_date = {
    calc_date: BreadthCoverageAccumulator()
    for calc_date in ordered_dates
}
overall_coverage = BreadthCoverageAccumulator()
```

After each batch load:

```python
overall_coverage.record_price_batch(
    batch_symbols,
    batch_cache_miss_symbols,
)
for coverage in coverage_by_date.values():
    coverage.record_price_batch(
        batch_symbols,
        batch_cache_miss_symbols,
    )
```

Record each date outcome on `coverage_by_date[calc_date]` and on
`overall_coverage`. Before ratio calculation/persistence, merge:

```python
metrics.update(
    coverage_by_date[calc_date].report().to_daily_dict()
)
```

Replace the existing cache-only result reconstruction with:

```python
if policy.cache_only:
    result.update(overall_coverage.report().to_backfill_dict())
```

Pass `policy=policy` from `fill_gaps` to `backfill_range`.

- [ ] **Step 6: Run breadth service tests**

Run the command from Step 3.

Expected: all selected tests pass.

- [ ] **Step 7: Commit the service integration**

```bash
git add \
  backend/app/services/breadth_calculator_service.py \
  backend/tests/unit/test_breadth_calculator.py \
  backend/tests/unit/test_breadth_calculator_service.py
git commit --no-verify -m "refactor: return typed breadth coverage"
```

---

### Task 4: Resolve breadth policy once at Celery boundaries

**Files:**
- Modify: `backend/app/tasks/breadth_tasks.py`
- Modify: `backend/app/tasks/daily_market_pipeline_tasks.py`
- Modify: `backend/tests/unit/test_breadth_tasks.py`
- Modify: `backend/tests/unit/test_daily_market_pipeline_tasks.py`

**Interfaces:**
- Public task signatures continue to accept legacy flags.
- New signatures accept `execution_policy: str | None`.
- Nested tasks receive only `execution_policy=policy.mode.value`.

- [ ] **Step 1: Update pipeline expectations to the new serialized mode**

In `backend/tests/unit/test_daily_market_pipeline_tasks.py`, replace both
guarded kwargs assertions with:

```python
assert signatures[2].kwargs == {
    "market": "HK",
    "calculation_date": "2026-03-16",
    "execution_policy": "refresh_guarded",
}
assert signatures[6].kwargs == {
    "market": "HK",
    "calculation_date": "2026-03-16",
    "execution_policy": "refresh_guarded",
}
```

- [ ] **Step 2: Extend existing compatibility and forwarding tests**

Replace `_breadth_metrics` in
`backend/tests/unit/test_breadth_tasks.py` with this typed helper and import the
two production result types:

```python
from app.services.breadth_coverage import (
    BreadthCalculationResult,
    BreadthCoverageReport,
)


def _breadth_result(
    *,
    scanned: int,
    skipped: int,
    misses: int,
    errors: int = 0,
) -> BreadthCalculationResult:
    candidates = scanned + skipped
    indicators = {
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
    }
    return BreadthCalculationResult(
        indicators=indicators,
        coverage=BreadthCoverageReport(
            candidate_stocks=candidates,
            symbols_with_cached_history=candidates - misses,
            cache_miss_stocks=misses,
            cache_miss_symbols_sample=(
                ("MISS",) if misses else ()
            ),
            cache_coverage_ratio=(
                (candidates - misses) / candidates
                if candidates
                else 0.0
            ),
            total_stocks_scanned=scanned,
            skipped_stocks=skipped,
            insufficient_data_stocks=max(
                skipped - misses - errors,
                0,
            ),
            error_stocks=errors,
            insufficient_history_observations=max(
                skipped - misses - errors,
                0,
            ),
        ),
    )
```

Use `_breadth_result(...)` for every mocked
`calculate_daily_breadth.return_value`, including the two raw dictionaries near
the same-day warmup and strict static-export tests.

In `test_guarded_historical_breadth_tolerates_cache_misses`, retain the legacy
task call and replace its service-call assertion with:

```python
call_kwargs = fake_calculator.calculate_daily_breadth.call_args.kwargs
assert call_kwargs["calculation_date"] == date(2026, 3, 19)
assert call_kwargs["policy"].mode.value == "refresh_guarded"
```

In
`test_guarded_breadth_wrapper_propagates_cache_only_to_gapfill_and_target`,
replace its service and nested-task assertions with:

```python
fill_kwargs = fake_calculator.fill_gaps.call_args.kwargs
assert fill_kwargs["policy"].mode.value == "refresh_guarded"
target_call.assert_called_once_with(
    market="US",
    calculation_date="2026-03-19",
    execution_policy="refresh_guarded",
)
```

Invoke this wrapper test with the new input:

```python
result = module.calculate_daily_breadth_with_gapfill.run(
    market="US",
    calculation_date="2026-03-19",
    execution_policy="refresh_guarded",
)
```

Update existing guarded tests to construct
`BreadthCalculationResult(indicators=..., coverage=...)` from the Task 2 types
instead of returning raw service metrics.

- [ ] **Step 3: Run task and pipeline tests and verify failures**

```bash
cd backend
/Users/admin/StockScreenClaude/backend/venv/bin/python -m pytest \
  tests/unit/test_breadth_tasks.py \
  tests/unit/test_daily_market_pipeline_tasks.py -q
```

Expected: failures because task signatures and service calls still use legacy
boolean branching.

- [ ] **Step 4: Resolve policy at the daily breadth task boundary**

Import:

```python
from ..services.breadth_coverage import BreadthCoverageReport
from ..services.derived_data_execution_policy import (
    DerivedDataExecutionMode,
    resolve_derived_data_execution_policy,
)
```

Append `execution_policy: str | None = None` to the public daily task
signature, preserving all existing parameter positions. Resolve after
`calc_date` and `today_local` are known:

```python
policy = resolve_derived_data_execution_policy(
    execution_policy=execution_policy,
    force_cache_only=force_cache_only,
    refresh_guarded_cache_only=refresh_guarded_cache_only,
    target_date=calc_date,
    current_date=today_local,
    allow_same_day_warmup_bypass=(
        _ALLOW_SAME_DAY_BREADTH_WARMUP_BYPASS.get()
    ),
)
```

Call:

```python
calculation = calculator.calculate_daily_breadth(
    calculation_date=calc_date,
    policy=policy,
)
metrics = calculation.to_metrics_dict()
coverage = calculation.coverage
```

Change strict and guarded validators to accept
`BreadthCoverageReport`. Delete `_breadth_cache_diagnostics`; serialize with
`coverage.to_daily_dict()`. Branch only on:

```python
if policy.cache_only:
    if policy.tolerates_partial_coverage:
        completeness_error = _validate_refresh_guarded_breadth(coverage)
    elif policy.requires_warmup_metadata:
        completeness_error = _validate_same_day_cache_only_breadth(
            calculator.price_cache,
            coverage,
            market=effective_market,
        )
    else:
        completeness_error = _validate_strict_cache_only_breadth(coverage)
```

Expose `cache_policy` only when:

```python
policy.mode is DerivedDataExecutionMode.REFRESH_GUARDED
```

- [ ] **Step 5: Normalize the breadth wrapper and pipeline**

Append `execution_policy: str | None = None` to
`calculate_daily_breadth_with_gapfill`. Resolve using `target_date` and the
market-local date. Pass `policy=policy` to `calculator.fill_gaps` and pass only:

```python
inner_kwargs["execution_policy"] = policy.mode.value
```

Change `_calculate_daily_breadth_in_process` to accept and forward only
`execution_policy`.

In `daily_market_pipeline_tasks.py`, emit:

```python
execution_policy="refresh_guarded"
```

for breadth and group wrapper signatures.

- [ ] **Step 6: Run breadth task and pipeline tests**

Run the command from Step 3.

Expected: all selected tests pass.

- [ ] **Step 7: Assert legacy policy names do not leak below the boundary**

Run:

```bash
rg -n "force_cache_only|refresh_guarded_cache_only" \
  backend/app/services/breadth_calculator_service.py
```

Expected: no output.

- [ ] **Step 8: Commit breadth orchestration**

```bash
git add \
  backend/app/tasks/breadth_tasks.py \
  backend/app/tasks/daily_market_pipeline_tasks.py \
  backend/tests/unit/test_breadth_tasks.py \
  backend/tests/unit/test_daily_market_pipeline_tasks.py
git commit --no-verify -m "refactor: normalize breadth execution policy"
```

---

### Task 5: Add typed group prefetch and calculation results

**Files:**
- Create: `backend/app/services/group_rank_models.py`
- Create: `backend/tests/unit/test_group_rank_models.py`
- Modify: `backend/app/services/ibd_group_rank_service.py`
- Modify: `backend/tests/unit/test_group_rank_service.py`

**Interfaces:**
- Produces:
  `GroupRankPrefetchStats`, `GroupRankPrefetchData`,
  `GroupRankCalculationResult`.
- Changes:
  `calculate_group_rankings(...) -> GroupRankCalculationResult`.

- [ ] **Step 1: Write model immutability and serialization tests**

Create `backend/tests/unit/test_group_rank_models.py`:

```python
from dataclasses import FrozenInstanceError

import pytest

from app.services.group_rank_models import GroupRankPrefetchStats
from app.services.group_rank_cache_policy import GroupRankCacheRequirement


def _stats():
    return GroupRankPrefetchStats(
        target_symbols=4,
        symbols_with_prices=3,
        cache_miss_symbols=1,
        cache_miss_symbols_sample=("MISS",),
        cache_coverage_ratio=0.75,
        benchmark_available=True,
        benchmark_cached=True,
        benchmark_symbol="SPY",
        benchmark_role="primary",
        market="US",
        cache_only=True,
        skipped_unsupported_symbols=0,
    )


def test_prefetch_stats_are_immutable():
    stats = _stats()

    with pytest.raises(FrozenInstanceError):
        stats.cache_miss_symbols = 2


def test_cache_requirement_returns_replaced_stats():
    stats = _stats()
    replaced = stats.with_cache_requirement(
        GroupRankCacheRequirement.minimum(0.8, reason="test"),
    )

    assert stats.cache_coverage_min is None
    assert replaced.cache_coverage_min == 0.8
    assert replaced.cache_requirement_reason == "test"


def test_prefetch_stats_preserve_external_dictionary_keys():
    assert _stats().to_dict() == {
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


def test_prefetch_stats_can_coerce_legacy_test_mapping():
    stats = GroupRankPrefetchStats.from_mapping({
        "target_symbols": 2,
        "symbols_with_prices": 1,
        "cache_miss_symbols": 1,
        "spy_cached": True,
    })

    assert stats.cache_coverage_ratio == 0.5
    assert stats.benchmark_available is True
    assert stats.cache_miss_symbols_sample == ()
```

- [ ] **Step 2: Run the new model tests and verify the missing module**

```bash
cd backend
/Users/admin/StockScreenClaude/backend/venv/bin/python -m pytest \
  tests/unit/test_group_rank_models.py -q
```

Expected: collection fails with
`ModuleNotFoundError: No module named 'app.services.group_rank_models'`.

- [ ] **Step 3: Implement the group model module**

Create `backend/app/services/group_rank_models.py`:

```python
"""Typed data contracts for group-ranking input and output."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Optional

import pandas as pd

from .group_rank_cache_policy import GroupRankCacheRequirement


@dataclass(frozen=True)
class GroupRankPrefetchStats:
    target_symbols: int
    symbols_with_prices: int
    cache_miss_symbols: int
    cache_miss_symbols_sample: tuple[str, ...]
    cache_coverage_ratio: float
    benchmark_available: bool
    benchmark_cached: bool
    benchmark_symbol: str
    benchmark_role: str
    market: str
    cache_only: bool
    skipped_unsupported_symbols: int
    cache_coverage_min: float | None = None
    cache_requirement_reason: str | None = None

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
    ) -> "GroupRankPrefetchStats":
        target_symbols = int(values.get("target_symbols", 0) or 0)
        symbols_with_prices = int(
            values.get("symbols_with_prices", 0) or 0
        )
        coverage = values.get("cache_coverage_ratio")
        return cls(
            target_symbols=target_symbols,
            symbols_with_prices=symbols_with_prices,
            cache_miss_symbols=int(
                values.get("cache_miss_symbols", 0) or 0
            ),
            cache_miss_symbols_sample=tuple(
                values.get("cache_miss_symbols_sample", ())
            ),
            cache_coverage_ratio=(
                float(coverage)
                if coverage is not None
                else (
                    symbols_with_prices / target_symbols
                    if target_symbols
                    else 1.0
                )
            ),
            benchmark_available=bool(
                values.get(
                    "benchmark_available",
                    values.get("spy_cached", False),
                )
            ),
            benchmark_cached=bool(
                values.get("benchmark_cached", False)
            ),
            benchmark_symbol=str(
                values.get("benchmark_symbol", "SPY")
            ),
            benchmark_role=str(
                values.get("benchmark_role", "primary")
            ),
            market=str(values.get("market", "US")).upper(),
            cache_only=bool(values.get("cache_only", False)),
            skipped_unsupported_symbols=int(
                values.get("skipped_unsupported_symbols", 0) or 0
            ),
            cache_coverage_min=values.get("cache_coverage_min"),
            cache_requirement_reason=values.get(
                "cache_requirement_reason"
            ),
        )

    def with_cache_requirement(
        self,
        requirement: GroupRankCacheRequirement,
    ) -> "GroupRankPrefetchStats":
        if not requirement.enabled:
            return self
        return replace(
            self,
            cache_coverage_min=requirement.min_coverage,
            cache_requirement_reason=requirement.reason,
        )

    def to_dict(self) -> dict[str, Any]:
        result = {
            "target_symbols": self.target_symbols,
            "symbols_with_prices": self.symbols_with_prices,
            "cache_miss_symbols": self.cache_miss_symbols,
            "cache_miss_symbols_sample": list(
                self.cache_miss_symbols_sample
            ),
            "cache_coverage_ratio": self.cache_coverage_ratio,
            "spy_cached": self.benchmark_available,
            "benchmark_cached": self.benchmark_cached,
            "benchmark_symbol": self.benchmark_symbol,
            "benchmark_role": self.benchmark_role,
            "market": self.market,
            "cache_only": self.cache_only,
            "skipped_unsupported_symbols": (
                self.skipped_unsupported_symbols
            ),
        }
        if self.cache_coverage_min is not None:
            result["cache_coverage_min"] = self.cache_coverage_min
        if self.cache_requirement_reason is not None:
            result["cache_requirement_reason"] = (
                self.cache_requirement_reason
            )
        return result


@dataclass(frozen=True)
class GroupRankPrefetchData:
    benchmark_prices: Optional[pd.DataFrame]
    prices_by_symbol: Mapping[str, Optional[pd.DataFrame]]
    active_symbols: frozenset[str]
    market_caps: Mapping[str, float]
    stats: GroupRankPrefetchStats
    symbols_by_group: Mapping[str, tuple[str, ...]]


@dataclass(frozen=True)
class GroupRankCalculationResult:
    rankings: tuple[Mapping[str, Any], ...]
    prefetch_stats: GroupRankPrefetchStats
```

- [ ] **Step 4: Run model tests**

Run the command from Step 2.

Expected: `4 passed`.

- [ ] **Step 5: Change service tests from output mutation to returned result**

In `backend/tests/unit/test_group_rank_service.py`, add:

```python
import inspect

from app.services.derived_data_execution_policy import (
    resolve_derived_data_execution_policy,
)
from app.services.group_rank_models import (
    GroupRankCalculationResult,
    GroupRankPrefetchData,
    GroupRankPrefetchStats,
)
```

Replace the guarded partial-cache assertion with:

```python
calculation = service.calculate_group_rankings(
    db_session,
    date(2026, 3, 20),
    market="US",
    policy=resolve_derived_data_execution_policy(
        execution_policy="refresh_guarded",
        target_date=date(2026, 3, 20),
        current_date=date(2026, 3, 20),
    ),
    cache_requirement=GroupRankCacheRequirement.disabled(),
)

assert len(calculation.rankings) == 1
assert calculation.rankings[0]["industry_group"] == "Software"
assert calculation.prefetch_stats.cache_miss_symbols == 1
assert calculation.prefetch_stats.cache_miss_symbols_sample == ("MISS",)
```

Delete the caller-owned `diagnostics` dictionary from this test. Add a source
contract test:

```python
def test_calculate_group_rankings_has_no_diagnostics_output_parameter():
    signature = inspect.signature(
        IBDGroupRankService.calculate_group_rankings
    )

    assert "diagnostics" not in signature.parameters
```

- [ ] **Step 6: Update `IBDGroupRankService` to return the typed result**

Import the new model types and `DerivedDataExecutionPolicy`. Remove the local
`GroupRankPrefetchData` class and remove the `diagnostics` parameter.

Use:

```python
def calculate_group_rankings(
    self,
    db: Session,
    calculation_date: date = None,
    *,
    market: str | None = None,
    policy: DerivedDataExecutionPolicy = (
        DerivedDataExecutionPolicy.provider_allowed()
    ),
    cache_requirement: GroupRankCacheRequirement = (
        GroupRankCacheRequirement.disabled()
    ),
) -> GroupRankCalculationResult:
```

Prefetch with `policy=policy`. Apply requirements immutably:

```python
prefetch_stats = prefetch.stats.with_cache_requirement(
    cache_requirement
)
```

Check typed properties rather than dictionary keys:

```python
if cache_requirement.enabled:
    if not prefetch_stats.benchmark_available:
        raise IncompleteGroupRankingCacheError(prefetch_stats)
    if (
        prefetch_stats.cache_coverage_ratio
        < cache_requirement.min_coverage
    ):
        raise IncompleteGroupRankingCacheError(prefetch_stats)
    if prefetch_stats.cache_miss_symbols:
        logger.warning(
            "Cache-only group ranking run has %d cache misses "
            "out of %d symbols (coverage %.1f%% >= %.1f%%)",
            prefetch_stats.cache_miss_symbols,
            prefetch_stats.target_symbols,
            prefetch_stats.cache_coverage_ratio * 100,
            cache_requirement.min_coverage * 100,
        )
```

Every no-ranking return is:

```python
return GroupRankCalculationResult(
    rankings=(),
    prefetch_stats=prefetch_stats,
)
```

After persistence, return:

```python
return GroupRankCalculationResult(
    rankings=tuple(group_metrics),
    prefetch_stats=prefetch_stats,
)
```

Update `IncompleteGroupRankingCacheError`:

```python
class IncompleteGroupRankingCacheError(RuntimeError):
    def __init__(self, stats: GroupRankPrefetchStats):
        self.stats = stats
        market_suffix = (
            f" for {stats.market}" if stats.market else ""
        )
        if not stats.benchmark_available:
            reason = (
                f"{stats.benchmark_symbol} benchmark data is "
                f"missing from cache{market_suffix}"
            )
        else:
            reason = (
                f"{stats.cache_miss_symbols} symbols are missing "
                "cached price data"
            )
        super().__init__(reason)
```

Task serialization calls `e.stats.to_dict()`.

Update internal `backfill_rankings` and `fill_gaps` callers:

```python
calculation = self.calculate_group_rankings(...)
if calculation.rankings:
    ...
```

Retain `_coerce_prefetch_data` for legacy tuple-based tests, but convert the
stats mapping explicitly:

```python
return GroupRankPrefetchData(
    benchmark_prices=benchmark_prices,
    prices_by_symbol=prices_by_symbol,
    active_symbols=frozenset(active_symbols),
    market_caps=market_caps,
    stats=GroupRankPrefetchStats.from_mapping(stats),
    symbols_by_group={},
)
```

Update optimized result dictionaries to use:

```python
"prefetch_stats": prefetch.stats.to_dict()
```

- [ ] **Step 7: Run group model and service tests**

```bash
cd backend
/Users/admin/StockScreenClaude/backend/venv/bin/python -m pytest \
  tests/unit/test_group_rank_models.py \
  tests/unit/test_group_rank_service.py -q
```

Expected: all selected tests pass.

- [ ] **Step 8: Commit the typed group result**

```bash
git add \
  backend/app/services/group_rank_models.py \
  backend/app/services/ibd_group_rank_service.py \
  backend/tests/unit/test_group_rank_models.py \
  backend/tests/unit/test_group_rank_service.py
git commit --no-verify -m "refactor: return typed group ranking results"
```

---

### Task 6: Extract group input loading from the ranking service

**Files:**
- Create: `backend/app/services/group_rank_input_loader.py`
- Create: `backend/tests/unit/test_group_rank_input_loader.py`
- Modify: `backend/app/services/ibd_group_rank_service.py`
- Modify: `backend/tests/unit/test_group_rank_service.py`

**Interfaces:**
- Consumes:
  `GroupRankInputLoader.load(db, market, policy)`.
- Produces: `GroupRankPrefetchData`.

- [ ] **Step 1: Move provider-exclusion expectations to loader tests**

Create `backend/tests/unit/test_group_rank_input_loader.py`. Move and adapt
these existing service tests:

- cached-only stock and benchmark reads;
- missing market benchmark;
- cached fallback benchmark;
- stale cache treated as missing;
- provider-capable historical reads;
- unsupported suffix exclusion.

Construct the loader directly:

```python
loader = GroupRankInputLoader(
    price_cache=price_cache,
    benchmark_cache=benchmark_cache,
)
prefetch = loader.load(
    db_session,
    market="US",
    policy=resolve_derived_data_execution_policy(
        execution_policy="refresh_guarded",
        target_date=date(2026, 3, 20),
        current_date=date(2026, 3, 20),
    ),
)
```

Use typed assertions:

```python
assert prefetch.stats.cache_miss_symbols == 1
assert prefetch.stats.cache_miss_symbols_sample == ("MISS",)
assert prefetch.stats.benchmark_cached is True
```

The guarded test must retain:

```python
price_cache.get_many.side_effect = AssertionError(
    "guarded group load must not call provider-capable stock reads"
)
benchmark_cache.get_benchmark_data.side_effect = AssertionError(
    "guarded group load must not call provider-capable benchmark reads"
)
```

- [ ] **Step 2: Run loader tests and verify the missing module**

```bash
cd backend
/Users/admin/StockScreenClaude/backend/venv/bin/python -m pytest \
  tests/unit/test_group_rank_input_loader.py -q
```

Expected: collection fails with
`ModuleNotFoundError: No module named 'app.services.group_rank_input_loader'`.

- [ ] **Step 3: Implement `GroupRankInputLoader`**

Create `backend/app/services/group_rank_input_loader.py` with:

```python
"""Load benchmark and constituent inputs for group-ranking calculations."""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
from sqlalchemy.orm import Session

from ..domain.providers.price_symbol_support import (
    is_unsupported_yahoo_price_symbol,
)
from ..models.stock_universe import StockUniverse
from .benchmark_cache_service import BenchmarkCacheService
from .derived_data_execution_policy import DerivedDataExecutionPolicy
from .group_rank_models import (
    GroupRankPrefetchData,
    GroupRankPrefetchStats,
)
from .ibd_industry_service import IBDIndustryService
from .price_cache_service import PriceCacheService


logger = logging.getLogger(__name__)
CACHE_MISS_SYMBOL_SAMPLE_LIMIT = 20


class GroupRankInputLoader:
    def __init__(
        self,
        *,
        price_cache: PriceCacheService,
        benchmark_cache: BenchmarkCacheService,
    ) -> None:
        self.price_cache = price_cache
        self.benchmark_cache = benchmark_cache

    def load(
        self,
        db: Session,
        *,
        market: str,
        policy: DerivedDataExecutionPolicy,
    ) -> GroupRankPrefetchData:
        from ..wiring.bootstrap import get_stock_universe_service

        normalized_market = (market or "US").upper()
        primary_benchmark = self.benchmark_cache.get_benchmark_symbol(
            normalized_market
        )
        benchmark_role = "primary"
        benchmark_symbol = primary_benchmark

        if policy.cache_only:
            benchmark_prices, benchmark_symbol, benchmark_role = (
                self._get_cached_benchmark(
                    normalized_market,
                    primary_benchmark,
                    period="2y",
                )
            )
        else:
            benchmark_prices = self.benchmark_cache.get_benchmark_data(
                market=normalized_market,
                period="2y",
            )

        if benchmark_prices is None or benchmark_prices.empty:
            return self._empty_prefetch(
                market=normalized_market,
                policy=policy,
                benchmark_symbol=benchmark_symbol,
                benchmark_role=benchmark_role,
            )

        active_symbols = frozenset(
            get_stock_universe_service().get_active_symbols(
                db,
                market=normalized_market,
            )
        )
        groups = IBDIndustryService.get_all_groups(
            db,
            market=normalized_market,
        )
        symbols_by_group: dict[str, tuple[str, ...]] = {}
        symbols_to_fetch: set[str] = set()
        unsupported_symbols: set[str] = set()

        for group in groups:
            validated: list[str] = []
            for symbol in IBDIndustryService.get_group_symbols(
                db,
                group,
                market=normalized_market,
            ):
                if symbol not in active_symbols:
                    continue
                if is_unsupported_yahoo_price_symbol(symbol):
                    unsupported_symbols.add(symbol)
                    continue
                validated.append(symbol)
            symbols_by_group[group] = tuple(validated)
            symbols_to_fetch.update(validated)

        ordered_symbols = sorted(symbols_to_fetch)
        if policy.cache_only:
            prices = self.price_cache.get_many_cached_only_fresh(
                ordered_symbols,
                period="2y",
            )
        else:
            prices = self.price_cache.get_many(
                ordered_symbols,
                period="2y",
            )

        missing = tuple(
            symbol
            for symbol in ordered_symbols
            if prices.get(symbol) is None or prices[symbol].empty
        )
        valid_count = len(ordered_symbols) - len(missing)
        market_caps = self._market_caps(db, ordered_symbols)
        stats = GroupRankPrefetchStats(
            target_symbols=len(ordered_symbols),
            symbols_with_prices=valid_count,
            cache_miss_symbols=len(missing),
            cache_miss_symbols_sample=missing[
                :CACHE_MISS_SYMBOL_SAMPLE_LIMIT
            ],
            cache_coverage_ratio=(
                valid_count / len(ordered_symbols)
                if ordered_symbols
                else 1.0
            ),
            benchmark_available=True,
            benchmark_cached=policy.cache_only,
            benchmark_symbol=benchmark_symbol,
            benchmark_role=benchmark_role,
            market=normalized_market,
            cache_only=policy.cache_only,
            skipped_unsupported_symbols=len(unsupported_symbols),
        )
        return GroupRankPrefetchData(
            benchmark_prices=benchmark_prices,
            prices_by_symbol=prices,
            active_symbols=active_symbols,
            market_caps=market_caps,
            stats=stats,
            symbols_by_group=symbols_by_group,
        )

    def _empty_prefetch(
        self,
        *,
        market: str,
        policy: DerivedDataExecutionPolicy,
        benchmark_symbol: str,
        benchmark_role: str,
    ) -> GroupRankPrefetchData:
        stats = GroupRankPrefetchStats(
            target_symbols=0,
            symbols_with_prices=0,
            cache_miss_symbols=0,
            cache_miss_symbols_sample=(),
            cache_coverage_ratio=0.0,
            benchmark_available=False,
            benchmark_cached=False,
            benchmark_symbol=benchmark_symbol,
            benchmark_role=benchmark_role,
            market=market,
            cache_only=policy.cache_only,
            skipped_unsupported_symbols=0,
        )
        return GroupRankPrefetchData(
            benchmark_prices=None,
            prices_by_symbol={},
            active_symbols=frozenset(),
            market_caps={},
            stats=stats,
            symbols_by_group={},
        )

    def _get_cached_benchmark(
        self,
        market: str,
        primary_symbol: str,
        *,
        period: str,
    ) -> tuple[Optional[pd.DataFrame], str, str]:
        candidates = [primary_symbol]
        candidate_fn = getattr(
            self.benchmark_cache,
            "get_benchmark_candidates",
            None,
        )
        if callable(candidate_fn):
            try:
                resolved = [
                    str(symbol)
                    for symbol in candidate_fn(market)
                    if symbol
                ]
                if resolved:
                    candidates = resolved
            except Exception:
                logger.debug(
                    "Could not resolve benchmark candidates for market=%s",
                    market,
                    exc_info=True,
                )

        for index, candidate in enumerate(candidates):
            data = self.price_cache.get_cached_only_fresh(
                candidate,
                period=period,
            )
            if data is not None and not data.empty:
                role = "primary" if index == 0 else "fallback"
                return data, candidate, role
        return None, primary_symbol, "primary"

    @staticmethod
    def _market_caps(
        db: Session,
        symbols: list[str],
    ) -> dict[str, float]:
        if not symbols:
            return {}
        rows = db.query(
            StockUniverse.symbol,
            StockUniverse.market_cap,
        ).filter(StockUniverse.symbol.in_(symbols)).all()
        return {
            symbol: market_cap
            for symbol, market_cap in rows
            if market_cap and market_cap > 0
        }
```

- [ ] **Step 4: Delegate service prefetching to the loader**

Add an optional `input_loader` constructor argument to
`IBDGroupRankService`. Default it to:

```python
self.input_loader = input_loader or GroupRankInputLoader(
    price_cache=price_cache,
    benchmark_cache=benchmark_cache,
)
```

Retain `_prefetch_all_data` as the existing test seam, but reduce it to:

```python
def _prefetch_all_data(
    self,
    db: Session,
    *,
    market: str | None = None,
    policy: DerivedDataExecutionPolicy = (
        DerivedDataExecutionPolicy.provider_allowed()
    ),
) -> GroupRankPrefetchData:
    return self.input_loader.load(
        db,
        market=(market or "US").upper(),
        policy=policy,
    )
```

Delete the old prefetch algorithm and cached benchmark helper from
`ibd_group_rank_service.py`. Update monkeypatch targets in moved tests to
`app.services.group_rank_input_loader`.

- [ ] **Step 5: Run loader and group service tests**

```bash
cd backend
/Users/admin/StockScreenClaude/backend/venv/bin/python -m pytest \
  tests/unit/test_group_rank_input_loader.py \
  tests/unit/test_group_rank_service.py -q
```

Expected: all selected tests pass.

- [ ] **Step 6: Confirm the service shrank**

```bash
wc -l backend/app/services/ibd_group_rank_service.py
```

Expected: fewer than 1,850 lines and at least 100 lines below the pre-refactor
1,958-line size.

- [ ] **Step 7: Commit the loader extraction**

```bash
git add \
  backend/app/services/group_rank_input_loader.py \
  backend/app/services/ibd_group_rank_service.py \
  backend/tests/unit/test_group_rank_input_loader.py \
  backend/tests/unit/test_group_rank_service.py
git commit --no-verify -m "refactor: extract group rank input loader"
```

---

### Task 7: Resolve group policy once and consume typed results

**Files:**
- Create: `backend/tests/unit/test_group_rank_execution_policy.py`
- Modify: `backend/app/tasks/group_rank_tasks.py`
- Modify: `backend/app/services/ibd_group_rank_service.py`
- Modify: `backend/tests/unit/test_group_rank_tasks.py`
- Modify: `backend/tests/unit/test_group_rank_service.py`

**Interfaces:**
- Public group tasks retain legacy arguments and add `execution_policy`.
- Services receive `DerivedDataExecutionPolicy`, not cache-mode booleans.
- Tasks consume `GroupRankCalculationResult`.

- [ ] **Step 1: Move execution-policy cases into a focused test module**

Create `backend/tests/unit/test_group_rank_execution_policy.py` and move these
tests from `test_group_rank_tasks.py`:

- same-day incomplete warmup;
- partial warmup acceptance and stale rejection;
- in-process warmup bypass;
- manual historical provider-capable behavior;
- manual strict cache-only behavior;
- guarded historical tolerant cache-only behavior;
- strict legacy precedence;
- guarded wrapper propagation.

Add these test-local helpers so the new module has no imports from another test
file:

```python
def _patch_serialized_lock(monkeypatch):
    fake_lock = MagicMock()
    fake_lock.acquire.return_value = (True, False)
    fake_lock.release.return_value = True
    fake_coordination = MagicMock()
    fake_coordination.acquire_market_workload.return_value = (
        True,
        False,
    )
    fake_coordination.release_market_workload.return_value = True
    monkeypatch.setattr(
        "app.wiring.bootstrap.get_data_fetch_lock",
        lambda: fake_lock,
    )
    monkeypatch.setattr(
        "app.wiring.bootstrap.get_workload_coordination",
        lambda: fake_coordination,
    )


def _patch_calendar_service(
    monkeypatch,
    now: datetime,
    *,
    is_trading_day: bool = True,
):
    fake = MagicMock()
    fake.is_trading_day.return_value = is_trading_day
    fake.market_now.return_value = now
    fake.last_completed_trading_day.return_value = now.date()
    monkeypatch.setattr(
        "app.tasks.group_rank_tasks.get_market_calendar_service",
        lambda: fake,
    )
    return fake
```

Import the typed results and add the following two result factories to both
`test_group_rank_execution_policy.py` and the remaining
`test_group_rank_tasks.py`:

```python
from app.services.group_rank_models import (
    GroupRankCalculationResult,
    GroupRankPrefetchStats,
)


def _prefetch_stats(
    *,
    misses: int = 0,
) -> GroupRankPrefetchStats:
    target = 100
    return GroupRankPrefetchStats(
        target_symbols=target,
        symbols_with_prices=target - misses,
        cache_miss_symbols=misses,
        cache_miss_symbols_sample=(
            ("MISS",) if misses else ()
        ),
        cache_coverage_ratio=(target - misses) / target,
        benchmark_available=True,
        benchmark_cached=True,
        benchmark_symbol="SPY",
        benchmark_role="primary",
        market="US",
        cache_only=True,
        skipped_unsupported_symbols=0,
    )


def _group_calculation(
    *,
    misses: int = 0,
) -> GroupRankCalculationResult:
    return GroupRankCalculationResult(
        rankings=(
            {
                "rank": 1,
                "industry_group": "Software",
                "avg_rs_rating": 90.0,
                "num_stocks": 3,
            },
        ),
        prefetch_stats=_prefetch_stats(misses=misses),
    )
```

Use `_group_calculation()` for every successful mocked
`calculate_group_rankings` return in both group task test modules. For the
guarded partial-cache test use `_group_calculation(misses=30)`. Construct
`IncompleteGroupRankingCacheError` with `_prefetch_stats(misses=100)` rather
than a raw dictionary.

In the moved
`test_guarded_historical_group_rankings_use_tolerant_cache_only_policy`, retain
the legacy `refresh_guarded_cache_only=True` invocation and assert:

```python
call_kwargs = fake_service.calculate_group_rankings.call_args.kwargs
assert call_kwargs["policy"].mode.value == "refresh_guarded"
assert call_kwargs["cache_requirement"] == (
    GroupRankCacheRequirement.disabled()
)
```

In the moved
`test_guarded_group_wrapper_propagates_cache_only_to_gapfill_and_target`, invoke
the wrapper with:

```python
result = module.calculate_daily_group_rankings_with_gapfill.run(
    market="US",
    calculation_date="2026-03-19",
    execution_policy="refresh_guarded",
)
```

and assert:

```python
fill_kwargs = fake_service.fill_gaps_optimized.call_args.kwargs
assert fill_kwargs["market"] == "US"
assert fill_kwargs["policy"].mode.value == "refresh_guarded"
target_call.assert_called_once_with(
    market="US",
    activity_lifecycle="daily_refresh",
    calculation_date="2026-03-19",
    execution_policy="refresh_guarded",
)
```

- [ ] **Step 2: Run group execution tests and verify failures**

```bash
cd backend
/Users/admin/StockScreenClaude/backend/venv/bin/python -m pytest \
  tests/unit/test_group_rank_execution_policy.py -q
```

Expected: failures because the group tasks do not accept or propagate the new
serialized mode and still expect list results.

- [ ] **Step 3: Normalize the daily group task**

Import:

```python
from ..services.derived_data_execution_policy import (
    DerivedDataExecutionMode,
    resolve_derived_data_execution_policy,
)
```

Append `execution_policy: str | None = None` to
`calculate_daily_group_rankings`. Resolve once using target/current date and
the group warmup-bypass context.

Set the cache requirement:

```python
cache_requirement = GroupRankCacheRequirement.disabled()
if policy.strict_completeness:
    cache_requirement = GroupRankCacheRequirement.strict()

if policy.requires_warmup_metadata:
    warmup_decision = evaluate_same_day_group_rank_warmup(
        service.price_cache,
        market=market,
    )
    cache_requirement = warmup_decision.cache_requirement
    if warmup_decision.error:
        _mark_market_activity_failed_safely(
            db,
            market=effective_market,
            stage_key="groups",
            lifecycle=activity_lifecycle,
            task_name=getattr(
                self,
                "name",
                "calculate_daily_group_rankings",
            ),
            task_id=getattr(
                getattr(self, "request", None),
                "id",
                None,
            ),
            message=warmup_decision.error,
        )
        return {
            "error": warmup_decision.error,
            "reason_code": GroupRankReasonCode.WARMUP_INCOMPLETE,
            "date": calc_date.strftime("%Y-%m-%d"),
            "timestamp": datetime.now().isoformat(),
            "cache_only": True,
        }
```

Call:

```python
calculation = service.calculate_group_rankings(
    db,
    calc_date,
    market=effective_market,
    policy=policy,
    cache_requirement=cache_requirement,
)
rankings = calculation.rankings
prefetch_stats = calculation.prefetch_stats.to_dict()
```

Replace every `results` read with `rankings`. Serialize caught cache errors
with:

```python
"prefetch_stats": e.stats.to_dict()
```

Expose guarded result fields when:

```python
policy.mode is DerivedDataExecutionMode.REFRESH_GUARDED
```

- [ ] **Step 4: Normalize the group wrapper and service gap-fill**

Append `execution_policy` to the wrapper. Resolve it after `target_date`.
Change `fill_gaps_optimized` to accept:

```python
policy: DerivedDataExecutionPolicy = (
    DerivedDataExecutionPolicy.provider_allowed()
)
```

Pass `policy=policy` through prefetch. The wrapper passes the resolved policy
object to `fill_gaps_optimized` and passes only
`execution_policy=policy.mode.value` to the nested task.

Change `_calculate_daily_group_rankings_in_process` to accept only the
serialized `execution_policy`.

- [ ] **Step 5: Run group execution and service tests**

```bash
cd backend
/Users/admin/StockScreenClaude/backend/venv/bin/python -m pytest \
  tests/unit/test_derived_data_execution_policy.py \
  tests/unit/test_group_rank_execution_policy.py \
  tests/unit/test_group_rank_tasks.py \
  tests/unit/test_group_rank_service.py -q
```

Expected: all selected tests pass.

- [ ] **Step 6: Assert service policy booleans are gone**

Add this regression to
`backend/tests/unit/test_derived_data_execution_policy.py`:

```python
import ast
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[2]
LEGACY_NAMES = {
    "force_cache_only",
    "refresh_guarded_cache_only",
}


def test_legacy_policy_names_do_not_branch_below_task_boundary():
    for relative_path in (
        "app/services/breadth_calculator_service.py",
        "app/services/ibd_group_rank_service.py",
        "app/services/group_rank_input_loader.py",
    ):
        source = (BACKEND_ROOT / relative_path).read_text()
        for legacy_name in LEGACY_NAMES:
            assert legacy_name not in source

    for relative_path in (
        "app/tasks/breadth_tasks.py",
        "app/tasks/group_rank_tasks.py",
    ):
        tree = ast.parse((BACKEND_ROOT / relative_path).read_text())
        for node in ast.walk(tree):
            if not isinstance(node, (ast.If, ast.IfExp, ast.While)):
                continue
            referenced = {
                child.id
                for child in ast.walk(node.test)
                if isinstance(child, ast.Name)
            }
            assert LEGACY_NAMES.isdisjoint(referenced), (
                relative_path,
                node.lineno,
                referenced,
            )
```

Use the shell check as a quick companion:

```bash
rg -n "refresh_guarded_cache_only|force_cache_only|cache_only: bool" \
  backend/app/services/ibd_group_rank_service.py \
  backend/app/services/group_rank_input_loader.py
```

Expected: no output.

- [ ] **Step 7: Commit group policy normalization**

```bash
git add \
  backend/app/tasks/group_rank_tasks.py \
  backend/app/services/ibd_group_rank_service.py \
  backend/tests/unit/test_derived_data_execution_policy.py \
  backend/tests/unit/test_group_rank_execution_policy.py \
  backend/tests/unit/test_group_rank_tasks.py \
  backend/tests/unit/test_group_rank_service.py
git commit --no-verify -m "refactor: normalize group execution policy"
```

---

### Task 8: Extract manual group backfill tasks without changing task names

**Files:**
- Create: `backend/app/tasks/group_rank_backfill_tasks.py`
- Create: `backend/tests/unit/test_group_rank_backfill_tasks.py`
- Modify: `backend/app/tasks/group_rank_tasks.py`
- Modify: `backend/tests/unit/test_group_rank_tasks.py`
- Verify: `backend/app/celery_app.py`
- Verify: `backend/app/api/v1/groups.py`

**Interfaces:**
- Produces the same Celery task objects:
  `backfill_group_rankings`, `gapfill_group_rankings`,
  `backfill_group_rankings_1year`.
- Registered names remain under `app.tasks.group_rank_tasks.*`.

- [ ] **Step 1: Move manual-task tests into their own module**

Create `backend/tests/unit/test_group_rank_backfill_tasks.py` with the existing
three market-forwarding tests plus:

```python
from app.tasks import group_rank_tasks


def test_extracted_tasks_keep_registered_names():
    assert group_rank_tasks.backfill_group_rankings.name == (
        "app.tasks.group_rank_tasks.backfill_group_rankings"
    )
    assert group_rank_tasks.gapfill_group_rankings.name == (
        "app.tasks.group_rank_tasks.gapfill_group_rankings"
    )
    assert group_rank_tasks.backfill_group_rankings_1year.name == (
        "app.tasks.group_rank_tasks.backfill_group_rankings_1year"
    )
```

Delete the moved tests from `test_group_rank_tasks.py`.

- [ ] **Step 2: Run extracted-task tests before moving production code**

```bash
cd backend
/Users/admin/StockScreenClaude/backend/venv/bin/python -m pytest \
  tests/unit/test_group_rank_backfill_tasks.py -q
```

Expected: the copied tests pass against the old module, establishing the
behavioral baseline.

- [ ] **Step 3: Create the extracted task module**

Create `backend/app/tasks/group_rank_backfill_tasks.py` with these imports:

```python
"""Manual and administrative IBD group-ranking backfill tasks."""

import logging
import time
from datetime import datetime, timedelta

from ..celery_app import celery_app
from ..database import SessionLocal
from ..wiring.bootstrap import (
    get_group_rank_service,
    get_market_calendar_service,
)
from .workload_coordination import serialized_market_workload


logger = logging.getLogger(__name__)


@celery_app.task(
    bind=True,
    name="app.tasks.group_rank_tasks.backfill_group_rankings",
)
@serialized_market_workload("backfill_group_rankings")
def backfill_group_rankings(
    self,
    start_date: str,
    end_date: str,
    market: str = "US",
):
    logger.info("=" * 60)
    logger.info("TASK: Backfill IBD Group Rankings (Optimized)")
    logger.info("Date range: %s to %s", start_date, end_date)
    logger.info("=" * 60)

    try:
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
        end = datetime.strptime(end_date, "%Y-%m-%d").date()
        if start > end:
            return {
                "error": "Invalid date range: start_date > end_date",
                "timestamp": datetime.now().isoformat(),
            }
    except ValueError as exc:
        return {
            "error": f"Invalid date format. Use YYYY-MM-DD: {exc}",
            "timestamp": datetime.now().isoformat(),
        }

    db = SessionLocal()
    start_time = time.time()
    try:
        service = get_group_rank_service()
        result = service.backfill_rankings_optimized(
            db,
            start,
            end,
            market=market,
        )
        duration = time.time() - start_time

        from ..services.group_rankings_cache import (
            bump_group_rankings_epoch,
        )

        bump_group_rankings_epoch(market)
        try:
            from ..services.ui_snapshot_service import (
                safe_publish_groups_bootstrap,
            )

            safe_publish_groups_bootstrap()
        except Exception as snapshot_error:
            logger.warning(
                "Group rankings snapshot publish failed after backfill: %s",
                snapshot_error,
            )

        return {
            "start_date": start_date,
            "end_date": end_date,
            "total_dates": result["total_dates"],
            "deleted": result.get("deleted", 0),
            "processed": result["processed"],
            "skipped": result["skipped"],
            "errors": result["errors"],
            "total_duration_seconds": round(duration, 2),
            "avg_duration_per_day": round(
                duration / max(result["processed"], 1),
                2,
            ),
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as exc:
        db.rollback()
        logger.error(
            "Error in backfill_group_rankings task: %s",
            exc,
            exc_info=True,
        )
        return {
            "error": str(exc),
            "timestamp": datetime.now().isoformat(),
        }
    finally:
        db.close()


@celery_app.task(
    bind=True,
    name="app.tasks.group_rank_tasks.gapfill_group_rankings",
)
@serialized_market_workload("gapfill_group_rankings")
def gapfill_group_rankings(
    self,
    max_days: int = 365,
    market: str = "US",
):
    logger.info("=" * 60)
    logger.info("TASK: Gap-Fill IBD Group Rankings (Optimized)")
    logger.info("Looking back %s days", max_days)
    logger.info("=" * 60)

    db = SessionLocal()
    start_time = time.time()
    try:
        service = get_group_rank_service()
        missing_dates = service.find_missing_dates(
            db,
            lookback_days=max_days,
            market=market,
        )
        if not missing_dates:
            return {
                "status": "complete",
                "gaps_found": 0,
                "message": "No gaps to fill",
                "timestamp": datetime.now().isoformat(),
            }

        result = service.fill_gaps_optimized(
            db,
            missing_dates,
            market=market,
        )
        duration = time.time() - start_time

        from ..services.group_rankings_cache import (
            bump_group_rankings_epoch,
        )

        bump_group_rankings_epoch(market)
        try:
            from ..services.ui_snapshot_service import (
                safe_publish_groups_bootstrap,
            )

            safe_publish_groups_bootstrap()
        except Exception as snapshot_error:
            logger.warning(
                "Group rankings snapshot publish failed after gapfill: %s",
                snapshot_error,
            )

        return {
            "status": "complete",
            "gaps_found": len(missing_dates),
            "processed": result["processed"],
            "errors": result["errors"],
            "total_duration_seconds": round(duration, 2),
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as exc:
        db.rollback()
        logger.error(
            "Error in gapfill_group_rankings task: %s",
            exc,
            exc_info=True,
        )
        return {
            "status": "error",
            "error": str(exc),
            "timestamp": datetime.now().isoformat(),
        }
    finally:
        db.close()


@celery_app.task(
    bind=True,
    name="app.tasks.group_rank_tasks.backfill_group_rankings_1year",
)
@serialized_market_workload("backfill_group_rankings_1year")
def backfill_group_rankings_1year(
    self,
    market: str = "US",
):
    from .market_queues import normalize_market

    effective_market = normalize_market(market)
    db = SessionLocal()
    start_time = time.time()
    try:
        service = get_group_rank_service()
        calendar_service = get_market_calendar_service()
        end_date = calendar_service.market_now(effective_market).date()
        start_date = end_date - timedelta(days=365)
        result = service.backfill_rankings_optimized(
            db,
            start_date,
            end_date,
            market=effective_market,
        )
        duration = time.time() - start_time

        try:
            from ..services.ui_snapshot_service import (
                safe_publish_groups_bootstrap,
            )

            safe_publish_groups_bootstrap()
        except Exception as snapshot_error:
            logger.warning(
                "Group rankings snapshot publish failed after "
                "1-year backfill: %s",
                snapshot_error,
            )

        result["total_duration_seconds"] = round(duration, 2)
        result["timestamp"] = datetime.now().isoformat()
        return result
    except Exception as exc:
        db.rollback()
        logger.error(
            "Error in backfill_group_rankings_1year: %s",
            exc,
            exc_info=True,
        )
        return {
            "error": str(exc),
            "timestamp": datetime.now().isoformat(),
        }
    finally:
        db.close()
```

- [ ] **Step 4: Re-export extracted task objects**

Delete the three definitions from `group_rank_tasks.py` and import:

```python
from .group_rank_backfill_tasks import (
    backfill_group_rankings,
    backfill_group_rankings_1year,
    gapfill_group_rankings,
)
```

In the three moved market-forwarding tests, change the module import to:

```python
import app.tasks.group_rank_backfill_tasks as module
```

This ensures monkeypatches of `SessionLocal`, `get_group_rank_service`, and
`get_market_calendar_service` target the module that owns the task function
globals. Keep the registration test importing `group_rank_tasks` so it verifies
the compatibility re-export.

Keep `backend/app/celery_app.py` and `backend/app/api/v1/groups.py` unchanged.
Their current imports continue to resolve through the re-exports.

- [ ] **Step 5: Run registration, API-import, and group task tests**

```bash
cd backend
/Users/admin/StockScreenClaude/backend/venv/bin/python -m pytest \
  tests/unit/test_group_rank_backfill_tasks.py \
  tests/unit/test_group_rank_tasks.py \
  tests/unit/test_group_rank_execution_policy.py -q
```

Expected: all selected tests pass.

- [ ] **Step 6: Verify module sizes**

```bash
wc -l \
  backend/app/tasks/group_rank_tasks.py \
  backend/app/tasks/group_rank_backfill_tasks.py \
  backend/tests/unit/test_group_rank_tasks.py \
  backend/tests/unit/test_group_rank_backfill_tasks.py \
  backend/tests/unit/test_group_rank_execution_policy.py
```

Expected:

- `group_rank_tasks.py` is below 800 lines.
- No new task test module exceeds 1,000 lines.

- [ ] **Step 7: Commit the task/test decomposition**

```bash
git add \
  backend/app/tasks/group_rank_tasks.py \
  backend/app/tasks/group_rank_backfill_tasks.py \
  backend/tests/unit/test_group_rank_tasks.py \
  backend/tests/unit/test_group_rank_backfill_tasks.py \
  backend/tests/unit/test_group_rank_execution_policy.py
git commit --no-verify -m "refactor: split group backfill tasks"
```

---

### Task 9: Run cross-layer regression gates and close the review findings

**Files:**
- Modify: `docs/superpowers/specs/2026-07-16-derived-data-execution-policy-refactor-design.md`
- Modify: `.beads/issues.jsonl`
- Modify: `.beads/interactions.jsonl`

**Interfaces:**
- Verifies all public task/result contracts and provider-exclusion guarantees.
- Produces a closed Bead and a pushed, clean branch.

- [ ] **Step 1: Run the focused issue-301 suite**

```bash
cd backend
/Users/admin/StockScreenClaude/backend/venv/bin/python -m pytest \
  tests/unit/test_derived_data_execution_policy.py \
  tests/unit/test_breadth_coverage.py \
  tests/unit/test_breadth_calculator.py \
  tests/unit/test_breadth_calculator_service.py \
  tests/unit/test_breadth_tasks.py \
  tests/unit/test_daily_market_pipeline_tasks.py \
  tests/unit/test_group_rank_models.py \
  tests/unit/test_group_rank_input_loader.py \
  tests/unit/test_group_rank_service.py \
  tests/unit/test_group_rank_execution_policy.py \
  tests/unit/test_group_rank_tasks.py \
  tests/unit/test_group_rank_backfill_tasks.py \
  tests/unit/test_group_rank_history_backfill_service.py -q
```

Expected: all selected tests pass.

- [ ] **Step 2: Run syntax and policy-leak checks**

```bash
cd backend
/Users/admin/StockScreenClaude/backend/venv/bin/python -m compileall -q app
rg -n "diagnostics.*Optional\\[Dict|diagnostics\\.clear|diagnostics\\.update" \
  app/services/ibd_group_rank_service.py
rg -n "refresh_guarded_cache_only|force_cache_only" \
  app/services/breadth_calculator_service.py \
  app/services/ibd_group_rank_service.py \
  app/services/group_rank_input_loader.py
```

Expected: compile succeeds and both searches produce no output.

- [ ] **Step 3: Run the backend unit suite**

```bash
cd backend
/Users/admin/StockScreenClaude/backend/venv/bin/python -m pytest \
  tests/unit -q
```

Expected: the same repository baseline as before this refactor—no new failures.
Known unrelated baseline failures remain tracked by Beads
`stockscreenclaude-cis`, `stockscreenclaude-w1l`, and `stockscreenclaude-fn7`.

- [ ] **Step 4: Run diff and size quality gates**

```bash
git diff --check
wc -l \
  backend/app/tasks/group_rank_tasks.py \
  backend/app/services/ibd_group_rank_service.py \
  backend/tests/unit/test_group_rank_tasks.py \
  backend/tests/unit/test_group_rank_service.py
git status --short
```

Expected:

- no whitespace errors;
- `group_rank_tasks.py` below 800 lines;
- `ibd_group_rank_service.py` at least 100 lines smaller than 1,958;
- group task and service tests are smaller than their original 1,493 and 1,483
  lines;
- only in-scope implementation, tests, docs, and Beads files are modified.

- [ ] **Step 5: Update the design status and close the Bead**

Change the design status to:

```markdown
**Status:** Implemented and verified
```

Run:

```bash
bd close stockscreenclaude-duw --reason="Addressed thermo-nuclear review findings with typed execution policy and compatibility resolver, typed group results and input loader, authoritative breadth coverage, decomposed group tasks/tests, and verified issue-301 provider exclusion."
bd export -o .beads/issues.jsonl
```

- [ ] **Step 6: Commit final verification metadata**

```bash
git add \
  docs/superpowers/specs/2026-07-16-derived-data-execution-policy-refactor-design.md \
  .beads/issues.jsonl \
  .beads/interactions.jsonl
git commit --no-verify -m "docs: verify derived-data policy refactor"
```

- [ ] **Step 7: Rebase, export Beads, push, and verify remote state**

```bash
git pull --rebase
bd export -o .beads/issues.jsonl
git status --short
git push
git status --short --branch
```

Expected final status:

```text
## codex/issue-301-cache-only-derived-data...origin/codex/issue-301-cache-only-derived-data
```

with no modified or untracked files.
