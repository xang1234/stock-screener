# Derived-data Thermo-review Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the issue-301 architecture by making execution policies closed and self-interpreting, separating breadth symbol coverage from per-date outcomes, isolating legacy group-prefetch compatibility, and turning `IBDGroupRankService` into a delegating compatibility facade.

**Architecture:** Public Celery and group-service APIs remain stable. The execution policy derives capabilities and gap-fill behavior from a small closed state; breadth composes shared price coverage with lightweight outcome reports; group input sources, calculation, persistence/query, historical calculation, and legacy conversion move into focused components that the existing facade delegates to.

**Tech Stack:** Python 3.12, dataclasses, enums, protocols, pandas, SQLAlchemy, Celery, pytest, Beads/Dolt, Git.

## Global Constraints

- Preserve existing Celery registered names and accepted keyword arguments.
- Preserve existing task result dictionary keys and public `IBDGroupRankService` method signatures.
- Preserve refresh-guarded provider exclusion and tolerant partial coverage.
- Preserve strict same-day/static-export completeness behavior.
- Preserve provider-capable manual historical behavior.
- Repositories must not commit independently; existing task/service transaction boundaries remain authoritative.
- Use red-green-refactor for every production change.
- Keep legacy tuple/mapping conversion only in `LegacyGroupRankPrefetchAdapter`.
- Do not modify `GroupRankHistoryBackfillService`; it remains the outer static-export workflow.

---

## File Structure

### Create

- `backend/app/services/group_rank_input_sources.py`
  - Protocols and default adapters for active universe, IBD taxonomy, and market caps.
- `backend/app/services/group_rank_legacy_adapter.py`
  - The only legacy tuple/mapping conversion boundary.
- `backend/app/services/group_ranking_calculator.py`
  - Pure RS and group-ranking calculation from typed prefetched inputs.
- `backend/app/services/group_ranking_repository.py`
  - IBD group-ranking persistence and ranking-table queries.
- `backend/app/services/group_rank_historical_calculator.py`
  - Range backfill, gap detection/fill, chunking, and historical calculation.
- `backend/tests/unit/test_group_rank_legacy_adapter.py`
- `backend/tests/unit/test_group_ranking_calculator.py`
- `backend/tests/unit/test_group_ranking_repository.py`
- `backend/tests/unit/test_group_rank_historical_calculator.py`

### Modify

- `backend/app/services/derived_data_execution_policy.py`
- `backend/app/tasks/breadth_tasks.py`
- `backend/app/tasks/group_rank_tasks.py`
- `backend/app/services/breadth_coverage.py`
- `backend/app/services/breadth_calculator_service.py`
- `backend/app/services/group_rank_models.py`
- `backend/app/services/group_rank_input_loader.py`
- `backend/app/services/ibd_group_rank_service.py`
- `backend/app/wiring/bootstrap.py`
- `backend/tests/unit/test_derived_data_execution_policy.py`
- `backend/tests/unit/test_breadth_tasks.py`
- `backend/tests/unit/test_group_rank_execution_policy.py`
- `backend/tests/unit/test_breadth_coverage.py`
- `backend/tests/unit/test_breadth_calculator_service.py`
- `backend/tests/unit/test_group_rank_models.py`
- `backend/tests/unit/test_group_rank_input_loader.py`
- `backend/tests/unit/test_group_rank_service.py`
- `backend/tests/unit/test_static_rrg_history_bundle.py`
- `docs/superpowers/specs/2026-07-16-derived-data-thermo-review-remediation-design.md`

---

### Task 1: Make execution policy a closed decision object

**Files:**

- Modify: `backend/app/services/derived_data_execution_policy.py`
- Modify: `backend/app/tasks/breadth_tasks.py`
- Modify: `backend/app/tasks/group_rank_tasks.py`
- Modify: `backend/tests/unit/test_derived_data_execution_policy.py`
- Modify: `backend/tests/unit/test_breadth_tasks.py`
- Modify: `backend/tests/unit/test_group_rank_execution_policy.py`

**Interfaces:**

- Produces:
  - `DerivedDataTargetKind`
  - `DerivedDataValidationProfile`
  - `DerivedDataExecutionPolicy.validation_profile`
  - `DerivedDataExecutionPolicy.allows_provider_reads`
  - `DerivedDataExecutionPolicy.requires_strict_completeness`
  - `DerivedDataExecutionPolicy.response_cache_policy`
  - `DerivedDataExecutionPolicy.for_gap_fill()`
  - `DerivedDataExecutionPolicy.annotate_response(result, include_cache_only=False)`
- Preserves:
  - `DerivedDataExecutionPolicy.provider_allowed()`
  - `resolve_derived_data_execution_policy(...)`
  - legacy request-flag precedence.

- [ ] **Step 1: Replace boolean-matrix assertions with closed-policy behavior tests**

Add these tests to `test_derived_data_execution_policy.py` before changing production code:

```python
from dataclasses import fields

from app.services.derived_data_execution_policy import (
    DerivedDataExecutionMode,
    DerivedDataTargetKind,
    DerivedDataValidationProfile,
)


def test_policy_stores_only_request_state():
    assert [field.name for field in fields(DerivedDataExecutionPolicy)] == [
        "mode",
        "target_kind",
        "same_day_warmup_bypassed",
    ]


@pytest.mark.parametrize(
    ("requested", "target", "profile"),
    [
        (None, HISTORICAL, DerivedDataValidationProfile.PROVIDER_ALLOWED),
        ("auto", TODAY, DerivedDataValidationProfile.STRICT_WITH_WARMUP),
        (
            "strict_cache_only",
            HISTORICAL,
            DerivedDataValidationProfile.STRICT_WITHOUT_WARMUP,
        ),
        (
            "refresh_guarded",
            HISTORICAL,
            DerivedDataValidationProfile.TOLERANT_CACHE_ONLY,
        ),
    ],
)
def test_policy_resolves_one_validation_profile(requested, target, profile):
    policy = resolve_derived_data_execution_policy(
        execution_policy=requested,
        target_date=target,
        current_date=TODAY,
    )

    assert policy.validation_profile is profile


def test_auto_same_day_gap_fill_becomes_provider_allowed_historical_policy():
    policy = resolve_derived_data_execution_policy(
        target_date=TODAY,
        current_date=TODAY,
    )

    gap_policy = policy.for_gap_fill()

    assert gap_policy.mode is DerivedDataExecutionMode.AUTO
    assert gap_policy.target_kind is DerivedDataTargetKind.HISTORICAL
    assert gap_policy.validation_profile is DerivedDataValidationProfile.PROVIDER_ALLOWED


def test_guarded_gap_fill_preserves_guarded_policy():
    policy = resolve_derived_data_execution_policy(
        execution_policy="refresh_guarded",
        target_date=TODAY,
        current_date=TODAY,
    )

    assert policy.for_gap_fill() is policy


def test_policy_owns_guarded_response_metadata():
    policy = resolve_derived_data_execution_policy(
        execution_policy="refresh_guarded",
        target_date=HISTORICAL,
        current_date=TODAY,
    )

    result = policy.annotate_response({"status": "failed"})

    assert result == {
        "status": "failed",
        "cache_only": True,
        "cache_policy": "refresh_guarded",
    }
```

Retain the existing legacy-precedence and invalid-string tests, but assert
derived properties/profile instead of stored booleans.

- [ ] **Step 2: Add source-level tests for centralized gap and response decisions**

Extend `test_legacy_policy_names_do_not_branch_below_task_boundary()`:

```python
for relative_path in (
    "app/tasks/breadth_tasks.py",
    "app/tasks/group_rank_tasks.py",
):
    source = (BACKEND_ROOT / relative_path).read_text()
    assert "DerivedDataExecutionMode.AUTO" not in source
    assert "DerivedDataExecutionMode.REFRESH_GUARDED" not in source
    assert "'policy' in locals()" not in source
```

Add wrapper assertions in breadth and group task tests:

```python
assert captured_gap_policy is resolved_policy.for_gap_fill()
assert result["cache_policy"] == "refresh_guarded"
```

Use object identity for guarded policies and profile equality for `AUTO`
historical derivation.

- [ ] **Step 3: Run the policy/task tests and verify RED**

Run:

```bash
cd backend
source venv/bin/activate
pytest \
  tests/unit/test_derived_data_execution_policy.py \
  tests/unit/test_breadth_tasks.py \
  tests/unit/test_group_rank_execution_policy.py -q
```

Expected: failures because target/profile enums, `for_gap_fill()`, and
`annotate_response()` do not exist and task modules still inspect modes.

- [ ] **Step 4: Implement the closed policy**

Replace the stored capability booleans with:

```python
class DerivedDataTargetKind(str, Enum):
    SAME_DAY = "same_day"
    HISTORICAL = "historical"


class DerivedDataValidationProfile(str, Enum):
    PROVIDER_ALLOWED = "provider_allowed"
    STRICT_WITH_WARMUP = "strict_with_warmup"
    STRICT_WITHOUT_WARMUP = "strict_without_warmup"
    TOLERANT_CACHE_ONLY = "tolerant_cache_only"


@dataclass(frozen=True)
class DerivedDataExecutionPolicy:
    mode: DerivedDataExecutionMode
    target_kind: DerivedDataTargetKind
    same_day_warmup_bypassed: bool = False

    def __post_init__(self) -> None:
        if (
            self.same_day_warmup_bypassed
            and (
                self.mode is not DerivedDataExecutionMode.AUTO
                or self.target_kind is not DerivedDataTargetKind.SAME_DAY
            )
        ):
            raise ValueError(
                "same-day warmup bypass is valid only for same-day AUTO policy"
            )

    @classmethod
    def provider_allowed(cls) -> "DerivedDataExecutionPolicy":
        return cls(
            mode=DerivedDataExecutionMode.AUTO,
            target_kind=DerivedDataTargetKind.HISTORICAL,
        )

    @property
    def validation_profile(self) -> DerivedDataValidationProfile:
        if self.mode is DerivedDataExecutionMode.STRICT_CACHE_ONLY:
            return DerivedDataValidationProfile.STRICT_WITHOUT_WARMUP
        if self.mode is DerivedDataExecutionMode.REFRESH_GUARDED:
            return DerivedDataValidationProfile.TOLERANT_CACHE_ONLY
        if self.target_kind is DerivedDataTargetKind.HISTORICAL:
            return DerivedDataValidationProfile.PROVIDER_ALLOWED
        if self.same_day_warmup_bypassed:
            return DerivedDataValidationProfile.STRICT_WITHOUT_WARMUP
        return DerivedDataValidationProfile.STRICT_WITH_WARMUP

    @property
    def allows_provider_reads(self) -> bool:
        return self.validation_profile is DerivedDataValidationProfile.PROVIDER_ALLOWED

    @property
    def cache_only(self) -> bool:
        return not self.allows_provider_reads

    @property
    def requires_strict_completeness(self) -> bool:
        return self.validation_profile in {
            DerivedDataValidationProfile.STRICT_WITH_WARMUP,
            DerivedDataValidationProfile.STRICT_WITHOUT_WARMUP,
        }

    @property
    def strict_completeness(self) -> bool:
        """Compatibility alias for existing consumers during this refactor."""
        return self.requires_strict_completeness

    @property
    def requires_warmup_metadata(self) -> bool:
        return self.validation_profile is DerivedDataValidationProfile.STRICT_WITH_WARMUP

    @property
    def tolerates_partial_coverage(self) -> bool:
        return self.validation_profile is DerivedDataValidationProfile.TOLERANT_CACHE_ONLY

    @property
    def response_cache_policy(self) -> str | None:
        if self.mode is DerivedDataExecutionMode.REFRESH_GUARDED:
            return "refresh_guarded"
        return None

    def for_gap_fill(self) -> "DerivedDataExecutionPolicy":
        if self.mode is DerivedDataExecutionMode.AUTO:
            return self.provider_allowed()
        return self

    def annotate_response(
        self,
        result: dict[str, object],
        *,
        include_cache_only: bool = False,
    ) -> dict[str, object]:
        if include_cache_only:
            result["cache_only"] = self.cache_only
        if self.response_cache_policy is not None:
            result["cache_only"] = True
            result["cache_policy"] = self.response_cache_policy
        return result
```

The resolver sets `target_kind` from `target_date == current_date` and passes
the bypass flag only for same-day `AUTO`.

- [ ] **Step 5: Refactor task decisions to one validation profile**

In breadth task code, replace nested capability checks with:

```python
profile = policy.validation_profile
if profile is DerivedDataValidationProfile.TOLERANT_CACHE_ONLY:
    completeness_error = _validate_refresh_guarded_breadth(coverage)
elif profile is DerivedDataValidationProfile.STRICT_WITH_WARMUP:
    completeness_error = _validate_same_day_cache_only_breadth(
        calculator.price_cache,
        coverage,
        market=effective_market,
    )
elif profile is DerivedDataValidationProfile.STRICT_WITHOUT_WARMUP:
    completeness_error = _validate_strict_cache_only_breadth(coverage)
else:
    completeness_error = None
```

Use:

```python
gap_policy = policy.for_gap_fill()
policy.annotate_response(task_result, include_cache_only=True)
policy.annotate_response(error_result)
```

In group task code, use the same profile enum to select strict requirements and
warmup evaluation. Replace every guarded-mode response branch with
`policy.annotate_response(...)`. Resolve policy before the task `try` block or
initialize it from the request before any failing work so error paths never
inspect `locals()`.

- [ ] **Step 6: Run focused tests and verify GREEN**

Run the Step 3 command.

Expected: all tests pass.

- [ ] **Step 7: Commit the closed policy**

```bash
git add \
  backend/app/services/derived_data_execution_policy.py \
  backend/app/tasks/breadth_tasks.py \
  backend/app/tasks/group_rank_tasks.py \
  backend/tests/unit/test_derived_data_execution_policy.py \
  backend/tests/unit/test_breadth_tasks.py \
  backend/tests/unit/test_group_rank_execution_policy.py
git commit -m "refactor: close derived-data execution policy"
```

---

### Task 2: Separate breadth price coverage from per-date outcomes

**Files:**

- Modify: `backend/app/services/breadth_coverage.py`
- Modify: `backend/app/services/breadth_calculator_service.py`
- Modify: `backend/tests/unit/test_breadth_coverage.py`
- Modify: `backend/tests/unit/test_breadth_calculator_service.py`

**Interfaces:**

- Produces:
  - `BreadthPriceCoverage`
  - `BreadthPriceCoverageAccumulator`
  - `BreadthOutcomeReport`
  - `BreadthOutcomeCounter`
  - `BreadthCoverageReport.from_parts(price_coverage, outcomes)`
- Preserves all existing `BreadthCoverageReport` convenience properties and
  serialized dictionary keys.

- [ ] **Step 1: Write failing unit tests for the composed report**

Replace direct mixed-accumulator construction in `test_breadth_coverage.py`
with:

```python
def _price_coverage():
    accumulator = BreadthPriceCoverageAccumulator()
    accumulator.record_batch(["AAA", "BBB", "MISS"], ["MISS"])
    return accumulator.report()


def test_composed_report_keeps_unique_symbols_separate_from_observations():
    outcomes = BreadthOutcomeCounter()
    outcomes.record_scanned()
    outcomes.record_insufficient()
    outcomes.record_cache_miss()

    report = BreadthCoverageReport.from_parts(
        _price_coverage(),
        outcomes.report(),
    )

    assert report.candidate_stocks == 3
    assert report.cache_miss_stocks == 1
    assert report.total_stocks_scanned == 1
    assert report.skipped_stocks == 2
    assert report.insufficient_history_observations == 1


def test_many_date_outcomes_share_one_price_coverage_value():
    price_coverage = _price_coverage()
    first = BreadthCoverageReport.from_parts(
        price_coverage,
        BreadthOutcomeReport(scanned=1),
    )
    second = BreadthCoverageReport.from_parts(
        price_coverage,
        BreadthOutcomeReport(cache_misses=1),
    )

    assert first.price_coverage is price_coverage
    assert second.price_coverage is price_coverage
```

Retain deterministic sample and serializer assertions.

- [ ] **Step 2: Add a breadth-backfill architecture regression test**

Add to `test_breadth_calculator_service.py`:

```python
import app.services.breadth_calculator_service as breadth_module


def test_backfill_allocates_symbol_coverage_once(monkeypatch):
    created = 0
    real_accumulator = breadth_module.BreadthPriceCoverageAccumulator

    class CountingAccumulator(real_accumulator):
        def __init__(self):
            nonlocal created
            created += 1
            super().__init__()

    monkeypatch.setattr(
        breadth_module,
        "BreadthPriceCoverageAccumulator",
        CountingAccumulator,
    )
    db = _make_db_session()
    db.add_all([
        StockUniverse(
            symbol="AAA",
            is_active=True,
            status=UNIVERSE_STATUS_ACTIVE,
        ),
        StockUniverse(
            symbol="BBB",
            is_active=True,
            status=UNIVERSE_STATUS_ACTIVE,
        ),
    ])
    db.commit()
    trading_dates = [date(2026, 3, 19), date(2026, 3, 20)]
    history = _make_price_df(trading_dates[-1])
    price_cache = MagicMock()
    price_cache.get_many_cached_only_fresh.return_value = {
        "AAA": history,
        "BBB": history,
    }
    service = BreadthCalculatorService(db, price_cache)

    service.backfill_range(
        trading_dates[0],
        trading_dates[-1],
        trading_dates=trading_dates,
        policy=_policy("refresh_guarded", trading_dates[-1]),
    )

    assert created == 1
```

Do not replace this with a fake that bypasses `backfill_range()`.

- [ ] **Step 3: Run breadth coverage/service tests and verify RED**

```bash
cd backend
source venv/bin/activate
pytest \
  tests/unit/test_breadth_coverage.py \
  tests/unit/test_breadth_calculator_service.py -q
```

Expected: import and allocation-count failures because the split types do not
exist and backfill still creates one set accumulator per date.

- [ ] **Step 4: Implement the split coverage types**

Refactor `breadth_coverage.py` around:

```python
@dataclass(frozen=True)
class BreadthPriceCoverage:
    candidate_stocks: int
    symbols_with_cached_history: int
    cache_miss_stocks: int
    cache_miss_symbols_sample: tuple[str, ...]
    cache_coverage_ratio: float


@dataclass
class BreadthPriceCoverageAccumulator:
    _candidate_symbols: set[str] = field(default_factory=set)
    _cached_symbols: set[str] = field(default_factory=set)
    _cache_miss_symbols: set[str] = field(default_factory=set)

    def record_batch(
        self,
        candidate_symbols: Iterable[str],
        cache_miss_symbols: Iterable[str],
    ) -> None:
        candidates = set(candidate_symbols)
        misses = set(cache_miss_symbols)
        self._candidate_symbols.update(candidates)
        self._cache_miss_symbols.update(misses)
        self._cached_symbols.update(candidates - misses)

    def report(self) -> BreadthPriceCoverage:
        candidate_count = len(self._candidate_symbols)
        cached_count = len(self._cached_symbols)
        return BreadthPriceCoverage(
            candidate_stocks=candidate_count,
            symbols_with_cached_history=cached_count,
            cache_miss_stocks=len(self._cache_miss_symbols),
            cache_miss_symbols_sample=tuple(
                sorted(self._cache_miss_symbols)[:CACHE_MISS_SYMBOL_SAMPLE_LIMIT]
            ),
            cache_coverage_ratio=(
                cached_count / candidate_count if candidate_count else 0.0
            ),
        )


@dataclass(frozen=True)
class BreadthOutcomeReport:
    scanned: int = 0
    cache_misses: int = 0
    insufficient: int = 0
    errors: int = 0

    @property
    def skipped(self) -> int:
        return self.cache_misses + self.insufficient + self.errors

    def __add__(self, other: "BreadthOutcomeReport") -> "BreadthOutcomeReport":
        return BreadthOutcomeReport(
            scanned=self.scanned + other.scanned,
            cache_misses=self.cache_misses + other.cache_misses,
            insufficient=self.insufficient + other.insufficient,
            errors=self.errors + other.errors,
        )


@dataclass
class BreadthOutcomeCounter:
    _scanned: int = 0
    _cache_misses: int = 0
    _insufficient: int = 0
    _errors: int = 0

    # record_* methods increment exactly one counter.

    def report(self) -> BreadthOutcomeReport:
        return BreadthOutcomeReport(
            scanned=self._scanned,
            cache_misses=self._cache_misses,
            insufficient=self._insufficient,
            errors=self._errors,
        )
```

`BreadthCoverageReport` stores `price_coverage` and `outcomes`, then exposes the
existing properties and serializers.

- [ ] **Step 5: Refactor daily and backfill accounting**

Daily calculation creates one price accumulator and one outcome counter.

Backfill uses:

```python
price_coverage = BreadthPriceCoverageAccumulator()
outcomes_by_date = {
    calc_date: BreadthOutcomeCounter()
    for calc_date in ordered_dates
}
```

Each batch calls `price_coverage.record_batch(...)` once. Each stock/date
calculation updates only the relevant outcome counter. Aggregate diagnostics
are created with:

```python
aggregate_outcomes = sum(
    (counter.report() for counter in outcomes_by_date.values()),
    start=BreadthOutcomeReport(),
)
overall_report = BreadthCoverageReport.from_parts(
    price_coverage.report(),
    aggregate_outcomes,
)
```

Each persisted daily record composes the same price coverage report with that
date's outcome report.

- [ ] **Step 6: Run breadth tests and verify GREEN**

Run the Step 3 command.

Expected: all tests pass, including current daily/backfill dictionary tests.

- [ ] **Step 7: Commit breadth coverage decomposition**

```bash
git add \
  backend/app/services/breadth_coverage.py \
  backend/app/services/breadth_calculator_service.py \
  backend/tests/unit/test_breadth_coverage.py \
  backend/tests/unit/test_breadth_calculator_service.py
git commit -m "refactor: split breadth coverage from outcomes"
```

---

### Task 3: Add explicit group input sources and isolate legacy conversion

**Files:**

- Create: `backend/app/services/group_rank_input_sources.py`
- Create: `backend/app/services/group_rank_legacy_adapter.py`
- Create: `backend/tests/unit/test_group_rank_legacy_adapter.py`
- Modify: `backend/app/services/group_rank_models.py`
- Modify: `backend/app/services/group_rank_input_loader.py`
- Modify: `backend/app/services/ibd_group_rank_service.py`
- Modify: `backend/app/wiring/bootstrap.py`
- Modify: `backend/tests/unit/test_group_rank_models.py`
- Modify: `backend/tests/unit/test_group_rank_input_loader.py`
- Modify: `backend/tests/unit/test_group_rank_service.py`
- Modify: `backend/tests/unit/test_static_rrg_history_bundle.py`

**Interfaces:**

- Produces:
  - `GroupRankUniverseSource`
  - `GroupRankTaxonomySource`
  - `GroupRankMarketCapSource`
  - `StockUniverseGroupRankSource`
  - `IBDIndustryTaxonomySource`
  - `SqlGroupRankMarketCapSource`
  - `LegacyGroupRankPrefetchAdapter.adapt(value)`
  - `GroupRankInputLoader.complete_legacy_symbols(...)`
- Core models accept typed values only.

- [ ] **Step 1: Write failing strict-model tests**

Replace permissive model tests with:

```python
def test_prefetch_data_rejects_legacy_stats_mapping():
    with pytest.raises(TypeError, match="GroupRankPrefetchStats"):
        GroupRankPrefetchData(
            benchmark_prices=pd.DataFrame({"Close": [100.0]}),
            prices_by_symbol={},
            active_symbols=frozenset(),
            market_caps={},
            stats={"spy_cached": True},
            symbols_by_group={},
        )


def test_prefetch_data_rejects_mutable_group_symbol_lists():
    with pytest.raises(TypeError, match="tuple"):
        GroupRankPrefetchData(
            benchmark_prices=None,
            prices_by_symbol={},
            active_symbols=frozenset(),
            market_caps={},
            stats=_stats(),
            symbols_by_group={"Software": ["AAA"]},
        )
```

- [ ] **Step 2: Write failing adapter tests**

Create `test_group_rank_legacy_adapter.py`:

```python
def _adapter_price_frame() -> pd.DataFrame:
    index = pd.bdate_range(end="2026-03-20", periods=260)
    return pd.DataFrame(
        {
            "Open": 100.0,
            "High": 101.0,
            "Low": 99.0,
            "Close": 100.0,
            "Volume": 1_000_000,
        },
        index=index,
    )


def _adapter_stats() -> GroupRankPrefetchStats:
    return GroupRankPrefetchStats(
        target_symbols=1,
        symbols_with_prices=1,
        cache_miss_symbols=0,
        cache_miss_symbols_sample=(),
        cache_coverage_ratio=1.0,
        benchmark_available=True,
        benchmark_cached=True,
        benchmark_symbol="SPY",
        benchmark_role="primary",
        market="US",
        cache_only=True,
        skipped_unsupported_symbols=0,
    )


def _typed_prefetch() -> GroupRankPrefetchData:
    prices = _adapter_price_frame()
    return GroupRankPrefetchData(
        benchmark_prices=prices,
        prices_by_symbol={"AAA": prices},
        active_symbols=frozenset({"AAA"}),
        market_caps={"AAA": 1_000_000},
        stats=_adapter_stats(),
        symbols_by_group={"Software": ("AAA",)},
    )


def test_adapter_converts_legacy_five_tuple():
    prices = _adapter_price_frame()
    legacy = (
        prices,
        {"AAA": prices},
        {"AAA"},
        {"AAA": 1_000_000},
        {
            "target_symbols": 2,
            "symbols_with_prices": 1,
            "cache_miss_symbols": 1,
            "spy_cached": True,
        },
    )

    adapted = LegacyGroupRankPrefetchAdapter().adapt(legacy)

    assert adapted.active_symbols == frozenset({"AAA"})
    assert adapted.stats.cache_coverage_ratio == 0.5
    assert adapted.stats.benchmark_available is True
    assert adapted.symbols_by_group == {}


def test_adapter_returns_typed_prefetch_unchanged():
    prefetch = _typed_prefetch()

    assert LegacyGroupRankPrefetchAdapter().adapt(prefetch) is prefetch


def test_adapter_rejects_unknown_legacy_shape():
    with pytest.raises(TypeError, match="legacy group prefetch"):
        LegacyGroupRankPrefetchAdapter().adapt(("too", "short"))
```

- [ ] **Step 3: Rewrite loader tests around fake source ports**

Define small fakes inside `test_group_rank_input_loader.py`:

```python
@dataclass
class FakeUniverseSource:
    symbols: tuple[str, ...]

    def active_symbols(self, db, market):
        return frozenset(self.symbols)


@dataclass
class FakeTaxonomySource:
    symbols_by_group: dict[str, tuple[str, ...]]

    def groups(self, db, market):
        return tuple(self.symbols_by_group)

    def symbols_for_group(self, db, group, market):
        return self.symbols_by_group[group]


@dataclass
class FakeMarketCapSource:
    values: dict[str, float]

    def market_caps(self, db, symbols):
        return {
            symbol: self.values[symbol]
            for symbol in symbols
            if symbol in self.values
        }
```

Construct the loader with all three sources. Remove monkeypatches of wiring,
static `IBDIndustryService`, `_market_caps`, and service-private methods.

Add a source-level test:

```python
def test_loader_has_no_service_location_or_facade_callback():
    source = Path(group_loader_module.__file__).read_text()
    assert "wiring.bootstrap" not in source
    assert "IBDGroupRankService" not in source
    assert "getattr(" not in source
```

- [ ] **Step 4: Run model/adapter/loader tests and verify RED**

```bash
cd backend
source venv/bin/activate
pytest \
  tests/unit/test_group_rank_models.py \
  tests/unit/test_group_rank_legacy_adapter.py \
  tests/unit/test_group_rank_input_loader.py \
  tests/unit/test_group_rank_service.py::test_calculate_group_rankings_rejects_incomplete_cache_only_inputs \
  tests/unit/test_static_rrg_history_bundle.py -q
```

Expected: missing modules/classes and strict-model failures.

- [ ] **Step 5: Implement source protocols and defaults**

Create `group_rank_input_sources.py`:

```python
class GroupRankUniverseSource(Protocol):
    def active_symbols(
        self,
        db: Session,
        market: str,
    ) -> frozenset[str]: ...


class GroupRankTaxonomySource(Protocol):
    def groups(self, db: Session, market: str) -> tuple[str, ...]: ...
    def symbols_for_group(
        self,
        db: Session,
        group: str,
        market: str,
    ) -> tuple[str, ...]: ...


class GroupRankMarketCapSource(Protocol):
    def market_caps(
        self,
        db: Session,
        symbols: Sequence[str],
    ) -> dict[str, float]: ...


@dataclass(frozen=True)
class StockUniverseGroupRankSource:
    service: StockUniverseService

    def active_symbols(self, db: Session, market: str) -> frozenset[str]:
        return frozenset(
            self.service.get_active_symbols(db, market=market)
        )


@dataclass(frozen=True)
class IBDIndustryTaxonomySource:
    def groups(self, db: Session, market: str) -> tuple[str, ...]:
        return tuple(IBDIndustryService.get_all_groups(db, market=market))

    def symbols_for_group(
        self,
        db: Session,
        group: str,
        market: str,
    ) -> tuple[str, ...]:
        return tuple(
            IBDIndustryService.get_group_symbols(
                db,
                group,
                market=market,
            )
        )


@dataclass(frozen=True)
class SqlGroupRankMarketCapSource:
    def market_caps(
        self,
        db: Session,
        symbols: Sequence[str],
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

- [ ] **Step 6: Make core models strict and add the legacy adapter**

`GroupRankPrefetchData.__post_init__()` validates without mutation:

```python
def __post_init__(self) -> None:
    if not isinstance(self.active_symbols, frozenset):
        raise TypeError("active_symbols must be frozenset[str]")
    if not isinstance(self.stats, GroupRankPrefetchStats):
        raise TypeError("stats must be GroupRankPrefetchStats")
    for group, symbols in self.symbols_by_group.items():
        if not isinstance(symbols, tuple):
            raise TypeError(
                f"symbols_by_group[{group!r}] must be tuple[str, ...]"
            )
```

Move all former `from_mapping()` logic into:

```python
@dataclass(frozen=True)
class LegacyGroupRankPrefetchAdapter:
    def adapt(self, value: object) -> GroupRankPrefetchData:
        if isinstance(value, GroupRankPrefetchData):
            return value
        if not isinstance(value, tuple) or len(value) != 5:
            raise TypeError(
                "unsupported legacy group prefetch value; expected typed "
                "GroupRankPrefetchData or five-item tuple"
            )
        benchmark, prices, active, market_caps, raw_stats = value
        if not isinstance(raw_stats, Mapping):
            raise TypeError("legacy group prefetch stats must be a mapping")
        return GroupRankPrefetchData(
            benchmark_prices=benchmark,
            prices_by_symbol=prices,
            active_symbols=frozenset(active),
            market_caps=market_caps,
            stats=self._stats(raw_stats),
            symbols_by_group={},
        )
```

The private `_stats()` method contains the exact old mapping/default conversion.

- [ ] **Step 7: Refactor the loader and composition root**

`GroupRankInputLoader.__init__()` requires the source ports:

```python
def __init__(
    self,
    *,
    price_cache: PriceCacheService,
    benchmark_cache: BenchmarkCacheService,
    universe_source: GroupRankUniverseSource,
    taxonomy_source: GroupRankTaxonomySource,
    market_cap_source: GroupRankMarketCapSource,
) -> None:
```

Use the source methods directly. Call
`benchmark_cache.get_benchmark_candidates(market)` directly; use the primary
symbol when it returns an empty list, and allow exceptions to propagate.

Add:

```python
def complete_legacy_symbols(
    self,
    db: Session,
    *,
    market: str,
    group_names: Sequence[str],
    prefetch: GroupRankPrefetchData,
) -> GroupRankPrefetchData:
    if prefetch.symbols_by_group:
        return prefetch
    symbols_by_group = self._validated_symbols_by_group(
        db,
        market=market,
        group_names=group_names,
        active_symbols=prefetch.active_symbols,
    )
    return replace(prefetch, symbols_by_group=symbols_by_group)
```

In `RuntimeServices.group_rank_service()`, build the loader with:

```python
input_loader = GroupRankInputLoader(
    price_cache=cache_bundle.price,
    benchmark_cache=cache_bundle.benchmark,
    universe_source=StockUniverseGroupRankSource(
        self.stock_universe_service()
    ),
    taxonomy_source=IBDIndustryTaxonomySource(),
    market_cap_source=SqlGroupRankMarketCapSource(),
)
```

Pass it to `IBDGroupRankService`. Remove the market-cap callback and duplicated
loader query. The facade uses `LegacyGroupRankPrefetchAdapter` at its
compatibility seam.

- [ ] **Step 8: Run focused tests and verify GREEN**

Run the Step 4 command.

Expected: all pass.

- [ ] **Step 9: Commit input ownership and adapter**

```bash
git add \
  backend/app/services/group_rank_input_sources.py \
  backend/app/services/group_rank_legacy_adapter.py \
  backend/app/services/group_rank_models.py \
  backend/app/services/group_rank_input_loader.py \
  backend/app/services/ibd_group_rank_service.py \
  backend/app/wiring/bootstrap.py \
  backend/tests/unit/test_group_rank_models.py \
  backend/tests/unit/test_group_rank_legacy_adapter.py \
  backend/tests/unit/test_group_rank_input_loader.py \
  backend/tests/unit/test_group_rank_service.py \
  backend/tests/unit/test_static_rrg_history_bundle.py
git commit -m "refactor: isolate group ranking input compatibility"
```

---

### Task 4: Extract pure group-ranking calculation

**Files:**

- Create: `backend/app/services/group_ranking_calculator.py`
- Create: `backend/tests/unit/test_group_ranking_calculator.py`
- Modify: `backend/app/services/ibd_group_rank_service.py`
- Modify: `backend/tests/unit/test_group_rank_service.py`

**Interfaces:**

- Produces:
  - `GroupRankingCalculator.calculate_for_date(...)`
  - `GroupRankingCalculator.calculate_for_dates(...)`
- Consumes typed `GroupRankPrefetchData` with completed
  `symbols_by_group`.
- Produces ranked tuple dictionaries without database writes.

- [ ] **Step 1: Move calculation behavior into characterization tests**

Create `test_group_ranking_calculator.py` using typed prefetch fixtures. Port
the existing vectorized parity tests and add:

```python
def _trend_frame(daily_step: float) -> pd.DataFrame:
    index = pd.bdate_range(end="2026-03-20", periods=260)
    closes = [
        100.0 + daily_step * item
        for item in range(len(index))
    ]
    return pd.DataFrame(
        {
            "Open": closes,
            "High": [value + 1 for value in closes],
            "Low": [value - 1 for value in closes],
            "Close": closes,
            "Volume": 1_000_000,
        },
        index=index,
    )


def _prefetch_for_two_groups() -> GroupRankPrefetchData:
    benchmark = _trend_frame(0.2)
    prices = {
        "AAA": _trend_frame(0.6),
        "BBB": _trend_frame(0.55),
        "CCC": _trend_frame(0.5),
        "DDD": _trend_frame(0.15),
        "EEE": _trend_frame(0.1),
        "FFF": _trend_frame(0.05),
    }
    return GroupRankPrefetchData(
        benchmark_prices=benchmark,
        prices_by_symbol=prices,
        active_symbols=frozenset(prices),
        market_caps={symbol: 1_000_000 for symbol in prices},
        stats=GroupRankPrefetchStats(
            target_symbols=6,
            symbols_with_prices=6,
            cache_miss_symbols=0,
            cache_miss_symbols_sample=(),
            cache_coverage_ratio=1.0,
            benchmark_available=True,
            benchmark_cached=True,
            benchmark_symbol="SPY",
            benchmark_role="primary",
            market="US",
            cache_only=True,
            skipped_unsupported_symbols=0,
        ),
        symbols_by_group={
            "Software": ("AAA", "BBB", "CCC"),
            "Retail": ("DDD", "EEE", "FFF"),
        },
    )


def test_calculate_for_date_returns_ranked_groups_without_mutating_prefetch():
    prefetch = _prefetch_for_two_groups()
    calculator = GroupRankingCalculator(
        rs_calculator=RelativeStrengthCalculator()
    )

    rankings = calculator.calculate_for_date(
        prefetch=prefetch,
        group_names=("Software", "Retail"),
        calculation_date=date(2026, 3, 20),
    )

    assert [row["rank"] for row in rankings] == [1, 2]
    assert rankings[0]["avg_rs_rating"] >= rankings[1]["avg_rs_rating"]
    assert prefetch.symbols_by_group == {
        "Software": ("AAA", "BBB", "CCC"),
        "Retail": ("DDD", "EEE", "FFF"),
    }


def test_calculator_performs_no_database_or_cache_reads():
    signature = inspect.signature(
        GroupRankingCalculator.calculate_for_date
    )
    assert "db" not in signature.parameters
```

Port:

- `test_vectorized_group_rs_matches_legacy_cache_path_and_excludes_short_history`
- `test_vectorized_group_rs_preserves_invalid_period_return_semantics`

The assertions must remain unchanged.

- [ ] **Step 2: Run calculator tests and verify RED**

```bash
cd backend
source venv/bin/activate
pytest tests/unit/test_group_ranking_calculator.py -q
```

Expected: module/class missing.

- [ ] **Step 3: Implement `GroupRankingCalculator`**

Move these helpers from `IBDGroupRankService` without changing formulas:

- `_calculate_pct_above_80`
- `_calculate_group_rs_from_cache`
- `_calculate_rs_by_symbol_for_dates`
- `_close_series`
- `_period_returns`
- `_positions_by_date`
- `_series_value_at`
- `_scale_relative_performance_to_rs`
- `_calculate_group_metrics_from_rs`

Expose:

```python
@dataclass(frozen=True)
class GroupRankingCalculator:
    rs_calculator: RelativeStrengthCalculator

    def calculate_for_date(
        self,
        *,
        prefetch: GroupRankPrefetchData,
        group_names: Sequence[str],
        calculation_date: date,
    ) -> tuple[Mapping[str, Any], ...]:
        by_date = self.calculate_for_dates(
            prefetch=prefetch,
            group_names=group_names,
            calculation_dates=(calculation_date,),
        )
        return by_date.get(calculation_date, ())

    def calculate_for_dates(
        self,
        *,
        prefetch: GroupRankPrefetchData,
        group_names: Sequence[str],
        calculation_dates: Sequence[date],
    ) -> dict[date, tuple[Mapping[str, Any], ...]]:
        rs_by_date = self._calculate_rs_by_symbol_for_dates(
            prefetch,
            list(calculation_dates),
        )
        result: dict[date, tuple[Mapping[str, Any], ...]] = {}
        for calculation_date in calculation_dates:
            metrics = [
                group_metrics
                for group_name in group_names
                if (
                    group_metrics := self._calculate_group_metrics_from_rs(
                        group_name,
                        prefetch.symbols_by_group.get(group_name, ()),
                        rs_by_date.get(calculation_date, {}),
                        prefetch.market_caps,
                        calculation_date,
                    )
                )
            ]
            metrics.sort(
                key=lambda item: item["avg_rs_rating"],
                reverse=True,
            )
            for rank, item in enumerate(metrics, start=1):
                item["rank"] = rank
            result[calculation_date] = tuple(metrics)
        return result
```

- [ ] **Step 4: Delegate facade calculation**

Inject `GroupRankingCalculator` into `IBDGroupRankService`. Its
`calculate_group_rankings()`:

1. obtains group names through the input loader/taxonomy source;
2. adapts any legacy prefetch only at the facade seam;
3. completes missing legacy `symbols_by_group`;
4. applies cache requirements;
5. calls `calculator.calculate_for_date(...)`;
6. delegates persistence to the still-existing facade `_store_rankings()` until
   Task 5;
7. returns `GroupRankCalculationResult`.

Delete the moved private helpers from the facade after all production calls use
the calculator.

- [ ] **Step 5: Run calculator and service tests**

```bash
cd backend
source venv/bin/activate
pytest \
  tests/unit/test_group_ranking_calculator.py \
  tests/unit/test_group_rank_service.py -q
```

Expected: all pass.

- [ ] **Step 6: Commit calculator extraction**

```bash
git add \
  backend/app/services/group_ranking_calculator.py \
  backend/app/services/ibd_group_rank_service.py \
  backend/tests/unit/test_group_ranking_calculator.py \
  backend/tests/unit/test_group_rank_service.py
git commit -m "refactor: extract group ranking calculator"
```

---

### Task 5: Extract ranking persistence and table queries

**Files:**

- Create: `backend/app/services/group_ranking_repository.py`
- Create: `backend/tests/unit/test_group_ranking_repository.py`
- Modify: `backend/app/services/ibd_group_rank_service.py`
- Modify: `backend/tests/unit/test_group_rank_service.py`

**Interfaces:**

- Produces:
  - `GroupRankingRepository.store_rankings(...)`
  - `GroupRankingRepository.delete_range(...)`
  - `GroupRankingRepository.current_rank_rows(...)`
  - `GroupRankingRepository.historical_ranks_batch(...)`
  - `GroupRankingRepository.group_rank_rows(...)`
- Repository methods receive `Session`; no constructor-owned session and no
  commit.

- [ ] **Step 1: Add repository characterization tests**

Create `test_group_ranking_repository.py`. Port the SQLite bulk-upsert test and
the current/historical date-selection tests. Add:

```python
def _ranking(group: str, *, rank: int) -> dict[str, object]:
    return {
        "industry_group": group,
        "rank": rank,
        "avg_rs_rating": 80.0,
        "median_rs_rating": 80.0,
        "weighted_avg_rs_rating": 80.0,
        "rs_std_dev": 0.0,
        "num_stocks": 3,
        "num_stocks_rs_above_80": 1,
        "top_symbol": "AAA",
        "top_rs_rating": 90.0,
    }


def _seed_rank(
    db_session,
    *,
    market: str,
    calculation_date: date,
) -> None:
    db_session.add(
        IBDGroupRank(
            market=market,
            date=calculation_date,
            industry_group="Software",
            rank=1,
            avg_rs_rating=80.0,
            median_rs_rating=80.0,
            weighted_avg_rs_rating=80.0,
            rs_std_dev=0.0,
            num_stocks=3,
            num_stocks_rs_above_80=1,
            top_symbol="AAA",
            top_rs_rating=90.0,
        )
    )


def test_store_rankings_does_not_commit(db_session, monkeypatch):
    repository = GroupRankingRepository()
    commit = Mock(side_effect=AssertionError("repository must not commit"))
    monkeypatch.setattr(db_session, "commit", commit)

    repository.store_rankings(
        db_session,
        calculation_date=date(2026, 3, 20),
        rankings=(_ranking("Software", rank=1),),
        market="US",
    )

    commit.assert_not_called()


def test_delete_range_is_market_scoped(db_session):
    _seed_rank(db_session, market="US", calculation_date=date(2026, 3, 20))
    _seed_rank(db_session, market="JP", calculation_date=date(2026, 3, 20))
    db_session.commit()

    deleted = GroupRankingRepository().delete_range(
        db_session,
        start_date=date(2026, 3, 20),
        end_date=date(2026, 3, 20),
        market="US",
    )

    assert deleted == 1
    assert db_session.query(IBDGroupRank).filter_by(market="JP").count() == 1
```

- [ ] **Step 2: Run repository tests and verify RED**

```bash
cd backend
source venv/bin/activate
pytest tests/unit/test_group_ranking_repository.py -q
```

Expected: module/class missing.

- [ ] **Step 3: Implement the repository**

Move persistence and direct ranking-table query logic from the facade:

- `_store_rankings`
- `_ranking_values`
- `_store_rankings_sqlalchemy_fallback`
- `_delete_rankings_for_range`
- latest-date and current-row query portion of `get_current_rankings`
- `get_historical_ranks_batch` and `_get_historical_ranks_batch`
- ranking-row portion of `get_group_history`

Use exact public repository signatures:

```python
class GroupRankingRepository:
    def store_rankings(
        self,
        db: Session,
        *,
        calculation_date: date,
        rankings: Sequence[Mapping[str, Any]],
        market: str,
    ) -> None: ...

    def delete_range(
        self,
        db: Session,
        *,
        start_date: date,
        end_date: date,
        market: str,
    ) -> int: ...

    def current_rank_rows(
        self,
        db: Session,
        *,
        limit: int,
        market: str,
        calculation_date: date | None,
    ) -> list[IBDGroupRank]: ...

    def historical_ranks_batch(
        self,
        db: Session,
        *,
        group_names: Sequence[str],
        current_date: date,
        period_days: Mapping[str, int],
        market: str,
    ) -> dict[tuple[str, str], int]: ...

    def group_rank_rows(
        self,
        db: Session,
        *,
        industry_group: str,
        start_date: date,
        market: str,
    ) -> list[IBDGroupRank]: ...
```

Preserve PostgreSQL upsert and SQLite fallback behavior exactly. Do not call
`db.commit()`.

- [ ] **Step 4: Delegate facade methods**

Inject the repository. Replace facade persistence and direct table reads with
repository calls. Keep facade-owned enrichment:

- symbol-name annotation;
- constituent detail payload construction;
- calendar offset selection;
- response dictionary formatting that combines ranking rows with constituent
  data.

Delete moved imports (`pg_insert`, direct ranking-write helpers) from the
facade.

- [ ] **Step 5: Run repository and service tests**

```bash
cd backend
source venv/bin/activate
pytest \
  tests/unit/test_group_ranking_repository.py \
  tests/unit/test_group_rank_service.py \
  tests/unit/test_group_detail_payloads.py -q
```

Expected: all pass.

- [ ] **Step 6: Commit repository extraction**

```bash
git add \
  backend/app/services/group_ranking_repository.py \
  backend/app/services/ibd_group_rank_service.py \
  backend/tests/unit/test_group_ranking_repository.py \
  backend/tests/unit/test_group_rank_service.py
git commit -m "refactor: extract group ranking repository"
```

---

### Task 6: Extract historical calculation and finish the compatibility facade

**Files:**

- Create: `backend/app/services/group_rank_historical_calculator.py`
- Create: `backend/tests/unit/test_group_rank_historical_calculator.py`
- Modify: `backend/app/services/ibd_group_rank_service.py`
- Modify: `backend/app/wiring/bootstrap.py`
- Modify: `backend/tests/unit/test_group_rank_service.py`
- Modify: `backend/tests/unit/test_group_rank_execution_policy.py`
- Modify: `backend/tests/unit/test_group_rank_backfill_tasks.py`
- Modify: `backend/tests/unit/test_group_rank_history_backfill_service.py`

**Interfaces:**

- Produces public-equivalent methods on `GroupRankHistoricalCalculator`:
  - `backfill_rankings_optimized`
  - `backfill_rankings`
  - `find_missing_dates`
  - `fill_gaps`
  - `fill_gaps_optimized`
  - `backfill_rankings_chunked`
- `IBDGroupRankService` retains the same methods as delegating wrappers.

- [ ] **Step 1: Port historical behavior to focused tests**

Create `test_group_rank_historical_calculator.py` and port these cases from
`test_group_rank_service.py`:

- optimized backfill accepts legacy adapter input;
- optimized backfill uses the market calendar;
- optimized backfill chunks RS date calculation;
- nonoptimized backfill checks existing rows by market;
- legacy tuple fallback completes validated group symbols;
- optimized gap-fill returns prefetch stats and propagates policy;
- optimized gap-fill uses prefetched group symbols without inner lookup;
- optimized gap-fill chunks RS calculation.

Construct the calculator with fake loader, calculator, repository, calendar,
and adapter. Add:

```python
def test_historical_calculator_does_not_locate_calendar_or_services():
    source = Path(historical_module.__file__).read_text()
    assert "wiring.bootstrap" not in source
    assert "get_market_calendar_service" not in source
    assert "IBDGroupRankService" not in source
```

- [ ] **Step 2: Run historical tests and verify RED**

```bash
cd backend
source venv/bin/activate
pytest tests/unit/test_group_rank_historical_calculator.py -q
```

Expected: module/class missing.

- [ ] **Step 3: Implement `GroupRankHistoricalCalculator`**

Use:

```python
@dataclass(frozen=True)
class GroupRankHistoricalCalculator:
    input_loader: GroupRankInputLoader
    ranking_calculator: GroupRankingCalculator
    repository: GroupRankingRepository
    calendar_service: MarketCalendarService
    legacy_adapter: LegacyGroupRankPrefetchAdapter
```

Move the six historical methods listed in Interfaces. Replace:

- direct `_prefetch_all_data()` with `input_loader.load(...)`;
- `_coerce_prefetch_data()` with `legacy_adapter.adapt(...)`;
- static group queries with loader taxonomy methods;
- direct group metric loops with
  `ranking_calculator.calculate_for_dates(...)`;
- direct persistence/deletion with repository methods;
- service-location calendar lookup with the injected calendar.

`fill_gaps_optimized()` retains its exact result keys, including
`prefetch_stats`. `backfill_rankings_optimized()` retains delete-before-rebuild
behavior. Legacy/nonoptimized methods retain skip-existing semantics.

- [ ] **Step 4: Wire and delegate**

Build the historical calculator in `RuntimeServices.group_rank_service()` and
pass it to the facade.

The facade wrappers are mechanically small:

```python
def fill_gaps_optimized(
    self,
    db: Session,
    missing_dates: List[date],
    *,
    market: str = "US",
    policy: DerivedDataExecutionPolicy = (
        DerivedDataExecutionPolicy.provider_allowed()
    ),
) -> Dict:
    return self.historical_calculator.fill_gaps_optimized(
        db,
        missing_dates,
        market=market,
        policy=policy,
    )
```

Repeat for all six public methods. Remove historical calculation loops,
`gc.collect()`, and calendar service-location imports from the facade.

- [ ] **Step 5: Add facade architecture assertions**

Add to `test_group_rank_service.py`:

```python
def test_group_rank_service_is_a_compatibility_facade():
    source = Path(group_rank_module.__file__).read_text()
    assert "gc.collect" not in source
    assert "get_market_calendar_service" not in source
    assert "pg_insert" not in source
    assert "StockUniverse" not in source
    assert source.count("\\n") < 900
```

Also assert that every existing public method is present on the facade:

```python
for method_name in (
    "calculate_group_rankings",
    "get_current_rankings",
    "get_historical_ranks_batch",
    "get_group_history",
    "get_rank_movers",
    "backfill_rankings_optimized",
    "backfill_rankings",
    "find_missing_dates",
    "fill_gaps",
    "fill_gaps_optimized",
    "backfill_rankings_chunked",
):
    assert hasattr(IBDGroupRankService, method_name)
```

- [ ] **Step 6: Run historical, facade, task, and static-export tests**

```bash
cd backend
source venv/bin/activate
pytest \
  tests/unit/test_group_rank_historical_calculator.py \
  tests/unit/test_group_rank_service.py \
  tests/unit/test_group_rank_execution_policy.py \
  tests/unit/test_group_rank_backfill_tasks.py \
  tests/unit/test_group_rank_history_backfill_service.py \
  tests/unit/test_static_rrg_history_bundle.py \
  tests/unit/test_export_static_site_script.py -q
```

Expected: all pass.

- [ ] **Step 7: Commit historical extraction and facade cleanup**

```bash
git add \
  backend/app/services/group_rank_historical_calculator.py \
  backend/app/services/ibd_group_rank_service.py \
  backend/app/wiring/bootstrap.py \
  backend/tests/unit/test_group_rank_historical_calculator.py \
  backend/tests/unit/test_group_rank_service.py \
  backend/tests/unit/test_group_rank_execution_policy.py \
  backend/tests/unit/test_group_rank_backfill_tasks.py \
  backend/tests/unit/test_group_rank_history_backfill_service.py
git commit -m "refactor: make group rank service a facade"
```

---

### Task 7: Run architecture and behavioral quality gates

**Files:**

- Modify: `docs/superpowers/specs/2026-07-16-derived-data-thermo-review-remediation-design.md`
- Modify: `.beads/issues.jsonl`
- Modify: `.beads/interactions.jsonl`

**Interfaces:**

- Verifies every acceptance criterion without adding new production behavior.

- [ ] **Step 1: Run formatting and compile checks**

```bash
git diff --check origin/main...HEAD
cd backend
source venv/bin/activate
python -m compileall \
  app/services/derived_data_execution_policy.py \
  app/services/breadth_coverage.py \
  app/services/breadth_calculator_service.py \
  app/services/group_rank_models.py \
  app/services/group_rank_input_sources.py \
  app/services/group_rank_legacy_adapter.py \
  app/services/group_rank_input_loader.py \
  app/services/group_ranking_calculator.py \
  app/services/group_ranking_repository.py \
  app/services/group_rank_historical_calculator.py \
  app/services/ibd_group_rank_service.py \
  app/tasks/breadth_tasks.py \
  app/tasks/group_rank_tasks.py
```

Expected: no whitespace errors and all modules compile.

- [ ] **Step 2: Run the complete issue-301 focused suite**

```bash
cd backend
source venv/bin/activate
pytest \
  tests/unit/test_derived_data_execution_policy.py \
  tests/unit/test_breadth_coverage.py \
  tests/unit/test_breadth_calculator_service.py \
  tests/unit/test_breadth_tasks.py \
  tests/unit/test_group_rank_models.py \
  tests/unit/test_group_rank_legacy_adapter.py \
  tests/unit/test_group_rank_input_loader.py \
  tests/unit/test_group_ranking_calculator.py \
  tests/unit/test_group_ranking_repository.py \
  tests/unit/test_group_rank_historical_calculator.py \
  tests/unit/test_group_rank_service.py \
  tests/unit/test_group_rank_execution_policy.py \
  tests/unit/test_group_rank_backfill_tasks.py \
  tests/unit/test_group_rank_history_backfill_service.py \
  tests/unit/test_daily_market_pipeline_tasks.py \
  tests/unit/test_static_rrg_history_bundle.py \
  tests/unit/test_export_static_site_script.py -q
```

Expected: all pass.

- [ ] **Step 3: Run the backend unit suite**

```bash
cd backend
source venv/bin/activate
pytest tests/unit -q
```

Expected: all tests pass, subject only to already documented unrelated
repository baseline failures. Any new failure in touched modules must be fixed
before proceeding.

- [ ] **Step 4: Re-run the thermo-nuclear review against the branch diff**

Review `origin/main...HEAD` and resulting full files. Confirm:

- policy contains no stored capability booleans;
- task modules use the validation profile and policy helpers;
- loader has explicit sources and no service location;
- model has no legacy coercion;
- legacy adapter is the only mapping/tuple converter;
- breadth backfill owns one symbol coverage accumulator;
- facade is under 900 lines and owns no ranking calculation, persistence, or
  historical loops.

If any blocker remains, return to the relevant task with a new failing test.

- [ ] **Step 5: Record completion in the spec**

Change the spec header to:

```markdown
**Status:** Implemented and verified
```

Append:

```markdown
## Implementation verification

- Closed execution-policy tests: passed.
- Breadth shared-coverage and per-date outcome tests: passed.
- Group input-source and legacy-adapter tests: passed.
- Calculator, repository, historical calculator, and facade tests: passed.
- Issue-301 provider-exclusion and compatibility suite: passed.
- Backend unit suite: passed, or only documented unrelated baseline failures remained.
```

Replace the final line with exact test counts and any baseline exception; do
not claim a clean suite if it was not clean.

- [ ] **Step 6: Close the Beads task**

```bash
bd close stockscreenclaude-2c1 --reason="Addressed all derived-data thermo review findings: closed policy state, shared breadth price coverage, explicit group input ports, isolated legacy adapter, extracted calculator/repository/historical calculation, compatibility facade, and passing focused quality gates."
```

- [ ] **Step 7: Commit completion records**

Because the repository’s Beads hook currently exports a stray root
`issues.jsonl`, bypass the hook for this documentation-only commit and ensure
the stray file is absent:

```bash
git add \
  docs/superpowers/specs/2026-07-16-derived-data-thermo-review-remediation-design.md \
  .beads/issues.jsonl \
  .beads/interactions.jsonl
git commit --no-verify -m "docs: verify derived-data review remediation"
test ! -e issues.jsonl
```

- [ ] **Step 8: Rebase, push, and verify remote synchronization**

The installed Beads CLI has no `bd sync`, and its Dolt database has no remote.
Record that limitation, then run:

```bash
git pull --rebase
bd dolt push || true
git push
git status --short --branch
```

Expected:

- `bd dolt push` reports that no Dolt remote is configured;
- Git push succeeds;
- status is clean and says the branch is up to date with
  `origin/codex/issue-301-cache-only-derived-data`.
