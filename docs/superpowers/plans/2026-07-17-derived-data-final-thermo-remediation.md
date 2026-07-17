# Derived-data Final Thermo Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make group historical replacement atomic and failure-safe, establish one typed/single-pass group-ranking path, and replace nested Celery task invocation with ordinary daily runners while preserving every public compatibility contract.

**Architecture:** Immutable `GroupRanking` values cross the calculator/repository boundary, and the input loader returns the authoritative ordered taxonomy snapshot. Historical replacement calculates before mutation and commits one date atomically. Celery-free breadth/group daily runners own domain calculation and persistence; public tasks remain compatibility and retry adapters.

**Tech Stack:** Python 3.11, dataclasses, pandas, SQLAlchemy, Celery, pytest, PostgreSQL/SQLite repository adapters.

## Global Constraints

- Preserve public Celery task names and existing keyword arguments.
- Preserve `force_cache_only`, `refresh_guarded_cache_only`, and `execution_policy` precedence.
- Preserve refresh-guarded provider exclusion and provider-capable manual historical defaults.
- Preserve existing serialized task response keys and reason codes.
- Preserve `LegacyGroupRankPrefetchAdapter` support for five-item tuples.
- Do not change the `ibd_group_ranks` database schema or API schemas.
- Calculate historical replacements before deleting existing rows.
- A failed date replacement must leave that date's previous rows intact.
- Use red-green-refactor for every production change.

---

## File Structure

### Create

- `backend/app/services/daily_group_rank_runner.py`
  - Celery-free daily group-ranking execution, validation, finalization, and typed outcome serialization.
- `backend/app/services/daily_breadth_runner.py`
  - Celery-free daily breadth execution, completeness validation, persistence, finalization, and typed outcome serialization.
- `backend/tests/unit/test_daily_group_rank_runner.py`
  - Runner success, strict validation, no-groups, and finalization tests.
- `backend/tests/unit/test_daily_breadth_runner.py`
  - Runner validation, persistence, and response-compatibility tests.

### Modify

- `backend/app/services/group_rank_models.py`
  - Add immutable `GroupRanking`; add ordered `group_names`; type calculation results.
- `backend/app/services/group_ranking_calculator.py`
  - Construct typed candidates/rankings and remove the production legacy comparator.
- `backend/app/services/group_ranking_repository.py`
  - Persist typed values and atomically replace one date.
- `backend/app/services/group_rank_input_loader.py`
  - Own production group discovery and partition each membership tuple once.
- `backend/app/services/group_rank_legacy_adapter.py`
  - Populate the new typed field for legacy tuples.
- `backend/app/services/group_rank_historical_calculator.py`
  - Replace successful dates atomically; never delete before calculation.
- `backend/app/services/ibd_group_rank_service.py`
  - Consume loader-owned group names and delete unused private implementations.
- `backend/app/services/breadth_calculator_service.py`
  - Expose canonical single-date persistence for the breadth runner.
- `backend/app/tasks/group_rank_tasks.py`
  - Adapt public tasks to the runner; delete task-object introspection and transient `ContextVar` propagation.
- `backend/app/tasks/breadth_tasks.py`
  - Adapt public tasks to the runner; delete task-object introspection.
- `backend/app/tasks/group_rank_backfill_tasks.py`
  - Publish only when at least one date was successfully replaced.
- `backend/app/wiring/bootstrap.py`
  - Wire runner collaborators if the final implementation uses dependency factories.
- Existing group/breadth unit tests
  - Migrate fixtures to typed rankings and patch runners rather than decorated task bodies.

---

### Task 1: Introduce the immutable group-ranking boundary

**Files:**
- Modify: `backend/app/services/group_rank_models.py`
- Modify: `backend/app/services/group_ranking_calculator.py`
- Modify: `backend/app/services/group_ranking_repository.py`
- Modify: `backend/app/services/ibd_group_rank_service.py`
- Test: `backend/tests/unit/test_group_rank_models.py`
- Test: `backend/tests/unit/test_group_ranking_calculator.py`
- Test: `backend/tests/unit/test_group_ranking_repository.py`
- Test: `backend/tests/unit/test_group_rank_service.py`

**Interfaces:**
- Produces: `GroupRanking` with explicit immutable fields.
- Produces: `GroupRankCalculationResult.rankings: tuple[GroupRanking, ...]`.
- Changes: `GroupRankingCalculator.calculate_for_date(...) -> tuple[GroupRanking, ...]`.
- Changes: `GroupRankingRepository.store_rankings(..., rankings: Sequence[GroupRanking], ...) -> None`.

- [ ] **Step 1: Write failing model and calculator tests**

Add a model test that describes the required public contract:

```python
def test_calculation_result_contains_immutable_group_rankings():
    ranking = GroupRanking(
        industry_group="Software",
        date=date(2026, 3, 20),
        rank=1,
        avg_rs_rating=88.0,
        median_rs_rating=87.0,
        weighted_avg_rs_rating=89.0,
        rs_std_dev=3.0,
        num_stocks=12,
        num_stocks_rs_above_80=8,
        top_symbol="AAA",
        top_rs_rating=96.0,
    )
    result = GroupRankCalculationResult(
        rankings=(ranking,),
        prefetch_stats=_stats(),
    )

    assert result.rankings == (ranking,)
    with pytest.raises(FrozenInstanceError):
        ranking.rank = 2
```

Update the calculator test to assert attributes rather than dictionary keys:

```python
rankings = calculator.calculate_for_date(
    prefetch=prefetch,
    group_names=("Software", "Retail"),
    calculation_date=calculation_date,
)

assert [ranking.rank for ranking in rankings] == [1, 2]
assert [ranking.industry_group for ranking in rankings] == ["Software", "Retail"]
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
cd backend
/Users/admin/StockScreenClaude/backend/venv/bin/python -m pytest -q \
  tests/unit/test_group_rank_models.py \
  tests/unit/test_group_ranking_calculator.py
```

Expected: collection/import failure because `GroupRanking` does not exist, or attribute assertions fail because rankings are mappings.

- [ ] **Step 3: Add the typed model and construct rankings without mutation**

Add to `group_rank_models.py`:

```python
@dataclass(frozen=True)
class GroupRanking:
    industry_group: str
    date: date
    rank: int
    avg_rs_rating: float
    median_rs_rating: float | None
    weighted_avg_rs_rating: float | None
    rs_std_dev: float | None
    num_stocks: int
    num_stocks_rs_above_80: int
    top_symbol: str | None
    top_rs_rating: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "industry_group": self.industry_group,
            "date": self.date,
            "rank": self.rank,
            "avg_rs_rating": self.avg_rs_rating,
            "median_rs_rating": self.median_rs_rating,
            "weighted_avg_rs_rating": self.weighted_avg_rs_rating,
            "rs_std_dev": self.rs_std_dev,
            "num_stocks": self.num_stocks,
            "num_stocks_rs_above_80": self.num_stocks_rs_above_80,
            "top_symbol": self.top_symbol,
            "top_rs_rating": self.top_rs_rating,
        }


@dataclass(frozen=True)
class GroupRankCalculationResult:
    rankings: tuple[GroupRanking, ...]
    prefetch_stats: GroupRankPrefetchStats
```

In the calculator, introduce a private frozen candidate with every field except `rank`. Sort candidates by `avg_rs_rating`, then create `GroupRanking` values with `rank=enumerated_rank`. Delete `_calculate_group_rs_from_cache`; retain parity assertions against stable expected fixtures rather than a second production implementation.

- [ ] **Step 4: Update the repository to accept attributes**

Replace the mapping contract with:

```python
def store_rankings(
    self,
    db: Session,
    *,
    calculation_date: date,
    rankings: Sequence[GroupRanking],
    market: str,
) -> None:
    values = [
        self._ranking_values(calculation_date, ranking, market=market)
        for ranking in rankings
    ]
```

and map `ranking.industry_group`, `ranking.rank`, and the remaining attributes in `_ranking_values`. Update service/task logging to use ranking attributes. Keep `to_dict()` only for explicit compatibility serialization.

Validate that every `ranking.date` equals the method's `calculation_date` before
building persistence values. This prevents the redundant date fields from
silently writing a ranking under the wrong replacement date.

- [ ] **Step 5: Run typed model/calculator/repository/service tests and verify GREEN**

Run:

```bash
cd backend
/Users/admin/StockScreenClaude/backend/venv/bin/python -m pytest -q \
  tests/unit/test_group_rank_models.py \
  tests/unit/test_group_ranking_calculator.py \
  tests/unit/test_group_ranking_repository.py \
  tests/unit/test_group_rank_service.py
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit the typed boundary**

```bash
git add backend/app/services/group_rank_models.py \
  backend/app/services/group_ranking_calculator.py \
  backend/app/services/group_ranking_repository.py \
  backend/app/services/ibd_group_rank_service.py \
  backend/tests/unit/test_group_rank_models.py \
  backend/tests/unit/test_group_ranking_calculator.py \
  backend/tests/unit/test_group_ranking_repository.py \
  backend/tests/unit/test_group_rank_service.py
git commit --no-verify -m "refactor: type group ranking persistence"
```

---

### Task 2: Make group taxonomy loading single-pass and loader-owned

**Files:**
- Modify: `backend/app/services/group_rank_models.py`
- Modify: `backend/app/services/group_rank_input_loader.py`
- Modify: `backend/app/services/group_rank_legacy_adapter.py`
- Modify: `backend/app/services/group_rank_historical_calculator.py`
- Modify: `backend/app/services/ibd_group_rank_service.py`
- Test: `backend/tests/unit/test_group_rank_input_loader.py`
- Test: `backend/tests/unit/test_group_rank_legacy_adapter.py`
- Test: `backend/tests/unit/test_group_rank_historical_calculator.py`
- Test: `backend/tests/unit/test_group_rank_service.py`

**Interfaces:**
- Adds: `GroupRankPrefetchData.group_names: tuple[str, ...]`.
- Production invariant: `groups()` once and `symbols_for_group()` once per group per load.
- Legacy invariant: five-item tuples adapt with `group_names=()` and complete through the explicit compatibility seam.

- [ ] **Step 1: Write the failing call-count and ownership tests**

Use a counting taxonomy source:

```python
@dataclass
class CountingTaxonomySource:
    symbols_by_group: dict[str, tuple[str, ...]]
    groups_calls: int = 0
    member_calls: list[str] = field(default_factory=list)

    def groups(self, db, market):
        self.groups_calls += 1
        return tuple(self.symbols_by_group)

    def symbols_for_group(self, db, group, market):
        self.member_calls.append(group)
        return self.symbols_by_group[group]


def test_load_reads_each_taxonomy_membership_once(db_session):
    taxonomy = CountingTaxonomySource({
        "Software": ("AAA", "MZYX-U"),
        "Retail": ("BBB",),
    })
    loader = _loader(taxonomy_source=taxonomy)

    prefetch = loader.load(db_session, market="US", policy=_policy("auto"))

    assert prefetch.group_names == ("Software", "Retail")
    assert taxonomy.groups_calls == 1
    assert taxonomy.member_calls == ["Software", "Retail"]
    assert prefetch.stats.skipped_unsupported_symbols == 1
```

Add a facade test that makes `taxonomy_source.groups` raise if called after the typed loader returns a non-empty `group_names` tuple.

- [ ] **Step 2: Run loader/service tests and verify RED**

Run:

```bash
cd backend
/Users/admin/StockScreenClaude/backend/venv/bin/python -m pytest -q \
  tests/unit/test_group_rank_input_loader.py \
  tests/unit/test_group_rank_service.py \
  tests/unit/test_group_rank_legacy_adapter.py
```

Expected: the new field is missing and each membership is called twice.

- [ ] **Step 3: Capture and partition membership once**

Add `group_names` to every `GroupRankPrefetchData` constructor. In `load()`, read
the ordered group names once before loading the benchmark. If the benchmark is
missing, pass those names into `_empty_prefetch()` so callers can distinguish
"taxonomy exists but prices are unavailable" from "taxonomy is absent" without
querying taxonomy again.

After the benchmark succeeds, load `active_symbols` once and replace the two
membership comprehensions with one loop:

```python
group_names = self.taxonomy_source.groups(db, normalized_market)
symbols_by_group: dict[str, tuple[str, ...]] = {}
unsupported_symbols: set[str] = set()

for group in group_names:
    raw_symbols = self.taxonomy_source.symbols_for_group(
        db,
        group,
        normalized_market,
    )
    supported: list[str] = []
    for symbol in raw_symbols:
        if symbol not in active_symbols:
            continue
        if is_unsupported_yahoo_price_symbol(symbol):
            unsupported_symbols.add(symbol)
            continue
        supported.append(symbol)
    symbols_by_group[group] = tuple(supported)
```

Ensure `_empty_prefetch()` receives and preserves `group_names`, even when the benchmark is unavailable. The typed facade obtains groups from `prefetch.group_names`; only `group_names == ()` from a legacy adapter may call the compatibility completion path.

The final production call order is therefore:

```text
taxonomy.groups() once
benchmark load once
universe.active_symbols() once (only when the benchmark is available)
taxonomy.symbols_for_group() once per group (only when the benchmark is available)
```

- [ ] **Step 4: Delete unused facade implementations**

Remove these unreferenced private methods and now-unused imports from `IBDGroupRankService`:

```text
_calculate_group_rs
_get_validated_group_symbols
_get_market_caps_for_symbols
```

Keep `_prefetch_all_data` and `_coerce_prefetch_data` as compatibility seams because existing tests and legacy callers patch/adapt them.

- [ ] **Step 5: Run the focused input/legacy/historical/service suite and verify GREEN**

Run:

```bash
cd backend
/Users/admin/StockScreenClaude/backend/venv/bin/python -m pytest -q \
  tests/unit/test_group_rank_input_loader.py \
  tests/unit/test_group_rank_legacy_adapter.py \
  tests/unit/test_group_rank_historical_calculator.py \
  tests/unit/test_group_rank_service.py
```

Expected: all selected tests pass and the counting taxonomy assertions prove one membership read per group.

- [ ] **Step 6: Commit single ownership**

```bash
git add backend/app/services/group_rank_models.py \
  backend/app/services/group_rank_input_loader.py \
  backend/app/services/group_rank_legacy_adapter.py \
  backend/app/services/group_rank_historical_calculator.py \
  backend/app/services/ibd_group_rank_service.py \
  backend/tests/unit/test_group_rank_input_loader.py \
  backend/tests/unit/test_group_rank_legacy_adapter.py \
  backend/tests/unit/test_group_rank_historical_calculator.py \
  backend/tests/unit/test_group_rank_service.py
git commit --no-verify -m "refactor: make group input loading single pass"
```

---

### Task 3: Replace historical rankings atomically per date

**Files:**
- Modify: `backend/app/services/group_ranking_repository.py`
- Modify: `backend/app/services/group_rank_historical_calculator.py`
- Modify: `backend/app/tasks/group_rank_backfill_tasks.py`
- Test: `backend/tests/unit/test_group_ranking_repository.py`
- Test: `backend/tests/unit/test_group_rank_historical_calculator.py`
- Test: `backend/tests/unit/test_group_rank_backfill_tasks.py`

**Interfaces:**
- Adds: `GroupRankingRepository.replace_rankings_for_date(...) -> int` without committing.
- Changes: optimized backfill never calls `delete_range()` before calculation.
- Invariant: the historical calculator commits or rolls back one date at a time.
- Invariant: zero successful dates do not bump epoch or publish snapshots.

- [ ] **Step 1: Write failing repository atomicity tests**

Add a SQLite-backed repository test:

```python
def test_replace_rankings_for_date_does_not_commit(db_session):
    _insert_existing_rank(db_session, group="Old", day=DAY, market="US")
    db_session.commit()
    repository = GroupRankingRepository()

    deleted = repository.replace_rankings_for_date(
        db_session,
        calculation_date=DAY,
        rankings=(_ranking(group="New", day=DAY),),
        market="US",
    )

    assert deleted == 1
    assert db_session.in_transaction()
    db_session.rollback()
    assert _groups_for_date(db_session, DAY) == ["Old"]
```

Add a historical test where the benchmark is missing and assert:

```python
repository.replace_rankings_for_date.assert_not_called()
repository.delete_range.assert_not_called()
db.commit.assert_not_called()
```

Add a store-failure test in which replacement raises, `db.rollback()` is called, the error count increments, and later dates continue.

- [ ] **Step 2: Run atomicity tests and verify RED**

Run:

```bash
cd backend
/Users/admin/StockScreenClaude/backend/venv/bin/python -m pytest -q \
  tests/unit/test_group_ranking_repository.py \
  tests/unit/test_group_rank_historical_calculator.py
```

Expected: `replace_rankings_for_date` is absent and optimized backfill still commits `delete_range()` before prefetch.

- [ ] **Step 3: Implement non-committing per-date replacement**

Add to the repository:

```python
def replace_rankings_for_date(
    self,
    db: Session,
    *,
    calculation_date: date,
    rankings: Sequence[GroupRanking],
    market: str,
) -> int:
    normalized_market = (market or "US").upper()
    deleted = (
        db.query(IBDGroupRank)
        .filter(
            IBDGroupRank.date == calculation_date,
            IBDGroupRank.market == normalized_market,
        )
        .delete(synchronize_session=False)
    )
    self.store_rankings(
        db,
        calculation_date=calculation_date,
        rankings=rankings,
        market=normalized_market,
    )
    return deleted
```

Do not commit inside the repository.

- [ ] **Step 4: Calculate first, then replace and commit successful dates**

Delete the initial `delete_range()` and commit from `backfill_rankings_optimized`. For each calculated non-empty date:

```python
try:
    replaced = self.repository.replace_rankings_for_date(
        db,
        calculation_date=calculation_date,
        rankings=rankings,
        market=normalized_market,
    )
    db.commit()
    deleted += replaced
    processed += 1
except Exception:
    db.rollback()
    errors += 1
    logger.exception("Error replacing %s", calculation_date)
```

Dates with no rankings increment `errors` without mutation. Apply the same repository operation to optimized gap fill only where replacement semantics are intended; simple missing-date insertion may continue using `store_rankings` because no existing date should be deleted.

- [ ] **Step 5: Guard task publication**

In each manual group backfill task, wrap epoch/snapshot publication:

```python
if result.get("processed", 0) > 0:
    bump_group_rankings_epoch(market)
    safe_publish_groups_bootstrap()
```

The one-year task receives the same guard. Do not publish when the service returns a benchmark error and zero processed dates.

- [ ] **Step 6: Run repository/historical/task tests and verify GREEN**

Run:

```bash
cd backend
/Users/admin/StockScreenClaude/backend/venv/bin/python -m pytest -q \
  tests/unit/test_group_ranking_repository.py \
  tests/unit/test_group_rank_historical_calculator.py \
  tests/unit/test_group_rank_backfill_tasks.py
```

Expected: all selected tests pass, including rollback preservation and zero-success publication guards.

- [ ] **Step 7: Commit atomic replacement**

```bash
git add backend/app/services/group_ranking_repository.py \
  backend/app/services/group_rank_historical_calculator.py \
  backend/app/tasks/group_rank_backfill_tasks.py \
  backend/tests/unit/test_group_ranking_repository.py \
  backend/tests/unit/test_group_rank_historical_calculator.py \
  backend/tests/unit/test_group_rank_backfill_tasks.py
git commit --no-verify -m "fix: replace historical group ranks atomically"
```

---

### Task 4: Extract a Celery-free daily group-ranking runner

**Files:**
- Create: `backend/app/services/daily_group_rank_runner.py`
- Modify: `backend/app/tasks/group_rank_tasks.py`
- Test: `backend/tests/unit/test_daily_group_rank_runner.py`
- Test: `backend/tests/unit/test_group_rank_tasks.py`
- Test: `backend/tests/unit/test_group_rank_in_process.py`
- Test: `backend/tests/unit/test_group_rank_execution_policy.py`

**Interfaces:**
- Adds: `DailyGroupRankRequest`.
- Adds: `DailyGroupRankOutcome.to_task_result(policy) -> dict[str, Any]`.
- Adds: `NoGroupRankingsCalculated` and `GroupRankWarmupIncomplete` domain failures.
- Adds: `run_daily_group_rankings(db, request, dependencies) -> DailyGroupRankOutcome`.
- Removes: `_calculate_daily_group_rankings_in_process` and `_PROPAGATE_IN_PROCESS_TRANSIENT_ERRORS`.

- [ ] **Step 1: Write failing runner contract tests**

Describe the normal callable API:

```python
def test_runner_returns_compatible_success_outcome():
    service = Mock()
    service.calculate_group_rankings.return_value = _typed_calculation()
    dependencies = _dependencies(service=service)

    outcome = run_daily_group_rankings(
        MagicMock(),
        DailyGroupRankRequest(
            calculation_date=date(2026, 3, 20),
            market="US",
            activity_lifecycle="daily_refresh",
            policy=_policy("refresh_guarded"),
        ),
        dependencies,
    )

    assert outcome.to_task_result(_policy("refresh_guarded"))["groups_ranked"] == 1
    dependencies.bump_epoch.assert_called_once_with("US")
```

Add tests proving strict warmup failure prevents calculation, no-groups raises a named failure carrying typed stats, and transient service errors propagate unchanged.

- [ ] **Step 2: Run the runner test and verify RED**

Run:

```bash
cd backend
/Users/admin/StockScreenClaude/backend/venv/bin/python -m pytest -q \
  tests/unit/test_daily_group_rank_runner.py
```

Expected: import failure because the runner module does not exist.

- [ ] **Step 3: Implement the request, dependencies, outcome, and runner**

Create these core contracts:

```python
@dataclass(frozen=True)
class DailyGroupRankRequest:
    calculation_date: date
    market: str
    activity_lifecycle: str
    policy: DerivedDataExecutionPolicy


@dataclass(frozen=True)
class DailyGroupRankDependencies:
    service: IBDGroupRankService
    bump_epoch: Callable[[str], None]
    publish_snapshot: Callable[[], object]
    repair_current_us_metadata: Callable[[date], object] | None = None


@dataclass(frozen=True)
class DailyGroupRankOutcome:
    calculation_date: date
    rankings: tuple[GroupRanking, ...]
    prefetch_stats: GroupRankPrefetchStats
    duration_seconds: float
    metadata_repair: object | None

    def to_task_result(self, policy: DerivedDataExecutionPolicy) -> dict[str, Any]:
        result = {
            "date": self.calculation_date.isoformat(),
            "groups_ranked": len(self.rankings),
            "top_group": self.rankings[0].industry_group,
            "top_avg_rs": self.rankings[0].avg_rs_rating,
            "calculation_duration_seconds": round(self.duration_seconds, 2),
            "metadata_repair": self.metadata_repair,
            "timestamp": datetime.now().isoformat(),
        }
        policy.annotate_response(result, include_cache_only=True)
        if policy.response_cache_policy is not None:
            result["prefetch_stats"] = self.prefetch_stats.to_dict()
        return result
```

The runner derives `GroupRankCacheRequirement` from `policy.validation_profile`, evaluates warmup when required, calls `service.calculate_group_rankings`, raises a named no-groups failure for an empty tuple, performs current-US metadata repair under the existing lifecycle/date rule, bumps the epoch, and publishes the snapshot best-effort.

- [ ] **Step 4: Adapt direct and gap-fill Celery tasks**

Keep date parsing, policy resolution, sessions, activity lifecycle, exception-to-response mapping, and retry scheduling in the public task. Replace the calculation/finalization block with one runner call.

Add one ordinary task-adapter helper, `_run_daily_group_rankings_response`, that
calls the runner and converts only the named domain failures into the exact
existing response dictionaries:

```text
GroupRankWarmupIncomplete / IncompleteGroupRankingCacheError
  -> reason_code=warmup_incomplete, cache_only, prefetch_stats when available
MissingIBDIndustryMappingsError
  -> reason_code=missing_ibd_mappings
NoGroupRankingsCalculated
  -> groups_ranked=0, warning, reason_code=no_groups_ranked, prefetch_stats when enabled
```

Do not catch transient infrastructure exceptions in this adapter. They must
reach the owning Celery task, so the direct task owns its retry and the gap-fill
task owns its retry without a `ContextVar` switch.

Both public entry points call `_run_daily_group_rankings_response` with the
resolved target date, policy, existing session, and dependencies. That helper
calls the ordinary runner—not the decorated daily task—and serializes the
outcome. The gap-fill task stores its returned dictionary in `result["today"]`
and preserves the current behavior of raising a wrapper-level failure when
that dictionary contains `error`.

Delete:

```text
_calculate_daily_group_rankings_in_process
_PROPAGATE_IN_PROCESS_TRANSIENT_ERRORS
disable_serialized_market_workload usage for nested group calculation
unittest.mock module detection
task.run() introspection
```

Keep `allow_same_day_group_rank_warmup_bypass()` for static export compatibility.

- [ ] **Step 5: Replace helper-patching tests with runner-patching tests**

Tests that formerly patched `_calculate_daily_group_rankings_in_process` should patch the task module's `run_daily_group_rankings` import and return `DailyGroupRankOutcome`. Replace the old in-process tests with assertions that:

```python
assert "_calculate_daily_group_rankings_in_process" not in source
assert "_PROPAGATE_IN_PROCESS_TRANSIENT_ERRORS" not in source
assert ".run(**kwargs)" not in source
```

Retain tests proving only the outer gap-fill task retries on a propagated transient exception.

- [ ] **Step 6: Run group runner/task policy tests and verify GREEN**

Run:

```bash
cd backend
/Users/admin/StockScreenClaude/backend/venv/bin/python -m pytest -q \
  tests/unit/test_daily_group_rank_runner.py \
  tests/unit/test_group_rank_tasks.py \
  tests/unit/test_group_rank_in_process.py \
  tests/unit/test_group_rank_execution_policy.py
```

Expected: all selected tests pass and source assertions prove no decorated task invocation remains.

- [ ] **Step 7: Commit the group runner**

```bash
git add backend/app/services/daily_group_rank_runner.py \
  backend/app/tasks/group_rank_tasks.py \
  backend/tests/unit/test_daily_group_rank_runner.py \
  backend/tests/unit/test_group_rank_tasks.py \
  backend/tests/unit/test_group_rank_in_process.py \
  backend/tests/unit/test_group_rank_execution_policy.py
git commit --no-verify -m "refactor: extract daily group ranking runner"
```

---

### Task 5: Extract a Celery-free daily breadth runner

**Files:**
- Create: `backend/app/services/daily_breadth_runner.py`
- Modify: `backend/app/services/breadth_calculator_service.py`
- Modify: `backend/app/tasks/breadth_tasks.py`
- Test: `backend/tests/unit/test_daily_breadth_runner.py`
- Test: `backend/tests/unit/test_breadth_calculator_service.py`
- Test: `backend/tests/unit/test_breadth_tasks.py`

**Interfaces:**
- Adds: `DailyBreadthRequest`.
- Adds: `DailyBreadthOutcome.to_task_result(policy) -> dict[str, Any]`.
- Adds: `IncompleteDailyBreadth` carrying a message and `BreadthCoverageReport`.
- Adds: `run_daily_breadth(db, request, dependencies) -> DailyBreadthOutcome`.
- Adds: `BreadthCalculatorService.store_daily_breadth(...) -> None` as the canonical persistence method.
- Removes: `_calculate_daily_breadth_in_process` and task-object introspection.

- [ ] **Step 1: Write failing runner and canonical persistence tests**

Add:

```python
def test_runner_persists_and_serializes_compatible_success():
    calculator = Mock()
    calculator.calculate_daily_breadth.return_value = _calculation()
    dependencies = _dependencies(calculator=calculator)

    outcome = run_daily_breadth(
        MagicMock(),
        DailyBreadthRequest(
            calculation_date=date(2026, 3, 20),
            market="US",
            policy=_policy("refresh_guarded"),
        ),
        dependencies,
    )

    calculator.store_daily_breadth.assert_called_once()
    result = outcome.to_task_result(_policy("refresh_guarded"))
    assert result["date"] == "2026-03-20"
    assert result["cache_policy"] == "refresh_guarded"
```

Add strict, tolerant, warmup-required, and zero-usable-stock validation tests. Add a service test proving `store_daily_breadth()` updates or inserts and commits one record using the existing market partition.

- [ ] **Step 2: Run breadth runner/service tests and verify RED**

Run:

```bash
cd backend
/Users/admin/StockScreenClaude/backend/venv/bin/python -m pytest -q \
  tests/unit/test_daily_breadth_runner.py \
  tests/unit/test_breadth_calculator_service.py
```

Expected: runner import failure and missing public persistence method.

- [ ] **Step 3: Implement the breadth runner**

Create these contracts:

```python
@dataclass(frozen=True)
class DailyBreadthRequest:
    calculation_date: date
    market: str
    policy: DerivedDataExecutionPolicy


@dataclass(frozen=True)
class DailyBreadthDependencies:
    calculator: BreadthCalculatorService
    publish_snapshot: Callable[[str], object]


@dataclass(frozen=True)
class DailyBreadthOutcome:
    calculation_date: date
    indicators: Mapping[str, Any]
    coverage: BreadthCoverageReport
    duration_seconds: float
```

Move `_validate_same_day_cache_only_breadth`, `_validate_strict_cache_only_breadth`, and `_validate_refresh_guarded_breadth` into the runner module. `run_daily_breadth` calculates, applies `policy.validation_profile`, raises `IncompleteDailyBreadth` before persistence on failure, persists through the canonical service method, and publishes the breadth snapshot best-effort. `to_task_result()` reproduces the existing `indicators`, counts, duration, cache policy, and diagnostics keys.

- [ ] **Step 4: Expose canonical breadth persistence**

Rename/generalize the existing `_store_breadth_record` to:

```python
def store_daily_breadth(
    self,
    calculation_date: date,
    metrics: Mapping[str, Any],
    *,
    duration_seconds: float,
) -> None:
```

Reuse its existing upsert-by-`(market, date)` behavior and commit. Delete the duplicated ORM assignment block from `calculate_daily_breadth` task.

- [ ] **Step 5: Adapt direct and gap-fill breadth tasks**

The direct task retains serialized date parsing, policy resolution, session lifecycle, exception mapping, and response compatibility. Both direct and gap-fill paths call `run_daily_breadth` directly. Delete:

```text
_calculate_daily_breadth_in_process
disable_serialized_market_workload usage for nested breadth calculation
unittest.mock module detection
task.run() introspection
```

Keep `allow_same_day_breadth_warmup_bypass()` because static export uses it.

Add one ordinary `_run_daily_breadth_response` adapter that converts only
`IncompleteDailyBreadth` into the exact current cache-completeness error payload
(`error`, `date`, `timestamp`, `cache_only`, `metrics`, optional
`cache_diagnostics`, and policy annotations). Both public tasks use that adapter;
the gap-fill wrapper stores it in `result["today"]` and preserves its existing
wrapper-level error behavior. All other exceptions propagate to the owning task,
so the adapter never schedules or suppresses retries.

- [ ] **Step 6: Migrate task tests to the runner seam**

Patch `run_daily_breadth` and return `DailyBreadthOutcome` instead of patching `_calculate_daily_breadth_in_process`. Add source assertions that reject the deleted helper and `.run(**kwargs)` task invocation. Retain guarded provider-exclusion, legacy flag precedence, historical manual, and transient outer-retry tests.

- [ ] **Step 7: Run breadth runner/service/task tests and verify GREEN**

Run:

```bash
cd backend
/Users/admin/StockScreenClaude/backend/venv/bin/python -m pytest -q \
  tests/unit/test_daily_breadth_runner.py \
  tests/unit/test_breadth_calculator.py \
  tests/unit/test_breadth_calculator_service.py \
  tests/unit/test_breadth_coverage.py \
  tests/unit/test_breadth_tasks.py
```

Expected: all selected tests pass and no nested decorated-task invocation remains.

- [ ] **Step 8: Commit the breadth runner**

```bash
git add backend/app/services/daily_breadth_runner.py \
  backend/app/services/breadth_calculator_service.py \
  backend/app/tasks/breadth_tasks.py \
  backend/tests/unit/test_daily_breadth_runner.py \
  backend/tests/unit/test_breadth_calculator_service.py \
  backend/tests/unit/test_breadth_tasks.py
git commit --no-verify -m "refactor: extract daily breadth runner"
```

---

### Task 6: Add architecture gates and complete verification

**Files:**
- Modify: `backend/tests/unit/test_derived_data_execution_policy.py`
- Modify: `backend/tests/unit/test_group_rank_in_process.py`
- Modify: `docs/superpowers/specs/2026-07-17-derived-data-final-thermo-remediation-design.md`
- Modify: `.beads/issues.jsonl`
- Modify: `.beads/interactions.jsonl`

**Interfaces:**
- Produces: source-level guards against regression to arbitrary ranking mappings, duplicated taxonomy traversal, and nested Celery invocation.
- Produces: final tracker closure and implementation evidence.

- [ ] **Step 1: Add architecture assertions**

Add AST/source tests equivalent to:

```python
def test_group_ranking_core_has_no_arbitrary_mapping_contract():
    for path in (
        "app/services/group_rank_models.py",
        "app/services/group_ranking_calculator.py",
        "app/services/group_ranking_repository.py",
    ):
        source = (BACKEND_ROOT / path).read_text()
        assert "rankings: tuple[Mapping[str, Any]" not in source
        assert "rankings: Sequence[Mapping[str, Any]" not in source


def test_derived_tasks_do_not_invoke_decorated_tasks_in_process():
    for path in (
        "app/tasks/breadth_tasks.py",
        "app/tasks/group_rank_tasks.py",
    ):
        source = (BACKEND_ROOT / path).read_text()
        assert "_calculate_daily_breadth_in_process" not in source
        assert "_calculate_daily_group_rankings_in_process" not in source
        assert "unittest.mock" not in source
        assert ".run(**kwargs)" not in source
```

- [ ] **Step 2: Run the complete focused derived-data suite**

Run:

```bash
cd backend
/Users/admin/StockScreenClaude/backend/venv/bin/python -m pytest -q \
  tests/unit/test_derived_data_execution_policy.py \
  tests/unit/test_daily_breadth_runner.py \
  tests/unit/test_breadth_calculator.py \
  tests/unit/test_breadth_calculator_service.py \
  tests/unit/test_breadth_coverage.py \
  tests/unit/test_breadth_tasks.py \
  tests/unit/test_daily_group_rank_runner.py \
  tests/unit/test_group_rank_models.py \
  tests/unit/test_group_rank_input_loader.py \
  tests/unit/test_group_rank_legacy_adapter.py \
  tests/unit/test_group_ranking_calculator.py \
  tests/unit/test_group_ranking_repository.py \
  tests/unit/test_group_rank_historical_calculator.py \
  tests/unit/test_group_rank_service.py \
  tests/unit/test_group_rank_tasks.py \
  tests/unit/test_group_rank_in_process.py \
  tests/unit/test_group_rank_execution_policy.py \
  tests/unit/test_group_rank_backfill_tasks.py \
  tests/unit/test_daily_market_pipeline_tasks.py
```

Expected: all selected tests pass.

- [ ] **Step 3: Run comprehensive backend verification**

Run:

```bash
cd backend
/Users/admin/StockScreenClaude/backend/venv/bin/python -m pytest tests/unit -q
```

Expected: the comprehensive unit suite passes with zero failures.

- [ ] **Step 4: Run repository quality gates**

Run from the repository root:

```bash
python3 scripts/generate_scan_filter_contract.py --check
make gate-identity
make gate-check
make gate-1
make gate-2
make gate-3
make gate-4
make gate-5
git diff --check
```

Expected: every command exits zero.

- [ ] **Step 5: Record implementation evidence and close the tracker item**

Update the design's implementation-verification section with exact test counts. Then run:

```bash
bd close stockscreenclaude-slh --reason="Made group historical replacement atomic per date, established typed single-pass group ranking inputs/results, removed nested Celery task invocation, and passed focused plus comprehensive verification"
```

Because the installed Beads version has no `bd sync`, ensure only canonical `.beads/*.jsonl` changes are staged and no root-level `issues.jsonl` is committed.

- [ ] **Step 6: Commit final gates and tracker state**

```bash
git add backend/tests/unit/test_derived_data_execution_policy.py \
  backend/tests/unit/test_group_rank_in_process.py \
  docs/superpowers/specs/2026-07-17-derived-data-final-thermo-remediation-design.md \
  .beads/issues.jsonl .beads/interactions.jsonl
git commit --no-verify -m "test: guard derived-data architecture"
```

- [ ] **Step 7: Rebase, push, and verify the pull request**

```bash
git pull --rebase
git push
git status
gh pr checks 303
```

Expected: the branch is up to date with its remote, the worktree is clean, and all required PR checks pass.
