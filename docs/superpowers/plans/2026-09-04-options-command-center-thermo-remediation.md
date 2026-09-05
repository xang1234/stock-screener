# Options Command Center Thermo-Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve the Options Command Center behavior while replacing its oversized, weakly typed orchestration and duplicated static/contract machinery with focused, testable components.

**Architecture:** A small run coordinator composes a typed cohort builder and candidate analyzer, then persists typed results through narrow repositories with named transaction boundaries. Static options become one section component using the shared atomic-directory publisher, while Pydantic-generated JSON Schema becomes the frontend validation source.

**Tech Stack:** Python 3.11, dataclasses/Protocols, Pydantic 2, SQLAlchemy, Pytest, React 18, Ajv 8, Vitest.

**Spec:** `docs/superpowers/specs/2026-09-04-options-command-center-thermo-remediation-design.md`

## Global Constraints

- Preserve candidate selection, metric formulas, publication thresholds, API payloads, task names, database schema, and static paths.
- Preserve the 40-per-source, strictly-above-USD-100-million liquidity rule, five-session continuity window, and 100-symbol maximum cohort.
- Preserve at most two concurrent symbols and three total attempts per symbol.
- Keep domain modules free of SQLAlchemy, FastAPI, Celery, yfinance, Redis, and React imports.
- Keep `backend/app/services/static_site_export_service.py` below 1,000 lines.
- Every production behavior change starts with a focused failing test.

---

### Task 1: Make the options application boundary truthful and typed

**Files:**
- Modify: `backend/app/domain/options_analytics/models.py`
- Replace: `backend/app/domain/options_analytics/ports.py`
- Modify: `backend/app/infra/query/options_candidate_source.py`
- Modify: `backend/app/wiring/use_case_factories.py`
- Test: `backend/tests/unit/domain/options_analytics/test_models.py`
- Test: `backend/tests/unit/use_cases/options_analytics/test_ports.py`

**Interfaces:**
- Produces: `CandidateSourceSnapshot`, `LastCurrentMembership`, `OptionsRunRecord`, `OptionsRunItemRecord`, `OptionsCandidateSource`, `OptionsRunWriter`, `PublishedOptionsReader`, `OptionsHistoryGateway`, `OptionsRetention`, `SessionCalendar`, and `CancellationToken`.
- Removes: unused `Clock`, `ProgressReporter`, and method names that no implementation provides.

- [ ] **Step 1: Write failing model and port-conformance tests**

```python
def test_candidate_source_protocol_matches_real_reader():
    assert isinstance(FakeCandidateSource(), OptionsCandidateSource)

def test_dividend_assumption_rejects_conflicting_source_and_value():
    with pytest.raises(ValueError):
        DividendAssumption(yield_value=None, source=DividendSource.PINNED_FEATURE_RUN)
```

- [ ] **Step 2: Run the tests and verify the expected missing-type failures**

Run: `cd backend && ./venv/bin/pytest -q tests/unit/domain/options_analytics/test_models.py tests/unit/use_cases/options_analytics/test_ports.py`

- [ ] **Step 3: Implement exact runtime-checkable protocols and DTOs**

```python
@runtime_checkable
class OptionsCandidateSource(Protocol):
    def read(self, source_feature_run_id: int) -> CandidateSourceSnapshot: ...
    def read_continuity_inputs(self, symbols: Sequence[str], as_of_date: date) -> Mapping[str, OptionCandidateInput]: ...

class DividendSource(str, Enum):
    PINNED_FEATURE_RUN = "pinned_feature_run"
    ZERO_ASSUMPTION = "zero_assumption"
```

Move source snapshot and last-membership DTOs out of infrastructure. Reuse `NeverCancelledToken`; replace the private US calendar with a shared `MarketSessionWindow` adapter.

- [ ] **Step 4: Run focused tests and commit**

Run: `cd backend && ./venv/bin/pytest -q tests/unit/domain/options_analytics tests/unit/infra/test_options_candidate_source.py tests/unit/use_cases/options_analytics/test_refresh.py`

Commit: `refactor: type options analytics boundaries`

---

### Task 2: Extract cohort construction and candidate analysis

**Files:**
- Create: `backend/app/use_cases/options_analytics/cohort.py`
- Create: `backend/app/use_cases/options_analytics/candidate_analysis.py`
- Create: `backend/app/use_cases/options_analytics/analysis_models.py`
- Modify: `backend/app/use_cases/options_analytics/refresh.py`
- Modify: `backend/app/infra/providers/yahoo_options.py`
- Test: `backend/tests/unit/use_cases/options_analytics/test_cohort.py`
- Test: `backend/tests/unit/use_cases/options_analytics/test_candidate_analysis.py`
- Modify: `backend/tests/unit/infra/test_yahoo_options_provider.py`
- Modify: `backend/tests/unit/use_cases/options_analytics/test_refresh.py`

**Interfaces:**
- Produces: `OptionsCandidateCohortBuilder.build(source_run_id) -> OptionsCohortSnapshot`.
- Produces: `OptionsCandidateAnalyzer.analyze(candidate, context) -> AvailableCandidateAnalysis | UnavailableCandidateAnalysis`.
- Consumes: the typed ports from Task 1 and existing pure metric/history/quality policies.

- [ ] **Step 1: Write failing tests for the wished-for typed APIs**

```python
def test_unavailable_analysis_cannot_contain_metrics():
    result = analyzer.analyze(candidate_without_spot, context)
    assert isinstance(result, UnavailableCandidateAnalysis)
    assert result.reason_codes == ("source_spot_unavailable",)

def test_returning_symbol_is_current_without_losing_lifetime_history():
    cohort = builder.build(source_run_id=12)
    assert cohort.by_symbol("AAPL").kind is CandidateKind.CURRENT
```

- [ ] **Step 2: Run new tests and verify they fail because the components do not exist**

Run: `cd backend && ./venv/bin/pytest -q tests/unit/use_cases/options_analytics/test_cohort.py tests/unit/use_cases/options_analytics/test_candidate_analysis.py`

- [ ] **Step 3: Move behavior without changing policy**

Move continuity assembly to `cohort.py`. Move retry, expiration, metrics, assumptions, evidence, readiness, warnings, and strike projection to `candidate_analysis.py`. Replace the nullable `_FetchResult` with two frozen result dataclasses. Remove retry loops and `max_attempts` from `YahooOptionsProvider`; one analyzer owns the complete three-attempt budget.

- [ ] **Step 4: Reduce the coordinator to workflow sequencing**

`RefreshOptionsAnalyticsUseCase.execute()` validates the command, gets the cohort, acquires/stages the run, resolves the run assumption, submits analyzer calls, persists results, ranks, and performs one terminal transition. It contains no metric/evidence/strike mapping helpers.

- [ ] **Step 5: Run focused suites and commit**

Run: `cd backend && ./venv/bin/pytest -q tests/unit/use_cases/options_analytics tests/unit/infra/test_yahoo_options_provider.py tests/unit/domain/options_analytics`

Commit: `refactor: separate options cohort and analysis`

---

### Task 3: Split persistence and make run transitions explicit

**Files:**
- Create: `backend/app/infra/db/repositories/options_run_writer.py`
- Create: `backend/app/infra/db/repositories/published_options_reader.py`
- Create: `backend/app/infra/db/repositories/options_history_repository.py`
- Create: `backend/app/infra/db/repositories/options_retention.py`
- Create: `backend/app/infra/db/repositories/options_row_mapper.py`
- Delete: `backend/app/infra/db/repositories/options_analytics_repo.py`
- Modify: `backend/app/use_cases/options_analytics/refresh.py`
- Modify: `backend/app/use_cases/options_analytics/queries.py`
- Modify: `backend/app/wiring/bootstrap.py`
- Modify: `backend/app/wiring/use_case_factories.py`
- Modify: `backend/tests/unit/repositories/test_options_analytics_repo.py`
- Modify: `backend/tests/unit/use_cases/options_analytics/test_refresh.py`
- Test: `backend/tests/unit/repositories/test_options_repository_boundaries.py`

**Interfaces:**
- `OptionsRunWriter.stage_run(run_id, candidates)`, `record_run_assumptions(run_id, assumptions)`, `save_analysis(run_id, analysis)`, `save_activity_ranks(run_id, ranks)`, `publish(run_id, summary)`, `fail_quality(run_id, summary)`, and `cancel(run_id)` each own their commit.
- `PublishedOptionsReader` owns only published command/detail/history queries.
- `OptionsHistoryRepository` owns typed aggregate export/import.
- `OptionsRetentionRepository.prune(aggregate_before, strike_history_run_limit=30)` owns retention.

- [ ] **Step 1: Write failing cancellation and transaction tests**

```python
def test_cancel_persists_terminal_status():
    result = use_case.execute(command)
    assert result["status"] == "cancelled"
    assert writer.run.status == OptionsRunStatus.CANCELLED.value

def test_save_analysis_replaces_strike_points_as_one_collection():
    writer.save_analysis(run_id, analysis)
    assert session.query(OptionsAnalyticsStrikePoint).count() == len(analysis.strike_points)
```

- [ ] **Step 2: Run tests and confirm current staged-state and missing-API failures**

Run: `cd backend && ./venv/bin/pytest -q tests/unit/use_cases/options_analytics/test_refresh.py tests/unit/repositories/test_options_repository_boundaries.py`

- [ ] **Step 3: Implement focused repositories and mapper**

Map `AvailableCandidateAnalysis` and `UnavailableCandidateAnalysis` explicitly. Delete dynamic metric-column `setattr` input and per-strike lookup loops. Delete public `commit()`. Keep SQLAlchemy row mutation private to repositories.

- [ ] **Step 4: Update composition and all repository tests**

Construct the four repositories with the same request/task session. Inject only the repository each use case/service needs.

- [ ] **Step 5: Run persistence/API/task suites and commit**

Run: `cd backend && ./venv/bin/pytest -q tests/unit/repositories/test_options_analytics_repo.py tests/unit/repositories/test_options_repository_boundaries.py tests/unit/use_cases/options_analytics tests/unit/test_options_analytics_api.py tests/unit/test_options_analytics_tasks.py tests/integration/test_options_analytics_publication.py`

Commit: `refactor: split options persistence responsibilities`

---

### Task 4: Replace dictionary history transfer with typed observations

**Files:**
- Create: `backend/app/schemas/options_history_transfer.py`
- Modify: `backend/app/services/options_history_transfer.py`
- Modify: `backend/app/infra/db/repositories/options_history_repository.py`
- Modify: `backend/tests/unit/test_options_history_transfer.py`
- Modify: `backend/tests/unit/test_options_history_scripts.py`

**Interfaces:**
- Produces: `OptionsHistoryBundle`, `OptionsHistoryObservation`, and `OptionsHistoryImportResult` Pydantic models.
- Bundle contains validated observations plus checksum; last-current memberships are derived, not serialized.

- [ ] **Step 1: Write failing typed round-trip and payload-simplification tests**

```python
def test_history_bundle_derives_memberships_from_observations():
    bundle = transfer.export_bundle(exported_at=now)
    assert "last_current_memberships" not in bundle
    assert transfer.import_bundle(bundle)["status"] == "imported"
```

- [ ] **Step 2: Verify the test fails against the duplicated membership payload**

Run: `cd backend && ./venv/bin/pytest -q tests/unit/test_options_history_transfer.py`

- [ ] **Step 3: Implement typed validation and repository mapping**

Validate dates, finite values, candidate/observation enums, run identity, forbidden raw fields, uniqueness, and checksum through model validators. Repository methods accept/return typed observations and do not parse arbitrary mappings.

- [ ] **Step 4: Run transfer/static workflow tests and commit**

Run: `cd backend && ./venv/bin/pytest -q tests/unit/test_options_history_transfer.py tests/unit/test_options_history_scripts.py tests/unit/test_static_site_workflow.py`

Commit: `refactor: type options history transfer`

---

### Task 5: Centralize atomic publishing and isolate the static options section

**Files:**
- Create: `backend/app/services/atomic_directory_publisher.py`
- Create: `backend/app/services/static_options_section.py`
- Modify: `backend/app/services/static_options_exporter.py`
- Modify: `backend/app/services/static_options_artifact_selector.py`
- Modify: `backend/app/services/static_artifact_combiner.py`
- Modify: `backend/app/services/static_site_export_service.py`
- Modify: `backend/app/scripts/download_static_market_fallbacks.py`
- Test: `backend/tests/unit/services/test_atomic_directory_publisher.py`
- Test: `backend/tests/unit/test_static_options_section.py`
- Modify: `backend/tests/unit/test_static_options_exporter.py`
- Modify: `backend/tests/unit/test_static_options_artifact_selector.py`
- Modify: `backend/tests/unit/test_static_site_export_service.py`
- Modify: `backend/tests/unit/services/test_static_artifact_combiner.py`

**Interfaces:**
- Produces: `AtomicDirectoryPublisher.publish(destination, populate, validate=None, clean=True)`.
- Produces: `StaticOptionsSection.compose_live(db, output_dir, generated_at, equity_entry, fallback_options_dir)` and `StaticOptionsSection.compose_combined(output_dir, manifest, current_options_dir, fallback_options_dir)` returning `StaticOptionsSectionResult`.

- [ ] **Step 1: Write failing rollback, parity, and file-size tests**

```python
def test_atomic_publisher_restores_destination_when_population_fails(tmp_path):
    destination = tmp_path / "published"
    destination.mkdir()
    (destination / "value.txt").write_text("previous")

    def fail(stage):
        (stage / "value.txt").write_text("partial")
        raise RuntimeError("population failed")

    with pytest.raises(RuntimeError, match="population failed"):
        AtomicDirectoryPublisher().publish(destination, fail)
    assert (destination / "value.txt").read_text() == "previous"

def test_static_exporter_remains_decomposed():
    assert len(Path(STATIC_SITE_EXPORTER).read_text().splitlines()) < 1000
```

- [ ] **Step 2: Run tests and confirm missing publisher plus size failure**

Run: `cd backend && ./venv/bin/pytest -q tests/unit/services/test_atomic_directory_publisher.py tests/unit/test_static_options_section.py tests/unit/test_static_site_export_service.py`

- [ ] **Step 3: Implement and adopt the atomic publisher**

Move stage/backup/swap/rollback/cleanup into one utility. Exporter, selector, combiner, and fallback installer provide only population and validation callbacks.

- [ ] **Step 4: Move all options orchestration behind `StaticOptionsSection`**

Remove options imports, dependency construction, fallback selection, stale marking, manifest mutation, and metadata rewriting from `StaticSiteExportService`. Direct and combined exports call the section through one result contract.

- [ ] **Step 5: Collapse fallback candidate downloading**

Add one `_download_candidate(run_id, artifact, wrapper, finder, date_reader)` helper used by both market and options artifact loops. Keep artifact-specific parsing and validation as injected functions.

- [ ] **Step 6: Run all static tests, verify line count, and commit**

Run: `cd backend && ./venv/bin/pytest -q tests/unit/test_static_options_exporter.py tests/unit/test_static_options_artifact_selector.py tests/unit/test_static_options_contract.py tests/unit/test_static_options_section.py tests/unit/test_static_site_export_service.py tests/unit/services/test_static_artifact_combiner.py tests/unit/test_static_site_workflow.py`

Run: `test $(wc -l < backend/app/services/static_site_export_service.py) -lt 1000`

Commit: `refactor: isolate static options publishing`

---

### Task 6: Generate and consume the frontend options contract

**Files:**
- Modify: `backend/app/schemas/options_analytics.py`
- Create: `backend/app/scripts/export_options_json_schema.py`
- Create: `frontend/src/features/options/optionsSchema.json`
- Modify: `frontend/src/features/options/optionsContract.js`
- Modify: `frontend/src/features/options/optionsContract.test.js`
- Modify: `frontend/package.json`
- Modify: `frontend/package-lock.json`
- Test: `backend/tests/unit/test_options_json_schema.py`

**Interfaces:**
- Produces: one deterministic JSON Schema document containing manifest, command-center, and symbol-detail definitions.
- Frontend retains semantic run-identity and safe-path checks but delegates payload shape/range validation to Ajv.

- [ ] **Step 1: Write failing schema-drift and frontend rejection tests**

```python
def test_committed_options_schema_matches_pydantic_models():
    assert committed_schema() == build_options_schema()
```

```javascript
expect(() => normalizeOptionsCommandCenter({ ...fixture, coverage: 2 })).toThrow();
```

- [ ] **Step 2: Run tests and verify missing generator/drift failures**

Run: `cd backend && ./venv/bin/pytest -q tests/unit/test_options_json_schema.py`

Run: `cd frontend && npm run test:run -- src/features/options/optionsContract.test.js`

- [ ] **Step 3: Add a strict static-manifest Pydantic model and deterministic exporter**

The exporter writes sorted, indented JSON with a trailing newline. Add `ajv@8` as a direct frontend dependency. Replace handwritten field validators with compiled schemas; retain uppercase-symbol uniqueness, unique static paths, run-context identity, and symbol identity checks as concise semantic checks.

- [ ] **Step 4: Run backend/frontend contract tests and commit**

Run: `cd backend && ./venv/bin/pytest -q tests/unit/test_options_analytics_schema.py tests/unit/test_static_options_contract.py tests/unit/test_options_json_schema.py`

Run: `cd frontend && npm run test:run -- src/features/options/optionsContract.test.js src/static/optionsClient.test.js src/features/options/optionsSurfaceParity.test.jsx`

Commit: `refactor: generate options wire contract`

---

### Task 7: Full verification and final structural audit

**Files:**
- Modify only files required to fix failures exposed by the commands below.

- [ ] **Step 1: Run the full backend unit suite**

Run: `cd backend && ./venv/bin/pytest -q tests/unit`

- [ ] **Step 2: Run options integration tests**

Run: `cd backend && ./venv/bin/pytest -q tests/integration/test_options_analytics_migration.py tests/integration/test_options_analytics_publication.py tests/integration/test_options_static_live_parity.py`

- [ ] **Step 3: Run complete frontend verification**

Run: `cd frontend && npm run test:run && npm run lint && npm run build`

- [ ] **Step 4: Run repository checks and structural assertions**

Run: `git diff --check`

Run: `test $(wc -l < backend/app/services/static_site_export_service.py) -lt 1000`

Run: `test $(wc -l < backend/app/use_cases/options_analytics/refresh.py) -lt 300`

Run: `test ! -f backend/app/infra/db/repositories/options_analytics_repo.py`

- [ ] **Step 5: Review the final diff against the remediation specification**

Confirm every public payload and route remains compatible, all six review findings have a corresponding structural change, and no replacement module recreates an omnibus file.

- [ ] **Step 6: Commit final verification-only repairs**

Commit only if verification required code changes: `fix: close options remediation verification gaps`
