# Options Command Center Review Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve the ten verified review findings on PR #358 without widening the Options Command Center cohort or introducing alternate data paths.

**Architecture:** Keep market validation at the pinned-feature-run gateway, serialize all SQL history reads before provider workers start, and calculate historical analytics in a pure domain module. Preserve one strict live/static response contract and keep manual refresh lifecycle state in the live page only.

**Tech Stack:** Python 3.11, FastAPI, SQLAlchemy, Alembic, pytest, React 18, TanStack Query, Vitest.

**Spec:** `docs/superpowers/specs/2026-09-04-options-command-center-design.md`

## Global Constraints

- US options only; reject any pinned feature run whose explicit Market is not `US`.
- At most three provider attempts and two provider workers.
- Never share a mutable SQLAlchemy `Session` across worker threads.
- Five-observation changes are absolute changes for the eight existing history series and preserve each metric's units.
- IV percentile and IV rank require 20 compatible observations in the trailing 30 US sessions.
- Live and static contracts remain identical and strict.

---

### Task 1: Source and metric correctness

**Files:**
- Modify: `backend/app/infra/query/options_candidate_source.py`
- Modify: `backend/app/domain/options_analytics/metrics/max_pain.py`
- Test: `backend/tests/unit/infra/test_options_candidate_source.py`
- Test: `backend/tests/unit/domain/options_analytics/metrics/test_max_pain.py`

**Interfaces:**
- Consumes: `feature_run_market(run)` and `NormalizedOptionContract.multiplier`.
- Produces: a US-only `CandidateSourceSnapshot` and multiplier-weighted Max Pain.

- [ ] Add a failing HK feature-run test expecting `LookupError`.
- [ ] Add a failing mixed-multiplier chain test with a hand-calculated minimizing strike.
- [ ] Require `feature_run_market(run) == "US"`; ignore contracts without a finite positive multiplier and weight payout by the multiplier.
- [ ] Run both focused test modules and confirm they pass.

### Task 2: Thread-safe analysis and Yahoo backoff

**Files:**
- Modify: `backend/app/domain/options_analytics/ports.py`
- Modify: `backend/app/use_cases/options_analytics/analysis_models.py`
- Modify: `backend/app/use_cases/options_analytics/candidate_analysis.py`
- Modify: `backend/app/use_cases/options_analytics/refresh.py`
- Modify: `backend/app/infra/providers/yahoo_options.py`
- Modify: `backend/app/wiring/use_case_factories.py`
- Test: `backend/tests/unit/use_cases/options_analytics/test_refresh.py`
- Test: `backend/tests/unit/use_cases/options_analytics/test_candidate_analysis.py`

**Interfaces:**
- Consumes: serial `PublishedOptionsReader.analysis_history(...)` and `RateBudgetPolicy.get_backoff_params("yfinance", "US")`.
- Produces: `AnalysisContext.historical_observations` and a callback invoked after throttled attempts 1 and 2.

- [ ] Add a failing test whose history reader rejects worker-thread access.
- [ ] Add a failing throttling test expecting policy delays before retry attempts.
- [ ] Preload per-symbol history before creating futures and move `ThrottledOptionsProviderError` to the domain provider port.
- [ ] Inject the US Yahoo backoff schedule through the use-case factory.
- [ ] Run the refresh, analyzer, provider, and factory tests and confirm they pass.

### Task 3: Latest-attempt history identity

**Files:**
- Modify: `backend/app/infra/db/repositories/options_history_repository.py`
- Modify: `backend/app/infra/db/repositories/published_options_reader.py`
- Test: `backend/tests/unit/repositories/test_options_analytics_repo.py`

**Interfaces:**
- Consumes: `OptionsAnalyticsRun.input_signature`, `attempt_number`, and `id`.
- Produces: one newest published observation per input identity and symbol for export and display.

- [ ] Add failing forced-attempt tests for transfer export and symbol history.
- [ ] Order newest attempts first within a session/input identity, deduplicate, then return deterministic chronological output.
- [ ] Run repository and transfer tests and confirm they pass.

### Task 4: Historical analytics contract

**Files:**
- Create: `backend/app/domain/options_analytics/metrics/history.py`
- Modify: `backend/app/domain/options_analytics/history.py`
- Modify: `backend/app/use_cases/options_analytics/analysis_models.py`
- Modify: `backend/app/use_cases/options_analytics/candidate_analysis.py`
- Modify: `backend/app/infra/db/models/options_analytics.py`
- Modify: `backend/alembic/versions/20260904_0034_add_options_analytics.py`
- Modify: `backend/app/infra/db/repositories/options_run_writer.py`
- Modify: `backend/app/infra/db/repositories/options_history_repository.py`
- Modify: `backend/app/infra/db/repositories/published_options_reader.py`
- Modify: `backend/app/schemas/options_analytics.py`
- Modify: `backend/app/schemas/options_history_transfer.py`
- Modify: `frontend/src/features/options/optionsSchema.json`
- Modify: `frontend/src/features/options/__fixtures__/optionsResponses.js`
- Modify: `frontend/src/features/options/OptionsSymbolDetailView.jsx`
- Test: domain, repository, schema, static-export, and frontend contract/detail modules.

**Interfaces:**
- Produces: `HistoricalMetricValues` containing `iv_percentile`, `iv_rank`, and `<series>_change_5` for `max_pain`, `net_gex`, `gamma_flip`, `atm_iv`, `skew_25_delta`, `realized_volatility`, `vrp`, and `activity_intensity`.

- [ ] Add failing pure-domain tests for readiness, gaps, percentile, zero-range rank, and all eight absolute changes.
- [ ] Implement finite-value historical calculations from compatible unique sessions only.
- [ ] Persist and transfer the ten values through typed columns and expose them as `historical_metrics` using `OptionsMetricResponse`.
- [ ] Regenerate the strict JSON schema and render historical context separately on ticker detail.
- [ ] Run all affected backend and frontend tests and confirm they pass.

### Task 5: Manual refresh lifecycle

**Files:**
- Modify: `frontend/src/pages/OptionsPage.jsx`
- Modify: `frontend/src/pages/OptionsPage.test.jsx`
- Test: `frontend/src/api/tasks.test.js`

**Interfaces:**
- Consumes: `getTaskStatus("daily-us-options-analytics", taskId)` and refresh requests with `sourceRunId: null`.
- Produces: retryable unavailable state and terminal task success/failure feedback.

- [ ] Add failing tests for stale refresh input, initial 404 bootstrap, terminal failed-quality, cancellation, and task exceptions.
- [ ] Dispatch without a source run, poll the accepted task, clear accepted state on every terminal result, and render Refresh when no publication exists.
- [ ] Invalidate live command/detail queries after successful publication and display terminal failure without permanently disabling Refresh.
- [ ] Run the page/API tests and confirm they pass.

### Task 6: Verification and review closure

**Files:**
- Modify only files required by failures found above.

**Interfaces:**
- Produces: a clean, mergeable PR with resolved inline review threads.

- [ ] Run focused backend and frontend options suites.
- [ ] Run all backend unit tests, frontend tests, lint, and build.
- [ ] Run `git diff --check` and confirm the worktree is clean after commits.
- [ ] Push the branch and reply to each GitHub inline comment with the verified resolution or technical disposition.
