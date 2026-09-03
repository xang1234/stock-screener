# Options Command Center Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task by task. Use superpowers:test-driven-development for every behavior change and superpowers:verification-before-completion before claiming completion.

**Goal:** Add a bounded US Options Command Center and ticker drill-down sourced from the published equity leaders, with truthful Yahoo-derived analytics, ticker-continuous history, and identical live/static presentation.

**Architecture:** Build a dedicated options-analytics bounded context. A published US feature run supplies at most 80 Current Candidates; up to 20 recent dropouts are collected for continuity. One selected monthly chain per symbol is normalized behind a provider port, all metrics are computed from that observation, and a relational run/item/strike model is published atomically at 90% Current Candidate coverage. Live APIs and static artifacts serialize one strict contract.

**Tech stack:** Python 3.11, FastAPI, SQLAlchemy/Alembic, Celery, PostgreSQL, yfinance 0.2.66, NumPy/SciPy, React 18, React Router, TanStack Query, MUI, Recharts, Pytest, Vitest/Testing Library.

**Design source:** docs/superpowers/specs/2026-09-04-options-command-center-design.md

## Execution rules

- Work only on branch feat/options-command-center in the isolated worktree.
- Do not copy, cherry-pick, or adapt code, migrations, fixtures, or file layout from PR #339.
- Begin each task with the named failing tests. Confirm the failure is for the missing behavior before adding implementation.
- Tests and CI must never call Yahoo, Redis, Celery workers, or a live server unless the step is explicitly marked as a manual smoke test.
- Keep domain modules free of SQLAlchemy, FastAPI, Celery, yfinance, Redis, and React imports.
- Persist a single successful Chain Observation per run item even if the provider needed retries.
- Commit after each green task. Do not combine unrelated cleanup with a task commit.
- Before creating migration 0034, run alembic heads and adjust the revision/down-revision if main has advanced.

---

### Task 1: Add domain language, US capability, and the disabled runtime flag

**Files:**

- Modify: CONTEXT.md
- Modify: backend/app/domain/markets/catalog.py
- Modify: backend/app/schemas/app_runtime.py
- Modify: backend/app/config/settings.py
- Modify: backend/.env.example
- Modify: .env.docker.example
- Modify: frontend/src/contexts/RuntimeContext.jsx
- Test: backend/tests/unit/domain/test_market_catalog.py
- Test: backend/tests/unit/test_app_runtime_endpoints.py
- Test: frontend/src/contexts/RuntimeContext.test.jsx

**Step 1: Write failing capability tests**

Add assertions that:

- MarketCapabilities exposes options_analytics.
- US reports options_analytics=true and every other Market reports false.
- settings.options_analytics_enabled defaults to false.
- GET /api/v1/app-capabilities reports features.options_analytics=false by default while the US Market capability remains true.
- the frontend fallback capability is false.

**Step 2: Run the focused tests and confirm red**

~~~bash
cd backend
pytest -q tests/unit/domain/test_market_catalog.py tests/unit/test_app_runtime_endpoints.py
cd ../frontend
npm run test:run -- src/contexts/RuntimeContext.test.jsx
~~~

Expected: failures for the missing field/flag, not import or fixture failures.

**Step 3: Implement the minimal capability surface**

- Add options_analytics to MarketCapabilities.
- Set it true only on the US catalog entry and false in shared/non-US capability constants.
- Add options_analytics_enabled: bool = False to Settings.
- Add options_analytics to settings.capability_flags().
- Add OPTIONS_ANALYTICS_ENABLED=false to both environment examples with a note that enabling it permits bounded Yahoo options traffic.
- Add Options Analytics Run, Candidate Cohort, Current Candidate, Continuity Candidate, Chain Observation, Published Options Run, and Model Estimate to CONTEXT.md.
- Add options_analytics=false to DEFAULT_CAPABILITIES.features.

Do not add configuration knobs for cohort size, thresholds, formulas, or retention.

**Step 4: Run tests and commit**

~~~bash
cd backend
pytest -q tests/unit/domain/test_market_catalog.py tests/unit/test_app_runtime_endpoints.py
cd ../frontend
npm run test:run -- src/contexts/RuntimeContext.test.jsx
cd ..
git add CONTEXT.md backend/app/domain/markets/catalog.py backend/app/schemas/app_runtime.py backend/app/config/settings.py backend/.env.example .env.docker.example frontend/src/contexts/RuntimeContext.jsx frontend/src/contexts/RuntimeContext.test.jsx backend/tests/unit/domain/test_market_catalog.py backend/tests/unit/test_app_runtime_endpoints.py
git commit -m "feat: declare options analytics capability"
~~~

---

### Task 2: Define options domain types and deterministic cohort policies

**Files:**

- Create: backend/app/domain/options_analytics/__init__.py
- Create: backend/app/domain/options_analytics/models.py
- Create: backend/app/domain/options_analytics/ports.py
- Create: backend/app/domain/options_analytics/selection.py
- Create: backend/app/domain/options_analytics/expiration.py
- Create: backend/app/domain/options_analytics/history.py
- Create: backend/app/domain/options_analytics/quality.py
- Create: backend/app/domain/scanning/leadership_policy.py
- Modify: backend/app/services/daily_snapshot_service.py
- Test: backend/tests/unit/domain/options_analytics/test_selection.py
- Test: backend/tests/unit/domain/options_analytics/test_expiration.py
- Test: backend/tests/unit/domain/options_analytics/test_history.py
- Test: backend/tests/unit/domain/options_analytics/test_quality.py
- Modify test: backend/tests/unit/test_daily_snapshot_service.py

**Step 1: Write failing policy tests**

Cover these exact cases:

- Top Candidates and Leaders are each ordered by composite score descending, then canonical symbol.
- The options cohort rejects exactly USD 100,000,000 and accepts values strictly above it.
- Each source is capped independently at 40; unused slots are not borrowed.
- A duplicate symbol keeps both flags and both source ranks.
- a symbol absent after Current membership is Continuity for sessions 1 through 5 and gone before session 6.
- Continuity is capped at 20 by most recent membership, best prior rank, then symbol.
- Current membership always overrides Continuity.
- the total cohort cannot exceed 100.
- the nearest standard monthly expiration in 14–45 calendar DTE is selected, including a Thursday holiday adjustment; weekly and out-of-window expirations are rejected.
- strike retention keeps closest-to-spot plus at most 30 lower and 30 higher distinct strikes.
- five compatible observations in the trailing seven US sessions enables short history; 20 in the trailing 30 enables IV history; gaps do not reset lifetime history.
- zero Current Candidates and coverage below 0.90 both block publication; Continuity never enters the denominator.

Use frozen dataclasses/enums for:

- OptionCandidateInput and OptionCandidate
- CandidateKind
- NormalizedOptionContract and ChainObservation
- ObservationState and reason codes
- OptionsRunStatus and OptionsRunSummary
- MetricValue with availability/reasons
- PublicationDecision and HistoryReadiness

Define provider, source-reader, repository, calendar, clock, progress, and cancellation protocols in ports.py.

**Step 2: Run tests and confirm red**

~~~bash
cd backend
pytest -q tests/unit/domain/options_analytics tests/unit/test_daily_snapshot_service.py
~~~

Expected: new-module import failures or missing behavior.

**Step 3: Implement pure policies**

- Put the shared leader thresholds and stable ordering keys in leadership_policy.py.
- Refactor daily_snapshot_service.py to import those constants/order definitions without changing its existing Top 20 response contract.
- Keep the Command Center-specific strict liquidity comparison and 40-item caps in options_analytics/selection.py.
- Make expiration selection consume listed dates plus an injected US-session calendar; never import a calendar service into the domain.
- Make history compatibility require the same calculation version.
- Return explicit reason codes such as expiration_unavailable, building_history, insufficient_core_coverage, and empty_current_cohort.

**Step 4: Run tests, check domain import boundaries, and commit**

~~~bash
cd backend
pytest -q tests/unit/domain/options_analytics tests/unit/test_daily_snapshot_service.py
python -m pytest -q tests/unit/domain/test_feature_store_no_io.py
git add app/domain/options_analytics app/domain/scanning/leadership_policy.py app/services/daily_snapshot_service.py tests/unit/domain/options_analytics tests/unit/test_daily_snapshot_service.py
git commit -m "feat: define options analytics domain policies"
~~~

---

### Task 3: Implement the metric calculators from normalized data

**Files:**

- Create: backend/app/domain/options_analytics/metrics/__init__.py
- Create: backend/app/domain/options_analytics/metrics/max_pain.py
- Create: backend/app/domain/options_analytics/metrics/gex.py
- Create: backend/app/domain/options_analytics/metrics/volatility.py
- Create: backend/app/domain/options_analytics/metrics/activity.py
- Create: backend/app/domain/options_analytics/metrics/aggregate.py
- Test: backend/tests/unit/domain/options_analytics/metrics/test_max_pain.py
- Test: backend/tests/unit/domain/options_analytics/metrics/test_gex.py
- Test: backend/tests/unit/domain/options_analytics/metrics/test_volatility.py
- Test: backend/tests/unit/domain/options_analytics/metrics/test_activity.py
- Test: backend/tests/unit/domain/options_analytics/metrics/test_aggregate.py

**Step 1: Write golden and boundary tests**

Use hand-built normalized chains, never Yahoo-shaped DataFrames. Pin expected values for:

- Max Pain over every usable strike in the full chain and lower-strike tie resolution.
- Black-Scholes unit gamma and dollar GEX per 1% move.
- regular 100 multiplier and unavailable non-regular multiplier.
- positive-call/negative-put dealer proxy labeling.
- genuine interpolated gamma crossing and no-crossing unavailable result.
- Estimated Call Wall and Estimated Put Wall.
- ATM IV requiring valid call and put IV at the closest strike.
- 25-delta skew requiring both selected deltas in 0.20–0.30.
- 20-return realized volatility from 21 closes with no fill and square-root-of-252 annualization.
- VRP as ATM IV minus realized volatility.
- volume/OI zero-denominator handling, plus/minus 5% concentration, 100-contract floor, activity intensity, and stable cross-sectional rank.
- NaN, infinity, negative OI, invalid IV, empty sides, and partial metric availability.

Use approximate numeric assertions with explicit tolerances.

**Step 2: Run tests and confirm red**

~~~bash
cd backend
pytest -q tests/unit/domain/options_analytics/metrics
~~~

**Step 3: Add minimal pure implementations**

- Use math/scipy only inside pure calculators.
- Return MetricValue objects instead of bare None.
- Calculate aggregates from the full normalized chain; truncate only persistence/chart points.
- Record assumption evidence with every GEX-family result.
- Do not emit directional flow, strategy, recommendation, or sentiment fields.
- Add json-finiteness invariants to the aggregate result.

**Step 4: Run tests and commit**

~~~bash
cd backend
pytest -q tests/unit/domain/options_analytics/metrics
git add app/domain/options_analytics/metrics tests/unit/domain/options_analytics/metrics
git commit -m "feat: calculate truthful options metrics"
~~~

---

### Task 4: Add relational options runs, items, strike points, and pointer

**Files:**

- Create: backend/app/infra/db/models/options_analytics.py
- Modify: backend/app/infra/db/models/__init__.py
- Modify: backend/app/models/__init__.py
- Create: backend/alembic/versions/20260904_0034_add_options_analytics.py
- Create: backend/tests/integration/test_options_analytics_migration.py
- Create: backend/tests/unit/test_options_analytics_schema.py

**Step 1: Verify the migration head**

~~~bash
cd backend
alembic heads
~~~

Expected before this task: 20260829_0033. If not, rename the new revision and set down_revision to the actual single head.

**Step 2: Write failing schema tests**

Require four tables:

1. options_analytics_runs
2. options_analytics_run_items
3. options_analytics_strike_points
4. options_analytics_pointers

Pin constraints:

- unique (input_signature, attempt_number) on runs;
- unique (run_id, security_symbol) on items;
- unique (item_id, strike) on strike points;
- primary/unique (market, calculation_version) on pointers;
- ordinary refresh runs require a source_feature_run_id foreign key;
- verified transfer-origin historical runs may leave the local foreign key null
  but must retain an external source_feature_run_key and origin=history_transfer;
- item and strike cascade only from their owning run/item;
- pointer deletion never cascades into a run.

Assert typed columns for frequently sorted summary metrics and JSON only for evidence, assumptions, warnings, and diagnostic detail.

**Step 3: Implement models and reversible migration**

The run stores source identity, origin, versions, attempt, lifecycle, risk-free input, counts, coverage, and timestamps. Add a database check so source_feature_run_id is non-null unless origin=history_transfer. The item stores candidate provenance, spot/expiry/state, current aggregates, history counts, retry count, evidence, and reasons. Strike points store call/put OI, volume, IV, and estimated GEX at one strike.

Do not store raw Yahoo payloads.

**Step 4: Prove upgrade/downgrade and commit**

~~~bash
cd backend
pytest -q tests/unit/test_options_analytics_schema.py tests/integration/test_options_analytics_migration.py
git add app/infra/db/models/options_analytics.py app/infra/db/models/__init__.py app/models/__init__.py alembic/versions/20260904_0034_add_options_analytics.py tests/unit/test_options_analytics_schema.py tests/integration/test_options_analytics_migration.py
git commit -m "feat: persist options analytics runs"
~~~

---

### Task 5: Implement repositories, candidate source query, history, and retention

**Files:**

- Create: backend/app/infra/db/repositories/options_analytics_repo.py
- Create: backend/app/infra/query/options_candidate_source.py
- Modify: backend/app/infra/db/repositories/__init__.py
- Modify: backend/app/infra/db/uow.py
- Test: backend/tests/unit/repositories/test_options_analytics_repo.py
- Test: backend/tests/unit/infra/test_options_candidate_source.py

**Step 1: Write failing repository tests**

Cover:

- start-or-reuse by input signature and attempt number;
- forced rerun increments an explicit attempt;
- stage exactly one item per symbol;
- save one observation and deterministic strike upsert;
- resume returns only terminally incomplete items;
- atomic status plus pointer advancement;
- failed quality leaves the old pointer unchanged;
- query the complete Published Options Run;
- ticker history spans absent-cohort gaps and ignores incompatible calculation versions;
- continuity reads last Current membership even if later runs contain only Continuity;
- prune aggregate items older than 252 US sessions and strike points beyond 30 published runs while preserving the pointed run;
- rollback leaves no partial pointer or duplicate observation.

For the candidate query, seed a published feature run and assert strict greater-than USD 100M, independent Top 40 caps, both provenance flags, source ranks, spot, dividend input, price closes, and canonical identity.

**Step 2: Run tests and confirm red**

~~~bash
cd backend
pytest -q tests/unit/repositories/test_options_analytics_repo.py tests/unit/infra/test_options_candidate_source.py
~~~

**Step 3: Implement adapters**

- Keep selection policy in the domain; the SQL source returns pinned feature/fundamental inputs.
- Use short transactions and flush before mapping generated IDs.
- Add options_analytics to SqlUnitOfWork without changing existing repository behavior.
- Use the source feature run ID, never “latest” again after the run starts.
- Expose read models that fetch the complete current set before any aggregate/rank calculation.
- Make retention accept injected trading-session cutoffs rather than doing calendar I/O inside the repository.

**Step 4: Run tests and commit**

~~~bash
cd backend
pytest -q tests/unit/repositories/test_options_analytics_repo.py tests/unit/infra/test_options_candidate_source.py
git add app/infra/db/repositories/options_analytics_repo.py app/infra/query/options_candidate_source.py app/infra/db/repositories/__init__.py app/infra/db/uow.py tests/unit/repositories/test_options_analytics_repo.py tests/unit/infra/test_options_candidate_source.py
git commit -m "feat: add options analytics repositories"
~~~

---

### Task 6: Add the Yahoo options adapter behind the provider port

**Files:**

- Create: backend/app/infra/providers/yahoo_options.py
- Modify: backend/app/infra/providers/__init__.py
- Test: backend/tests/unit/infra/test_yahoo_options_provider.py
- Create fixture: backend/tests/fixtures/options/yahoo_chain_normalized_source.json

**Step 1: Write adapter contract tests**

Patch yfinance.Ticker with a fake. Assert:

- expiration strings become dates and the domain selector chooses one monthly date;
- calls/puts normalize strike, bid, ask, last price, volume, OI, IV, lastTradeDate, contract size, and multiplier;
- NaN differs from zero and non-finite values never reach the domain object;
- missing columns and empty frames produce explicit provider/schema failures;
- rate limiting occurs before each network attempt;
- at most three attempts are made;
- only the first successful payload is returned;
- a run-level IRX close is resolved on or before the pinned date;
- provider spot is diagnostic and cannot replace source spot;
- no adapter test opens Redis or makes a network call.

The fixture must be newly authored from the documented field contract. Do not copy a PR #339 fixture or persist a raw Yahoo response.

**Step 2: Run tests and confirm red**

~~~bash
cd backend
pytest -q tests/unit/infra/test_yahoo_options_provider.py
~~~

**Step 3: Implement the synchronous adapter**

- Inject the yfinance ticker factory, rate limiter, retry policy, and clock.
- Make expiration discovery and selected-chain fetch separate port operations.
- Convert timestamps to UTC and record fetched_at.
- Treat contractSize=REGULAR as multiplier 100; mark other/unknown sizes unavailable for GEX rather than assuming.
- Raise typed transient, throttled, schema, and unavailable errors.
- Never catch an exception and return fabricated market data.

**Step 4: Run tests and commit**

~~~bash
cd backend
pytest -q tests/unit/infra/test_yahoo_options_provider.py
git add app/infra/providers/yahoo_options.py app/infra/providers/__init__.py tests/unit/infra/test_yahoo_options_provider.py tests/fixtures/options/yahoo_chain_normalized_source.json
git commit -m "feat: normalize Yahoo option chains"
~~~

---

### Task 7: Build the refresh use case and publication transaction

**Files:**

- Create: backend/app/use_cases/options_analytics/__init__.py
- Create: backend/app/use_cases/options_analytics/refresh.py
- Create: backend/app/use_cases/options_analytics/queries.py
- Modify: backend/app/wiring/use_case_factories.py
- Modify: backend/app/wiring/bootstrap.py
- Test: backend/tests/unit/use_cases/options_analytics/test_refresh.py
- Test: backend/tests/unit/use_cases/options_analytics/test_queries.py

**Step 1: Write failing orchestration tests with fakes**

Cover the full use-case contract:

- pin one published US feature run and build the deterministic Current/Continuity cohort;
- resolve risk-free input once per run;
- fetch at most two symbols concurrently but persist on the calling thread;
- one successful selected-expiration payload feeds all metrics;
- retry a transient symbol up to three attempts without duplicating an observation;
- resume skips successful items;
- cancellation leaves a resumable staged run;
- per-symbol unavailable results do not discard successful siblings;
- compute historical readiness from compatible published items with genuine gaps;
- publish at exactly 90% Current coverage;
- fail quality at 89%, empty Current cohort, or invalid source run while preserving the prior pointer;
- Continuity failures do not affect coverage;
- run retention only after a successful commit;
- all returned task values are JSON-safe.

Use an in-memory fake repository/provider/calendar/clock. Assert no SQLAlchemy, yfinance, Celery, or Redis import is needed to run the use case.

**Step 2: Run tests and confirm red**

~~~bash
cd backend
pytest -q tests/unit/use_cases/options_analytics
~~~

**Step 3: Implement refresh and read use cases**

Refresh phases:

1. validate capability/market and pin the source run;
2. start or reuse the Options Analytics Run and stage its cohort;
3. resolve run-level assumptions;
4. fetch/calculate items with maximum concurrency two;
5. persist each completed item in a short transaction;
6. calculate complete-set activity ranks/history fields;
7. evaluate Current coverage;
8. atomically publish or mark failed quality;
9. prune by the approved retention policy.

queries.py must expose:

- get_published_command_center(market, calculation_version);
- get_published_symbol_detail(symbol, market, calculation_version);
- get_run_diagnostics(run_id).

Reads never call the provider or wait for a task.

**Step 4: Run tests and commit**

~~~bash
cd backend
pytest -q tests/unit/use_cases/options_analytics
git add app/use_cases/options_analytics app/wiring/use_case_factories.py app/wiring/bootstrap.py tests/unit/use_cases/options_analytics
git commit -m "feat: refresh and publish options analytics"
~~~

---

### Task 8: Expose strict protected live API contracts

**Files:**

- Create: backend/app/schemas/options_analytics.py
- Create: backend/app/api/v1/options_analytics.py
- Modify: backend/app/api/v1/router.py
- Test: backend/tests/unit/test_options_analytics_api.py
- Modify test: backend/tests/unit/test_api_route_registration.py
- Modify test: backend/tests/unit/test_server_auth_api.py

**Step 1: Write failing API tests**

Pin schema version and response shape for:

- GET /api/v1/options-analytics/command-center
- GET /api/v1/options-analytics/symbols/{symbol}
- GET /api/v1/options-analytics/runs/{run_id}/diagnostics
- POST /api/v1/options-analytics/refresh

Assert:

- read endpoints return only the Published Options Run;
- no published run returns a typed 404/unavailable response, never fake rows;
- summary statistics are based on the complete current set;
- unavailable, building_history, insufficient_quality, and stale remain distinct;
- NaN/Infinity fail serialization;
- refresh returns HTTP 202 with task/run identity immediately;
- diagnostics/refresh require the existing server session;
- route handlers do not import yfinance, call Redis, wait for AsyncResult, or execute calculations.

**Step 2: Run tests and confirm red**

~~~bash
cd backend
pytest -q tests/unit/test_options_analytics_api.py tests/unit/test_api_route_registration.py tests/unit/test_server_auth_api.py
~~~

**Step 3: Implement schemas and thin routes**

- Set model_config extra=forbid on every public model.
- Include equity source date/run, options observation time, provider, versions, coverage, staleness, provenance ranks, metric availability, assumptions, and reason codes.
- URL-decode then resolve symbols through canonical SecurityMaster behavior.
- Inject use cases/task dispatcher through wiring.
- Keep all routes under the router’s existing authentication dependency.

**Step 4: Run tests and commit**

~~~bash
cd backend
pytest -q tests/unit/test_options_analytics_api.py tests/unit/test_api_route_registration.py tests/unit/test_server_auth_api.py
git add app/schemas/options_analytics.py app/api/v1/options_analytics.py app/api/v1/router.py tests/unit/test_options_analytics_api.py tests/unit/test_api_route_registration.py tests/unit/test_server_auth_api.py
git commit -m "feat: expose options analytics API"
~~~

---

### Task 9: Wire the non-blocking daily follow-on and Operations visibility

**Files:**

- Create: backend/app/interfaces/tasks/options_analytics_tasks.py
- Modify: backend/app/celery_app.py
- Modify: backend/app/tasks/data_fetch_lock.py
- Modify: backend/app/tasks/daily_market_pipeline_tasks.py
- Modify: backend/app/services/task_registry_service.py
- Modify: backend/app/services/runtime_activity_contract.py
- Modify: frontend/src/contexts/RuntimeContext.jsx
- Test: backend/tests/unit/test_options_analytics_tasks.py
- Modify test: backend/tests/unit/test_data_fetch_lock.py
- Modify test: backend/tests/unit/test_daily_market_pipeline_tasks.py
- Modify test: backend/tests/unit/test_celery_config.py
- Modify test: backend/tests/unit/test_market_worker_config.py
- Modify test: backend/tests/unit/test_operations_job_service.py
- Modify test: frontend/src/contexts/RuntimeContext.test.jsx

**Step 1: Write failing task/orchestration tests**

Assert:

- the data-fetch decorator accepts a pre-lock enabled predicate, so the disabled
  setting returns skipped before constructing Yahoo or Redis dependencies;
- only US plus Market capability may run;
- manual invocation without source_run_id resolves the current published US feature pointer;
- the task uses serialized_data_fetch_task and routes to data_fetch_us;
- no worker family or queue is added;
- guard_snapshot_result preserves run_id, as_of_date, and auto_scan_id;
- a final dispatch_options_after_snapshot signature queues the options task only after a published US feature result;
- the dispatch is fire-and-report: an options failure cannot change the already-published equity result or fail the completed equity chain;
- Operations shows one Daily US Options Analytics task and progress fields expected/completed/core-valid/failed/retried/coverage;
- the runtime stage label is Options Analytics.

**Step 2: Run tests and confirm red**

~~~bash
cd backend
pytest -q tests/unit/test_options_analytics_tasks.py tests/unit/test_data_fetch_lock.py tests/unit/test_daily_market_pipeline_tasks.py tests/unit/test_celery_config.py tests/unit/test_market_worker_config.py tests/unit/test_operations_job_service.py
~~~

**Step 3: Implement task wiring**

- Add an optional pre-lock enabled predicate to serialized_data_fetch_task and
  regression-test existing callers with no predicate.
- Register the new interface task in Celery include and the existing market-scoped data-fetch routing set.
- Decorate it with serialized_data_fetch_task and the options-enabled predicate
  so enabled work participates in both external-fetch and Market Workload
  coordination while disabled work exits before Redis.
- Emit market activity using stage_key=options and lifecycle=daily_refresh.
- Add dispatch_options_after_snapshot as the last daily-pipeline signature; it sends the options task to data_fetch_us and returns a JSON-safe dispatch result.
- Do not add a second Beat entry: the daily follow-on is driven by the published equity run.
- Add a task-registry entry for manual refresh, enabled only when OPTIONS_ANALYTICS_ENABLED is true.

**Step 4: Run tests and commit**

~~~bash
cd backend
pytest -q tests/unit/test_options_analytics_tasks.py tests/unit/test_data_fetch_lock.py tests/unit/test_daily_market_pipeline_tasks.py tests/unit/test_celery_config.py tests/unit/test_market_worker_config.py tests/unit/test_operations_job_service.py
cd ../frontend
npm run test:run -- src/contexts/RuntimeContext.test.jsx
git add ../backend/app/interfaces/tasks/options_analytics_tasks.py ../backend/app/celery_app.py ../backend/app/tasks/data_fetch_lock.py ../backend/app/tasks/daily_market_pipeline_tasks.py ../backend/app/services/task_registry_service.py ../backend/app/services/runtime_activity_contract.py ../backend/tests/unit/test_options_analytics_tasks.py ../backend/tests/unit/test_data_fetch_lock.py ../backend/tests/unit/test_daily_market_pipeline_tasks.py ../backend/tests/unit/test_celery_config.py ../backend/tests/unit/test_market_worker_config.py ../backend/tests/unit/test_operations_job_service.py src/contexts/RuntimeContext.jsx src/contexts/RuntimeContext.test.jsx
git commit -m "feat: schedule US options analytics follow-on"
~~~

---

### Task 10: Export and validate atomic static options artifacts

**Files:**

- Create: backend/app/services/static_options_contract.py
- Create: backend/app/services/static_options_exporter.py
- Create: backend/app/services/static_options_artifact_selector.py
- Modify: backend/app/services/static_site_export_service.py
- Modify: backend/app/tasks/static_export_tasks.py
- Modify: backend/app/scripts/export_static_site.py
- Test: backend/tests/unit/test_static_options_contract.py
- Test: backend/tests/unit/test_static_options_exporter.py
- Test: backend/tests/unit/test_static_options_artifact_selector.py
- Modify test: backend/tests/unit/test_static_site_export_service.py
- Modify test: backend/tests/unit/test_export_static_site_script.py
- Modify test: backend/tests/unit/test_export_static_site_refresh.py

**Step 1: Write failing artifact tests**

Require:

- options/manifest.json
- options/command-center.json
- options/symbols/{url-safe-key}.json

Validate schema/calculation version, published run ID, source feature run/date, provider/observation timestamps, current coverage, symbol/path mapping, at most 80 current rows, unique symbols/strikes, finite numbers, and matching run identity in every file.

Assert:

- Continuity-only symbols are absent.
- Every Current Candidate has a summary, including unavailable states.
- a returning ticker detail includes compatible history with missing dates unfilled.
- exporter output is staged then promoted as one directory.
- current valid options win; otherwise a compatible last-good options artifact is copied and marked stale relative to fresh equity; absent options do not block equity export.
- static export never invokes Yahoo or live HTTP.

**Step 2: Run tests and confirm red**

~~~bash
cd backend
pytest -q tests/unit/test_static_options_contract.py tests/unit/test_static_options_exporter.py tests/unit/test_static_options_artifact_selector.py tests/unit/test_static_site_export_service.py tests/unit/test_export_static_site_script.py tests/unit/test_export_static_site_refresh.py
~~~

**Step 3: Implement isolated static components**

- Keep options writing/validation out of the already-large StaticSiteExportService; add only composition hooks.
- Serialize from the same Pydantic response models used by live reads.
- Have the root manifest advertise Options only after the complete options directory validates.
- In server export, use the existing target directory as last-good fallback.
- In combine mode, accept separate current/fallback options artifact roots and select them independently of Market artifacts.
- When fallback options are used, preserve their options observation/source date and set explicit stale-relative-to-equity metadata.

**Step 4: Run tests and commit**

~~~bash
cd backend
pytest -q tests/unit/test_static_options_contract.py tests/unit/test_static_options_exporter.py tests/unit/test_static_options_artifact_selector.py tests/unit/test_static_site_export_service.py tests/unit/test_export_static_site_script.py tests/unit/test_export_static_site_refresh.py
git add app/services/static_options_contract.py app/services/static_options_exporter.py app/services/static_options_artifact_selector.py app/services/static_site_export_service.py app/tasks/static_export_tasks.py app/scripts/export_static_site.py tests/unit/test_static_options_contract.py tests/unit/test_static_options_exporter.py tests/unit/test_static_options_artifact_selector.py tests/unit/test_static_site_export_service.py tests/unit/test_export_static_site_script.py tests/unit/test_export_static_site_refresh.py
git commit -m "feat: export static options command center"
~~~

---

### Task 11: Preserve options history across ephemeral GitHub Pages builds

**Files:**

- Create: backend/app/services/options_history_transfer.py
- Create: backend/app/scripts/import_options_history.py
- Create: backend/app/scripts/export_options_history.py
- Modify: backend/app/scripts/download_static_market_fallbacks.py
- Modify: backend/app/scripts/validate_static_market_artifacts.py
- Modify: .github/workflows/static-site.yml
- Test: backend/tests/unit/test_options_history_transfer.py
- Test: backend/tests/unit/test_options_history_scripts.py
- Modify test: backend/tests/unit/test_export_static_site_script.py
- Modify test: backend/tests/unit/test_static_site_workflow.py

**Step 1: Write failing transfer/fallback tests**

Assert the transfer bundle:

- has a schema version, calculation version, market, export timestamp, canonical payload checksum, observations, and last-Current membership dates;
- contains aggregate published observations only—no raw contracts or strike points;
- imports idempotently through OptionsAnalyticsRepository;
- recreates transferred historical runs with origin=history_transfer, preserves
  their external source-run keys, and never advances the local published pointer;
- rejects bad checksum, wrong Market, unsupported schema, incompatible calculation version, duplicate observation identity, non-finite values, and future dates;
- missing/incompatible input returns a visible fresh-history result instead of synthesizing rows;
- exports only after a newly Published Options Run.

Assert the static fallback downloader recognizes a separate static-options-US artifact, chooses the newest compatible valid artifact, and does not affect Market artifact validation.

**Step 2: Run tests and confirm red**

~~~bash
cd backend
pytest -q tests/unit/test_options_history_transfer.py tests/unit/test_options_history_scripts.py tests/unit/test_export_static_site_script.py tests/unit/test_static_site_workflow.py
~~~

**Step 3: Implement the portable handoff and workflow**

Update static-site.yml to:

1. ensure an options-analytics-data release exists;
2. for the US matrix job, download options-history-us-v1.json.gz if present;
3. import it after migrations and before the US daily refresh;
4. set OPTIONS_ANALYTICS_ENABLED=true only for that US static build;
5. run the options refresh in-process after feature publication;
6. upload /tmp/static-data/options as static-options-US only when valid;
7. export and replace the history release asset only when a new Options Run published;
8. in combine-and-build, download current and prior compatible options artifacts and pass both roots to combine mode.

Use workflow permissions already present. History-release upload may retry, but it must never publish an unvalidated bundle or block fresh equity pages.

**Step 4: Run tests and commit**

~~~bash
cd backend
pytest -q tests/unit/test_options_history_transfer.py tests/unit/test_options_history_scripts.py tests/unit/test_export_static_site_script.py tests/unit/test_static_site_workflow.py
git add app/services/options_history_transfer.py app/scripts/import_options_history.py app/scripts/export_options_history.py app/scripts/download_static_market_fallbacks.py app/scripts/validate_static_market_artifacts.py ../.github/workflows/static-site.yml tests/unit/test_options_history_transfer.py tests/unit/test_options_history_scripts.py tests/unit/test_export_static_site_script.py tests/unit/test_static_site_workflow.py
git commit -m "feat: preserve static options history"
~~~

---

### Task 12: Add matching live and static frontend data adapters

**Files:**

- Create: frontend/src/features/options/optionsContract.js
- Create: frontend/src/features/options/optionsContract.test.js
- Create: frontend/src/api/optionsAnalytics.js
- Create: frontend/src/api/optionsAnalytics.test.js
- Create: frontend/src/static/optionsClient.js
- Create: frontend/src/static/optionsClient.test.js

**Step 1: Write failing adapter tests**

Use one canonical fixture to prove both adapters expose the same view model. Test:

- command-center and symbol-detail query keys include run/path/symbol identity;
- live adapter uses only /v1/options-analytics routes;
- static adapter uses only paths advertised by options/manifest.json;
- no static refresh mutation exists;
- stale run, quality reasons, model-estimate labels, source ranks, zero, null, and unavailable survive normalization;
- a late AAPL detail response cannot populate an MSFT query key;
- malformed schema/run identity/path traversal is rejected.

**Step 2: Run tests and confirm red**

~~~bash
cd frontend
npm run test:run -- src/features/options/optionsContract.test.js src/api/optionsAnalytics.test.js src/static/optionsClient.test.js
~~~

**Step 3: Implement thin clients**

- Keep response validation/normalization in optionsContract.js.
- Return the same shape from live and static clients.
- Use manifest-provided url-safe paths; never derive a static filename from raw ticker input.
- Key symbol queries by canonical symbol plus published run ID/path.
- Keep static staleTime/gcTime infinite and live reads finite.

**Step 4: Run tests and commit**

~~~bash
cd frontend
npm run test:run -- src/features/options/optionsContract.test.js src/api/optionsAnalytics.test.js src/static/optionsClient.test.js
git add src/features/options/optionsContract.js src/features/options/optionsContract.test.js src/api/optionsAnalytics.js src/api/optionsAnalytics.test.js src/static/optionsClient.js src/static/optionsClient.test.js
git commit -m "feat: add options data clients"
~~~

---

### Task 13: Build shared Command Center and ticker-detail presentation

**Files:**

- Create: frontend/src/features/options/OptionsStatusBanner.jsx
- Create: frontend/src/features/options/OptionsQualityBadge.jsx
- Create: frontend/src/features/options/OptionsSourceBadges.jsx
- Create: frontend/src/features/options/OptionsCommandCenterView.jsx
- Create: frontend/src/features/options/OptionsMetricTable.jsx
- Create: frontend/src/features/options/OptionsSymbolDetailView.jsx
- Create: frontend/src/features/options/OptionsStrikeCharts.jsx
- Create: frontend/src/features/options/OptionsHistoryChart.jsx
- Test: frontend/src/features/options/OptionsCommandCenterView.test.jsx
- Test: frontend/src/features/options/OptionsSymbolDetailView.test.jsx
- Test: frontend/src/features/options/optionsAccessibility.test.jsx

**Step 1: Write failing component tests**

Cover:

- header shows separate equity date/run and options observation time, provider, calculation version, coverage, and stale/building state;
- one table changes focused columns for Gamma, Volatility, Skew, and Activity views rather than rendering multiple cohorts;
- all Current Candidates remain visible, while metric sorting places unavailable values last and excludes them from rank numbering;
- Candidate/Leader/Both badges and source ranks are visible;
- GEX, gamma flip, and walls say Estimated;
- missing values show a neutral marker with an accessible reason, while numeric zero displays as zero;
- stale banner is prominent and does not imply fresh options;
- row Enter/Space navigation and sortable headers are keyboard accessible;
- symbol detail renders OI/volume, estimated GEX, IV smile, history gaps, observation counts, assumptions, coverage, and warnings;
- the view has no provider/API logic and renders identically for equivalent live/static props.

**Step 2: Run tests and confirm red**

~~~bash
cd frontend
npm run test:run -- src/features/options/OptionsCommandCenterView.test.jsx src/features/options/OptionsSymbolDetailView.test.jsx src/features/options/optionsAccessibility.test.jsx
~~~

**Step 3: Implement presentation-only components**

- Use existing MUI density/theme conventions.
- Use Recharts for detail views already justified by the data relationships.
- Keep the default Command Center view compact; do not show all possible columns simultaneously.
- Put model assumptions and quality evidence in progressive disclosure on ticker detail.
- Never label activity as inflow, buying, selling, bullish, bearish, or dealer fact.

**Step 4: Run tests, lint, and commit**

~~~bash
cd frontend
npm run test:run -- src/features/options/OptionsCommandCenterView.test.jsx src/features/options/OptionsSymbolDetailView.test.jsx src/features/options/optionsAccessibility.test.jsx
npx eslint src/features/options
git add src/features/options
git commit -m "feat: build options command center views"
~~~

---

### Task 14: Connect live/static pages, routes, navigation, and refresh

**Files:**

- Create: frontend/src/pages/OptionsPage.jsx
- Create: frontend/src/pages/OptionsSymbolPage.jsx
- Create: frontend/src/static/pages/StaticOptionsPage.jsx
- Create: frontend/src/static/pages/StaticOptionsSymbolPage.jsx
- Modify: frontend/src/App.jsx
- Modify: frontend/src/components/Layout/Layout.jsx
- Modify: frontend/src/static/StaticAppShell.jsx
- Modify: frontend/src/static/StaticLayout.jsx
- Modify: frontend/src/static/dataClient.js
- Test: frontend/src/pages/OptionsPage.test.jsx
- Test: frontend/src/pages/OptionsSymbolPage.test.jsx
- Test: frontend/src/static/pages/StaticOptionsPage.test.jsx
- Test: frontend/src/static/pages/StaticOptionsSymbolPage.test.jsx
- Modify test: frontend/src/App.live.test.jsx
- Modify test: frontend/src/App.static.test.jsx
- Modify test: frontend/src/components/Layout/Layout.test.jsx

**Step 1: Write failing route and capability tests**

Assert:

- live /options and /options/:symbol lazy-load only when runtime feature flag and selected US Market capability are true;
- non-US or disabled deployments do not show Options navigation;
- static navigation appears only when the selected US manifest advertises a valid options page;
- static routes never render or call Refresh;
- live Refresh posts once, shows accepted task identity, and invalidates reads only after the task/run status changes;
- deep-linking to a missing/unavailable symbol gives a clear state and a route back to Command Center;
- command-center row navigation URL-encodes the canonical symbol;
- stale async detail data cannot overwrite the newly selected ticker.

**Step 2: Run tests and confirm red**

~~~bash
cd frontend
npm run test:run -- src/pages/OptionsPage.test.jsx src/pages/OptionsSymbolPage.test.jsx src/static/pages/StaticOptionsPage.test.jsx src/static/pages/StaticOptionsSymbolPage.test.jsx src/App.live.test.jsx src/App.static.test.jsx src/components/Layout/Layout.test.jsx
~~~

**Step 3: Implement pages and routes**

- Live pages own TanStack queries/mutation and pass pure props to the shared views.
- Static pages own manifest/file queries and pass the same props.
- Add lazy imports in App.jsx.
- Add /options and /options/:symbol to both routers.
- Determine active nav state by prefix so ticker detail keeps Options selected.
- Gate static navigation from root manifest metadata and selected Market=US.

**Step 4: Run tests, lint, build, and commit**

~~~bash
cd frontend
npm run test:run -- src/pages/OptionsPage.test.jsx src/pages/OptionsSymbolPage.test.jsx src/static/pages/StaticOptionsPage.test.jsx src/static/pages/StaticOptionsSymbolPage.test.jsx src/App.live.test.jsx src/App.static.test.jsx src/components/Layout/Layout.test.jsx
npm run lint
npm run build
git add src/pages/OptionsPage.jsx src/pages/OptionsSymbolPage.jsx src/static/pages/StaticOptionsPage.jsx src/static/pages/StaticOptionsSymbolPage.jsx src/App.jsx src/components/Layout/Layout.jsx src/static/StaticAppShell.jsx src/static/StaticLayout.jsx src/static/dataClient.js src/pages/OptionsPage.test.jsx src/pages/OptionsSymbolPage.test.jsx src/static/pages/StaticOptionsPage.test.jsx src/static/pages/StaticOptionsSymbolPage.test.jsx src/App.live.test.jsx src/App.static.test.jsx src/components/Layout/Layout.test.jsx
git commit -m "feat: connect options live and static routes"
~~~

---

### Task 15: Add end-to-end contract parity and clean-room regression guards

**Files:**

- Create: backend/tests/integration/test_options_analytics_publication.py
- Create: backend/tests/integration/test_options_static_live_parity.py
- Create: backend/tests/unit/test_options_clean_room_policy.py
- Create: frontend/src/features/options/optionsSurfaceParity.test.jsx
- Modify: docs/OPERATIONS.md
- Modify: README.md

**Step 1: Write failing integration/parity tests**

Build a synthetic end-to-end scenario:

1. publish a US feature run with Candidates, Leaders, overlap, and one exact-$100M exclusion;
2. run refresh with a fake provider containing successes, building history, and one unavailable symbol;
3. cross the 90% gate and assert atomic pointer movement;
4. query the live contracts;
5. export the static bundle;
6. load static contracts and assert semantic equality with live contracts;
7. run the next day with a dropped ticker in Continuity;
8. return it after a gap and assert its history counter did not reset;
9. fail a later options run and assert fresh equity plus stale prior options export.

The clean-room test should scan production application paths and reject PR-specific legacy names such as net_premium_inflow, option_flow_signal, force-release endpoints, or separate max-pain/GEX task pipelines. Exclude documentation and the guard test itself.

**Step 2: Run tests and confirm red where wiring remains**

~~~bash
cd backend
pytest -q tests/integration/test_options_analytics_publication.py tests/integration/test_options_static_live_parity.py tests/unit/test_options_clean_room_policy.py
cd ../frontend
npm run test:run -- src/features/options/optionsSurfaceParity.test.jsx
~~~

**Step 3: Close only the demonstrated gaps and document operations**

Document:

- OPTIONS_ANALYTICS_ENABLED default and Yahoo traffic ceiling;
- existing data_fetch_us worker requirement;
- Daily US Options Analytics progress/diagnostics;
- manual canary procedure;
- 90% gate and stale-pointer behavior;
- retention and static history release behavior;
- rollback by disabling collection/navigation without deleting history;
- Yahoo limitations and Model Estimate labels.

Do not add new product behavior during this task.

**Step 4: Run integration tests and commit**

~~~bash
cd backend
pytest -q tests/integration/test_options_analytics_publication.py tests/integration/test_options_static_live_parity.py tests/unit/test_options_clean_room_policy.py
cd ../frontend
npm run test:run -- src/features/options/optionsSurfaceParity.test.jsx
git add ../backend/tests/integration/test_options_analytics_publication.py ../backend/tests/integration/test_options_static_live_parity.py ../backend/tests/unit/test_options_clean_room_policy.py src/features/options/optionsSurfaceParity.test.jsx ../docs/OPERATIONS.md ../README.md
git commit -m "test: prove options command center parity"
~~~

---

### Task 16: Full verification, manual Yahoo canary, and handoff

**Files:**

- No feature files should be added in this task.
- Update only documentation if verification exposes an inaccurate command or operational note.

**Step 1: Run backend quality gates**

~~~bash
cd backend
pytest -q
~~~

Run the repository’s configured backend lint/type checks if present in CI. Resolve failures in the task that introduced them; do not weaken gates.

**Step 2: Run frontend quality gates**

~~~bash
cd frontend
npm run test:run
npm run lint
npm run build
~~~

**Step 3: Run static and Compose contract checks**

Run the same static workflow, migration, route-registration, worker-config, and Compose-health commands used by CI. Confirm:

- OPTIONS_ANALYTICS_ENABLED=false produces no Yahoo options calls;
- no test requires Redis unless it explicitly supplies a fake;
- both live and static builds compile;
- no new worker/queue family exists;
- git diff --check is clean.

**Step 4: Perform the opt-in manual canary**

With explicit network access and OPTIONS_ANALYTICS_ENABLED=true:

- run no more than three liquid US symbols first;
- inspect selected expiration, request count, coverage, latest trade evidence, and all unavailable reasons;
- verify GEX inputs include recorded IRX/dividend assumptions;
- then run the full Current plus Continuity cohort;
- require at least 90% core-valid Current coverage before accepting publication;
- export static artifacts and load /options and one /options/:symbol route locally.

The canary is observational. Do not edit fixtures from live Yahoo output and do not replace failed values with fallbacks.

**Step 5: Review branch scope**

~~~bash
git status --short
git diff origin/main...HEAD --stat
git diff origin/main...HEAD --check
git log --oneline origin/main..HEAD
~~~

Confirm every changed file belongs to the approved Options Command Center design and no PR #339 code was introduced.

**Step 6: Use the finishing workflow**

Invoke superpowers:verification-before-completion, then superpowers:requesting-code-review. After review findings are resolved and all verification is fresh, invoke superpowers:finishing-a-development-branch and present merge/PR choices to the user.
