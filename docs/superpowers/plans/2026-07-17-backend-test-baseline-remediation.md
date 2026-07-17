# Backend Test Baseline Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore a green, hermetic backend test baseline and make it a required CI signal without changing breadth/group cache-only behavior.

**Architecture:** Repair tests at their canonical contract boundaries rather than restoring retired production behavior. Move pure domain concepts out of schema/infra layers while preserving compatibility re-exports, and make live-service harnesses opt-in before their modules import side-effecting dependencies.

**Tech Stack:** Python 3.11, pytest, FastAPI/httpx, SQLAlchemy, pandas, GitHub Actions, Beads.

## Global Constraints

- Preserve breadth/group cache-only execution and existing public compatibility imports.
- Do not restore direct Gemini extraction fallback; provider fallback remains owned by `LLMService`.
- Do not remove Daily Snapshot key-market instruments from price refresh plans.
- Do not weaken complete finite OHLC normalization.
- Default pytest collection must perform no network, Redis, or database mutation.
- Live, load, and chaos checks remain opt-in and excluded from pull-request CI.

---

### Task 1: Make live-service test modules safe to collect

**Files:**
- Modify: `backend/pytest.ini`
- Modify: `backend/tests/integration/test_production_performance.py`
- Modify: `backend/tests/integration/test_cache_warming.py`
- Modify: `backend/tests/integration/test_scan_api.py`

**Interfaces:**
- Consumes: `RUN_LIVE_SERVICE_TESTS=1` as the explicit opt-in switch.
- Produces: the registered `live_service` pytest marker and import-safe module skips.

- [ ] **Step 1: Verify the current collection failure**

Run:

```bash
cd backend
/Users/admin/StockScreenClaude/backend/venv/bin/python -m pytest --collect-only -q
```

Expected: collection performs localhost HTTP work and aborts in a live-service module.

- [ ] **Step 2: Add an import-time opt-in guard before side-effecting imports**

Add immediately after each module docstring:

```python
import os

import pytest

pytestmark = pytest.mark.live_service

if os.getenv("RUN_LIVE_SERVICE_TESTS") != "1":
    pytest.skip(
        "requires RUN_LIVE_SERVICE_TESTS=1 and a running backend",
        allow_module_level=True,
    )
```

Register the marker in `backend/pytest.ini`:

```ini
    live_service: marks tests that require an explicitly started live backend and external services
```

- [ ] **Step 3: Verify collection no longer executes the live harnesses**

Run:

```bash
cd backend
/Users/admin/StockScreenClaude/backend/venv/bin/python -m pytest \
  tests/integration/test_production_performance.py \
  tests/integration/test_cache_warming.py \
  tests/integration/test_scan_api.py \
  --collect-only -q
```

Expected: exit 0 with three skipped modules and no localhost, Redis, or database errors.

- [ ] **Step 4: Commit the safe-collection boundary**

```bash
git add backend/pytest.ini backend/tests/integration/test_production_performance.py backend/tests/integration/test_cache_warming.py backend/tests/integration/test_scan_api.py
git commit -m "test: make live service harnesses opt in"
```

### Task 2: Repair canonical catalog, bootstrap, docs, and policy assertions

**Files:**
- Modify: `backend/tests/unit/domain/test_bootstrap_plan.py`
- Modify: `backend/tests/unit/golden/snapshots/mcp_market_overview.json`
- Modify: `backend/tests/unit/test_market_drift_guards.py`
- Modify: `backend/tests/unit/test_operational_flags.py`
- Modify: `backend/tests/unit/test_pct_change_policy.py`
- Modify: `backend/tests/unit/test_provider_routing_policy.py`
- Modify: `backend/tests/unit/test_release_docs.py`
- Modify: `backend/tests/unit/test_zai_env_docs.py`

**Interfaces:**
- Consumes: `BootstrapOperation.CALCULATE_MARKET_EXPOSURE`, the runtime market catalog, typed invalidation payloads, Python AST call nodes, and parsed Compose YAML.
- Produces: contract assertions that fail only when canonical behavior actually drifts.

- [ ] **Step 1: Re-run the eleven-file cluster and preserve the red evidence**

```bash
cd backend
/Users/admin/StockScreenClaude/backend/venv/bin/python -m pytest \
  tests/unit/domain/test_bootstrap_plan.py \
  tests/unit/golden/test_mcp_market_copilot.py \
  tests/unit/test_market_drift_guards.py \
  tests/unit/test_operational_flags.py \
  tests/unit/test_pct_change_policy.py \
  tests/unit/test_provider_routing_policy.py \
  tests/unit/test_release_docs.py \
  tests/unit/test_zai_env_docs.py -q
```

Expected: ten failures after excluding the architecture and price-cache tests handled later.

- [ ] **Step 2: Update bootstrap and catalog assertions**

Add the exposure stage and operation to the expected bootstrap sequence. Replace the deleted frontend `MARKET_LABELS` parser assertion with an assertion against the canonical frontend fallback catalog. Make the provider-routing test compare with `KNOWN_MARKETS` while retaining explicit US/Asia/AU membership assertions.

- [ ] **Step 3: Regenerate and review the MCP market-overview task array**

Use the test fixture's `MarketCopilotService` and `_normalized_payload` helper to
write the actual `tasks` array to a temporary JSON value. Replace only the
`tasks` array in `mcp_market_overview.json`; retain the other nine top-level
values unchanged. Run the single snapshot test and inspect the diff:

```bash
cd backend
/Users/admin/StockScreenClaude/backend/venv/bin/python -m pytest \
  tests/unit/golden/test_mcp_market_copilot.py::test_market_overview_snapshot -q
git diff -- tests/unit/golden/snapshots/mcp_market_overview.json
```

Expected: the test passes and the diff contains only current daily-market task
definitions in canonical market order.

- [ ] **Step 4: Replace the line-based pandas policy with AST inspection**

Implement the policy test around parsed calls:

```python
tree = ast.parse(path.read_text(), filename=str(path))
for node in ast.walk(tree):
    if not isinstance(node, ast.Call):
        continue
    if not isinstance(node.func, ast.Attribute) or node.func.attr != "pct_change":
        continue
    keyword_names = {keyword.arg for keyword in node.keywords}
    if "fill_method" not in keyword_names:
        offenders.append(f"{path.relative_to(backend_app.parent)}:{node.lineno}")
```

- [ ] **Step 5: Repair typed-payload and documentation contracts**

Assert the invalidation payload dictionary fields rather than string membership. Make the release-doc test validate a version-shaped `APP_IMAGE_TAG=vN.N.N` and consistency across release instructions instead of pinning `v1.2.0`. Parse `docker-compose.yml` with `yaml.safe_load`, resolve the environment mapping used by each theme service, and assert `ZAI_API_KEY`, `ZAI_API_KEYS`, and `ZAI_API_BASE` are present.

- [ ] **Step 6: Verify the repaired cluster**

Run the Step 1 command again. Expected: all selected tests pass.

- [ ] **Step 7: Commit canonical contract repairs**

```bash
git add backend/tests/unit/domain/test_bootstrap_plan.py backend/tests/unit/golden/snapshots/mcp_market_overview.json backend/tests/unit/test_market_drift_guards.py backend/tests/unit/test_operational_flags.py backend/tests/unit/test_pct_change_policy.py backend/tests/unit/test_provider_routing_policy.py backend/tests/unit/test_release_docs.py backend/tests/unit/test_zai_env_docs.py
git commit -m "test: align policy and catalog contracts"
```

### Task 3: Repair cache-refresh planning and provider test doubles

**Files:**
- Modify: `backend/tests/unit/test_cache_refresh_unification.py`
- Modify: `backend/tests/unit/test_price_refresh_execution.py`
- Modify: `backend/tests/unit/test_price_refresh_live_runner.py`
- Modify: `backend/tests/unit/test_yahoo_batch_ingestion.py`
- Modify: `backend/tests/unit/test_market_cache_policy.py`
- Modify: `backend/tests/unit/test_cleanup_orphaned_scans.py`

**Interfaces:**
- Consumes: current `progress_callback` callable signatures, key-market instruments, finite OHLCV frames, and the single-active-scan database invariant.
- Produces: refresh tests that cover complete planned work and realistic collaborators.

- [ ] **Step 1: Reproduce the seventeen refresh failures and cache fallback failure**

```bash
cd backend
/Users/admin/StockScreenClaude/backend/venv/bin/python -m pytest \
  tests/unit/test_cache_refresh_unification.py \
  tests/unit/test_cleanup_orphaned_scans.py \
  tests/unit/test_price_refresh_execution.py \
  tests/unit/test_price_refresh_live_runner.py \
  tests/unit/test_yahoo_batch_ingestion.py \
  tests/unit/test_market_cache_policy.py -q
```

Expected: eighteen failures matching the diagnosed target expansion, callback signatures, OHLC fixtures, and unique index setup.

- [ ] **Step 2: Make refresh totals derive from the actual planned symbols**

For explicit-market tests, construct `expected_symbols` as universe symbols plus the supported symbols from `key_market_instruments(market)`. Assert progress totals and retry lists against that sequence. For shared refresh, assert that success/retry accounting contains the markets represented by the resulting symbol-market plan instead of only the two seed rows.

- [ ] **Step 3: Update provider doubles to implement the current callable protocol**

Every fake passed as `fetch_batch`, `fetch_with_backoff`, or Yahoo fallback accepts:

```python
def fake_fetch(
    symbols,
    period="2y",
    market=None,
    progress_callback=None,
    **kwargs,
):
    if progress_callback is not None:
        progress_callback(len(symbols))
    return results
```

Retain each test's existing result semantics and assert delegated progress only in the tests whose subject is progress.

- [ ] **Step 4: Use complete price frames**

Replace `Close`-only Redis and pipeline-fallback fixtures with Open, High, Low, Close, Adj Close, and Volume columns. Keep the existing 200-row freshness and market-scoped key assertions.

- [ ] **Step 5: Respect the single-active-scan invariant in cleanup setup**

Keep cancelled, stale-running, and completed rows in the deletion test, but move fresh-running preservation into the existing zero-match test so no fixture inserts two active rows concurrently.

- [ ] **Step 6: Remove the dead in-process smart-refresh bypass**

Run:

```bash
rg -n "allow_smart_refresh_time_window_bypass" backend/app backend/tests
```

The repository search currently returns only the production definition and its
unit test. Remove both; static export now uses `StaticDailyPriceRefreshService`
and no supported caller invokes the bypass.

- [ ] **Step 7: Verify the cache-refresh cluster**

Run the Step 1 command again. Expected: all selected tests pass.

- [ ] **Step 8: Commit cache-refresh test repairs**

```bash
git add backend/app/tasks/cache_tasks.py backend/app/services/price_refresh_activity.py backend/tests/unit/test_cache_refresh_unification.py backend/tests/unit/test_cleanup_orphaned_scans.py backend/tests/unit/test_price_refresh_execution.py backend/tests/unit/test_price_refresh_live_runner.py backend/tests/unit/test_yahoo_batch_ingestion.py backend/tests/unit/test_market_cache_policy.py
git commit -m "test: align price refresh contracts"
```

### Task 4: Repair auth, theme, and legacy scan behavior tests

**Files:**
- Modify: `backend/tests/unit/test_symbol_validation_endpoints.py`
- Modify: `backend/tests/unit/test_theme_identity_normalization.py`
- Modify: `backend/tests/unit/test_theme_matching_telemetry_api.py`
- Modify: `backend/tests/unit/test_theme_pipeline_observability_api.py`
- Modify: `backend/tests/unit/test_theme_reprocessing.py`
- Modify: `backend/tests/unit/test_get_filter_options_use_case.py`
- Modify: `backend/tests/unit/test_get_peers_use_case.py`

**Interfaces:**
- Consumes: protected API auth dependency, synchronous FastAPI route functions, `ContentItem.source_language`, `LLMService` fallback ownership, and unbound-scan fallback to `scan_results`.
- Produces: endpoint and use-case tests for supported public behavior.

- [ ] **Step 1: Reproduce the twenty-six failures**

```bash
cd backend
/Users/admin/StockScreenClaude/backend/venv/bin/python -m pytest \
  tests/unit/test_symbol_validation_endpoints.py \
  tests/unit/test_theme_identity_normalization.py \
  tests/unit/test_theme_matching_telemetry_api.py \
  tests/unit/test_theme_pipeline_observability_api.py \
  tests/unit/test_theme_reprocessing.py \
  tests/unit/test_get_filter_options_use_case.py \
  tests/unit/test_get_peers_use_case.py -q
```

Expected: twenty-six failures matching auth interception, stale async calls, missing language, removed Gemini seam, and contradictory legacy expectations.

- [ ] **Step 2: Isolate symbol validation from server authentication**

Use `monkeypatch` in the `client` fixture to set `settings.server_auth_enabled = False` for the fixture lifetime, restoring it automatically after the test. Keep the real application router and SQLite database so validation remains end-to-end.

- [ ] **Step 3: Call synchronous route functions synchronously**

Remove `async def`, `@pytest.mark.asyncio`, and `await` from the twelve direct route-function tests. Do not change the production endpoint definitions.

- [ ] **Step 4: Complete the content fixture and retire the private Gemini seam**

Add `source_language=None` to the `SimpleNamespace` content item. Remove the test that patches `_try_generate_gemini`; provider fallback is already exercised through `LLMService` tests and the extraction service should expose only sanctioned LiteLLM routing.

- [ ] **Step 5: Remove contradictory unbound-scan assertions**

Delete the two `TestUnboundScanRaises` cases from the legacy files. Retain the newer tests under `tests/unit/use_cases/` that assert fallback to `scan_results` and missing-symbol errors.

- [ ] **Step 6: Verify the repaired endpoint/use-case cluster**

Run the Step 1 command again. Expected: all selected tests pass.

- [ ] **Step 7: Commit endpoint and legacy contract repairs**

```bash
git add backend/tests/unit/test_symbol_validation_endpoints.py backend/tests/unit/test_theme_identity_normalization.py backend/tests/unit/test_theme_matching_telemetry_api.py backend/tests/unit/test_theme_pipeline_observability_api.py backend/tests/unit/test_theme_reprocessing.py backend/tests/unit/test_get_filter_options_use_case.py backend/tests/unit/test_get_peers_use_case.py
git commit -m "test: align endpoint and legacy scan contracts"
```

### Task 5: Restore use-case layer boundaries with compatibility re-exports

**Files:**
- Create: `backend/app/domain/markets/price_coverage.py`
- Create: `backend/app/domain/feature_store/bootstrap_coverage.py`
- Create: `backend/app/domain/common/normalization.py`
- Modify: `backend/app/services/price_coverage_policy.py`
- Modify: `backend/app/services/bootstrap_cache_coverage.py`
- Modify: `backend/app/use_cases/feature_store/build_daily_snapshot.py`
- Modify: `backend/app/wiring/bootstrap.py`
- Modify: `backend/app/domain/universe/__init__.py`
- Modify: `backend/app/schemas/universe.py`
- Modify: `backend/app/use_cases/scanning/create_scan.py`
- Modify: `backend/app/infra/serialization.py`
- Modify: `backend/app/use_cases/scanning/export_scan_results.py`
- Test: `backend/tests/unit/test_layer_boundaries.py`
- Test: `backend/tests/unit/test_bootstrap_cache_coverage.py`
- Test: `backend/tests/unit/test_universe_schema.py`

**Interfaces:**
- Consumes: current price-coverage thresholds and `UniverseType` values.
- Produces: `app.domain.markets.price_coverage` policy API,
  `app.domain.feature_store.bootstrap_coverage.normalize_bootstrap_gate_report`,
  `app.domain.common.normalization.normalize_string_list`, and compatibility
  imports from the previous service/schema/infra locations.

- [ ] **Step 1: Preserve the red architecture signal**

```bash
cd backend
/Users/admin/StockScreenClaude/backend/venv/bin/python -m pytest \
  tests/unit/test_layer_boundaries.py::TestLayerBoundaries::test_use_cases_imports_only_stdlib_and_domain -q
```

Expected: one failure listing four forbidden imports.

- [ ] **Step 2: Add compatibility tests before moving domain concepts**

Add assertions that the old and new import locations return identical objects:

```python
from app.domain.universe import UniverseType as DomainUniverseType
from app.schemas.universe import UniverseType as SchemaUniverseType
assert DomainUniverseType is SchemaUniverseType

from app.domain.common.normalization import normalize_string_list as domain_normalize
from app.infra.serialization import normalize_string_list as infra_normalize
assert domain_normalize([" A ", None, 2]) == infra_normalize([" A ", None, 2])
```

Run the new assertions before implementation. Expected: import failure because the domain exports do not exist.

- [ ] **Step 3: Move price coverage policy into the domain**

Create `app.domain.markets.price_coverage` with the existing constants,
`PriceCoveragePolicy`, `normalize_market_code`, and
`price_coverage_policy_for_market`. Make `app.services.price_coverage_policy`
re-export those names. Move the pure `normalize_bootstrap_gate_report` function
to `app.domain.feature_store.bootstrap_coverage`, make
`app.services.bootstrap_cache_coverage` re-export it, and update the use case to
import the domain function. Update bootstrap coverage service imports to the
domain policy module.

- [ ] **Step 4: Move `UniverseType` and string-list normalization into the domain**

Define `UniverseType` in `app.domain.universe`, import and re-export it from
`app.schemas.universe`, and update `create_scan.py` to import the domain enum.
Define `normalize_string_list` in `app.domain.common.normalization`, re-export it
from `app.infra.serialization`, and update `export_scan_results.py` to import the
domain helper.

- [ ] **Step 5: Inject trading-day and bootstrap coverage collaborators**

Remove `pandas_market_calendars` and service imports from the use-case module.
Extend the constructor with the evaluator callable:

```python
bootstrap_coverage_evaluator: Callable[..., Mapping[str, object]] | None = None
```

Use the already-injected `MarketCalendarPort` for production trading-day checks.
Retain `_is_us_trading_day` as a dependency-free compatibility hook for existing
unit tests, raising `ValidationError` when neither a calendar nor a patched hook
is available. Wire `evaluate_bootstrap_cache_coverage` as the evaluator in
`get_build_daily_snapshot_use_case`; normalization comes from the new domain
module.

- [ ] **Step 6: Verify compatibility and architecture**

```bash
cd backend
/Users/admin/StockScreenClaude/backend/venv/bin/python -m pytest \
  tests/unit/test_layer_boundaries.py \
  tests/unit/test_bootstrap_cache_coverage.py \
  tests/unit/test_universe_schema.py \
  tests/unit/use_cases/feature_store/test_build_daily_snapshot.py -q
```

Expected: all selected tests pass and the boundary checker reports no use-case violations.

- [ ] **Step 7: Commit the boundary restoration**

```bash
git add backend/app/domain/markets/price_coverage.py backend/app/domain/feature_store/bootstrap_coverage.py backend/app/domain/common/normalization.py backend/app/services/price_coverage_policy.py backend/app/services/bootstrap_cache_coverage.py backend/app/use_cases/feature_store/build_daily_snapshot.py backend/app/wiring/bootstrap.py backend/app/domain/universe/__init__.py backend/app/schemas/universe.py backend/app/use_cases/scanning/create_scan.py backend/app/infra/serialization.py backend/app/use_cases/scanning/export_scan_results.py backend/tests/unit/test_layer_boundaries.py backend/tests/unit/test_bootstrap_cache_coverage.py backend/tests/unit/test_universe_schema.py backend/tests/unit/use_cases/feature_store/test_build_daily_snapshot.py
git commit -m "refactor: restore use case layer boundaries"
```

### Task 6: Add a comprehensive hermetic backend CI job

**Files:**
- Modify: `.github/workflows/ci.yml`

**Interfaces:**
- Consumes: the registered `live_service`, `load`, and existing pytest markers.
- Produces: a required `Backend Unit Suite` job and a publish dependency on that job.

- [ ] **Step 1: Add the CI job**

Add a job using the backend job's checkout, Python 3.11, dependency install, and database environment. Its commands are:

```yaml
      - name: Safe test collection
        run: cd backend && python -m pytest --collect-only -q -m "not live_service and not load"

      - name: Comprehensive backend unit suite
        run: cd backend && python -m pytest tests/unit -q
```

Add `backend-unit` to `publish-images.needs` so main/tag image publication cannot bypass it.

- [ ] **Step 2: Validate workflow syntax and dependency names**

```bash
/Users/admin/StockScreenClaude/backend/venv/bin/python -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml'))"
rg -n "backend-unit|Safe test collection|Comprehensive backend unit suite|needs:" .github/workflows/ci.yml
```

Expected: YAML parse exit 0 and the publish job depends on `backend-unit`.

- [ ] **Step 3: Commit CI coverage**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: run comprehensive backend unit suite"
```

### Task 7: Full verification and issue closure

**Files:**
- Modify: `.beads/issues.jsonl` through `bd` commands.

**Interfaces:**
- Consumes: all preceding commits.
- Produces: fresh verification evidence, closed Beads issues, and the pushed PR branch.

- [ ] **Step 1: Run the complete backend unit suite**

```bash
cd backend
/Users/admin/StockScreenClaude/backend/venv/bin/python -m pytest tests/unit -q
```

Expected: zero failures.

- [ ] **Step 2: Run safe default collection**

```bash
cd backend
/Users/admin/StockScreenClaude/backend/venv/bin/python -m pytest --collect-only -q -m "not live_service and not load"
```

Expected: exit 0 without live-service activity.

- [ ] **Step 3: Run focused breadth/group compatibility tests**

```bash
cd backend
/Users/admin/StockScreenClaude/backend/venv/bin/python -m pytest \
  tests/unit/test_breadth_calculator.py \
  tests/unit/test_breadth_calculator_service.py \
  tests/unit/test_breadth_coverage.py \
  tests/unit/test_breadth_tasks.py \
  tests/unit/test_daily_market_pipeline_tasks.py \
  tests/unit/test_derived_data_execution_policy.py \
  tests/unit/test_group_rank_backfill_tasks.py \
  tests/unit/test_group_rank_execution_policy.py \
  tests/unit/test_group_rank_historical_calculator.py \
  tests/unit/test_group_rank_in_process.py \
  tests/unit/test_group_rank_input_loader.py \
  tests/unit/test_group_rank_legacy_adapter.py \
  tests/unit/test_group_rank_models.py \
  tests/unit/test_group_rank_service.py \
  tests/unit/test_group_rank_tasks.py \
  tests/unit/test_group_ranking_calculator.py \
  tests/unit/test_group_ranking_repository.py \
  tests/unit/test_static_rrg_history_bundle.py -q
```

Expected: zero failures.

- [ ] **Step 4: Run curated backend gates**

```bash
make gate-identity gate-1 gate-2 gate-3 gate-4 gate-5
```

Expected: every make target exits 0.

- [ ] **Step 5: Inspect the final diff and close tracked work**

```bash
git diff origin/main...HEAD --check
git status --short
bd close stockscreenclaude-cis --reason "Comprehensive hermetic backend unit suite restored and added to CI"
bd close stockscreenclaude-w1l --reason "Live-service harnesses made explicit opt-in and collection-safe"
bd close stockscreenclaude-fn7 --reason "Resolved by stockscreenclaude-w1l"
```

- [ ] **Step 6: Commit Beads state and push**

```bash
git add .beads/issues.jsonl .beads/interactions.jsonl
git commit -m "chore: close backend test remediation issues"
git pull --rebase
git push
git status --branch --short
```

Expected: the branch is up to date with `origin/codex/issue-301-cache-only-derived-data`.
