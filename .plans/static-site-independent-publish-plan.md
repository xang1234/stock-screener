# Static Site Independent Publish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A finished market reaches the live site without waiting for slower markets, through a stateless serialized publisher woken by producer jobs.

**Architecture:** Move `combine-and-build` + `deploy` out of `static-site.yml` into a new `static-site-publish.yml` that downloads the newest valid artifact per market by name (#540 downloader) into one selection directory, validates, combines, builds and deploys under the `static-site-publisher` concurrency group. Producer jobs wake it with `gh workflow run` after uploading; a final always-run job wakes it once more. No cron.

**Tech Stack:** GitHub Actions YAML, `gh` CLI, existing backend scripts (`download_static_market_fallbacks`, `validate_static_market_artifacts`, `export_static_site`, `report_static_market_freshness`), pytest string/YAML structure tests.

**Spec:** `.plans/static-site-independent-publish.md`

## Global Constraints

- No `schedule` trigger on the publisher (user decision: wake-ups only).
- Publisher concurrency for production runs: group `static-site-publisher`, `cancel-in-progress: false`.
- Deploy only for `workflow_dispatch` on the default branch; `environment: github-pages`.
- Artifact lookup always uses the default branch (`github.event.repository.default_branch`), never the triggering ref.
- Wake-up failures never fail a producer job (`continue-on-error: true`).
- `static-site.yml` keeps `select-markets`, `ensure_daily_price_release`, `build-cot`, `build-market`, the `market_group` input values and its `static-site-${ref}` concurrency.
- Worktree: `/Users/admin/StockScreenClaude/.claude/worktrees/issue-499-independent-publish`; backend tests from `<wt>/backend` with `/Users/admin/StockScreenClaude/backend/venv/bin/python -m pytest`; git as `/usr/bin/git -C <wt> …`, one command per call.
- Conventional Commits ending with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Ruling against the spec

`workflow_dispatch` only triggers workflow files present on the default branch, so the spec's "dispatch the publisher on the feature branch" rehearsal is impossible pre-merge. The publisher also runs build-only on `pull_request` for same-repo PRs that touch its files, in a per-PR concurrency group, never deploying. `gh run download` does not check run status (cli/cli `run/download/download.go` lists artifacts directly), so the spec's artifact-ID download fallback is not needed.

## Review Focus

1. A PR rehearsal sharing the production concurrency group could replace a pending production publish → PR runs use a per-ref group (Task 1 test `test_publisher_pr_runs_never_share_the_production_group`).
2. A producer dispatched on a feature branch must not wake the production publisher → wake steps gated on the default branch (Task 2 test `test_wake_steps_only_run_on_the_default_branch`).
3. A failure in a build-market step after the artifact upload must not skip the wake-up → wake step uses `always()` + upload outcome (Task 2 test `test_market_wake_runs_after_any_later_failure`).
4. On a PR run the lookup must still read main's artifacts, not the PR branch's → `BRANCH_NAME` is the default branch (Task 1 test `test_publisher_reads_default_branch_artifacts_into_one_selection`).
5. Fork PRs must not run the publisher build → same-repo guard on the build job (Task 1 test `test_publisher_runs_only_on_dispatch_and_same_repo_prs`).

---

### Task 1: Publisher workflow

**Files:**
- Create: `.github/workflows/static-site-publish.yml`
- Test: `backend/tests/unit/test_static_site_workflow.py` (add publisher tests; move combine-job assertions to the publisher)

**Interfaces:**
- Produces: workflow file name `static-site-publish.yml` (Task 2 dispatches it by this name); dispatch input `rs_formula_overrides` (string, default `'{}'`).

- [ ] **Step 1: Write the failing tests** (append; add `import yaml` at the top)

```python
PUBLISH_WORKFLOW = ROOT / ".github" / "workflows" / "static-site-publish.yml"


def _publish_workflow() -> dict:
    return yaml.safe_load(PUBLISH_WORKFLOW.read_text(encoding="utf-8"))


def _publish_step(name: str) -> dict:
    steps = _publish_workflow()["jobs"]["build"]["steps"]
    return next(step for step in steps if step.get("name") == name)


def test_publisher_runs_only_on_dispatch_and_same_repo_prs() -> None:
    workflow = _publish_workflow()
    triggers = workflow[True]  # PyYAML parses the `on:` key as True
    assert set(triggers) == {"workflow_dispatch", "pull_request"}
    assert "rs_formula_overrides" in triggers["workflow_dispatch"]["inputs"]
    build_if = workflow["jobs"]["build"]["if"]
    assert "github.event.pull_request.head.repo.full_name == github.repository" in build_if
    assert "default_branch" in build_if


def test_publisher_pr_runs_never_share_the_production_group() -> None:
    concurrency = _publish_workflow()["concurrency"]
    assert concurrency["cancel-in-progress"] is False
    group = concurrency["group"]
    assert "static-site-publisher" in group
    assert "pull_request" in group and "github.ref" in group


def test_publisher_reads_default_branch_artifacts_into_one_selection() -> None:
    step = _publish_step("Download newest market artifacts")
    assert "continue-on-error" not in step
    assert step["env"]["BRANCH_NAME"] == "${{ github.event.repository.default_branch }}"
    assert step["env"]["CURRENT_RUN_ID"] == "${{ github.run_id }}"
    run = step["run"]
    assert "python -m app.scripts.download_static_market_fallbacks" in run
    assert "--fallback-dir /tmp/static-market-artifacts" in run
    assert "--fallback-options-dir /tmp/static-options" in run
    assert "--fallback-cot-dir /tmp/static-cot" in run


def test_publisher_validates_and_combines_one_selection() -> None:
    validate = _publish_step("Validate market artifacts")["run"]
    assert "--current-dir /tmp/static-market-artifacts" in validate
    assert "--selected-markets '[]'" in validate
    combine = _publish_step("Combine static data bundle")["run"]
    assert "--combine-artifacts-dir /tmp/static-market-artifacts" in combine
    assert "--fallback-artifacts-dir" not in combine
    assert "--options-artifacts-dir /tmp/static-options" in combine
    assert "--cot-artifacts-dir /tmp/static-cot" in combine
    report = _publish_step("Report market freshness")
    assert report["continue-on-error"] is True


def test_publisher_deploys_only_from_default_branch_dispatch() -> None:
    jobs = _publish_workflow()["jobs"]
    deploy_if = jobs["deploy"]["if"]
    assert "github.event_name == 'workflow_dispatch'" in deploy_if
    assert "default_branch" in deploy_if
    assert jobs["deploy"]["environment"]["name"] == "github-pages"
    for name in ("Configure Pages", "Prune duplicate Pages artifacts"):
        assert "workflow_dispatch" in _publish_step(name)["if"]
```

- [ ] **Step 2: Run, expect FAIL** — `FileNotFoundError` for `static-site-publish.yml`.

Run: `…/python -m pytest tests/unit/test_static_site_workflow.py -q -k publisher`

- [ ] **Step 3: Create `.github/workflows/static-site-publish.yml`**

```yaml
name: Static Site Publish

# Stateless publisher (#499): every run publishes the newest valid artifact per
# market, found by name, so wake-ups may be lost, duplicated or coalesced and
# the next run converges. Producers in static-site.yml wake it; no schedule.
on:
  workflow_dispatch:
    inputs:
      rs_formula_overrides:
        description: 'Optional per-market RS rollback JSON, e.g. {"HK":"legacy-linear-v1"}'
        type: string
        default: '{}'
  # Build-only rehearsal for PRs that change the publisher; never deploys.
  pull_request:
    paths:
      - .github/workflows/static-site-publish.yml
      - backend/app/scripts/download_static_market_fallbacks.py
      - backend/app/scripts/validate_static_market_artifacts.py
      - backend/app/services/static_artifact_combiner.py
      - backend/app/services/static_advertised_paths.py

concurrency:
  # One production publish at a time; a newer pending run replaces an older
  # pending one, which is safe because every run publishes "latest per market".
  # PR rehearsals get their own group so they can never displace a production run.
  group: ${{ github.event_name == 'pull_request' && format('static-site-publisher-pr-{0}', github.ref) || 'static-site-publisher' }}
  cancel-in-progress: false

permissions:
  actions: write
  contents: read
  pages: write
  id-token: write

jobs:
  build:
    if: ${{ (github.event_name == 'workflow_dispatch' && github.ref == format('refs/heads/{0}', github.event.repository.default_branch)) || (github.event_name == 'pull_request' && github.event.pull_request.head.repo.full_name == github.repository) }}
    runs-on: ubuntu-latest
    env:
      REDIS_ENABLED: "false"
      DATABASE_URL: postgresql://stockscanner:stockscanner@localhost:5432/stockscanner
      RS_FORMULA_OVERRIDES: ${{ inputs.rs_formula_overrides || '{}' }}
    steps:
      - uses: actions/checkout@v7

      - uses: actions/setup-python@v7
        with:
          python-version: "3.11"
          cache: pip
          cache-dependency-path: backend/requirements.txt

      - uses: actions/setup-node@v7
        with:
          node-version: 22
          cache: npm
          cache-dependency-path: frontend/package-lock.json

      - name: Configure Pages
        if: ${{ github.event_name == 'workflow_dispatch' }}
        uses: actions/configure-pages@v6
        with:
          enablement: true

      - name: Install backend dependencies
        run: pip install -r backend/requirements.txt

      - name: Install frontend dependencies
        run: cd frontend && npm ci

      - name: Download newest market artifacts
        env:
          GH_TOKEN: ${{ github.token }}
          REPOSITORY: ${{ github.repository }}
          CURRENT_RUN_ID: ${{ github.run_id }}
          BRANCH_NAME: ${{ github.event.repository.default_branch }}
        run: |
          mkdir -p /tmp/static-empty
          cd backend
          python -m app.scripts.download_static_market_fallbacks \
            --current-dir /tmp/static-empty \
            --fallback-dir /tmp/static-market-artifacts \
            --fallback-options-dir /tmp/static-options \
            --fallback-cot-dir /tmp/static-cot \
            --fallback-rs-formula-overrides-json "$RS_FORMULA_OVERRIDES"

      - name: Validate market artifacts
        run: |
          mkdir -p /tmp/static-market-artifacts
          cd backend
          python -m app.scripts.validate_static_market_artifacts \
            --current-dir /tmp/static-market-artifacts \
            --fallback-dir /tmp/static-empty \
            --current-options-dir /tmp/static-options \
            --current-cot-dir /tmp/static-cot \
            --selected-markets '[]'

      - name: Combine static data bundle
        run: |
          cd backend
          python -m app.scripts.export_static_site \
            --output-dir ../frontend/public/static-data \
            --combine-artifacts-dir /tmp/static-market-artifacts \
            --options-artifacts-dir /tmp/static-options \
            --cot-artifacts-dir /tmp/static-cot \
            --rs-formula-overrides-json "$RS_FORMULA_OVERRIDES"

      # Reporting only (#484): never blocks the deploy.
      - name: Report market freshness
        continue-on-error: true
        env:
          GH_TOKEN: ${{ github.token }}
        run: |
          mkdir -p /tmp/daily-price-manifests
          gh release download daily-price-data \
            --repo "${{ github.repository }}" \
            --pattern 'daily-price-latest-*.json' \
            --dir /tmp/daily-price-manifests \
            || echo "Daily price manifests unavailable; bundle freshness will be blank."
          cd backend
          python -m app.scripts.report_static_market_freshness \
            --manifest ../frontend/public/static-data/manifest.json \
            --artifacts-dir /tmp/static-market-artifacts \
            --price-manifest-dir /tmp/daily-price-manifests \
            --selected-markets '[]'

      - name: Build static frontend
        env:
          VITE_STATIC_SITE: "true"
          VITE_BASE_PATH: /${{ github.event.repository.name }}/
        run: cd frontend && npm run build

      - name: Upload Pages artifact
        id: upload-pages-artifact
        uses: actions/upload-pages-artifact@v5
        with:
          path: frontend/dist

      - name: Prune duplicate Pages artifacts
        if: ${{ github.event_name == 'workflow_dispatch' }}
        env:
          GH_TOKEN: ${{ github.token }}
          REPOSITORY: ${{ github.repository }}
          RUN_ID: ${{ github.run_id }}
          KEEP_ARTIFACT_ID: ${{ steps.upload-pages-artifact.outputs.artifact_id }}
        run: |
          <copy the existing "Prune duplicate Pages artifacts" python heredoc
           from static-site.yml combine-and-build verbatim>

  deploy:
    if: ${{ github.event_name == 'workflow_dispatch' && github.ref == format('refs/heads/{0}', github.event.repository.default_branch) }}
    needs: build
    runs-on: ubuntu-latest
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - id: deployment
        uses: actions/deploy-pages@v5
```

The prune step body is the existing heredoc in `static-site.yml` (lines under `- name: Prune duplicate Pages artifacts` in `combine-and-build`), copied byte for byte.

- [ ] **Step 4: Move the combine-job assertions to the publisher**

In `test_static_site_workflow.py`, point the combine assertions at the publisher:
- `test_static_site_workflow_publishes_and_combines_global_cot_artifact`: replace the four `/tmp/static-cot-current` / `-fallback` assertions and the `needs: [...]` assertion with `assert "--fallback-cot-dir /tmp/static-cot" in PUBLISH_WORKFLOW.read_text()` and `assert "--cot-artifacts-dir /tmp/static-cot" in PUBLISH_WORKFLOW.read_text()`.
- `test_static_site_preserves_and_publishes_us_options_history`: replace the four options-dir assertions on `combine_job` with the publisher equivalents (`--fallback-options-dir /tmp/static-options`, `--options-artifacts-dir /tmp/static-options`).
- Delete `test_static_site_combine_downloads_current_and_per_market_fallback_artifacts`, `test_static_site_reports_market_freshness_without_blocking_deploy`, `test_static_site_validation_uses_python_module_not_inline_control_plane` and the helpers `_combine_and_build_job` / `_fallback_download_step` only in Task 2 (when `combine-and-build` is removed); in Task 1 they still pass.

- [ ] **Step 5: Run, expect PASS**

Run: `…/python -m pytest tests/unit/test_static_site_workflow.py tests/unit/test_static_workflow_markets.py -q`
Also: `python -c "import yaml,sys; yaml.safe_load(open('../.github/workflows/static-site-publish.yml'))"`.

- [ ] **Step 6: Commit** — `feat(static): add a stateless static-site publisher workflow (#499)`

---

### Task 2: Producers wake the publisher; legacy combine/deploy removed

**Files:**
- Modify: `.github/workflows/static-site.yml` (select-markets `needs`, permissions, build-cot and build-market wake steps, remove `combine-and-build` + `deploy`, add `wake-publisher`)
- Test: `backend/tests/unit/test_static_site_workflow.py`

**Interfaces:** Consumes `static-site-publish.yml` from Task 1.

- [ ] **Step 1: Write the failing tests**

```python
SITE_WORKFLOW = ROOT / ".github" / "workflows" / "static-site.yml"
WAKE_COMMAND = "gh workflow run static-site-publish.yml"


def _site_workflow() -> dict:
    return yaml.safe_load(SITE_WORKFLOW.read_text(encoding="utf-8"))


def _wake_step(job: str) -> dict:
    steps = _site_workflow()["jobs"][job]["steps"]
    return next(step for step in steps if step.get("name") == "Wake static-site publisher")


def test_static_site_no_longer_combines_or_deploys() -> None:
    workflow = _site_workflow()
    assert "combine-and-build" not in workflow["jobs"]
    assert "deploy" not in workflow["jobs"]
    assert "pages" not in workflow["permissions"]
    assert "id-token" not in workflow["permissions"]
    assert "needs" not in workflow["jobs"]["select-markets"]


def test_wake_steps_only_run_on_the_default_branch() -> None:
    for job in ("build-cot", "build-market"):
        step = _wake_step(job)
        assert step["continue-on-error"] is True
        assert "default_branch" in step["if"]
        assert WAKE_COMMAND in step["run"]
        assert "--ref" in step["run"]
    wake_job = _site_workflow()["jobs"]["wake-publisher"]
    assert "always()" in wake_job["if"] and "default_branch" in wake_job["if"]
    assert set(wake_job["needs"]) >= {"build-cot", "build-market"}
    assert WAKE_COMMAND in wake_job["steps"][0]["run"]


def test_market_wake_runs_after_any_later_failure() -> None:
    steps = _site_workflow()["jobs"]["build-market"]["steps"]
    names = [step.get("name") for step in steps]
    assert names[-1] == "Wake static-site publisher"
    wake_if = steps[-1]["if"]
    assert "always()" in wake_if
    assert "steps.upload-market-artifact.outcome == 'success'" in wake_if
    upload = next(step for step in steps if step.get("name") == "Upload market artifact")
    assert upload["id"] == "upload-market-artifact"


def test_cot_wake_follows_its_upload() -> None:
    steps = _site_workflow()["jobs"]["build-cot"]["steps"]
    names = [step.get("name") for step in steps]
    assert names.index("Wake static-site publisher") > names.index(
        "Upload current global COT artifact"
    )
```

- [ ] **Step 2: Run, expect FAIL** (`combine-and-build` still present; no wake steps).

- [ ] **Step 3: Edit `static-site.yml`**

1. `select-markets`: delete the line `    needs: calendar-audit`. (Calendar audit keeps running as a reporting job; the weekly `market-calendar-audit.yml` is unchanged.)
2. `permissions`: keep `actions: write`, `contents: write`; delete `pages: write` and `id-token: write`.
3. `build-cot`: append after `Upload current global COT artifact`:

```yaml
      - name: Wake static-site publisher
        if: ${{ github.ref == format('refs/heads/{0}', github.event.repository.default_branch) }}
        continue-on-error: true
        env:
          GH_TOKEN: ${{ github.token }}
        run: gh workflow run static-site-publish.yml --repo "${{ github.repository }}" --ref "${{ github.event.repository.default_branch }}"
```

4. `build-market`: add `id: upload-market-artifact` to `Upload market artifact`, and append as the job's last step:

```yaml
      # Runs after any later step failure so an uploaded market is never left
      # waiting for the next export run.
      - name: Wake static-site publisher
        if: ${{ always() && steps.upload-market-artifact.outcome == 'success' && github.ref == format('refs/heads/{0}', github.event.repository.default_branch) }}
        continue-on-error: true
        env:
          GH_TOKEN: ${{ github.token }}
        run: gh workflow run static-site-publish.yml --repo "${{ github.repository }}" --ref "${{ github.event.repository.default_branch }}"
```

5. Delete the `combine-and-build` and `deploy` jobs; add:

```yaml
  # Catch-all wake-up: covers a market that uploaded and was cancelled before
  # its own wake step. Duplicate wake-ups coalesce in the publisher.
  wake-publisher:
    if: ${{ always() && github.ref == format('refs/heads/{0}', github.event.repository.default_branch) }}
    needs: [select-markets, build-cot, build-market]
    runs-on: ubuntu-latest
    steps:
      - name: Wake static-site publisher
        env:
          GH_TOKEN: ${{ github.token }}
        run: gh workflow run static-site-publish.yml --repo "${{ github.repository }}" --ref "${{ github.event.repository.default_branch }}"
```

6. Update the `select-markets` comment ("The combine job back-fills…") to say the publisher selects the newest artifact per market.

- [ ] **Step 4: Update helpers and old tests**

- `_build_market_job()`: end marker `"\n  combine-and-build:"` → `"\n  wake-publisher:"`.
- The two `upload_market_step` splits ending at `"\n\n  combine-and-build:"` → `"\n\n  wake-publisher:"`.
- Delete `_combine_and_build_job`, `_fallback_download_step`, `test_static_site_combine_downloads_current_and_per_market_fallback_artifacts`, `test_static_site_reports_market_freshness_without_blocking_deploy`, `test_static_site_validation_uses_python_module_not_inline_control_plane` (their behaviour is pinned on the publisher by Task 1 tests).
- `test_static_site_workflow_publishes_and_combines_global_cot_artifact`: drop the `needs: [select-markets, build-cot, build-market]` assertion if Task 1 left it.

- [ ] **Step 5: Run, expect PASS**

Run: `…/python -m pytest tests/unit/test_static_site_workflow.py tests/unit/test_static_workflow_markets.py -q` and the YAML load check for both workflows.

- [ ] **Step 6: Commit** — `feat(static): producers wake the publisher; drop the matrix-wide combine barrier (#499)`

---

### Task 3: Verification and rollout evidence

- [ ] Broad backend static suite: `…/python -m pytest tests/unit -q -p no:warnings -k "static or combiner or cot or options_artifact or freshness or workflow"`.
- [ ] `actionlint` if available (`/opt/homebrew/bin/actionlint .github/workflows/static-site.yml .github/workflows/static-site-publish.yml`); otherwise the YAML load checks above.
- [ ] Strict report-only reviewer agent on `git diff origin/main` (project practice), fix confirmed findings test-first.
- [ ] Push and open the PR: the `pull_request` trigger runs the publisher build-only; confirm in its log that every available market downloads (in-progress runs included if one is running), validation and combine succeed, the build succeeds, and `deploy` is skipped.
- [ ] After merge: watch the first scheduled export run; confirm a publisher run per uploaded market, coalescing of overlapping wake-ups, and the live `manifest.json` advancing per market before the slowest market finishes.
