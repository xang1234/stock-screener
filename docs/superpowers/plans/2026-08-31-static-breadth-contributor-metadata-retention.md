# Static Breadth Contributor Metadata Retention Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish company names and stable IBD groups for all latest-20 static breadth contributor sessions, including cold-start builds, without changing breadth formulas or live-app behavior.

**Architecture:** A compact per-market rolling metadata bundle is restored before each static market build. After the feature snapshot and IBD enrichment finish, a static-only finalizer merges restored date/symbol metadata with current reference metadata for new entries, updates only contributor metadata columns, applies a nonblank coverage gate, and writes the next rolling bundle.

**Tech Stack:** Python 3.11, SQLAlchemy, Pydantic v2, pytest, GitHub Actions, GitHub Releases, gzip JSON.

**Spec:** `docs/superpowers/specs/2026-08-31-static-breadth-contributor-metadata-retention-design.md`

## Global Constraints

- Keep `calculation_revision = 3`; do not change any breadth formula, aggregate, eligibility rule, qualifying value, or contributor calculation signature.
- Retain and advertise at most the latest 20 completed contributor sessions.
- Restored metadata for an exact date/symbol wins over current metadata and remains frozen.
- Current metadata is used only when a retained date/symbol has no restored value.
- The finalizer may update only `company_name` and `ibd_industry_group`.
- A non-empty bundle with zero company names or zero classified IBD groups must not be published.
- Restore or state-publication failure suppresses the current market artifact so combine uses the last-known-good fallback.
- Do not alter live breadth calculation, API, frontend, or database schema.
- Preserve unrelated working-tree files.

---

### Task 1: Define and validate the rolling metadata contract

**Files:**
- Create: `backend/app/services/static_breadth_contributor_metadata_contract.py`
- Create: `backend/tests/unit/test_static_breadth_contributor_metadata_contract.py`

**Interfaces:**
- Produces: `STATIC_BREADTH_CONTRIBUTOR_METADATA_SCHEMA_VERSION = "static-breadth-contributor-metadata-v1"`.
- Produces: `STATIC_BREADTH_CONTRIBUTOR_METADATA_RETENTION_DATES = 20`.
- Produces: `FrozenBreadthContributorMetadata`, `FrozenBreadthContributorSession`, and `StaticBreadthContributorMetadataState`.
- Produces: `StaticBreadthContributorMetadataPlan` and `build_static_breadth_contributor_metadata_plan(*, market: str, directory: Path, market_catalog: MarketCatalog | None = None) -> StaticBreadthContributorMetadataPlan`.
- Produces: `read_static_breadth_contributor_metadata(path, expected_market)` and `write_static_breadth_contributor_metadata(path, state)`.

- [ ] **Step 1: Write the failing contract tests**

Cover deterministic market/path planning, gzip round-trip, newest-first unique dates, the 20-date limit, sorted unique normalized symbols, blank-group normalization, wrong-market rejection, unsupported schemas, and corrupt gzip. Include:

```python
def test_metadata_state_round_trips_as_deterministic_gzip(tmp_path):
    state = StaticBreadthContributorMetadataState(
        schema_version=STATIC_BREADTH_CONTRIBUTOR_METADATA_SCHEMA_VERSION,
        market="US",
        generated_at=datetime(2026, 8, 31, 4, tzinfo=UTC),
        sessions=(
            FrozenBreadthContributorSession(
                date=date(2026, 8, 28),
                contributors=(
                    FrozenBreadthContributorMetadata(
                        symbol="BTAI",
                        company_name="BioXcel Therapeutics Inc",
                        ibd_industry_group="Medical-Biomed/Biotech",
                    ),
                ),
            ),
        ),
    )
    path = tmp_path / "breadth-contributor-metadata-us.json.gz"

    write_static_breadth_contributor_metadata(path, state)
    restored = read_static_breadth_contributor_metadata(path, expected_market="US")

    assert restored == state
    assert path.read_bytes()[:2] == b"\x1f\x8b"
```

- [ ] **Step 2: Run the contract tests and verify RED**

Run: `cd backend && ./venv/bin/pytest tests/unit/test_static_breadth_contributor_metadata_contract.py -q`

Expected: FAIL during collection because the contract module does not exist.

- [ ] **Step 3: Implement the contract and gzip helpers**

Use frozen, extra-forbid Pydantic models. Normalize markets and symbols to uppercase, trim text, normalize blank groups to `No Group`, require sessions newest-first and unique, require contributors sorted by unique symbol, and cap sessions at 20. Serialize canonical sorted-key compact JSON with gzip `mtime=0`.

The plan builder returns:

```python
StaticBreadthContributorMetadataPlan(
    enabled=catalog.get(normalized_market).capabilities.breadth,
    market=normalized_market,
    asset_name=f"breadth-contributor-metadata-{normalized_market.lower()}.json.gz",
    source_path=directory / asset_name,
    output_path=directory / "current" / asset_name,
)
```

- [ ] **Step 4: Run the contract tests and verify GREEN**

Run: `cd backend && ./venv/bin/pytest tests/unit/test_static_breadth_contributor_metadata_contract.py -q`

Expected: PASS with no warnings.

- [ ] **Step 5: Commit the contract**

```bash
git add backend/app/services/static_breadth_contributor_metadata_contract.py backend/tests/unit/test_static_breadth_contributor_metadata_contract.py
git commit -m "feat: define static breadth metadata state"
```

### Task 2: Restore rolling metadata safely from GitHub Releases

**Files:**
- Create: `backend/app/services/static_breadth_contributor_metadata_release.py`
- Create: `backend/app/scripts/describe_static_breadth_contributor_metadata.py`
- Create: `backend/app/scripts/restore_static_breadth_contributor_metadata.py`
- Create: `backend/tests/unit/test_static_breadth_contributor_metadata_release.py`

**Interfaces:**
- Consumes Task 1's plan and contract.
- Produces: `StaticBreadthContributorMetadataRestoreStatus` values `RESTORED`, `MISSING`, and `FAILED`.
- Produces: `StaticBreadthContributorMetadataRestoreResult(status, asset_name, output_path, detail=None)` with `safe_to_publish` and `as_dict()`.
- Produces: `StaticBreadthContributorMetadataReleaseRestorer.restore(*, repository_full_name, release_tag, asset_name, output_path, github_token, request_timeout_seconds, attempts=3, retry_delay_seconds=5)`.
- Produces description and restore CLIs used by GitHub Actions.

- [ ] **Step 1: Write failing restore and CLI tests**

Mirror the RRG release tests and require:

```python
@pytest.mark.parametrize(
    ("status", "exit_code", "safe"),
    [
        (StaticBreadthContributorMetadataRestoreStatus.RESTORED, 0, True),
        (StaticBreadthContributorMetadataRestoreStatus.MISSING, 0, True),
        (StaticBreadthContributorMetadataRestoreStatus.FAILED, 1, False),
    ],
)
def test_restore_cli_reports_publication_safety(
    tmp_path, capsys, status, exit_code, safe
):
    output_path = tmp_path / "breadth-contributor-metadata-us.json.gz"
    fake_restorer = SimpleNamespace(
        restore=lambda **_kwargs: StaticBreadthContributorMetadataRestoreResult(
            status=status,
            asset_name=output_path.name,
            output_path=output_path,
            detail="fixture",
        )
    )
    argv = ["--asset-name", output_path.name, "--output-path", str(output_path)]
    assert main(argv, restorer=fake_restorer) == exit_code
    assert json.loads(capsys.readouterr().out)["safe_to_publish"] is safe
```

Also prove one retryable network failure followed by success retries once, while a missing asset is not retried.

- [ ] **Step 2: Run the release tests and verify RED**

Run: `cd backend && ./venv/bin/pytest tests/unit/test_static_breadth_contributor_metadata_release.py -q`

Expected: FAIL during collection because the service and CLIs do not exist.

- [ ] **Step 3: Implement the release restorer and CLIs**

Use `GitHubReleaseSyncService.fetch_named_asset` and `retry_github_operation`, following the RRG restorer. The restore CLI defaults to release tag `breadth-contributor-metadata-data`, prints compact JSON, and returns nonzero only for `FAILED`. The description CLI prints Task 1's plan.

- [ ] **Step 4: Run release and contract tests and verify GREEN**

Run: `cd backend && ./venv/bin/pytest tests/unit/test_static_breadth_contributor_metadata_contract.py tests/unit/test_static_breadth_contributor_metadata_release.py -q`

Expected: PASS.

- [ ] **Step 5: Commit release restoration**

```bash
git add backend/app/services/static_breadth_contributor_metadata_release.py backend/app/scripts/describe_static_breadth_contributor_metadata.py backend/app/scripts/restore_static_breadth_contributor_metadata.py backend/tests/unit/test_static_breadth_contributor_metadata_release.py
git commit -m "feat: restore static breadth metadata state"
```

### Task 3: Finalize persisted contributor metadata after enrichment

**Files:**
- Create: `backend/app/services/static_breadth_contributor_metadata_finalizer.py`
- Create: `backend/tests/unit/test_static_breadth_contributor_metadata_finalizer.py`

**Interfaces:**
- Consumes Task 1's state reader/writer and `BreadthContributorMetadataLoader.current(db: Session, market: str, symbols: Sequence[str]) -> Mapping[str, BreadthContributorMetadata]`.
- Produces: `StaticBreadthContributorMetadataCoverageError`.
- Produces: immutable `StaticBreadthContributorMetadataFinalizationReport` with `market`, `retained_dates`, `contributors`, `restored`, `bootstrapped`, `named`, `classified`, and `source_status`.
- Produces: `StaticBreadthContributorMetadataFinalizer(db).finalize(market, source_path, output_path, source_status, limit=20)`.

- [ ] **Step 1: Write failing finalizer tests**

Use SQLite with `MarketBreadth`, `MarketBreadthContributorSnapshot`, and `MarketBreadthContributor`. Inject a metadata-loader callable. Cover:

1. Restored exact-date/symbol metadata wins over a changed current classification.
2. New dates and newly qualifying symbols bootstrap from current metadata.
3. Only the newest 20 snapshots are serialized.
4. Blank current groups become `No Group`.
5. Aggregates, signals, daily changes, snapshot identity, and calculation signature stay identical.
6. Non-empty zero-name data raises and rolls back.
7. Non-empty zero-real-group data raises and rolls back.
8. Empty contributors serialize without tripping coverage.

Core assertion:

```python
report = finalizer.finalize(
    market="US",
    source_path=restored_path,
    output_path=output_path,
    source_status="restored",
)
db.expire_all()
row = db.query(MarketBreadthContributor).filter_by(symbol="AAA").one()
assert row.company_name == "Alpha Old"
assert row.ibd_industry_group == "Old Group"
assert row.signals_json == original_signals
assert report.restored == 1
```

- [ ] **Step 2: Run the finalizer tests and verify RED**

Run: `cd backend && ./venv/bin/pytest tests/unit/test_static_breadth_contributor_metadata_finalizer.py -q`

Expected: FAIL during collection because the finalizer does not exist.

- [ ] **Step 3: Implement the minimal finalizer**

Query newest snapshots with `joinedload(MarketBreadthContributorSnapshot.contributors)`, limit to 20, and load current metadata once for the symbol union. Read prior state only for `source_status == "restored"`. Select metadata with:

```python
frozen = restored_by_date_symbol.get((snapshot.date, contributor.symbol))
current = current_by_symbol.get(contributor.symbol, BreadthContributorMetadata())
selected = frozen or current
contributor.company_name = selected.company_name
contributor.ibd_industry_group = selected.ibd_industry_group or NO_GROUP_LABEL
```

Flush, count coverage, raise and roll back when non-empty data has `named == 0` or `classified == 0`, commit once, and serialize newest-first state. Never assign another model field.

- [ ] **Step 4: Run finalizer tests and verify GREEN**

Run: `cd backend && ./venv/bin/pytest tests/unit/test_static_breadth_contributor_metadata_finalizer.py -q`

Expected: PASS.

- [ ] **Step 5: Run adjacent persistence and metadata tests**

Run: `cd backend && ./venv/bin/pytest tests/unit/test_breadth_contributor_metadata.py tests/unit/test_breadth_contributor_backfill.py tests/unit/test_breadth_persistence.py tests/unit/test_static_breadth_contributor_exporter.py -q`

Expected: PASS, proving live historical no-look-ahead and contributor reconciliation remain intact.

- [ ] **Step 6: Commit the finalizer**

```bash
git add backend/app/services/static_breadth_contributor_metadata_finalizer.py backend/tests/unit/test_static_breadth_contributor_metadata_finalizer.py
git commit -m "fix: finalize static breadth contributor metadata"
```

### Task 4: Integrate finalization into static refresh ordering

**Files:**
- Modify: `backend/app/scripts/export_static_site.py`
- Modify: `backend/tests/unit/test_export_static_site_script.py`
- Modify: `backend/tests/unit/test_export_static_site_refresh.py`

**Interfaces:**
- Consumes Tasks 1 and 3.
- Extends `_run_daily_refresh` with keyword-only parameters `breadth_contributor_metadata_plan: StaticBreadthContributorMetadataPlan | None = None` and `breadth_contributor_metadata_restore_status: str = "missing"`.
- Adds result: `results["breadth_contributor_metadata"][market]`.
- Adds CLI: `--breadth-contributor-metadata-dir`, valid only with `--market`.

- [ ] **Step 1: Write a failing orchestration-order regression test**

Extend the existing event recorder with a fake finalizer and require:

```python
assert events == [
    "universe_refresh",
    "fundamentals_refresh",
    "feature_snapshot",
    "group_rank:US",
    "enrich:77",
    "finalize_contributors:US",
]
assert results["breadth_contributor_metadata"]["US"]["classified"] == 3
```

Add tests proving finalization is skipped for an unpublishable feature snapshot, a non-breadth market, or no plan. Add CLI validation proving the directory requires one `--market`.

- [ ] **Step 2: Run targeted orchestration tests and verify RED**

Run: `cd backend && ./venv/bin/pytest tests/unit/test_export_static_site_script.py -q -k 'breadth_contributor_metadata or builds_snapshot_before_group_rank'`

Expected: FAIL because the refresh driver does not accept or run the finalizer.

- [ ] **Step 3: Wire the finalizer after IBD enrichment**

Build the plan in `main()` when the directory is supplied. Pass it and `BREADTH_CONTRIBUTOR_METADATA_RESTORE_STATUS` into `_run_daily_refresh`. After `ibd_metadata_refresh`, finalize each publishable breadth market with a fresh `SessionLocal()` and record `report.as_dict()`.

A failed restore raises before current output. An unpublishable feature snapshot records a structured skip and writes no state. Coverage/contract errors propagate and quarantine current output.

- [ ] **Step 4: Run orchestration tests and verify GREEN**

Run: `cd backend && ./venv/bin/pytest tests/unit/test_export_static_site_script.py tests/unit/test_export_static_site_refresh.py -q`

Expected: PASS.

- [ ] **Step 5: Run static contributor regressions**

Run: `cd backend && ./venv/bin/pytest tests/unit/test_static_site_export_service.py tests/unit/test_static_breadth_contributor_exporter.py tests/unit/test_export_static_site_no_current_artifacts.py -q`

Expected: PASS.

- [ ] **Step 6: Commit refresh integration**

```bash
git add backend/app/scripts/export_static_site.py backend/tests/unit/test_export_static_site_script.py backend/tests/unit/test_export_static_site_refresh.py
git commit -m "fix: finalize breadth metadata after static enrichment"
```

### Task 5: Add GitHub Actions restore and publication lifecycle

**Files:**
- Modify: `.github/workflows/static-site.yml`
- Modify: `backend/tests/unit/test_static_workflow_markets.py`
- Modify: `docs/STATIC_SITE.md`

**Interfaces:**
- Consumes Tasks 2 and 4.
- Produces release: `breadth-contributor-metadata-data`.
- Publishes: `breadth-contributor-metadata-<market>.json.gz` plus its durable
  `breadth-contributor-metadata-<market>.previous.json.gz` recovery asset.

- [ ] **Step 1: Write failing workflow contract tests**

Require these literal properties:

```python
assert "gh release view breadth-contributor-metadata-data" in content
assert "app.scripts.describe_static_breadth_contributor_metadata" in content
assert "app.scripts.restore_static_breadth_contributor_metadata" in content
assert "--breadth-contributor-metadata-dir" in content
assert "BREADTH_CONTRIBUTOR_METADATA_RESTORE_STATUS" in content
assert "steps.restore-breadth-contributor-metadata.outputs.safe_to_publish == 'true'" in content
assert "gh release upload breadth-contributor-metadata-data" in content
```

Also require restore before export, state publication after successful export,
and state publication before market artifact upload.

- [ ] **Step 2: Run workflow tests and verify RED**

Run: `cd backend && ./venv/bin/pytest tests/unit/test_static_workflow_markets.py -q`

Expected: FAIL because no metadata lifecycle exists.

- [ ] **Step 3: Implement the workflow lifecycle**

Create the release in `ensure_daily_price_release`. Each market job plans paths,
restores and validates its canonical asset (falling back to `.previous` only
when canonical is absent), passes directory/status to export, and suppresses
current market publication on failed restore. After a successful candidate
export, first preserve a restored canonical asset as `.previous`, then upload
finalized canonical state with `--clobber` and three bounded retries. Never
attempt canonical replacement when preserving `.previous` fails. Record a

`metadata_state_published` step output and run `actions/upload-artifact` for the
market only when it is `true`. A failed state upload must exit the publication
step successfully with `metadata_state_published=false` so the combine job uses
the prior fallback without marking the whole matrix job failed.

- [ ] **Step 4: Document cold-start and rolling behavior**

Document that the first build bootstraps all retained dates from current reference metadata, later builds preserve exact date/symbol values, only new entries adopt current metadata, failed restore preserves fallback, and live is unaffected.

- [ ] **Step 5: Run workflow and publish-policy tests**

Run: `cd backend && ./venv/bin/pytest tests/unit/test_static_workflow_markets.py tests/unit/test_export_static_site_script.py tests/unit/test_static_market_publish_policy.py -q`

Expected: PASS.

- [ ] **Step 6: Commit workflow and documentation**

```bash
git add .github/workflows/static-site.yml backend/tests/unit/test_static_workflow_markets.py docs/STATIC_SITE.md
git commit -m "ci: retain static breadth contributor metadata"
```

### Task 6: Verify the complete change

**Files:**
- Modify only when verification exposes an in-scope defect.

**Interfaces:**
- Consumes all previous tasks.
- Produces fresh verification evidence and a clean branch diff.

- [ ] **Step 1: Run the focused backend suite**

```bash
cd backend
./venv/bin/pytest \
  tests/unit/test_static_breadth_contributor_metadata_contract.py \
  tests/unit/test_static_breadth_contributor_metadata_release.py \
  tests/unit/test_static_breadth_contributor_metadata_finalizer.py \
  tests/unit/test_export_static_site_script.py \
  tests/unit/test_export_static_site_refresh.py \
  tests/unit/test_static_breadth_contributor_exporter.py \
  tests/unit/test_static_site_export_service.py \
  tests/unit/test_static_workflow_markets.py -q
```

Expected: PASS with zero failures.

- [ ] **Step 2: Run the full backend unit suite**

Run: `cd backend && ./venv/bin/pytest tests/unit/ -q`

Expected: PASS with zero failures.

- [ ] **Step 3: Run frontend regressions and production build**

```bash
cd frontend
npm run test:run
npm run lint
npm run build
```

Expected: all commands exit 0.

- [ ] **Step 4: Validate repository state and diff**

```bash
git diff --check origin/main...HEAD
git status --short
git log --oneline origin/main..HEAD
```

Expected: no whitespace errors; only planned tracked files plus the user's pre-existing unrelated untracked files.

- [ ] **Step 5: Commit verification-only corrections separately**

Only if Steps 1-4 exposed and you repaired an in-scope defect, stage the exact planned files that changed:

```bash
git add backend/app/services/static_breadth_contributor_metadata_contract.py backend/app/services/static_breadth_contributor_metadata_release.py backend/app/services/static_breadth_contributor_metadata_finalizer.py backend/app/scripts/describe_static_breadth_contributor_metadata.py backend/app/scripts/restore_static_breadth_contributor_metadata.py backend/app/scripts/export_static_site.py .github/workflows/static-site.yml docs/STATIC_SITE.md backend/tests/unit/test_static_breadth_contributor_metadata_contract.py backend/tests/unit/test_static_breadth_contributor_metadata_release.py backend/tests/unit/test_static_breadth_contributor_metadata_finalizer.py backend/tests/unit/test_export_static_site_script.py backend/tests/unit/test_export_static_site_refresh.py backend/tests/unit/test_static_workflow_markets.py
git commit -m "fix: address static breadth metadata verification"
```
