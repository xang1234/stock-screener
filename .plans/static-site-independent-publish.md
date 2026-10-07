# Static site: independent market publication (#499, revised)

Issue: #499. Parent: #497. Builds on #498 (PR #540: by-name artifact lookup,
`unavailable_markets`, per-market `publication`, advertised-path validation).

## Goal

A finished market reaches the live site without waiting for slower markets in
the same run, and the system converges to "newest valid artifact per market is
live" despite GitHub Actions cancelling, delaying or coalescing runs.

## Design: stateless publisher, wake-ups only

The issue's candidate/request/catalog/CAS machinery is dropped (it depended on
#498's dropped durable store). The uploaded Actions artifacts are the record of
pending work: the newest artifact per market is discoverable by name (#540).
Every publisher run computes the same answer from the same artifacts, so any
run may be cancelled, duplicated or coalesced and the next run repairs it.

### Producers (`.github/workflows/static-site.yml`)

- `select-markets`, `ensure_daily_price_release`, `build-cot`, `build-market`
  and the `market_group` dispatch input (`all`/`asia`/`china`/`us`) keep their
  behaviour.
- `calendar-audit` no longer gates `select-markets` (drop
  `needs: calendar-audit`); it stays as a reporting job. The weekly
  `market-calendar-audit.yml` is unchanged. A market's export still refuses an
  invalid session for that market (existing exit 78/79 handling).
- **Wake-up:** after `Upload market artifact` succeeds, a step runs
  `gh workflow run static-site-publish.yml --ref <default branch>`
  (`continue-on-error: true`; a failed wake-up is logged, never fails the job).
  `build-cot` does the same after uploading `static-cot-global`.
- **Final wake-up:** a new job `wake-publisher` (`needs` every producer job,
  `if: always()`) dispatches the publisher once more, catching a market that
  uploaded and was cancelled before its own wake-up.
- `combine-and-build` and `deploy` are removed from this workflow.
- Workflow `concurrency` (`static-site-${ref}`, `queue: max`) is unchanged; it
  now orders exports only.
- Wake-ups only on default-branch runs (same `if` as today's jobs): a
  dispatched producer on another branch uploads artifacts the publisher
  ignores (#540 trust filter) and does not wake it.

### Publisher (new `.github/workflows/static-site-publish.yml`)

- Triggers: `workflow_dispatch` only. No schedule. (`workflow_dispatch` created
  with `GITHUB_TOKEN` does start a run; it is the documented exception to the
  no-recursion rule.)
- `concurrency: { group: static-site-publisher, cancel-in-progress: false }`:
  one run at a time; a newer pending run replaces an older pending one, which
  is safe because every run publishes "latest per market".
- Permissions: `actions: write` (prune Pages artifacts), `contents: read`,
  `pages: write`, `id-token: write`.
- Job `build` (the moved `combine-and-build` steps):
  1. checkout, setup Python/Node, install deps, `configure-pages`.
  2. `download_static_market_fallbacks` into one selection directory
     (markets, `static-options-US`, `static-cot-global`), `CURRENT_RUN_ID` =
     this run, `BRANCH_NAME` = default branch.
  3. `validate_static_market_artifacts` with `--selected-markets '[]'` (no
     producer statuses in this run; US stays required).
  4. `export_static_site` combine with the selection directory as
     `--combine-artifacts-dir` (no separate fallback dir) and the options/COT
     selections as their directories.
  5. Freshness report (non-blocking, as today), `npm run build`,
     `upload-pages-artifact`, prune duplicate Pages artifacts.
- Job `deploy`: `deploy-pages`, `environment: github-pages`, only when the run
  is on the default branch. On another branch the publisher rehearses
  everything except the deploy.
- `rs_formula_overrides` dispatch input passes through to download/combine as
  today (default `{}`).

### Artifacts from in-progress runs

upload-artifact v4 artifacts are finalized at upload and served by the REST API
while their run continues, so HK publishes while CN is still exporting. The
downloader uses `gh run download <run_id> --name <artifact>`; if that refuses an
in-progress run, switch `_download_candidate` to
`gh api repos/{repo}/actions/artifacts/{id}/zip` + unzip (the artifact ID is
already in the by-name listing). Verify on the first branch rehearsal while a
producer run is in progress.

### Publication semantics

With one selection directory every served market is labelled
`publication.source: "current"`; freshness is carried by `publication.state`
(`current`/`stale`/`unknown`, judged against the market's last completed
session). The freshness report has no producer statuses in the publisher run,
so its Reason column is empty; per-market reasons stay in each producer run's
status/diagnostics artifacts.

## Correctness under unreliable Actions

| Event | Outcome |
|---|---|
| Market export cancelled before upload | That market keeps its last artifact (`stale`); others unaffected |
| Cancelled after upload, before its wake-up | Next wake-up from any market, or the run's final `wake-publisher`, publishes it; if the whole run died, the next export run's wake-ups do |
| Publisher cancelled / build or deploy fails | Live site keeps its last deploy (Pages deploy is atomic); next wake-up republishes without re-exporting |
| Late or out-of-order producer | Selection is by session date (strictly newer wins; same session: newest upload) and publishes are serialized, so older data cannot replace newer |
| Duplicate / coalesced wake-ups | Idempotent |
| No run starts for hours | Out of scope here; #503's external watchdog triggers producers/publisher via the API |

Accepted cost: a stranded artifact waits for the next export run's wake-ups
when every wake-up of its own run was lost (no cron backstop, by choice).

## Mid-session behaviour

The publisher reads no prices. Market exports keep their existing session
handling; `publication.state` is judged against `last_completed_trading_day`
at publish time, so a publish during a session judges against the previous
close, and a market whose session closes between two publishes turns `stale`
on the later one until its export lands.

## Tests

`backend/tests/unit/test_static_site_workflow.py` (YAML structure tests, as
existing ones do):
- `static-site.yml` has no `combine-and-build`/`deploy` jobs; `select-markets`
  does not need `calendar-audit`.
- each producer upload is followed by a non-fatal publisher wake-up on the
  default branch; `wake-publisher` needs all producers and runs `always()`.
- `static-site-publish.yml`: `workflow_dispatch` only (no `schedule`), group
  `static-site-publisher` without cancel-in-progress, downloads markets/options/
  COT by name into one directory, validates with no selected markets, combines
  without a fallback dir, deploys only on the default branch.
- Existing combine/download/validator/freshness tests stay green.

Rollout: dispatch the publisher on the feature branch (no deploy) while a
producer run is in progress to verify in-progress artifact download and a full
build; merge; watch the first scheduled export run publish per market.
Rollback: revert the PR (restores `combine-and-build`).

## Out of scope

Durable requests, catalog, CAS, `recovery_id`, cron backstop, build reuse
(#504), overdue-market recovery (#503).
