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
  4. `export_static_site` combine with an empty `--combine-artifacts-dir`
     and the selection as `--fallback-artifacts-dir` (options/COT likewise as
     fallback dirs). Stored artifacts are last-good inputs, so they get the
     fallback policy: an artifact that predates the balanced RS rollout or the
     current breadth revision is accepted or skipped instead of failing every
     market's publish; only an explicit `rs_formula_overrides` input constrains
     it. (Review finding: passing them as current applied the full balanced
     policy, so an RS rollback would have blocked every publish.)
  5. Freshness report (non-blocking, as today), `npm run build`,
     `upload-pages-artifact`, prune duplicate Pages artifacts.
- Job `deploy`: `deploy-pages`, `environment: github-pages`, only for a
  `workflow_dispatch` on the default branch. A dispatch on another branch, or
  a same-repo PR touching the publisher's files, rehearses everything except
  Configure Pages, pruning and the deploy, in a per-ref concurrency group so it
  can never replace a pending production publish.
- `rs_formula_overrides` dispatch input passes through to download/combine as
  today (default `{}`).

### Artifacts from in-progress runs

upload-artifact v4 artifacts are finalized at upload and served by the REST API
while their run continues, so HK publishes while CN is still exporting. The
downloader uses `gh run download <run_id> --name <artifact>`, which lists the
run's artifacts without checking the run's status (cli/cli
`pkg/cmd/run/download/download.go`), so no fallback download path is needed.

### Publication semantics

Every served market is labelled `publication.source: "fallback"` (it comes
from the artifact store, not from this run's export), and the combiner skips
its per-market "reused" warning when there are no current artifacts; freshness is carried by `publication.state`
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
| Late or out-of-order producer | Selection is by session date (strictly newer wins; same session: newest upload) and publishes are serialized, so a late producer cannot replace newer data |
| Transient download/lookup failure in a publish | The downloader moves to the next candidate, so that publish can show an older session for one market (or omit an optional one); the next publish restores it. Accepted for now |
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
- `static-site-publish.yml`: `workflow_dispatch` + same-repo PRs (no
  `schedule`); the exact concurrency expression (production group only for a
  default-branch dispatch); downloads markets/options/COT by name into one
  selection; validates and combines it as fallbacks; deploys only on a
  default-branch dispatch.
- An end-to-end test runs `export_static_site.main()` with the combine
  arguments parsed from the workflow, over a balanced US and a legacy HK
  artifact.
- Existing combine/download/validator/freshness tests stay green.

Rollout: the PR's build-only rehearsal verifies download/validate/combine/
build against main's artifacts; merge while no `static-site.yml` run is queued
or running (queued runs keep the old definition and would deploy alongside the
publisher); watch the first scheduled export run publish per market.
Rollback: revert the PR (restores `combine-and-build`).

## Out of scope

Durable requests, catalog, CAS, `recovery_id`, cron backstop, build reuse
(#504), overdue-market recovery (#503).
