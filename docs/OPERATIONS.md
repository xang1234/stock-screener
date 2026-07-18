# Operations Guide

This guide covers live runtime operations for the server-backed app: starting and stopping the stack, first-run bootstrap, resetting to a clean bootstrap, enabled-market workers, runtime activity, job controls, telemetry, scheduled tasks, and common recovery paths.

## Runtime Model

The live app stores runtime choices in PostgreSQL and runs market work through Redis/Celery queues.

- `runtime.primary_market` controls the market opened first and used for startup defaults.
- `runtime.enabled_markets` controls which markets the app should hydrate and expose.
- `runtime.bootstrap_state` moves from `not_started` to `running` to `ready`.
- Runtime activity rows report per-market lifecycle, stage, progress, owner task, and warning/failure state.

The UI surfaces this through the header status chip and `/operations`.

## Starting the Stack

All start/stop goes through `scripts/docker-compose-enabled-markets.sh` — a thin wrapper around `docker compose` that derives the right Compose **profiles** from `ENABLED_MARKETS` and forwards every other argument unchanged.

**Prerequisites:** Docker (with Compose v2), Python 3.11+ on `PATH` (the wrapper uses it to compute profiles), and a `.env` (or `.env.docker`) holding at least `SERVER_AUTH_PASSWORD` and `ENABLED_MARKETS`.

```bash
# Start (detached), enabling US + four Asian markets
ENABLED_MARKETS=US,HK,JP,TW,KR scripts/docker-compose-enabled-markets.sh up -d

# Foreground (watch logs); Ctrl-C stops
ENABLED_MARKETS=US,HK,JP,TW,KR scripts/docker-compose-enabled-markets.sh up

# Stop and remove containers
ENABLED_MARKETS=US,HK,JP,TW,KR scripts/docker-compose-enabled-markets.sh down
```

Open the app at the deployment URL and sign in with `SERVER_AUTH_PASSWORD`. The wrapper prints the resolved `ENABLED_MARKETS` and `COMPOSE_PROFILES` before running.

### Options

| Capability | How |
|------------|-----|
| Select market workers | `ENABLED_MARKETS=US,HK,JP,TW,KR` as an env var, or set it in `.env` / `.env.docker`. Default: `US`. |
| Any Compose subcommand | Forwarded verbatim: `up`, `up -d`, `down`, `pull`, `ps`, `logs -f`, `restart <service>`. |
| Compose overlays | Append `-f docker-compose.yml -f docker-compose.prod.yml …` for prod/HTTPS/release layers. |
| Env file | Auto-uses `.env`, then `.env.docker`; override with `--env-file <path>`. |
| Extra profiles | `COMPOSE_PROFILES=…` is merged with the derived market profiles. |
| Python interpreter | Pin with `STOCKSCREEN_PYTHON=/path/to/python3.11+` if the default isn't 3.11+. |

### Selecting enabled markets

The base Compose file defines workers for every supported market, but market-specific workers sit behind Compose profiles. The wrapper starts only the profiles required by `ENABLED_MARKETS`.

For `ENABLED_MARKETS=US,HK,JP,TW,KR`, Docker starts the US/HK/JP/TW/KR market-job and user-scan workers; CN/IN/DE/CA/SG/MY/AU containers are not created, and the global data-fetch worker listens only to those markets' `data_fetch_*` queues.

Keep the first-run wizard's enabled markets within the deployment `ENABLED_MARKETS` set. To add a market later, update `ENABLED_MARKETS` and recreate:

```bash
ENABLED_MARKETS=US,HK,JP,TW,KR,CN scripts/docker-compose-enabled-markets.sh up -d
```

> **Note:** `down` always tears down **all** market profiles (not just the enabled ones) and auto-adds `--remove-orphans`, so it fully stops the stack regardless of the `ENABLED_MARKETS` value on that line.

## First-Run Bootstrap

On a fresh (empty) database the app opens to the first-run wizard. Choose a primary market and any secondary markets to hydrate in the background. The workspace opens when the primary market is ready; secondary markets continue on their own queues.

| | |
|---|---|
| ![First-run primary-market picker](screenshots/bootstrap-setup.jpg) | ![Staged bootstrap progress](screenshots/bootstrap-progress.jpg) |
| *Primary-market picker* | *Staged hydration progress* |

Bootstrap stages:

1. **Universe refresh** — seeds the market symbol list. US uses S&P 500 / Russell / NDX via `refresh_stock_universe`; HK / IN / JP / KR / TW / CN / CA / DE / SG / MY / AU use official exchange feeds via `refresh_official_market_universe`.
2. **Benchmark + price refresh** — imports the GitHub daily price bundle first, accepts recent stale bundles during bootstrap, then live-fetches missing/current-session gaps (`7d` top-up for stale symbols, `2y` for no-history symbols). Under `live_only` this stage skips the bundle and fetches live — see [Market Data Source Mode](#market-data-source-mode).
3. **Fundamentals refresh** — loads quarterly and annual financials.
4. **Breadth calculation** — computes StockBee-style advance/decline data with gap-fill.
5. **Group rankings** — computes IBD-style relative strength group ranks.
6. **Feature snapshot** — US-only daily feature rollup for the Setup Engine.
7. **Initial autoscan** — publishes the first default-profile scan.

Selecting many enabled markets multiplies this work. On smaller hosts, start with one primary market and add markets after the workspace is ready.

## Market Data Source Mode

`MARKET_DATA_SOURCE_MODE` controls **where market data is sourced from**. It is a Pydantic setting read once at process startup by every app and worker container (from `.env` / `.env.docker`); there is no runtime toggle.

| Value | Behavior |
|-------|----------|
| `github_first` *(default)* | Pull prebuilt data from the project's GitHub release bundles first, then live-fetch only what's missing or stale. |
| `live_only` | Skip the GitHub bundles; fetch live from yfinance / Finviz — **except IBD classification, which has no live source and simply stops refreshing** (see below). |

The valid values are `github_first` and `live_only` (not `live`).

**What `github_first` pulls from GitHub** — published release bundles are tried before live providers, both during bootstrap and on the weekly schedule:

- **Daily prices** — a per-market 2-year OHLCV bundle (release tag `daily-price-data`). Imported when the manifest is fresh (within ~4 days) and the checksum matches, then live top-ups fill stale / no-history symbols.
- **Weekly universe** — `refresh_stock_universe` / `refresh_official_market_universe` check the GitHub weekly-reference bundle first; if current it no-ops, if missing/stale it falls back to live universe sources.
- **Weekly fundamentals** — `refresh_all_fundamentals` tries the GitHub weekly-reference bundle before the live provider path.
- **Weekly IBD classification** — syncs the GitHub `ibd-classification-data` bundle (Sunday, after the weekly classifier publishes). **GitHub-only: there is no live provider for IBD classification**, and it is how non-US markets without a curated CSV get IBD coverage.

For prices, universe, and fundamentals, any GitHub miss (missing/stale manifest, checksum mismatch, network error) silently falls back to live, so `github_first` is safe to leave on; its benefit is speed and fewer provider rate-limit hits (the heavy history downloads as one bundle instead of symbol-by-symbol). This is also why, under `github_first`, some Celery jobs still reach GitHub *after* bootstrap — expected behavior, see [issue #266](https://github.com/xang1234/stock-screener/issues/266).

`live_only` routes prices, universe, and fundamentals through live providers. **IBD classification is the exception**: it has no live source, so under `live_only` its weekly sync is a clean no-op and classifications simply stop updating — mainly affecting non-US markets, whose IBD coverage comes entirely from the GitHub bundle. Under `github_first`, the IBD sync no-ops only when the bundle is already `up_to_date`; a genuine GitHub miss (missing/stale manifest, checksum mismatch, network error) instead **fails the IBD sync and raises an alert** (unlike prices/universe/fundamentals, which quietly fall back to live), leaving the prior classifications in place until the next successful run.

**Changing it on a running app.** The value is baked into the containers at startup, so a running stack must be **stopped** for the change to take effect — recreating alone is not enough:

```bash
# 1. Put it in the env file you pass via --env-file — Compose interpolates the
#    x-app-env entry from there: .env for the base stack, .env.docker for the
#    prod overlay (the README / INSTALL_DOCKER prod commands already pass
#    --env-file .env.docker). Note: that environment: entry outranks the prod
#    overlay's own env_file, so setting it ONLY in .env.docker without passing
#    --env-file will not take effect.
#      MARKET_DATA_SOURCE_MODE=live_only      # or github_first

# 2. Stop the stack — running containers will not pick up the change
ENABLED_MARKETS=US,HK,JP,TW,KR scripts/docker-compose-enabled-markets.sh down

# 3. Start again with the new value
ENABLED_MARKETS=US,HK,JP,TW,KR scripts/docker-compose-enabled-markets.sh up -d
```

No cache flush is needed — the setting changes the *fetch strategy*, not data already stored. Switching to `live_only` keeps existing data and refreshes prices, universe, and fundamentals live going forward (IBD classification will not refresh); switching back to `github_first` resumes bundle syncing on the next refresh.

## Reset to a Clean Bootstrap

To re-run the first-run wizard from scratch (e.g. corrupt state, schema reset, or a clean demo), stop the stack, clear DB/cache/scheduler state, and start again. The Postgres directory is **moved, not deleted**, so the reset is reversible.

```bash
# 1. Stop the stack
ENABLED_MARKETS=US,HK,JP,TW,KR scripts/docker-compose-enabled-markets.sh down

# 2. Preserve the current DB → forces an empty one (this is what re-triggers bootstrap)
mv docker-data/postgres docker-data/postgres.saved.$(date +%Y%m%d_%H%M%S)

# 3. Drop the Redis volume (Celery broker + results + app cache)
docker volume rm $(docker volume ls -q | grep '_redis_data$')

# 4. Reset the Celery Beat schedule
rm -f data/celerybeat-schedule

# 5. Start again — the app opens to the first-run wizard
ENABLED_MARKETS=US,HK,JP,TW,KR scripts/docker-compose-enabled-markets.sh up
```

| Step | Effect |
|------|--------|
| `down` | Stops all containers (and orphans). |
| `mv docker-data/postgres …saved.<timestamp>` | Empties the live DB while keeping a timestamped backup (e.g. `postgres.saved.20260624_110650`). |
| `docker volume rm …_redis_data` | Clears the Celery broker/results and the application cache. |
| `rm -f data/celerybeat-schedule` | Forces the scheduler to rebuild its run state. |
| `up` | Boots into first-run bootstrap; watch progress on `/operations`. |

> **Warning:** steps 2–4 wipe the live database, cache, and scheduler state, and step 5 re-hydrates **every** enabled market — heavy on large `ENABLED_MARKETS` sets. The data loss is recoverable only from the moved Postgres directory (and any external backups).

**Roll back** to the pre-reset database:

```bash
ENABLED_MARKETS=US,HK,JP,TW,KR scripts/docker-compose-enabled-markets.sh down
rm -rf docker-data/postgres                                   # discard the fresh bootstrap DB
mv docker-data/postgres.saved.<timestamp> docker-data/postgres
ENABLED_MARKETS=US,HK,JP,TW,KR scripts/docker-compose-enabled-markets.sh up -d
```

## Runtime Activity

The header chip summarizes runtime state:

- **OK** — all markets are idle or ready.
- **Sync / percent** — bootstrap is running.
- **count** — one or more markets have active work.
- **Warn** — a market is stale, stuck, failed, or runtime activity cannot be checked.

Click the chip to open `/operations`.

## Operations Page

![Operations — per-market activity cards and the job console with cancel controls](screenshots/operations.png)
*Operations — per-market activity (states, messages, timestamps) above the filterable job console*

The Operations page includes:

- **Market activity** — per-market lifecycle, stage, message, task name, progress, and updated time.
- **Telemetry alerts** — warning/critical alerts with acknowledge controls.
- **Market health summaries** — freshness lag, benchmark age, universe drift, and completeness distribution.
- **Job console** — queued, waiting, reserved, running, stale, stuck, failed, and cancelled jobs across Celery queues.
- **Lease view** — current external-fetch and market-workload ownership.
- **Safe cancellation controls** — revoke, scan cancel, force refresh cancel, or queue removal when the backend marks an action as supported.

Use the filters (state, queue, market, task text) to narrow the job console before cancelling anything.

## Scheduled Tasks

When the tasks feature is enabled, the header settings icon opens **Scheduled Tasks**.

![Scheduled Tasks — registered jobs with schedules, last run, and run-now actions](screenshots/scheduled-tasks.png)
*Scheduled Tasks — registered jobs with schedule, last-run, status, and run-now*

The dialog shows each task's display name and description, schedule, last run time and duration, last status, and a run-now action (with polling while a task is active). Tasks are feature-gated; deployments without task support do not show this control.

## Balanced Market RS Rollout

`balanced-horizon-percentile-v2` is activated independently for each Market. Its five same-set excess-return percentiles are weighted **1M 20%**, **3M 30%**, **6M 20%**, **9M 15%**, and **12M 15%** before the composite is re-ranked. Keep the prior `legacy-linear-v1` Market formula pointer and legacy Feature-run ID until the rollout is accepted.

### 1. Record the rollback state

Before activation, record both active pointers. Substitute the target Market in both predicates:

```sql
SELECT market, formula_version
FROM market_rs_formula_pointers
WHERE market = 'US';

SELECT p.key, p.run_id, r.status, r.as_of_date, r.config_json
FROM feature_run_pointers AS p
JOIN feature_runs AS r ON r.id = p.run_id
WHERE p.key = 'latest_published_market:US';
```

The saved Feature run must remain present with status `published`; activation does not delete it.

### 2. Run the resumable shadow backfill

Run from `backend` with its virtual environment active. `--start-date` is optional when resuming a repaired range:

```bash
python -m app.scripts.backfill_market_rs \
  --market US \
  --through-date 2026-07-17
```

This leaves the active pointers unchanged. Inspect the JSON report: `formula_version` must be `balanced-horizon-percentile-v2`, `failed_count` and `validation_errors` must be empty/zero, every required trading date must be completed, and the latest run must reach `through_date`. A failed date is isolated for repair and the command exits nonzero.

### 3. Validate and activate atomically

Choose an absolute, empty directory that is not `STATIC_EXPORT_OUTPUT_DIR` and is not served by nginx or another web server. The activation command refuses a relative, non-empty, or configured serving directory.

```bash
python -m app.scripts.backfill_market_rs \
  --market US \
  --through-date 2026-07-17 \
  --static-staging-dir /var/tmp/stockscreen-rs-us-20260717 \
  --activate
```

The command resumes the backfill, builds a balanced Feature snapshot, stages `static-site-v3`, and checks stock/Group coverage, 1–99 ranges, contiguous deterministic Group ranks, formula/run/universe metadata, live/static stock and Group parity, and formula-isolated RRG state. Any failed gate exits nonzero without changing either active pointer. A successful validation updates the Market formula pointer and `latest_published_market:<MARKET>` Feature pointer in one database transaction, then invalidates Group caches and republishes the US bootstrap snapshot when applicable.

After success, verify the JSON has `activated: true`, the expected formula/run IDs, and no validation errors. In the live app, refresh Groups and a Scan and confirm:

- the response metadata names `balanced-horizon-percentile-v2` and the activation date;
- Scan overall/1M/3M/12M RS matches the staged sample;
- Group overall/1M/3M RS and ranks match staged `markets/<market>/groups.json`;
- the Group page shows 1M RS and 3M RS columns; and
- RRG either uses balanced history or explicitly reports insufficient balanced history.

### 4. Promote static artifacts

Do not copy an unvalidated staging directory into the serving directory. After the code and Market activation are on the default branch, use the existing Static Site workflow so its combine, fallback-compatibility, frontend-build, and Pages deployment gates remain in force:

```bash
gh workflow run static-site.yml -f market_group=us
# Use market_group=asia or market_group=all for the corresponding rollout set.
```

Confirm the workflow publishes `static-site-v3`, updates the Market's `static-rrg-history-v4` release asset where applicable, and deploys Pages successfully.

### 5. Roll back one Market

Rollback must restore both saved pointers together. First verify the recorded legacy Feature run is still `published`, then replace `<LEGACY_FEATURE_RUN_ID>` in this transaction:

```sql
BEGIN;

UPDATE market_rs_formula_pointers
SET formula_version = 'legacy-linear-v1', updated_at = CURRENT_TIMESTAMP
WHERE market = 'US';

UPDATE feature_run_pointers
SET run_id = <LEGACY_FEATURE_RUN_ID>, updated_at = CURRENT_TIMESTAMP
WHERE key = 'latest_published_market:US';

COMMIT;
```

If either row was not updated, roll back the transaction and investigate rather than leaving split pointers. After restoration, invalidate the affected Market's Group-ranking cache and republish/reload the live Groups bootstrap. Regenerate rollback artifacts with the workflow's explicit formula input (and the appropriate `market_group`):

```bash
gh workflow run static-site.yml \
  -f market_group=us \
  -f rs_formula_version=legacy-linear-v1
```

Verify live/static metadata says `legacy-linear-v1`. Retain balanced rows for diagnosis; rollback changes pointers, not history. A later return to balanced static output must use a newly validated live activation and the workflow's default `balanced-horizon-percentile-v2` input.

## Common Recovery Paths

### Bootstrap Is Slow

- Check `/operations` for active market stages and queue ownership.
- Confirm the host has workers for every market selected in the wizard.
- Reduce enabled markets and restart the stack if the host is resource constrained.
- Check upstream data-provider throttling if many symbols are stuck in price/fundamental refresh.

### Scan Is Blocked

Scans can return a market-refresh blocker while a selected market is hydrating. Wait for the relevant market to leave active refresh state, or inspect `/operations` for stale/stuck work.

### Runtime Activity Looks Stale

Use `/operations` to confirm whether a live worker owns the task. If no worker owns stale running work, restart the affected worker profile and re-run the market refresh or bootstrap step.

### Job Cancellation Fails

Cancellation is intentionally conservative. If a job has no supported cancel strategy, inspect the queue/worker state first, then restart only the affected worker profile if necessary.

### Bootstrap Won't Re-Trigger

The wizard only appears on an empty database. If a stale DB persists, confirm `docker-data/postgres` was moved/emptied (see [Reset to a Clean Bootstrap](#reset-to-a-clean-bootstrap)) before restarting.

### API Docs Are Missing

Interactive API docs are disabled by default in server deployments. Set `SERVER_EXPOSE_API_DOCS=true` only for trusted local development or private environments.

## Related Docs

- [Live App Guide](LIVE_APP_GUIDE.md)
- [Docker Deployment](INSTALL_DOCKER.md)
- [Environment Variables](ENVIRONMENT.md)
- [Architecture](ARCHITECTURE.md)
