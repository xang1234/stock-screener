# ASIA v2 Operator Runbooks and Incident Templates

- Date: 2026-04-15
- Status: Publication artifact for `StockScreenClaude-asia.10.3`
- Scope: diagnosis + rollback procedures for every alert defined in `asia_v2_flag_matrix_and_rollback_runbook.md` (flag matrix) and `backend/app/services/telemetry/alert_thresholds.py` (thresholds)
- Required by: Launch Gate **G8** (Observability and Operations Readiness) and **G9** (Rollback Control Validation) in `asia_v2_launch_gate_charter.md`
- Drill record: `asia_v2_runbook_drill_2026-04-15.md`

## How to Read This Document

Every alert that the evaluator can raise (`MarketTelemetryAlert`) maps to exactly one **runbook entry** below. Each entry has the same five sections so operators can execute without rereading prose:

1. **Trigger** — which metric + severity fires the alert, owner team, and source payload fields.
2. **Diagnosis** — first three checks to narrow the fault to a subsystem.
3. **Rollback** — numbered flag toggles (copy-pasteable) in the order from the flag-matrix rollback playbook.
4. **Verification checkpoints** — the telemetry events you must see after a flag flip, with expected payload deltas and max latency.
5. **Recovery** — gate-style re-enable checklist.

### Common References

- **Flag source of truth**: `docs/asia/asia_v2_flag_matrix_and_rollback_runbook.md`
- **Threshold source of truth**: `backend/app/services/telemetry/alert_thresholds.py`
- **Payload schemas**: `backend/app/services/telemetry/schema.py`
- **API entry points**: `/api/v1/telemetry/alerts`, `/api/v1/telemetry/markets/{market}`, `/api/v1/telemetry/markets/{market}/{metric_key}`
- **Dashboard**: `/operations` (frontend `OperationsPage`)
- **Severity bands**: SEV-1/2/3 as defined in the flag-matrix "Severity Bands" section.

### Owner Routing (from `OWNERS` map)

| Market | Owner team |
|---|---|
| US | `us-ops` |
| HK, JP, TW | `asia-ops` |
| SHARED (cross-market) | `platform-ops` |

The owner is stamped onto the alert row at trigger time, so later `OWNERS` edits do not rewrite history. Page the team named on the alert row, not the current map.

---

## Alert → Runbook Index

| Alert (metric_key) | Markets | Triggers runbook |
|---|---|---|
| `freshness_lag` | US, HK, JP, TW | [RB-01](#rb-01-freshnesslag-stale-price-refresh) |
| `benchmark_age` | US, HK, JP, TW | [RB-02](#rb-02-benchmarkage-benchmark-cache-not-warming) |
| `universe_drift` | US, HK, JP, TW | [RB-03](#rb-03-universedrift-universe-growing-or-shrinking-abnormally) |
| `completeness_distribution` | US, HK, JP, TW | [RB-04](#rb-04-completenessdistribution-too-many-symbols-with-low-provenance) |
| `extraction_success` | SHARED | [RB-05](#rb-05-extractionsuccess-theme-extraction-ratio-collapse) |

---

## RB-01 `freshness_lag` — Stale price refresh

### Trigger

- **Metric**: `freshness_lag` (seconds since last successful price refresh for the market)
- **Severity ladder**: US 2h/6h, HK+JP 3h/8h, TW 4h/10h (warning / critical)
- **Owner**: `us-ops` for US, `asia-ops` for HK/JP/TW
- **Payload fields read**: `payload.last_refresh_at_epoch`, `payload.source`, `payload.symbols_refreshed`
- **Likely causes**: upstream data provider outage, Celery `data_fetch` worker stall, market-specific ingestion flag off, DB write failure

### Diagnosis

1. Pull the most recent `freshness_lag` event for the market:
   ```bash
   curl -sS "$HOST/api/v1/telemetry/markets/$MARKET/freshness_lag" | jq '.events[0]'
   ```
   Check `payload.last_refresh_at_epoch`. If it's very old, refresh is stalled; if it's recent, the gauge is stale but the job ran — suspect alert-evaluator input.
2. Confirm the `data_fetch` queue is alive:
   ```bash
   celery -A app.celery_app inspect ping -d datafetch@%h
   celery -A app.celery_app inspect active -d datafetch@%h
   ```
3. If US is fine but Asia is lagging, the issue is market-scoped — jump straight to subsystem rollback. If all markets are lagging, suspect global Celery / provider outage and treat as SEV-1.

### Rollback

Execute in this order; stop at the first step that clears the alert.

1. **Stop new scan creation for the market** to contain user-visible impact:
   ```
   asia_scans_<MARKET>_enabled=false
   ```
2. **Pause ingestion writes** (quarantine stays on, evaluation continues):
   ```
   asia_ingestion_<MARKET>_enabled=false
   ```
3. **If systemic (all markets lagging)**, disable destructive writes globally:
   ```
   asia_universe_apply_destructive_enabled=false
   ```
4. **SEV-1 escalation** (all non-US markets breaching critical):
   ```
   asia_master_enabled=false
   ```

### Verification checkpoints

After each rollback step, confirm the following within **5 minutes** (max 1 Celery Beat cycle + one telemetry emit):

| Check | What to look for | Where |
|---|---|---|
| Ingestion halted | no new `freshness_lag` events with `source=prices` for the affected market | `/api/v1/telemetry/markets/$MARKET/freshness_lag` |
| Scans blocked | POST to `/api/v1/scans` for that market returns 4xx with "market disabled" | backend access log |
| Lag gauge stops climbing | `/api/v1/telemetry/markets/$MARKET` → `freshness_lag.lag_seconds` plateaus | OperationsPage or curl |
| Alert state | alert transitions to `acknowledged` once operator acknowledges via UI | `/api/v1/telemetry/alerts` |

### Recovery

Re-enable in **reverse order** of rollback. For each step:

- Root cause written into the incident ticket.
- At least one successful `freshness_lag` event with `payload.symbols_refreshed > 0` observed.
- Alert closed automatically by evaluator (it will, because `freshness_lag` gauge < `warning` threshold and `value is None` no-op rule preserves state across Redis blips).
- Dashboards quiet for **2 consecutive market sessions** before raising `asia_ui_exposure_<MARKET>_enabled` back to `true`.

---

## RB-02 `benchmark_age` — Benchmark cache not warming

### Trigger

- **Metric**: `benchmark_age` (seconds since per-market benchmark cache was last warmed)
- **Severity ladder**: 1d warn / 2d critical across all markets
- **Owner**: market owner team
- **Payload fields**: `payload.last_warmed_at_epoch`, `payload.benchmark_symbol`
- **Per-market benchmark**: US→SPY, HK→^HSI, JP→^N225, TW→^TWII
- **Likely causes**: Celery Beat misfire, benchmark cache flag disabled, Redis cluster outage, yfinance / provider benchmark endpoint failing

### Diagnosis

1. Confirm the benchmark symbol in the alert payload matches the market's expected symbol — a mismatch indicates a regression in the benchmark registry (see `benchmark_registry.get_primary_symbol`). This is SEV-2 immediately because US parity contract (Charter G3) forbids SPY leakage.
2. Inspect Celery Beat schedule for the warm task:
   ```bash
   celery -A app.celery_app inspect scheduled
   ```
3. Check Redis availability — benchmark cache relies on DB 2 in the shared pool (`services/redis_pool.py`).

### Rollback

1. **Disable benchmark refresh for the affected market** to stop re-emitting stale cache warm events:
   ```
   asia_benchmark_cache_<MARKET>_enabled=false
   ```
2. **Disable scans for that market** (scans depend on benchmark for RS/percentile semantics):
   ```
   asia_scans_<MARKET>_enabled=false
   ```
3. **If symbol mismatch** (SPY returned for HK/JP/TW), also:
   ```
   asia_ui_exposure_<MARKET>_enabled=false
   ```
   to hide incorrect comparator data from users while the registry is fixed.

### Verification checkpoints

| Check | Expected delta | When |
|---|---|---|
| No more cache warms | no new `benchmark_age` events for market | within 1 Beat cycle (≤60s) |
| Scan rejection | scan API 4xx for market | immediate |
| Registry audit | `benchmark_registry.get_primary_symbol(market)` returns correct symbol in unit test | before re-enable |

### Recovery

- Re-run `tests/unit/test_benchmark_registry.py` and the Charter G3 benchmark regression set.
- Re-enable `asia_benchmark_cache_<MARKET>_enabled=true`, observe one successful warm event with correct `benchmark_symbol`, then flip scans and UI back on.

---

## RB-03 `universe_drift` — Universe growing or shrinking abnormally

### Trigger

- **Metric**: `universe_drift` (`|delta| / prior_size`, capped at 1.0)
- **Severity ladder**: 5% warn / 15% critical, all markets
- **Owner**: market owner team
- **Payload fields**: `payload.current_size`, `payload.prior_size`, `payload.delta`
- **Likely causes**: upstream index-membership diff bug, reconciliation applied without quarantine, provider snapshot corruption, schema-level change (new share class, delisting cascade)

### Diagnosis

1. Compute direction: `sign(payload.delta)`. A sudden **negative** delta of >15% is a mass deactivation event → treat as SEV-1 until proven otherwise. A **positive** delta of the same size is typically a listing dump — still SEV-2.
2. Confirm quarantine gate is active:
   ```
   asia_reconciliation_quarantine_enforced == true
   ```
3. Look at the most recent reconciliation artifact for the market (diff JSON emitted by `universe_tasks`). If quarantine did not fire on a breach, that is a control-plane bug (quarantine was bypassed) and requires a SEV-1 rollback regardless of data validity.

### Rollback

1. **Emergency destructive-write stop** — the canonical response to universe drift:
   ```
   asia_universe_apply_destructive_enabled=false
   asia_reconciliation_quarantine_enforced=true
   ```
2. **Pause ingestion** for the affected market:
   ```
   asia_ingestion_<MARKET>_enabled=false
   ```
3. **Pause scans** that depend on that universe:
   ```
   asia_scans_<MARKET>_enabled=false
   ```
4. **If cross-market drift signature** (two or more markets breach within one hour):
   ```
   asia_master_enabled=false
   ```

### Verification checkpoints

| Check | Expected delta | When |
|---|---|---|
| No destructive applies | reconciliation tasks log `apply=dry_run` only | next reconciliation run |
| Drift gauge plateau | new `universe_drift` events show the same `current_size` as last (post-quarantine size) | within 1 ingestion cycle |
| Quarantine artifacts present | diff JSON with `quarantined=true` for breaching symbols | reconciliation output dir |

### Recovery

- Manual review of the quarantined diff by Data Platform.
- Reconciliation re-run in dry mode with a clean diff artifact.
- Lift flags in reverse order, with **no destructive apply re-enable for at least 1 market session** after UI exposure returns.

---

## RB-04 `completeness_distribution` — Too many symbols with low provenance

### Trigger

- **Metric**: `completeness_distribution` (fraction of universe in the `0-25` completeness bucket)
- **Severity ladder**: 30% warn / 50% critical, all markets
- **Owner**: market owner team
- **Payload fields**: `payload.bucket_counts`, `payload.symbols_total`
- **Likely causes**: fundamentals provider outage for a locale, unsupported-field behavior regression, new market universe loaded before fundamentals backfill completes

### Diagnosis

1. Break down the payload — if `bucket_counts["0-25"] / symbols_total > 0.5` but the rest of the distribution is normal, a subset of symbols is failing fundamental enrichment (provider-specific, likely locale). If the whole distribution is shifted left, the fundamentals pipeline is broken globally.
2. Confirm the graceful-degrade flag is still on (we want unsupported fields surfaced, not silently zeroed):
   ```
   asia_non_us_unsupported_fields_graceful_degrade == true
   ```
3. Check Alpha Vantage / provider rate limit counters — on a 25 req/day tier, hitting the cap can silently wipe completeness for a session.

### Rollback

1. **Keep ingestion going** (fundamentals gaps don't require halting price data), but **block scans** that require high-completeness inputs:
   ```
   asia_scans_<MARKET>_enabled=false
   ```
2. **If the regression is in the unsupported-field display path**, force graceful-degrade on globally:
   ```
   asia_non_us_unsupported_fields_graceful_degrade=true
   ```
3. **If themes depend on the affected fundamentals**, also:
   ```
   asia_themes_<MARKET>_enabled=false
   ```

### Verification checkpoints

| Check | Expected delta | When |
|---|---|---|
| Scans blocked | scan API 4xx for market | immediate |
| Unsupported fields surface reason codes | sampled `/api/v1/fundamentals/...` responses include `unsupported_reason` | next request |
| Distribution improves | new `completeness_distribution` events show `0-25` bucket shrinking after provider recovers | within one fundamentals refresh cycle |

### Recovery

- Fundamentals backfill completes for the market.
- Sampled API contract test (Charter G4) passes.
- Re-enable scans, then themes, then UI exposure.

---

## RB-05 `extraction_success` — Theme extraction ratio collapse

### Trigger

- **Metric**: `extraction_success` (per-day success ratio across all languages, SHARED scope)
- **Severity ladder**: 0.85 warn / 0.70 critical (smaller is worse)
- **Owner**: `platform-ops`
- **Payload fields**: `payload.language`, `payload.success`, `payload.latency_ms`, `payload.provider`
- **Aggregation**: evaluator computes `sum(success) / sum(total)` across `by_language` buckets for the current day
- **Likely causes**: Minimax outage, Z.AI fallback disabled, LLM provider key revoked, latency spike causing timeout-induced failures, multilingual QA harness regression

### Diagnosis

1. Break down by language. `extraction_success` is recorded under `SHARED` scope and the `/markets/{market}/{metric_key}` endpoint rejects `SHARED` (scoped to `SUPPORTED_MARKETS={US,HK,JP,TW}`), so operators read directly from the event log:
   ```sql
   SELECT payload->>'language' AS lang,
          COUNT(*) AS n,
          SUM(CASE WHEN (payload->>'success')::bool THEN 1 ELSE 0 END) AS ok
   FROM market_telemetry_events
   WHERE market='SHARED'
     AND metric_key='extraction_success'
     AND recorded_at >= NOW() - INTERVAL '1 day'
   GROUP BY 1;
   ```
   Single-language collapse → provider / locale issue. Global collapse → Minimax outage or key issue. The SHARED-scope alert itself is always visible via `/api/v1/telemetry/alerts`.
2. Check the configured primary/fallback providers. Remember: Groq is the chatbot/research path; Minimax is primary for theme extraction/merge; Z.AI is the extraction fallback. Do not route theme extraction to Groq as a workaround.
3. Verify `LLM_FALLBACK_ENABLED` and `ZAI_API_KEY` are both set — without them, a Minimax outage cannot self-heal.

### Rollback

1. **Disable themes for the most-affected market** first to reduce noise:
   ```
   asia_themes_<MARKET>_enabled=false
   ```
2. **If the collapse is global (all markets, all languages)**, disable themes for all non-US markets:
   ```
   asia_themes_HK_enabled=false
   asia_themes_JP_enabled=false
   asia_themes_TW_enabled=false
   ```
3. **US is not a valid bypass** — do not silently re-route extraction to Groq; that would violate the sanctioned-provider contract (see CLAUDE.md LLM routing). Instead, ride out the outage with themes disabled.

### Verification checkpoints

| Check | Expected delta | When |
|---|---|---|
| Extraction paused | no new `extraction_success` events for disabled markets | immediate |
| Success ratio recovers | once provider healthy + flags re-enabled, day ratio climbs back above `warning` (0.85) | within 1 extraction cycle |
| Alert auto-closes | evaluator sees gauge above all thresholds and sets `state=closed` | next `/alerts` poll (≤30s) |

### Recovery

- Provider incident closed (Minimax or Z.AI).
- Dry-run extraction against the multilingual QA golden set (Charter G5) passes precision ≥ 0.85 / recall ≥ 0.75.
- Re-enable themes market-by-market, not all at once.

---

## Subsystem Diagnosis Flows

These are short per-subsystem checks that feed into the per-alert runbooks above. Use when the alert is ambiguous (e.g., `freshness_lag` that could be either ingestion or Celery).

### Ingestion (`asia_ingestion_<MARKET>_enabled`)

- Celery: `celery -A app.celery_app inspect active -d datafetch@%h`
- Beat: `celery -A app.celery_app inspect scheduled`
- DB: `SELECT market, MAX(recorded_at) FROM market_telemetry_events WHERE metric_key='freshness_lag' GROUP BY market;`
- Redis hot path: `rtk proxy redis-cli -n 2 GET "market:${MARKET}:last_refresh_epoch"`

### Benchmark cache (`asia_benchmark_cache_<MARKET>_enabled`)

- Registry check: in backend shell, `from app.services.benchmark_registry_service import benchmark_registry; benchmark_registry.get_primary_symbol("HK")` → must return `^HSI`.
- Cache key: `rtk proxy redis-cli -n 2 GET "benchmark:${SYMBOL}:ohlcv"`
- Daily warm log line: `grep 'benchmark_age_payload' backend.log`

### Scanning (`asia_scans_<MARKET>_enabled`)

- Attempt a scan POST against the disabled market — must reject with 4xx.
- Confirm `scan_orchestrator` logs the skipped-market line.

### Themes (`asia_themes_<MARKET>_enabled`)

- Confirm theme extraction tasks stop enqueuing: `celery -A app.celery_app inspect scheduled` should show no theme tasks for market.
- Last `extraction_success` event `payload.provider` — should be `minimax` under normal, `zai` under fallback; never `groq`.

### UI exposure (`asia_ui_exposure_<MARKET>_enabled`)

- Frontend: market option absent from universe selector.
- API: `/api/v1/stocks/markets` omits disabled market from the list.

### Reconciliation safety

- `asia_universe_apply_destructive_enabled=false` blocks apply step.
- `asia_reconciliation_quarantine_enforced=true` routes breaching diffs to quarantine output.

---

## Incident Templates

### SEV-1 Declaration

```
[SEV-1] <metric_key> <market> <yyyy-mm-dd hh:mm UTC>

Alert ID: <id from /api/v1/telemetry/alerts>
Owner team: <from alert row>
First fired: <opened_at>
Current severity: critical
Current gauge value: <value>
Threshold crossed: <critical threshold from alert_thresholds.py>

Suspected subsystem: <ingestion|benchmark|scans|themes|reconciliation>
Rollback steps executed: <list of flag toggles + timestamps>
Next check: <when + what payload delta we expect>

IC: <name>
Comms: <name>
Scribe: <name>
```

### SEV-2 / SEV-3 Declaration

Drop IC/Comms/Scribe roles; keep alert ID, rollback steps, and next check.

### Postmortem Skeleton (24h after resolution)

```
# Postmortem — <incident id>

## Summary
<1-paragraph plain-language description>

## Timeline (UTC)
- T-N: <first telemetry signal>
- T0: <alert fired>
- T+X: <rollback step 1>
- T+Y: <recovery confirmed>

## Impact
- Affected markets:
- User-visible impact:
- Data integrity impact:

## Root cause
<single-sentence root cause, then 3-5 sentence explanation>

## What went well
- <alert fired within expected window>
- <rollback executed from runbook without improvisation>

## What went poorly
- <missed signals, misrouted pages, unclear runbook steps>

## Action items
- [ ] <owner> <description> <due date>
- [ ] Runbook update — amend RB-XX section Y

## Links
- Alert row: /api/v1/telemetry/alerts?id=<id>
- Telemetry export: /api/v1/telemetry/markets/<market>/<metric_key>
- Drill record (most recent): asia_v2_runbook_drill_<date>.md
```

---

## Expected Telemetry Deltas Reference

This is the table operators consult after flipping a flag to confirm the rollback landed. Row N is "after you do X, you should see Y in the event log within Z seconds."

| Flag change | Event evidence | Max latency |
|---|---|---|
| `asia_ingestion_<M>_enabled=false` | no new `freshness_lag` events for `<M>` | 1 Beat cycle (60s) |
| `asia_benchmark_cache_<M>_enabled=false` | no new `benchmark_age` events for `<M>` | 1 Beat cycle (60s) |
| `asia_scans_<M>_enabled=false` | scan API rejects `<M>` | immediate (next request) |
| `asia_themes_<M>_enabled=false` | no new `extraction_success` events attributable to `<M>` | 1 extraction cycle (≤5 min) |
| `asia_universe_apply_destructive_enabled=false` | reconciliation logs `apply=dry_run` | next reconciliation (≤15 min) |
| `asia_reconciliation_quarantine_enforced=true` | breaching diffs have `quarantined=true` | next reconciliation |
| `asia_ui_exposure_<M>_enabled=false` | `/api/v1/stocks/markets` omits `<M>` | immediate |
| `asia_master_enabled=false` | all non-US subsystem flags effectively off | immediate |

If the expected delta does **not** appear within the listed latency, escalate one severity band and assume the flag did not take effect (misrouted env var, stale cache, wrong environment).

---

## Drill Exercise Procedure

To keep this document trustworthy, the runbook must be **exercised** before each canary stage (Charter G8, G9). Procedure:

1. Schedule a 30-minute window with on-call and a second operator as observer.
2. Pick **one** alert from RB-01..RB-05 to drill. Rotate weekly.
3. In staging, synthetically breach the threshold (e.g., force `freshness_lag` by pausing the price refresh task).
4. Confirm the alert appears via `/api/v1/telemetry/alerts`.
5. Execute the runbook **only using this document** — if you need to look up anything else, that is a documentation defect. Note it.
6. Execute each verification checkpoint and time the delta.
7. Recover using the Recovery section.
8. Record the drill in `asia_v2_runbook_drill_<yyyy-mm-dd>.md` with:
   - Which runbook was drilled
   - Start/end times
   - Each step's actual latency vs the expected latency in the table above
   - Documentation defects found
   - Operator sign-offs

The drill record is the evidence artifact for launch-gate **G8**. The most recent drill is listed at the top of this file.

---

## Change Control

- This document is the **single source of truth** for incident response on ASIA telemetry alerts.
- Changes require a PR review from at least one owner team lead (`us-ops`, `asia-ops`, or `platform-ops`).
- When `alert_thresholds.py` is edited, this runbook's "Severity ladder" lines in the affected RB-XX section must be updated in the same PR.
- When new metrics or flags are introduced, this document must be amended **before** the flag is shipped to production.
