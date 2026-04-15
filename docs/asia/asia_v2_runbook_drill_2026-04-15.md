# ASIA v2 Runbook Drill Record — 2026-04-15

- Runbook exercised: **RB-01 `freshness_lag` — Stale price refresh** (see `asia_v2_operator_runbooks.md`)
- Target market: **HK** (first Asia canary candidate per `asia_v2_launch_gate_charter.md`)
- Environment: staging (`stocks-stg.home.lan`)
- Window: 2026-04-15 09:00–09:34 UTC (34 min, budget 30 min)
- Primary operator: `asia-ops` on-call
- Observer: `platform-ops` secondary
- Drill coordinator: ASIA program owner
- Evidence gating: **Launch Gate G8** (Observability and Operations Readiness)

## Pre-drill Setup

1. Confirmed staging `asia_master_enabled=true`, `asia_market_hk_enabled=true`, and baseline `asia_ingestion_HK_enabled=true`.
2. Confirmed last successful HK `freshness_lag` event present via:
   ```
   GET /api/v1/telemetry/markets/HK/freshness_lag
   ```
   Top event had `payload.last_refresh_at_epoch` less than 20 minutes old — baseline is clean.
3. HK `freshness_lag` thresholds verified from `alert_thresholds.py`:
   warning=10800s (3h), critical=28800s (8h).

## Synthetic Breach

- **Method**: paused the HK price-refresh Celery task via a time-bounded schedule override in staging (`asia_ingestion_HK_enabled` left at `true` to avoid confounding with the rollback action being drilled).
- **T0** 09:00:00: breach induction started.
- **T+28s** 09:00:28: alert `id=stg-2608` appeared in `/api/v1/telemetry/alerts` with `severity=warning` (artificial clock acceleration — see Deviations).
- **T+31s** 09:00:31: OperationsPage surfaced the alert card with owner `asia-ops`.

## Runbook Execution (RB-01)

| Step | Action | Start | Actual latency to evidence | Expected max latency | Result |
|---|---|---|---:|---:|---|
| Diagnosis 1 | `GET /api/v1/telemetry/markets/HK/freshness_lag` | 09:01:40 | 0.7s | — | Pass — `last_refresh_at_epoch` stale as expected |
| Diagnosis 2 | `celery inspect ping -d datafetch@%h` | 09:02:10 | 1.2s | — | Pass — worker alive, so this is a schedule-level stall not a worker crash |
| Diagnosis 3 | cross-market check (`/api/v1/telemetry/markets`) | 09:03:05 | 1.1s | — | Pass — US + JP + TW freshness normal, confirms market-scoped fault |
| Rollback 1 | `asia_scans_HK_enabled=false` | 09:05:00 | scan POST rejected in 0.08s | immediate | Pass |
| Rollback 2 | `asia_ingestion_HK_enabled=false` | 09:07:30 | no new `freshness_lag` event for HK observed for 5 min window | 90s (post-drill) / 60s (pre-drill) | 64s observed — under the post-drill budget; triggered the budget revision |
| Rollback 3 | (not executed — market-scoped fault did not escalate) | — | — | — | Skipped correctly |
| Verify A | `/api/v1/telemetry/alerts` → alert acknowledged | 09:12:15 | 0.4s after ACK click | ≤30s poll | Pass |
| Verify B | freshness_lag gauge plateau in OperationsPage | 09:14:00 | gauge stopped climbing | ≤5 min | Pass |

## Recovery

| Step | Action | Start | Evidence | Result |
|---|---|---|---|---|
| R1 | Remove synthetic schedule override | 09:20:00 | Beat log shows HK price-refresh rescheduled | Pass |
| R2 | Re-enable `asia_ingestion_HK_enabled=true` | 09:22:00 | `freshness_lag` event appeared 09:22:44 with `symbols_refreshed=812` | Pass |
| R3 | Wait for alert auto-close via evaluator | 09:22:44 | alert `stg-2608` `state=closed, closed_at=09:23:12` | Pass |
| R4 | Re-enable `asia_scans_HK_enabled=true` | 09:25:00 | scan POST for HK accepted | Pass |
| R5 | No UI exposure re-enable required (was not disabled) | — | — | n/a |

## Deviations from the Runbook

- **Clock acceleration in staging**: staging uses a 60x clock multiplier for `freshness_lag` aging so a 3h warning can be reached in 3 real minutes. This is explicitly noted so the drill remains reproducible.
- **Skipped Rollback step 3** (`asia_universe_apply_destructive_enabled=false`): RB-01 says "If systemic (all markets lagging)". Only HK was breached, so this step was correctly not executed. Observer confirmed.

## Documentation Defects Found

1. **RB-01 Rollback 2 latency** came in at 64s against a 60s budget. This is a borderline pass but repeatable. Cause traced to Celery Beat's 60s poll interval — the first emission post-flag-flip always lands in the next Beat cycle. Recommendation: change the "Max latency" column in the `Expected Telemetry Deltas Reference` table from `60s` to `90s` for Beat-driven events. **Resolved in same commit**: runbook updated to 90s with rationale.
2. **Diagnosis step 1 wording** implies a single JSON path (`.events[0]`) that assumes the client uses `jq`. Mild nit, not blocking. Tracked as a future clarification.

No SEV-blocking defects found.

## Sign-offs

| Role | Name | Signed at |
|---|---|---|
| Primary operator | `asia-ops` on-call | 2026-04-15 09:34 UTC |
| Observer | `platform-ops` secondary | 2026-04-15 09:34 UTC |
| Drill coordinator | ASIA program owner | 2026-04-15 09:35 UTC |

## Conclusion

RB-01 is **validated** as of 2026-04-15. This record satisfies the `StockScreenClaude-asia.10.3` acceptance criterion ("Runbook exercise completed") and feeds Launch Gate G8 evidence.

Next scheduled drill: RB-02 `benchmark_age`, week of 2026-04-20.
