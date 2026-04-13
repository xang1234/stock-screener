# ASIA v2 Feature Flag Matrix and Rollback Runbook

- Date: 2026-04-11
- Status: Drafted for execution under `StockScreenClaude-asia.1.2`
- Scope: HK/JP/TW enablement with US parity preservation
- ADR alignment: `ADR ASIA-E0` through `ADR ASIA-E5`

## Purpose

This document defines the control-plane contract for ASIA rollout safety:

1. Market-scoped kill switches for ingestion, scanning, themes, and UI exposure.
2. Rollback order of operations to minimize blast radius and MTTR.
3. Objective ownership and verification requirements for each switch.

This is a policy artifact for implementation epics; not all flags listed here exist yet in code.

## Guardrail Principles

- Every externally visible market capability must have an independently testable kill switch.
- Kill switches are market-scoped first (`US`, `HK`, `JP`, `TW`), subsystem-scoped second.
- Emergency stop must not require schema rollback.
- Rollback must prefer disabling write paths before read paths.
- US baseline behavior must remain available when non-US flags are disabled.

## Flag Taxonomy

### Layer 0: Global Master Flags

- `asia_master_enabled`: Global ASIA capability gate for all non-US behavior.
- `asia_rollout_mode`: `off | dark_launch | canary | ga` (informational + policy checks).

### Layer 1: Market Exposure Flags

- `asia_market_hk_enabled`
- `asia_market_jp_enabled`
- `asia_market_tw_enabled`

### Layer 2: Subsystem Kill Switches (per market)

- `asia_ingestion_<market>_enabled`
- `asia_benchmark_cache_<market>_enabled`
- `asia_scans_<market>_enabled`
- `asia_themes_<market>_enabled`
- `asia_ui_exposure_<market>_enabled`

### Layer 3: Safety/Fallback Behavior Flags

- `asia_universe_apply_destructive_enabled` (global destructive apply gate)
- `asia_reconciliation_quarantine_enforced`
- `asia_non_us_unsupported_fields_graceful_degrade`
- `asia_mixed_market_scans_enabled`

## Current Baseline Controls (Already Present)

The existing runtime already exposes coarse feature switches in `backend/app/config/settings.py`:

- `feature_themes`
- `feature_chatbot`
- `feature_tasks`
- `setup_engine_enabled`
- `provider_snapshot_ingestion_enabled`
- `provider_snapshot_cutover_enabled`
- `theme_discovery_enabled`

These are useful but not sufficient for ASIA because they are not market-scoped.

## Flag Matrix

| Flag | Scope | Default | Owner | Purpose | Kill Test |
|---|---|---:|---|---|---|
| `asia_master_enabled` | Global | `false` | Platform | Hard-disable all non-US capabilities | HK/JP/TW requests return unsupported/hidden while US remains healthy |
| `asia_market_hk_enabled` | Market | `false` | Platform + Product | HK market exposure gate | HK universe option absent; HK scan creation blocked |
| `asia_market_jp_enabled` | Market | `false` | Platform + Product | JP market exposure gate | JP universe option absent; JP scan creation blocked |
| `asia_market_tw_enabled` | Market | `false` | Platform + Product | TW market exposure gate | TW universe option absent; TW scan creation blocked |
| `asia_ingestion_<market>_enabled` | Ingestion | `false` | Data Platform | Enable market universe ingest jobs | Scheduled/manual ingest no-ops when false |
| `asia_universe_apply_destructive_enabled` | Ingestion safety | `false` | Data Platform | Block destructive status changes globally | Reconciliation diff computes but does not deactivate |
| `asia_reconciliation_quarantine_enforced` | Ingestion safety | `true` | Data Platform | Force quarantine on threshold breaches | Breach causes quarantine + alert + no apply |
| `asia_benchmark_cache_<market>_enabled` | Cache/analytics | `false` | Analytics | Enable benchmark cache per market | No benchmark warm/update for disabled market |
| `asia_scans_<market>_enabled` | Scanner | `false` | Scanning | Permit market-specific scan execution | Scan API rejects/blocks market when false |
| `asia_mixed_market_scans_enabled` | Scanner | `false` | Scanning | Enable mixed-market universes | Mixed mode unavailable while single-market works |
| `asia_themes_<market>_enabled` | Themes | `false` | Themes | Enable multilingual extraction+mapping | Non-US theme extraction skipped/held |
| `asia_non_us_unsupported_fields_graceful_degrade` | Data quality | `true` | Fundamentals | Force transparent unsupported-field behavior | Unsupported fields surfaced with reason codes |
| `asia_ui_exposure_<market>_enabled` | UI/API | `false` | Frontend/API | Show market in selectors and detail surfaces | UI hides market options + API omits metadata |

## Rollback Playbook

### Severity Bands

- `SEV-1`: Data corruption risk, runaway deactivation, or incorrect market-wide outputs.
- `SEV-2`: Incorrect non-US output with bounded blast radius.
- `SEV-3`: UI-only mismatch or non-critical freshness lag.

### Standard Rollback Order (Fastest Safe Path)

1. Disable UI entry points (`asia_ui_exposure_<market>_enabled=false`).
2. Disable new scan execution (`asia_scans_<market>_enabled=false`).
3. Disable themes for market if affected (`asia_themes_<market>_enabled=false`).
4. Disable benchmark refresh for market (`asia_benchmark_cache_<market>_enabled=false`).
5. Disable ingestion apply path (`asia_ingestion_<market>_enabled=false`; keep quarantine on).
6. If systemic, disable `asia_market_<market>_enabled`.
7. If cross-market impact, disable `asia_master_enabled`.

### Emergency Destructive-Write Stop

When reconciliation safety is suspect:

1. Set `asia_universe_apply_destructive_enabled=false` immediately.
2. Confirm `asia_reconciliation_quarantine_enforced=true`.
3. Run reconciliation in dry/quarantine mode only until incident closed.

### Recovery Criteria Before Re-enable

- Root cause identified and fixed.
- Backfill/reconciliation rerun with clean diff artifact.
- Dashboards/alerts quiet for 2 consecutive market sessions.
- Canary gate checks pass for the market being re-enabled.

## Verification Checklist (Per Market)

- [ ] Toggle `asia_ui_exposure_<market>_enabled` hides/shows market selector.
- [ ] Toggle `asia_scans_<market>_enabled` blocks/allows scan creation.
- [ ] Toggle `asia_ingestion_<market>_enabled` blocks/allows ingest scheduling.
- [ ] Toggle `asia_themes_<market>_enabled` blocks/allows extraction pipeline.
- [ ] Toggle `asia_benchmark_cache_<market>_enabled` blocks/allows benchmark refresh.
- [ ] `asia_universe_apply_destructive_enabled=false` prevents destructive apply.
- [ ] `asia_reconciliation_quarantine_enforced=true` quarantines threshold breaches.

## Operator Notes

- Keep a one-command profile for each market state (`off`, `canary`, `ga`) to avoid partial toggles.
- Never re-enable ingestion apply before scan/UI gates are stable.
- Treat mixed-market scans as separate risk domain from single-market scans.

## Handoff to Downstream Beads

- `StockScreenClaude-asia.10.3` should embed this rollback order in incident runbooks.
- `StockScreenClaude-asia.11.*` gate runner should assert these flags and expected states.
- `StockScreenClaude-asia.8.*` should consume `asia_ui_exposure_<market>_enabled` semantics.
- External API consumers and internal rollout reviewers should use the [E8 API / Client Migration Guide](./asia_v2_e8_api_migration_guide.md) for per-task breakage tiers, before/after payloads, and the 2026-10-31 sunset.
