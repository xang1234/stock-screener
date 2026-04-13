# ASIA Multi-Market ADR Pack

- Program: `StockScreenClaude-asia`
- Governance epic: `StockScreenClaude-asia.1`
- ADR publication task: `StockScreenClaude-asia.1.1`
- Date: 2026-04-11

This folder contains the authoritative architecture decision records for ASIA expansion (HK/JP/TW + US parity).

## ADR Index

1. [ADR ASIA-E0: Scope and Non-Goals (v1)](./adr_asia_e0_scope_non_goals_v1.md)
2. [ADR ASIA-E1: SecurityMaster Identity Model (v1)](./adr_asia_e1_security_master_identity_v1.md)
3. [ADR ASIA-E2: BenchmarkRegistry Policy (v1)](./adr_asia_e2_benchmark_registry_policy_v1.md)
4. [ADR ASIA-E3: MarketCalendarService Engine and Semantics (v1)](./adr_asia_e3_market_calendar_service_v1.md)
5. [ADR ASIA-E4: TranslationPipeline and Multilingual Extraction Contract (v1)](./adr_asia_e4_translation_pipeline_v1.md)
6. [ADR ASIA-E5: ReconciliationCircuitBreaker Policy (v1)](./adr_asia_e5_reconciliation_circuit_breaker_v1.md)
7. [ASIA v2 Feature Flag Matrix and Rollback Runbook](./asia_v2_flag_matrix_and_rollback_runbook.md)
8. [ASIA v2 Symbol Constraint Inventory Matrix (ST1)](./asia_v2_symbol_constraint_inventory_matrix.md)
9. [ASIA v2 Objective Launch-Gate Charter](./asia_v2_launch_gate_charter.md)
10. [ASIA v2 E2 Migration Rehearsal Report (ST3 + T2, 2026-04-11)](./asia_v2_e2_st3_t2_migration_rehearsal_report_2026-04-11.md)
11. [ASIA v2 JP Ingestion Adapter Notes](./asia_v2_jp_ingestion_adapter_notes.md)
12. [ASIA v2 Legacy Universe Compat + Deprecation Policy](./asia_v2_legacy_universe_compat_deprecation_policy.md)
13. [ASIA v2 E8 API / Client Migration Guide](./asia_v2_e8_api_migration_guide.md)

## Consumption Rules

- These ADRs are normative for ASIA beads under `StockScreenClaude-asia.*`.
- Any behavior change that violates a v1 contract requires a superseding ADR (`v2`) plus migration/rollout notes.
- Rollout gates in `StockScreenClaude-asia.11.*` should reference these ADRs for go/no-go evidence.
