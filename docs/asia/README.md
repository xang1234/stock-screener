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

## Consumption Rules

- These ADRs are normative for ASIA beads under `StockScreenClaude-asia.*`.
- Any behavior change that violates a v1 contract requires a superseding ADR (`v2`) plus migration/rollout notes.
- Rollout gates in `StockScreenClaude-asia.11.*` should reference these ADRs for go/no-go evidence.
