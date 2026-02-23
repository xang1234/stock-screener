# ADR LL2-E1: Canonical Price Contract (v1)

- Date: 2026-02-24
- Status: Accepted
- Issue: `StockScreenClaude-ofh.1.2`
- Depends on audit: `docs/learning_loop/ll2_e1_t1_price_ingestion_audit.md`
- Machine-check spec: `docs/learning_loop/adr_ll2_e1_canonical_price_contract_v1.invariants.json`

## Context

Learning Loop outcome analytics require deterministic raw/adjusted semantics across all price ingestion paths. The LL2-E1-T1 audit found critical defects:

- `adj_close` is currently persisted as an alias of `close`.
- Same-day correction is blocked by insert-only persistence.
- yfinance adjustment behavior is implicit (not parameterized explicitly).

This ADR defines the canonical contract that downstream implementation (`StockScreenClaude-ofh.1.3`) and reconciliation (`StockScreenClaude-ofh.1.4`) must satisfy.

## Decision

### 1) Canonical price fields

The canonical logical model for each `(symbol, trade_date)` row is:

- `raw_open`
- `raw_high`
- `raw_low`
- `raw_close`
- `raw_volume`
- `adj_close`
- `adj_factor_close` (derived as `adj_close / raw_close` when both non-null and `raw_close != 0`)
- `adj_open` (derived)
- `adj_high` (derived)
- `adj_low` (derived)
- `split_ratio` (event context, nullable)
- `dividend_cash` (event context, nullable)

`adj_open`, `adj_high`, `adj_low` are derived only, not independently fetched:

- `adj_open = raw_open * adj_factor_close`
- `adj_high = raw_high * adj_factor_close`
- `adj_low = raw_low * adj_factor_close`

### 2) Fetch-mode contract (Yahoo)

All daily Yahoo price ingestions MUST explicitly set adjustment semantics in code:

- `auto_adjust=False`
- `actions=True`
- `interval="1d"`

Applies to both:

- `Ticker.history(...)`
- `yf.download(...)`

No ingestion path may rely on yfinance defaults for adjustment behavior.

### 3) Allowed transformations for outcome calculations

Outcome labels and forward-return analytics MUST use adjusted series (`adj_*`) and must not directly use raw close-to-close returns across corporate-action boundaries.

Allowed:

- adjusted return calculations from `adj_close`
- adjusted ATR/range calculations from `adj_open/high/low/close`
- event-window diagnostics using raw + corporate-action fields

Disallowed:

- treating `raw_close` as adjusted
- treating `adj_close` as raw
- mixing raw and adjusted fields in one return expression without an explicit factor transform

### 4) Temporal semantics and correction rules

- `symbol + trade_date` is a logical upsert key.
- Post-close refresh MUST be able to replace same-day rows (not append-only).
- Freshness requires both:
  - expected market date coverage, and
  - stale-intraday replacement completion after close buffer.

### 5) Corporate-action semantics

- Split/dividend context must be preserved or derivable for each row.
- Adjustment factor provenance must be explicit and reproducible.
- A row where `adj_close == raw_close` is valid only when factor is effectively `1.0`; it is not a default fill rule.

## Invariant Set (Machine-Check IDs)

Normative invariant definitions are versioned in:

- `docs/learning_loop/adr_ll2_e1_canonical_price_contract_v1.invariants.json`

Each invariant declares a machine-check type:

- `check_kind = "sql"` uses `sql_check` and must return one scalar `violations` value.
- `check_kind = "repo"` uses `repo_check` path/literal assertions against repository code.

For SQL checks that depend on future schema, `violations = -1` means "required schema not present yet" and is treated as non-compliant until the migration is shipped.

Required IDs for v1:

- `PRICE-INV-001`
- `PRICE-INV-002`
- `PRICE-INV-003`
- `PRICE-INV-004`
- `PRICE-INV-005`
- `PRICE-INV-006`
- `PRICE-INV-007`
- `PRICE-INV-008`

## Rollout Notes

- This ADR defines target contract semantics.
- Known current non-compliance is documented in the LL2-E1-T1 audit.
- LL2-E1-T3 must implement schema/write-path/read-path changes to make these invariants pass.
- LL2-E1-T4 must backfill/reconcile historical rows against this contract.
