# ADR E1: Pipeline-Scoped Processing State and Migration Contract (v1)

- Date: 2026-02-24
- Status: Accepted
- Issue: `StockScreenClaude-bv9.1.1`
- Machine-check spec: `docs/theme_identity/adr_e1_pipeline_scoped_processing_state_v1.invariants.json`

## Context

Theme extraction currently uses global `ContentItem` processing flags (`is_processed`, `processed_at`, `extraction_error`).
Both `technical` and `fundamental` pipelines read from the same global state, which creates cross-pipeline interference:

- One pipeline can mark an item processed before the other pipeline runs.
- Parse and provider failures can be represented as global success/failure rather than per-pipeline outcomes.
- Reprocessing and silent-failure recovery become ambiguous when pipeline ownership is implicit.

This ADR defines the contract for a per-pipeline processing state model that downstream E1 tasks must implement.

## Decision

### 1) Processing state must be pipeline-scoped

Introduce `content_item_pipeline_state` keyed by `(content_item_id, pipeline)`.
Global `ContentItem` processing fields remain compatibility fields during staged cutover and must not be the source of truth for per-pipeline eligibility.

Minimum columns for `content_item_pipeline_state`:

- `id` (PK)
- `content_item_id` (FK logical reference)
- `pipeline` (`technical` or `fundamental`)
- `status` (`pending`, `in_progress`, `processed`, `failed_retryable`, `failed_terminal`)
- `attempt_count`
- `error_code`
- `error_message`
- `last_attempt_at`
- `processed_at`
- `created_at`
- `updated_at`

Required uniqueness:

- `UNIQUE(content_item_id, pipeline)`

### 2) State transition contract

Allowed transitions:

- `pending -> in_progress`
- `in_progress -> processed`
- `in_progress -> failed_retryable`
- `in_progress -> failed_terminal`
- `failed_retryable -> in_progress`
- `failed_retryable -> failed_terminal`

Disallowed direct transitions:

- `pending -> processed` (must pass through `in_progress`)
- `processed -> pending` (except explicit operator backfill/reset flow)
- Any terminal-to-terminal direct jump that skips `in_progress`

### 3) Failure and retry semantics

- Parse failures and transient LLM/provider failures are not success states.
- Parse/provider failures default to `failed_retryable` unless classified non-retryable.
- Retry policy uses bounded attempts with backoff.
- Attempt metadata (`attempt_count`, `last_attempt_at`, `error_code`) is mandatory for diagnosis and SLO tracking.

### 4) Compatibility and staged cutover

Compatibility period:

- Existing `ContentItem` fields remain populated so current APIs do not break.
- New extraction/reprocess logic reads and writes `content_item_pipeline_state` as authoritative pipeline state.

Cutover phases:

1. Add schema and indexes idempotently at startup.
2. Backfill pipeline-state rows from historical items and mentions.
3. Route extraction and reprocessing eligibility to pipeline-state table.
4. Keep compatibility writes to global fields until API/reporting migration completes.
5. Remove dependence on global fields for orchestration decisions.

### 5) Observability contract

Per-pipeline metrics are required:

- pending count and age percentiles
- in-progress count
- processed count
- failed-retryable count and retry age
- failed-terminal count
- processed-without-mentions ratio

## Invariant Set (Machine-Check IDs)

Normative invariants are versioned in:

- `docs/theme_identity/adr_e1_pipeline_scoped_processing_state_v1.invariants.json`

Required IDs for v1:

- `E1-INV-001`
- `E1-INV-002`
- `E1-INV-003`
- `E1-INV-004`
- `E1-INV-005`
- `E1-INV-006`
- `E1-INV-007`

## Downstream Task References

This ADR is normative for all E1 implementation tasks:

- `StockScreenClaude-bv9.1.2` (`E1-T2`) schema and migration implementation
- `StockScreenClaude-bv9.1.3` (`E1-T3`) extraction/reprocess refactor
- `StockScreenClaude-bv9.1.4` (`E1-T4`) backfill from historical data
- `StockScreenClaude-bv9.1.5` (`E1-T5`) pipeline reassignment integrity handling
- `StockScreenClaude-bv9.1.6` (`E1-T6`) observability and drift detection

## Rollout Notes

- This ADR defines contract semantics first; implementation follows in E1-T2 through E1-T6.
- During migration, prefer conservative correctness over aggressive inference for ambiguous historical rows.
- Any policy exceptions must be documented with rationale and linked issue IDs.
