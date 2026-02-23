# E1-T2 Runbook: Pipeline-State Migration Verification and Rollback

## Scope

This runbook covers post-migration verification and rollback guidance for:

- `content_item_pipeline_state` schema creation
- index/constraint validation
- operator sanity checks after deploy

Related ADR:

- `docs/theme_identity/adr_e1_pipeline_scoped_processing_state_v1.md`

## Pre-Migration Safety

1. Ensure a current database backup exists before deploying migration code.
2. Verify application instance count and planned deployment window.
3. Confirm no manual schema edits are in progress.

## Migration Execution

The migration runs automatically at API startup via:

- `backend/app/main.py` -> `run_theme_pipeline_state_migration()`

Manual verification command:

```bash
cd backend
source venv/bin/activate
python scripts/verify_theme_pipeline_state_migration.py --json
```

Expected success:

- output field `ok` is `true`
- process exits with code `0`

If you need to apply migration first in a non-startup context:

```bash
python scripts/verify_theme_pipeline_state_migration.py --apply-migration --json
```

## SQL Verification Checks

### 1) Table exists

```sql
SELECT name
FROM sqlite_master
WHERE type='table' AND name='content_item_pipeline_state';
```

### 2) Required columns exist

```sql
PRAGMA table_info(content_item_pipeline_state);
```

Required column names:

- `id`
- `content_item_id`
- `pipeline`
- `status`
- `attempt_count`
- `error_code`
- `error_message`
- `last_attempt_at`
- `processed_at`
- `created_at`
- `updated_at`

### 3) Required indexes exist

```sql
PRAGMA index_list(content_item_pipeline_state);
```

Required indexes:

- `uix_cips_content_item_pipeline`
- `idx_cips_pipeline_status_last_attempt`
- `idx_cips_pipeline_status_created`
- `idx_cips_content_item_pipeline_status`
- `idx_cips_error_code`
- `idx_cips_updated_at`

### 4) Duplicate key sanity check

```sql
SELECT COUNT(*) AS duplicates
FROM (
  SELECT content_item_id, pipeline, COUNT(*) AS c
  FROM content_item_pipeline_state
  GROUP BY content_item_id, pipeline
  HAVING c > 1
) d;
```

Expected: `duplicates = 0`

### 5) Invalid status sanity check

```sql
SELECT COUNT(*) AS invalid_status_rows
FROM content_item_pipeline_state
WHERE status NOT IN (
  'pending',
  'in_progress',
  'processed',
  'failed_retryable',
  'failed_terminal'
);
```

Expected: `invalid_status_rows = 0`

## Rollback Triggers

Rollback investigation should be initiated if any of the following occur:

1. Table creation fails and startup migration emits warnings repeatedly.
2. Verification script returns `ok=false`.
3. Duplicate `(content_item_id, pipeline)` rows are detected.
4. Extraction workers show widespread SQL errors targeting `content_item_pipeline_state`.
5. Queue growth/regressions indicate scheduler queries are not using expected indexes.

## Rollback Guidance

Because this migration is additive (new table + indexes), prefer controlled disablement over destructive rollback:

1. Stop rollout and keep existing global-flag orchestration path active.
2. Keep table intact for forensic analysis.
3. Fix migration logic and redeploy.
4. Re-run verification script until `ok=true`.

Avoid dropping `content_item_pipeline_state` in production unless explicitly required by an incident commander.
Dropping the table can remove forensic state useful for recovery.
