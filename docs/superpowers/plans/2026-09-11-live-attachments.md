# Live Attachment Evidence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Automatically recover and prepare ordinary live post attachments and refresh grounded extraction when evidence arrives.

**Architecture:** Persist a bounded set of child attachment records, processed by a durable scheduled worker. Share immutable prepared evidence snapshots with legacy grounding and social work hashing; successful re-extraction replaces legacy mentions while social publications keep their frozen input history.

**Tech Stack:** FastAPI, SQLAlchemy/Alembic, Celery, existing public fetch/BeautifulSoup/PDF components, Kimi K2.6.

**Spec:** docs/superpowers/specs/2026-09-11-live-attachments.md

## Global Constraints

- No full article chunking; preserve bounded extraction limits and truncation warnings.
- Only approved Kimi/OpenCode Go preparation; social extraction remains inside its existing budget wrapper.
- No deployment, migration application or benchmark artifact changes.
- One parent source family; attachment availability cannot inflate corroboration.

### Task 1: Preserve provider references

Files: domain/social_signals/records.py, infra/providers/{xui_cli_social_provider,official_x_social_provider}.py, services/twitter_ingestion_providers.py and their existing tests under backend/.

Interface: `SocialAttachmentRef(kind, url)` and `SocialPostRecord.attachments`, default empty tuple; legacy fetch dictionaries emit the equivalent list. Serialize empty fields without changing historical observations.

- [x] Add failing provider fixtures for actual XUI photo/link fields, official media expansions and deduplication.
- [x] Implement bounded references excluding avatars, video previews and X navigation.
- [x] Verify existing no-attachment provider tests and new cases.

### Task 2: Prepare attachments

Files: backend/app/services/live_attachment_preparation.py and backend/tests/unit/test_live_attachment_preparation.py.

Interface: `LiveAttachmentPreparer(api_key)(kind, url) -> PreparedAttachment`, holding text, original_text, content_sha256, final_url, status and provenance.

- [x] Test image validation, readable article recovery, translation and restricted/unavailable responses using fake transports/models.
- [x] Reuse existing safe fetch, image, article and translation primitives, retaining originals and deterministic quantity normalization.
- [x] Run focused tests; verify truncation and partial states are explicit.

### Task 3: Persist, schedule and ground

Files: backend/app/models/theme.py, migration 0041, services/live_attachment_service.py, tasks/live_attachment_tasks.py, celery_app.py, content_ingestion_service.py, theme_extraction_service.py and new unit tests.

Interfaces: `record_attachments(db,item,refs,observed_at=None)` adds child references in the caller transaction; `attachment_snapshot(db,item_id,as_of=None)` returns revision, aggregate status and bounded evidence; preparation worker claims one row with an expiring lease, prepares outside the transaction, and publishes only if it still owns that lease.

- [x] Test idempotent recording and preparation, failure retry/backoff, revision changes, and same-family grounding with original hashes.
- [x] Add schema and transactional preparation service; register a periodic bounded sweep so committed ingestion survives enqueue failures.
- [x] Load evidence into default grounding. Reconcile evidence revisions with per-pipeline states; successful re-extraction replaces only legacy mentions for this post/pipeline atomically.
- [x] Test pending-to-prepared, late evidence races and preservation of prior mentions on model failure.

### Task 4: Freeze social evidence per generation

Files: social_extraction_service.py, social_signal_writer.py and social refresh integration with existing tests.

- [x] Add tests proving prepared evidence changes new work hashes while old saved observations remain unchanged.
- [x] Freeze explicit evidence in new observations, with same-parent citation validation and exact source quotes.
- [x] Reuse existing budget/work machinery for new evidence; preserve historical publications and replay inputs.
- [x] Verify no-attachment compatibility and no additional source count.

### Task 5: Surface status and validate integration

Files: themes_queries.py, schemas/theme.py, theme source/detail views and operational documentation.

- [x] Expose pending/complete/partial/failed evidence plus source URLs and warnings.
- [x] Run representative live ingestion-to-grounding tests without external model calls, targeted lint, frontend checks and migration checks.
- [x] Review all changed integration paths, repair findings and document deployment/activation requirements.

## Execution and verification

Implemented in the existing theme-detection-evaluation worktree. Independent review identified and repaired unbounded queue selection, stale-item dispatch starvation, source authorization gaps and duplicate URL/content evidence. Additional regressions cover a 100-item recovery backlog and authorization-deferred queue entries.

Validation: 807 backend tests passed in the broad related suite; the final 14-case live service suite (including two additional recovery regressions) passed. Frontend attachment-status test and production build passed. New-module Ruff checks and `git diff --check` pass. Migration upgrade/downgrade was exercised in isolated SQLite only. No live provider calls, deployed schema changes or running-service changes were made.

Operational activation and supported capture limits are documented in `docs/theme_evaluation/live_attachments.md`.
