# Theme Consolidation and Developments Implementation Plan

> Execute inline, as requested by the user. Use the approved spec and test each slice before integration.

**Goal:** Reversible equivalent-theme grouping and source-bound development timelines.
**Architecture:** Preserve original theme/mention identities; derive current groups from audited operations. Persist immutable event observations and supersede prior source revisions. Focused services own grouping, read aggregation, development matching, and model preparation.
**Tech Stack:** Existing SQLAlchemy/Alembic, FastAPI, Celery, React/React Query/MUI, pytest/Vitest.
**Spec:** docs/superpowers/specs/2026-09-12-theme-consolidation-developments-design.md

## Constraints
Names remain open. Initial semantic grouping requires review. Preserve HBM/Memory and pipeline distinctions. No destructive regrouping, unbounded backfill, new model provider, or ranking-weight changes. Original evidence and earlier event observations remain attributable.

## Tasks
- [x] 1. Add grouping operation and event/observation persistence in backend/app/models/theme_intelligence.py, register models, and additive Alembic migration. Test SQLite upgrade/downgrade and constraints.
- [x] 2. Implement backend/app/services/theme_equivalence_service.py: preview, reviewed apply, idempotency, dependency-safe undo, hierarchy/pipeline checks, preserved source IDs. Add behavioral tests in backend/tests/unit/test_theme_equivalence.py. Guard existing destructive merges for grouped members.
- [x] 3. Implement backend/app/services/theme_group_reads.py and connect current ranking inputs, details, constituents, mentions and search. Rebuild current caches after membership changes and expose grouping version. Test parent-post deduplication and undo visibility.
- [x] 4. Implement backend/app/services/theme_development_service.py with structured, cited event observations, conservative matching, revisions, repeat/update/contradiction classification and source times. Add model preparation through the configured extraction route and a bounded retryable task. Test attribution, repetition, independent events, corrections, and failures.
- [x] 5. Add backend/app/api/v1/themes_intelligence.py for group previews/apply/history/undo and development timelines/backfill preview. Reuse existing API authentication and register static routes before theme IDs. Add API tests.
- [x] 6. Add frontend API functions, reviewed grouping controls and undo history within merge review, and a development timeline in theme detail. Add Vitest tests, run frontend build.
- [x] 7. Run related backend integration/unit and performance gates, update CONTEXT.md and operator documentation, commit the implementation on the feature branch. Do not deploy or merge main.

For each task: write a failing behavior test, reproduce, implement the smallest complete change, then rerun it and related regressions. Inspect the final diff for atomicity, duplicate evidence, unavailable-service behavior and destructive legacy paths.

## Verification record

- Complete theme unit selection: 321 passed (before the last source/history regression was added).
- Final integration/category checks and existing performance gates: 27 passed, including the added representative-history test.
- Theme UI and page tests: 5 passed.
- Frontend production build and focused lint: passed.
- Additive migration upgrade/downgrade was exercised against a disposable SQLite database. No live database migration or paid model calls were performed.
- See docs/theme-consolidation-and-developments.md for activation and the explicit standard-pipeline/native-Social boundary.
