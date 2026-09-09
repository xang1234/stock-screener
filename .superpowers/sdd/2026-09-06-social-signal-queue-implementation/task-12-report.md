# Task 12 report — Social Queue and Evidence Drawer UI

## Outcome

Added a capability-gated Social Signals tab directly after Daily Snapshot in the
existing Market Scan surface. The tab reads only the authenticated public API and
supports 1D/7D/14D windows, Actionable/All Signals views, Blended/Pure Social
ordering, five local filters, and independent pagination for ranked, Context, and
global Needs resolution sections. Missing values remain em dashes; stale,
limited-history, partial-source, empty, disabled, reauthentication, and first-run
states are explicit.

The dense queue preserves Confirmation while Pure Social changes ordering. It
shows published-run setup/readiness, RS, group, Theme, coverage, state, and score
facts. The public query projection was extended to expose these values from the
immutable prepared run context rather than mutable current Market or Theme data.

Row selection opens an evidence-first drawer with frozen score/component weights,
state reasons, acceleration availability, related listings, and at most three
plain-text, canonical X links with source badges. Existing chart/setup and
watchlist controls are reused. “Send visible to Scan” creates a URL-safe,
Market-aware custom symbol selection; Scan consumes, validates, deduplicates, and
caps that selection and lets the user clear it.

## Verification

- Public Social API regression: **15 passed**, inherited dependency warnings only.
- Social client/presentation/tab plus Market Scan and Scan interactions:
  **28 passed** across five files.
- Final drawer/Scan focused rerun: **21 passed** across two files.
- Full frontend lint: **0 errors**, four pre-existing warnings outside this work.
- Production frontend build: succeeded, 2,546 modules transformed; Social Signals
  remains a lazy chunk.
- `git diff --check`: clean.
- No live X access, private package import, paid model request, production database,
  image publication, registry push, merge, or deployment mutation was used.

## Deferred ownership

- Task 13 adds the Daily summary card, Theme Social Pulse, and administrator health,
  source, identity, and budget controls.
- Task 14 proves Social code and data remain absent from the static application.
- Tasks 15–17 retain Docker/private worker packaging, CI guardrails, PostgreSQL,
  and fixture-only end-to-end verification.
