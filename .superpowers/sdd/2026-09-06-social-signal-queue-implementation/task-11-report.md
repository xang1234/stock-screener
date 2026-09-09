# Task 11 report — authenticated Social Signal API

## Outcome

Added server-session-protected public Social Signal reads for summary, ranked queue,
context, unresolved identities, run-frozen evidence, and Theme pulse. Public reads
use only the latest published run; disabled and warming-up installations return
typed availability responses instead of errors. Queue controls are Market-aware,
bounded, and keep blended, Pure Social, and unranked sections independent.

Added an administrator surface protected by both the server session and
`X-Admin-Key`: shared runtime policy, source creation/rename/test/transition and
audit history, health, run history, validation previews, saved analysis and
budgeted retry dispatch, Theme association decisions, company identities, and
manual refresh. Provider diagnostics and run projections expose only allowlisted
codes and metadata; credentials, session state, raw provider output, and private
package details are not returned.

Legacy Theme source mutations now reject every Social-owned source before any
field, pipeline, or lifecycle change. Equivalent `x.com`/`twitter.com` list URLs
cannot duplicate a Social-owned list through the legacy endpoint. Ordinary Theme
sources retain their existing behavior. Live Theme detail uses the explicit union
of legacy membership and accepted Social membership, while non-live/static
readers remain legacy-only.

## Verification

- Task-focused API/boundary suite: **23 passed**, 20 inherited warnings.
- Broad Social plus affected Theme regression checkpoint: **561 passed**, 21
  inherited warnings.
- Final focused suite after excerpt canonicalization and frozen Theme-pulse
  projection: **23 passed**, 20 inherited warnings.
- Public application import with an isolated SQLite test database: succeeded.
- `git diff --check` and Python compilation: clean.
- No live X request, private xui session, paid model request, production database,
  image build, registry push, merge, or deployment mutation was used.

## Deferred ownership

- Tasks 12–13 consume these contracts in the live queue, evidence drawer, Daily,
  Theme Pulse, and Operations UI.
- Task 14 proves the static build graph and exported data stay Social-free.
- Task 15 adds public/private Docker worker targets and operator documentation.
- Task 17 retains disposable PostgreSQL and fixture-only end-to-end validation.
