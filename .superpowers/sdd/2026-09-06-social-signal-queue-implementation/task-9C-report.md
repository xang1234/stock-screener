# Task 9C report — Refresh orchestration and production wiring

## Outcome

Implemented `RefreshSocialSignals.execute(origin, now, saved_run_id=None)` and the production factory path. The workflow now:

- reads DB-authoritative mode, provider, version, model, and readable enabled source pins;
- does no X/model work while off/disabled and requires at least two enabled lists;
- creates deterministic generation IDs, resumes crash-before-completion with the same generation, and creates a new replay generation for budget resumption without reading X;
- holds the provider lease only around independently bounded source reads and releases it before metered extraction;
- uses fourteen-day/1,000-post initial requests until a committed initial success, then application-progress incremental requests capped at 200;
- persists failed source outcomes for audit and blocks analysis/publication if any enabled source fails;
- scopes backlog claims to the exact generation work set and checks DB runtime immediately before every actual model dispatch;
- retains canonical prior-run evidence and non-null observed metrics across capped incremental reads, pins retained semantic work separately from current source participation, and revalidates production social scores from those immutable records during prepare/publish;
- uses the shared administrator-attested identity configuration, shared Theme projection/application, one coherent confirmation batch per Market, existing three-window scoring, confirmation, and state rules;
- saves ranked, context, and unresolved sections distinctly;
- stages validation generations without pointer/catalog publication and atomically publishes live generations only.

The production factory lazily selects exactly `official` or `xui`; there is no fallback and no private package import. Official quota reservations use the durable registry service. Source names are frozen beside list IDs and source versions for readable evidence badges.

## Public additions

- `RefreshSocialSignals.execute(origin, now, *, saved_run_id=None)`
- `SocialWriter.current_inputs_ready(run_id)`
- `ProcessSocialBacklog.execute(..., work_ids=())` for generation-scoped claims
- `SocialCurrentInputManifest.scoring_work_ids`
- `SocialPublicationContext.scoring_input_version`
- `SqlSocialRefreshCatalog`
- `SocialScoringEvidenceReader.read`, `.retained_posts`, and `.read_in_session`
- session-safe Theme and confirmation facades
- `get_refresh_social_signals_use_case(...)`

## Verification

- RED: refresh module missing — 5 expected failures.
- Focused refresh workflow and rolling-evidence tests: 11 passed.
- Real fixture-backed flow completed through actual writer, metered backlog, Theme preparation, frozen confirmation validation, and pointer publication with synthetic provider/LLM only.
- Publication/writer regression: 116 passed.
- Backlog regression: 22 passed.
- Full affected Social suite: **456 passed, 4 inherited dependency warnings in 180.33s**.
- `git diff --check`: clean.
- New/modified Python modules compile successfully.

No live X request, private xui session, paid model request, production database, registry push, or deployment mutation was used.

## Deferred ownership

- Task 10 supplies the Social-specific Redis lease/cooldown/scheduler and six-hour job entry point. The factory currently adapts the existing shared external-fetch coordination seam.
- XUI history remains conservative (`limited`/warming) because its current fixture contract cannot prove exhaustive fourteen-day coverage; Official X marks `observed_window` only when it actually crosses the requested boundary.
- Task 11 exposes frozen read/admin APIs; Tasks 12–13 consume the three persisted sections and readable source pins.
- Task 17 retains the disposable PostgreSQL independent-producer concurrency proof and full fixture-only end-to-end verification.
