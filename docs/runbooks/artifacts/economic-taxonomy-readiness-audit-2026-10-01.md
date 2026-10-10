# Economic Taxonomy readiness and legacy retirement audit — 2026-10-01

Audit for #430 (part of #410). Line numbers refer to `main` @ `93f22b30`.
It checks deployments against the [cutover runbook](../economic-taxonomy-cutover.md),
defines when legacy rollback support can be retired, and inventories the
readers and writers that still depend on legacy, shadow or dual mode. It
changes no code.

The inventory was built by reading the code, starting from searches for legacy
theme models and authority-mode checks. It is a starting point, not proof that
nothing is missing. Retirement criterion 4's CI gate is the check that makes it
complete.

## Summary

- **Local Docker deployment:** never seeded. There is no `taxonomy_authority`
  row, so it runs as implicit legacy authority. It is not ready to enter
  shadow; see the checklist below.
- **Three gaps in the code apply to every deployment.** They are not
  configuration problems:
  1. **G1:** news and RSS content is never admitted as economic evidence. After
     cutover, only Social saved work would feed the economic catalog.
  2. **G2:** several legacy theme writers bypass the authority fence. They
     would keep changing legacy tables after cutover, and the fenced tasks
     would log an error on every scheduled run.
  3. **G3:** the runbook's seed step refuses an authority row that the fence
     itself creates on the first fenced legacy write.
- **Retirement:** not yet possible for the local Docker deployment, because it
  has not run in economic mode. Other deployments were not checked; apply the
  criteria below to each one. G1–G3 come first.

## 1. Readiness audit

### Deployments checked

Only the local Docker stack (`stockscreenclaude-*`) was reachable from this
audit. Other deployments, such as a homelab or VPS, should run the read-only
check at the end of this section and fill in their own copy of the table.

| Runbook item | Local status | Evidence |
|---|---|---|
| Migrations at head (step 1) | **One behind**: `20260926_0060`; head is `20260928_0061_workload_fences` (not taxonomy-related) | `alembic_version` |
| Required PostgreSQL gate (step 1) | Passes in CI on every PR; **not yet run** as release evidence for this deployment | `.github/workflows/ci.yml:94-98` |
| Admin identity (`ADMIN_API_KEY`, `ADMIN_PRINCIPAL_ID`) | **`ADMIN_PRINCIPAL_ID` empty**: publication would stop with `admin_identity_not_configured` | `scripts/publish_economic_taxonomy.py:32-33` |
| Extraction-provider credential | MiniMax and Z.AI keys are set | `.env` (names only) |
| Synthetic processing request (`processed: 1`) | **Not run.** It calls a paid provider, so it was left for the operator | runbook step 1 |
| Seeded V1 snapshot and authority (step 2) | **Not seeded**: no `taxonomy_authority` row | `SELECT * FROM taxonomy_authority` → 0 rows |
| Reviewed migration coverage and sealed snapshot (step 3) | Not started | — |
| Reader capability and shadow benchmark (step 4) | Not started | — |
| Full-procedure rehearsal | Done on a disposable clone, 2026-09-21 | [rehearsal artifact](economic-taxonomy-rehearsal-2026-09-21.md) |

**Next steps for this deployment:**
- Run `alembic upgrade head` and set `ADMIN_PRINCIPAL_ID`.
- Resolve G3 before seeding: the first scheduled theme metrics run will create
  the authority row.
- Resolve G1 before dual mode, so that shadow comparisons include news-derived
  themes.

### Read-only check for other deployments

```sql
SELECT mode, authority_epoch, writes_fenced, rollback_state,
       processing_taxonomy_version_id IS NOT NULL AS seeded,
       serving_generation_id IS NOT NULL AS has_generation
FROM taxonomy_authority;
SELECT version_num FROM alembic_version;
```

No rows means "implicit legacy, not seeded". A row with `seeded = false` has
no processing taxonomy version and hits the G3 seed blocker. The query does not
show what created the row, but `lock_authority()` is the code path that creates
rows in this state.

### G1: news content is never admitted as economic evidence

`EconomicSourceAdmissionService.admit_content()`
(`app/services/economic_source_admission.py:124`) is called only from tests.
The only production admission path is Social saved work
(`app/services/economic_social_taxonomy_adapter.py:984`, `admit_social_work`).

- **Expected:** the design says legacy content and Social work for the same
  post share one source family and lineage (design spec line 224), and the plan
  delivers `admit_content()` for that purpose.
- **What happens instead:** content ingestion (`poll_due_sources`, `ingest`)
  feeds only the legacy pipeline. In economic mode, news, RSS and Substack
  evidence would stop reaching the serving catalog.
- **What already works:** the admission service itself is tested (for example,
  `tests/unit/test_economic_source_admission.py`). Only the call from ingestion
  is missing.
- **When it blocks:** dual and economic. Shadow can still be entered, and shadow
  comparisons are where this gap would show up.

### G2: legacy writers that bypass the authority fence

The design requires that "a legacy writer racing cutover cannot commit after the
authority switch" (design spec line 728).

**Paths that comply** (they use
`EconomicTaxonomyRuntimeService.legacy_producer_write`, which allows only
legacy, shadow and dual, at `app/services/economic_taxonomy_runtime.py:872`):
- `ThemeDiscoveryService._fenced_legacy_mutation`
  (`app/services/theme_discovery_service.py:111`) and its call sites: metrics,
  candidate promotion, lifecycle policies and relationship inference.
- Social projection and decisions. They switch to the economic adapter in
  economic mode (`social_theme_projection_service.py:433`, `:727`), which makes
  them the model for the paths below.

**Paths with no authority check:**

| Writer | Entry points | Legacy rows written |
|---|---|---|
| `ThemeExtractionService` | beat `extract_themes` (:10, :40) and `reprocess_failed_themes` (:05); task `refresh_attachment_themes` (`live_attachment_tasks.py:15-33`); `POST /themes/extract`, `POST /themes/pipeline/run` | `ThemeMention`, `ThemeCluster`, `ThemeConstituent` (`theme_extraction_service.py:917`, `:1719`, `:1808`) |
| `ThemeCorrelationService.create_theme_from_cluster` | `POST /themes/create-from-cluster` | `ThemeCluster`, `ThemeConstituent` (`theme_correlation_service.py:427-477`) |
| `ThemeCorrelationService.validate_theme` | task `validate_themes`, `POST /themes/validate-all` (the per-theme `GET /{id}/validate` is guarded) | cluster validation fields (`theme_correlation_service.py:160`, `:169`) |
| `ThemeMergingService` | task `consolidate_themes` (`theme_discovery_tasks.py:1421`); `POST /themes/merge-suggestions/{id}/approve` and `/reject` (`theme_merging_service.py:2919`, `:2964-2986`), `/consolidate`, `/merge-wave/*` | merges, `ThemeMergeSuggestion` (`theme_merging_service.py:1446`) |
| `ThemeTaxonomyService.run_full_taxonomy_assignment` | task `run_taxonomy_assignment` (the API is guarded by `_reject_economic_mode`; the task is not) | L1 `ThemeCluster` and L2 assignments (`theme_taxonomy_service.py:178`) |
| `ThemeTaxonomyService.compute_all_l1_metrics` | task `compute_l1_metrics` (`theme_discovery_tasks.py:1504`), also called from `run_full_pipeline` (`:1221`) after the fenced L2 metrics step | L1 `ThemeMetrics` (`theme_taxonomy_service.py:1088`) |
| Theme review and merge API | `POST /themes/{id}/add-constituents`, `DELETE /themes/{id}`, `/candidates/review`, `/alerts/*` | `ThemeConstituent` (`themes_review_merge.py:384`), cluster state, alerts |
| Equivalence API | `POST /themes/equivalence`, `/equivalence/{id}/undo` | legacy identity equivalence |
| `ThemeTaxonomyService.compute_l1_centroid_embeddings` | task `recompute_l1_centroid_embeddings` (`theme_discovery_tasks.py:1607`) | L1 `ThemeEmbedding` (`theme_taxonomy_service.py:1102-1159`) |
| Theme content corruption recovery: `reset_corrupt_theme_content_storage` (`theme_content_recovery_service.py:53-72`) | content list and export, when corruption survives REINDEX (`api/v1/themes.py:140-171`) | **Drops and recreates** `theme_mentions`, together with the shared `content_items` and `content_item_pipeline_state`. No authority check: in economic mode it deletes rollback data, and after retirement it would recreate `theme_mentions` |
| `theme_group_refresh.refresh_groups` | beat `theme-group-refresh` every 60 s (`celery_app.py:536-540`, task `theme_intelligence_tasks.py:33`) | `ThemeEquivalenceOperation` status (`theme_group_refresh.py:29-44`) |
| One-off maintenance | `theme_alias_backfill_service` (`ThemeAlias`, `:227-246`; CLI `scripts/backfill_theme_aliases.py`), `app/scripts/repair_jp_alpha_universe_symbols.py` (`ThemeConstituent`, `ThemeMention`, `ThemeAlert`) | legacy theme rows, run by an operator |
| Operator CLI `scripts/backfill_l1_taxonomy.py` | `run_full_taxonomy_assignment` (`:59`), then `compute_l1_centroid_embeddings` and `compute_all_l1_metrics` with commits (`:107-113`) | L1 `ThemeCluster`, `ThemeEmbedding`, `ThemeMetrics` |
| Operator CLI `scripts/backfill_silent_failures.py` | reprocesses content through `ThemeExtractionService` | same rows as extraction |
| `ThemeMergingService` embeddings | task `recompute_stale_theme_embeddings` (`theme_discovery_tasks.py:603`), `POST /themes/embeddings/refresh-campaign` | `ThemeEmbedding` for legacy clusters (`theme_merging_service.py:573-601`, `:630-708`, `:840`) |
| `ThemeDiscoveryService.check_for_alerts` | task `check_alerts`, `run_full_pipeline`, `POST /themes/alerts/check` | `ThemeAlert` (`theme_discovery_service.py:1157`) |

**Effect in economic mode:**
- These paths keep changing legacy tables with no source revision and no
  compatibility event. A later rollback would serve legacy rows that are partly
  rebuilt projections and partly direct writes.
- The fenced tasks fail closed but loudly. `calculate_theme_metrics` (:20, :50),
  `promote_candidate_themes` (04:30) and `apply_lifecycle_policies` (04:45) raise
  `AuthorityModeRejected`, which their handlers log as an ERROR on every run
  (for example `theme_discovery_tasks.py:584-590` and `:756-759`).
- They need a mode-aware skip: a reason in the task result, and 409
  `economic_generation_endpoint_required` on the APIs, as `themes_queries` and
  `themes_taxonomy` already return.

### G3: the seed step conflicts with the authority row the fence creates

- `EconomicTaxonomyPublicationRepository.lock_authority()`
  (`app/infra/db/repositories/economic_taxonomy_publication_repo.py:43-69`)
  inserts a `legacy` authority row with `processing_taxonomy_version_id = NULL`
  the first time any fenced legacy write runs, for example a theme metrics run.
- Runbook step 2 is meant to run "when `processing_taxonomy_version_id` is null".
  However, its script stops on any existing row (`economic-taxonomy-cutover.md:122`).
- Nothing else can set the processing taxonomy in legacy mode. Taxonomy
  operations require shadow mode or later and an existing base version
  (`economic_taxonomy_operations.py:282`, `:314`).
- So a deployment that has run theme metrics since the fence shipped cannot
  follow the runbook as written.
- **Fix:** the seed should accept an existing `legacy` row with a null taxonomy
  version, and fill it under the exclusive fence.

## 2. Retirement criteria

The legacy authority modes and compatibility paths are the **rollback** design.
They may be retired only when all of the following hold, and only through a
separate, explicitly approved change. These criteria turn the runbook's
"Remove legacy Theme/Social storage or compatibility writes" entry into
something checkable.

1. **Gaps closed:** G1–G3 are resolved and released.
2. **Stable economic authority:** every deployment has served in `economic`
   mode for at least **8 consecutive weeks**, covering at least two monthly
   lifecycle and metrics cycles. During that time:
   - no rollback publication;
   - no `rollback_recovery` and no `recovery_failed`;
   - no structural `held` state left unresolved for more than 7 days.
3. **Healthy compatibility delivery throughout the window:**
   `deliver_taxonomy_outbox` stays drained (`failures: 0`) and
   `rollback_state = ready`. This is the evidence that a rollback would still
   have worked if it had been needed.
4. **Zero legacy reads, checked by a machine:** a CI test enumerates API routes,
   Celery tasks and the MCP tools, and fails if any of them reads the legacy
   theme tables without routing through `EconomicThemeReader`. Every entry in
   the reader inventory below must be routed, moved to economic data, or
   removed. *Implemented in #474 as
   `backend/tests/unit/test_legacy_theme_read_gate.py`; its `ALLOWLIST` is the
   enforced form of the inventory and must be empty at retirement.*
5. **A recovery path that doesn't need legacy projections:** a documented and
   rehearsed restore, from a backup or an economic snapshot export, replaces
   "roll back to legacy". It is exercised on a disposable clone, as the
   2026-09-21 rehearsal was.
6. **Retention and approval:**
   - a retention and export policy for legacy Theme and Social history;
   - a rollback-support end date;
   - an explicit irreversible-cleanup approval from the owner.

## 3. Inventory of legacy, shadow and dual dependencies

### Rollback machinery (keep until retirement)

- `AuthorityMode` (`app/domain/economic_taxonomy/contracts.py:42`), the writer
  fence (`app/services/economic_taxonomy_fence.py`), and `lock_authority`'s
  implicit-legacy default.
- `EconomicTaxonomyRuntimeService.legacy_producer_write` and compatibility
  projection staging (`economic_taxonomy_runtime.py:86-140`, `:857-903`).
- `economic_social_taxonomy_adapter.py`,
  `economic_taxonomy_publication_compatibility.py` and
  `economic_taxonomy_rollback_recovery.py`.
- The publication coordinator's legacy target mode
  (`economic_taxonomy_publication.py:192`).

### Writers

| Writer | Allowed modes | In economic mode |
|---|---|---|
| Fenced theme pipeline (`_fenced_legacy_mutation`) | legacy, shadow, dual | Rejected (`AuthorityModeRejected`) |
| Social projection: `apply_live` and `decide` | all | Routes to the economic adapter |
| Economic producers: source admission, migration, processor, operations, work repo | listed per call site (`allowed_modes=`) | Mode-aware |
| Development recording (`theme_development_service.py:155-160`), from beat `prepare_developments` and `POST /themes/developments/backfill` | all, including economic | **Writes legacy `ThemeDevelopmentTheme` links next to economic links** (`:319-328`) in every mode |
| Unfenced legacy writers (G2 table) | not checked | **Keep writing legacy tables** |

### Readers routed by authority (`EconomicThemeReader`)

- `themes_queries.py` (except `/matching/telemetry`, listed below):
  - rankings and emerging switch to economic data;
  - alerts return an empty list;
  - detail, history, mentions, correlation, validate, entrants, similar,
    lifecycle transitions and alert dismissal return 409.
- `themes_taxonomy.py`: every endpoint returns 409.
- `stocks.py:322`, `economic_themes.py`, `economic_taxonomy.py`.
- `digest_service`, `ui_snapshot_service`, `social_confirmation_reader`,
  `social_theme_market_service`, `stock_universe_service`.
- The MCP `market_copilot` theme tools (partly; its alert reads are not
  routed, see below).
- Frontend: `ThemesPageContainer.jsx:114-129` switches on
  `generation.authority_mode`. The legacy review, settings, sources and article
  dialogs render only in the legacy branch (after line 522). This means the
  content-source management UI disappears in economic mode.

### Readers with no routing (they read legacy tables in every mode)

| Reader | What it reads | In economic mode |
|---|---|---|
| `themes_review_merge.py` GETs: merge suggestions, merge history, merge-plan dry run, candidate queue, relationship graph | legacy clusters and suggestions | Serves legacy data (UI hidden; API still reachable). *Since #557 these return 409 (`economic_generation_endpoint_required`) under economic authority.* |
| `themes_intelligence.py` GETs: equivalence preview, history and search; `/{id}/developments` | legacy identities | Same. *Since #557 the equivalence GETs return 409 under economic authority; `/{id}/developments` is to be routed to `EconomicThemeDevelopment`.* |
| `watchlist_stewardship_service.py:328` | `ThemeAlert` | **Serves legacy-derived alerts**: `check_for_alerts` is unfenced (G2), but lifecycle-transition alerts stop because they come from the fenced lifecycle path |
| `validation_service.py:235` (`/validation`, stock validation) | `ThemeAlert`, `ThemeCluster` | Same |
| MCP `market_copilot._recent_alerts` (`market_copilot.py:1495`), used by `market_overview` (`:239`) and a second tool (`:719`) | `ThemeAlert` | Same |
| `GET /themes/matching/telemetry` (`themes_queries.py:409-439`; the module's other endpoints are routed) | `ThemeMention` | Serves legacy matcher statistics. *Since #557 it returns 409 (`economic_generation_endpoint_required`) under economic authority.* |
| `theme_development_worker.discover` (`:43-57`), from beat `theme-development-preparation` every minute (`celery_app.py:541-545`, task `theme_intelligence_tasks.py:16-29`) and `POST /themes/developments/backfill` with `apply=true` (`themes_intelligence.py:177-195`) | `ThemeMention` | Keeps reading legacy mentions; no authority check on either path |
| Content listing mention annotations (`api/v1/themes.py:106-115`) | `ThemeMention` | Serves legacy annotations |
| Social publication preparation: `social_signal_writer.prepare_run()` and `publish()` (`social_signal_writer.py:716`, `:784`) → `SocialThemeProjectionService.prepare_application()` (`social_theme_projection_service.py:153-175`, `:189-226`) | `ThemeCluster`, `ThemeMention`, `ThemeAlias`, `ThemeConstituent`, `SocialThemeAssociation` | **Legacy, shadow and dual modes:** runs for every live Social run, before `apply_live` makes its economic-mode check (`:433`); Social publication in these modes still needs the legacy tables. **Economic authority (since #515):** preparation maps theme keys through the economic catalog, reads accepted economic memberships and reads no legacy theme table; the processor applies automatic acceptance. Retirement is not blocked here once a deployment serves economic authority. |
| One-off `theme_pipeline_state_backfill_service` (`:122-130`; CLI `scripts/backfill_theme_pipeline_state.py`) | `ThemeMention`, to infer status | Writes only shared `ContentItemPipelineState`, so the G2 fence must **not** block it; adapt it before `ThemeMention` is removed |
| **Economic** reader snapshot builder (`economic_taxonomy_snapshot_builder.py:698-717`) | `ThemeDevelopmentTheme`, mapped to economic themes | **The economic side itself depends on legacy development links.** Retirement must migrate these links first |
| `GET /themes/pipeline/state-health`, `/themes/pipeline/observability` (`themes_content_pipeline.py:215-245`, via `theme_pipeline_state_service.py:338-352`, `:430-475`) | `ThemeMention`, `ThemeCluster`, `ThemeMergeSuggestion` | Serves legacy-only diagnostics. *Since #557 these return 409 (`economic_generation_endpoint_required`) under economic authority.* |
| `SocialSignalOperationsService.snapshot` (`social_signal_operations_service.py:106-111`), used by `GET /operations/social-signals` and `GET /social-signals/admin/health` | counts `SocialThemeAssociation` | Counts legacy Social associations |
| `GET /social-signals/admin/associations` (`social_signals.py:422-442`) | `SocialThemeAssociation` joined to `ThemeCluster` | Serves legacy associations; the decision endpoint beside it is mode-aware, but this list is not. *Since #515 the list serves economic associations under economic authority without reading `SocialThemeAssociation`.* |
| `theme_development_preparation`, `theme_platform/content_browser_queries`, `social_refresh_support` | legacy clusters and mentions | Serves legacy data |

`ContentItem` and content-source endpoints are shared ingestion inputs, not
legacy authority. Retirement must keep them.

**Reconciled with the #474 gate.** The gate's allowlist covers every row above
that a route, task or MCP tool reaches, with these differences:
- Also listed: the assistant routes and `POST /mcp/`, which reach the MCP
  `market_overview` alert read; the rollback machinery tasks
  `deliver_taxonomy_outbox` and `process_economic_taxonomy_work` (through the
  Social taxonomy adapter); and the Social association decision endpoint, whose
  economic-mode branch (`_decide_economic`) still reads and revises the bridged
  `SocialThemeAssociation` row, so "mode-aware" above does not mean "no legacy
  read" (*since #515 the legacy-id endpoint refuses under economic authority;
  `POST /social-signals/admin/economic-associations/{id}/decision` decides by
  economic id, and reads bridged legacy rows only to keep their mirror in step*);
  and the daily digest (`/digest/daily`, its markdown variant and the MCP
  `daily_digest` tool), which routes its theme section but builds its
  validation section through `validation_service` in every mode.
- Not listed: the MCP `theme_state` alert read (`market_copilot.py:719`) runs
  only in the legacy branch, after the tool's economic-mode return, so it is
  routed. `theme_pipeline_state_backfill_service` runs only from a CLI script,
  which is not a gate entry point.

### Coverage check

**Legacy authority models:** `ThemeCluster`, `ThemeMention`,
`ThemeConstituent`, `ThemeAlias`, `ThemeMetrics`, `ThemeAlert`,
`ThemeEmbedding`, `ThemeMergeSuggestion`, `ThemeMergeHistory`,
`ThemeRelationship`, `ThemeLifecycleTransition`, `ThemeEquivalenceOperation`,
`ThemeDevelopmentTheme`, `SocialThemeAssociation` and `SocialThemeDecision`.

**Shared ingestion models** (kept at retirement): `ContentSource`,
`ContentItem`, `ContentAttachment`, `ContentItemPipelineState`,
`ThemePipelineRun`, and the Social work, attempt and budget tables.

Every module under `backend/app` that imports a theme, theme-intelligence or
Social-analysis model package was checked. So was every script under
`backend/scripts` that names a legacy model or legacy theme service (the
static checker `check_phase2_type_contracts.py` is out of scope). Modules that use only shared models
are out of scope. The rest are either listed above or fall into one of these
groups:
- **Helpers reached only through the services listed above:**
  `theme_lifecycle_service`, `theme_group_reads`, `theme_group_snapshot`,
  `theme_mention_replacement`, `theme_embedding_service`,
  `theme_development_facts`, `infra/db/repositories/theme_alias_repo` (used by extraction) and
  `api/v1/themes_common`.
- **Model definitions, and historical schema migrations** under
  `app/db_migrations/`.

Every Celery task in `app/tasks/` that reaches a legacy authority model is
classified above. That covers `theme_discovery_tasks`,
`theme_intelligence_tasks` and `live_attachment_tasks`. The remaining theme
tasks write only shared ingestion state (`ingest_content`,
`poll_due_sources`, `prepare_live_attachments`).

This check works at the module and task level. A module listed as routed may
still contain an unrouted endpoint, as `themes_queries` does with
`/matching/telemetry`. Endpoint-level completeness is criterion 4's job.

## Proposed follow-ups

1. **G1:** admit ingested content through `admit_content()`, with lineage shared
   with Social. Also decide where content-source management lives in economic
   mode.
2. **G2:** make the unfenced writers and the fenced tasks mode-aware: skip with a
   reason in Celery and return 409 from the APIs. Add economic-mode tests.
3. **G3:** let the seed adopt an implicit-legacy authority row, and update
   runbook step 2.
4. **Retirement criterion 4:** a CI consumer-inventory gate for legacy reads,
   starting from the tables above.
5. Move the `ThemeAlert` readers (watchlist stewardship, validation, the MCP
   copilot's alert reads) to an
   economic signal, or label them legacy-only. In economic mode they
   currently mix legacy-derived alerts with missing lifecycle alerts.
6. Move development links off `ThemeDevelopmentTheme` before any retirement.
   Today the economic snapshot builder reads them, and development recording
   keeps writing them in every mode, so both sides need to change together.
7. Move Social publication's basket preparation (`prepare_application`) off
   the legacy theme tables. In economic mode only its final apply step
   switches today.
