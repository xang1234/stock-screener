# Company exposure map — execution baseline (Task 00)

Recorded 2026-09-26 while executing `docs/superpowers/plans/2026-09-25-company-exposure-map.md`.

## Approval record

| Artifact | SHA-256 | Status |
|---|---|---|
| Original design | `a182065bf6b1e5e07026045b50e6ae9124501893b383e3a940efb5f9f13f8249` | Approved by the user |
| R1 design | `a5f407209777c6e48b9a58311b090aafebec418ad370184455cdd2182ed526d9` | Superseded by R2 |
| R2 design (`docs/superpowers/specs/2026-09-25-company-exposure-map-design.md`) | `84784bdea4d789c2f7bb0f1e5529054259ab2fa8c66c504f30678efcc8fae2df` | Accepted by the user; implementation authorized to start |
| R2 plan (`docs/superpowers/plans/2026-09-25-company-exposure-map.md`) | `ebbf8349f6480af16ebe19451d682a4911191d67efe35e69b9e7805c10f077ce` | Accepted with the R2 design |

`tests/unit/company_exposure/test_contracts.py` verifies the installed spec hash and that the plan pins it.

## Execution base

- Branch `claude/brave-carson-9i90ws`, based on `main` at `28c220e4e4ca5afcb5a380678bb2b80dd7f388b7`; spec/plan commit `065b52f`.
- Alembic head at start: `20260925_0056` (`20260925_0056_drop_economic_theme_embeddings.py`). No revision numbers are reserved; each migration owner rechecks `alembic heads` at commit.
- Test environment: Python 3.11, dependencies from `requirements-runtime.txt` + `requirements-test.txt` (the CI set). Unit tests use the repository's shared SQLite harness; PostgreSQL tests use `DATABASE_URL=postgresql://ci:ci@localhost:5432/ci STOCKSCANNER_TEST_ALLOW_POSTGRES=1` (a local PostgreSQL 16 in this environment).
- Baseline regression `tests/unit/test_economic_theme_observations.py` and `tests/unit/theme_evaluation/test_kimi_translation.py`: 30 passed.

## Verified reuse seams

| Seam | Observed | Responsibility |
|---|---|---|
| `theme_evaluation/kimi_client.py` `OpenCodeGoKimi` | provider `opencode-go`, model `kimi-k2.6`, `settings.opencode_go_api_key`, endpoint from `opencode_go_endpoint()`; collapses `httpx.TimeoutException` → `model_timeout` and other `httpx.HTTPError` → `model_connection_failed` | Task 04 adds `dispatch_phase` and `complete_json_response` compatibly |
| `theme_evaluation/image_preparation.py` `OpenCodeGoVision` | `policy_version="image-v1"`; tests `tests/unit/theme_evaluation/test_image_preparation.py`, `test_image_stage.py` | Task 08A (not in slice S1) |
| `theme_evaluation/kimi_translation.py` `OpenCodeGoTranslator` | `policy_version="translation-v3"`; tests `test_kimi_translation.py` | Task 08A |
| `theme_evaluation/multilingual_v2.py` | `assess_language`, `segment_text_v2`, `prepare_text_v2`, `preparation_cache_policy`; tests `test_multilingual_v2.py` | Task 07 language identity; Task 08A |
| `theme_evaluation/multilingual_preparation.py` | `detect_language`, `segment_text`, `numerical_warnings`, `prepare_text`, `finalize_translation`; tests `test_multilingual_preparation.py` | Task 08A |
| `theme_evaluation/translation_normalization.py`, `translation_quality.py` | `normalize_text`, `quantity_counter`, `quantity_associations`, `assess_translation`; tests `test_translation_quality.py` | Task 08A |
| `theme_evaluation/quantity_display.py`, `quantity_review.py` | `normalize_quantities`, `normalized_segments`, `render_normalized_segment`; tests `test_quantity_display.py` | Task 07 unit display; Task 08A |
| `theme_evaluation/public_fetch.py` `fetch_public` | per-hop checked public IP, pinned connection, fixed UA, 200-only | Task 06 extracts compatible primitives; public wrapper unchanged |
| `rate_budget_policy.py` / `rate_limiter.py` | `RateBudgetPolicy.provider_key`, `get_rate_interval`; `RedisRateLimiter.wait` falls back to a process-local limiter when Redis is unavailable; `sec_edgar` global interval 0.15 s | Task 06 strict distributed option; no other `sec_edgar` caller exists in `app/` |
| `economic_taxonomy_fence.py` | `producer_write`, `exclusive_publication` | Authoritative research writes join this protocol (later tasks) |
| `social_company_identity_service.py` | `SocialCompanyIdentityService.read/replace`; `replace` owns its transaction | Task 05 import; Task 20 facade |
| `api/v1/config.py` `require_admin` → `AdminPrincipal` | admin key/bearer; actor from configuration | Task 21A authorization |
| `start_celery.sh`, `docker-compose.yml`, `docker-compose.prod.yml` | all Compose workers use `--pool=prefork`; Redis runs without authentication; only the frontend publishes a host port | Task 04 worker; Task 08B broker isolation |

Search adapters: `settings.py` declares `tavily_api_key` and `serper_api_key`; no Tavily or Serper adapter code exists in `app/`. Tavily remains the selected optional adapter (Task 13), disabled by default. No credentials were inspected or printed.

CIK data: no model, service or test references a CIK. Task 09 supplies it from SEC's ticker-to-CIK file (spec §4.2).

## Corpus ownership

- Corpus owner: the user (product owner).
- Independent second reviewer: **not yet assigned**. This is an explicit hold on the automatic-admission gate (Task 25B); it does not block the US verify-only shadow slice.
- US case collection starts with the shadow slice.

## Deployment credentials

Not available in this execution environment and not required for offline tests: OpenCode Go key, SEC identifying User-Agent declaration, EDINET key, Tavily key. Live probes (Task 27) are unexecuted until an operator supplies them.

## Recorded deviations

- **Compose layout (Task 04).** Plan Appendix F.1 places the research worker in `docker-compose.yml` and adds a read-only evidence mount to the `backend` service there. A new bind mount in the base file would make Docker create a root-owned `./data/exposure-evidence` on every deployment, including those that never enable research. The worker, its resource limits and the backend's read-only mount therefore live in the opt-in overlay `docker-compose.exposure.yml` (worker still behind the `exposure-research` profile). Behaviour, queue, limits and narrow environment are as specified.
- **Migration numbering.** `main` added `20260925_0057` after the plan baseline; company-exposure migrations start at `20260925_0058` (see `company-exposure-migrations.json`).
- **SEC reachability (Task 09).** `www.sec.gov` and `data.sec.gov` are blocked by this implementation sandbox's egress proxy (HTTP 403 on CONNECT, 2026-09-26). US fixtures are therefore synthetic, labelled as such, and follow SEC's published response shapes (`tests/fixtures/company_exposure/routes/us.json`). The live SEC route, ticker-file schema and exchange-qualified file availability remain unverified until the opt-in operator probe (Task 27).
- **S1 scope additions.** The research request accepts a US `symbol` as an alternative to `security_id`, and `EXPOSURE_RESEARCH_MODE=live` is refused (`live_mode_not_installed`) because nothing in S1 publishes. The operator CLI implements the S1 subset (`status`, `job`, `resolve-issuer`, `resume`, `process`, `inspect-holds`, `refresh-holds`); `import-identities`, `set-mode`, `set-stage`, `discover`, `prepare-preview`, `probe-market`, `disable-acquisition` and `reconcile-compatibility` belong to later slices (mode and stages come from configuration in S1).

- **Deferred until a caller exists.** An over-engineering pass removed code that S1 never calls: the Social attestation import (`import-identities`), authorized original reads, the blocking-only safety check for automatic use, job pause/cancel, the evidence-DAG check, the paid-search/dispatch predicates, and the non-overlapping sum. Each returns with the task that first calls it (Tasks 13, 18–20, 23–24, 27B). Task 23A brought back storage garbage collection and period close with scheduled callers (below). Cross-listing tests now build links through the reviewed `propose_link`/`apply_link` path, and I09 is covered by generated or classifier output never counting as primary support.

- **Scheduled maintenance (Task 23A).** Celery beat sends three entries to the `exposure_research` queue, each with a message `expires` so nothing piles up while the opt-in worker is not deployed: `company-exposure-work` (every 5 minutes, up to 3 stages; claims nothing while research is disabled), `company-exposure-holds` (hourly; provider-free hold refresh plus closing every ended allocation period, which marks uncertain reservations `expired_uncertain`) and `company-exposure-evidence-gc` (daily at 03:41 in the Celery beat timezone; removes unreferenced originals older than 30 days and temp files older than 24 hours, writing a tombstone for each). The hold refresh and GC run in every research mode because they make no network or model call. GC takes an exclusive PostgreSQL advisory lock; acquisition takes it shared while binding a revision to a blob.

## S1 delivery record (US verify-only shadow)

- Tasks: 00–07, 09, 14–16, 17A, 21A, 22A, 23A, 26A, 27A on branch `claude/brave-carson-9i90ws`.
- Backend S1 gate: `scripts/run_required_company_exposure_postgres.py --slice S1` on PostgreSQL 16 (local disposable cluster) — 283 passed, all required case/layer pairs and named invariants passed.
- Frontend: full Vitest suite 824 passed; `npm run build` succeeded.
- Execution mode: saved synthetic SEC fixtures, mocked SEC transport and a fake subscription transport. **No live SEC probe and no real model call were made**; both remain unverified until an operator with credentials and SEC egress runs them.
- Not installed in S1 (reported `not_installed`): other markets, generation-bound reads, automatic admission, discovery, paid search, classifier grounding, browser rendering.
