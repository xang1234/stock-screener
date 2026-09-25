# Evidence-Backed Company Exposure Map — Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add issuer-centric, claim-level primary-backed exposure research to the live app, supporting verification, bounded candidate discovery, reviewed-role membership additions, and targeted classification grounding without manufacturing source observations.

**Architecture:** Add a dedicated `company_exposure` subsystem inside the existing backend. Original documents, exact claims, assessments, listing decisions, and grounding uses have distinct identities and immutable revisions. Reuse Economic Taxonomy's writer fence, serving generations, snapshots and ordered compatibility delivery; there is no second live exposure pointer or new theme catalog.

**Tech Stack:** Existing Python 3.11/FastAPI/SQLAlchemy/Alembic/Celery/PostgreSQL stack and React/MUI/TanStack Query/Vitest. Add isolated, pinned document-processing dependencies only in the tasks that use them; preserve existing server behavior. Routine tests are offline; PostgreSQL concurrency and bounded four-market live probes are separate gates.

**Spec:** `docs/superpowers/specs/2026-09-25-company-exposure-map-design.md`  
**Paired source artifact:** `2026-09-26-company-exposure-map-design-r2.md` (install at the canonical Spec path above).  
**Paired artifact SHA-256:** `84784bdea4d789c2f7bb0f1e5529054259ab2fa8c66c504f30678efcc8fae2df`  
**Prior artifacts SHA-256:** original approved `a182065bf6b1e5e07026045b50e6ae9124501893b383e3a940efb5f9f13f8249`; R1 `a5f407209777c6e48b9a58311b090aafebec418ad370184455cdd2182ed526d9`.  
**Approval status:** The original written spec was approved. The user accepted the R1 amendments/defaults as corrected by R2's four fixes (see "R2 changes" below). Record that acceptance against the paired R2 hash in Task 00; preserve the original and R1 artifacts in the approval record.  
**Revision:** R2 — 2026-09-26, second implementer-review corrections (R1 — 2026-09-26, first corrections); no product code was changed.

### R2 changes

1. **US CIK sourcing** — Task 05 adds the `official_registry_single_listing` acceptance policy; Task 09 adds `USIssuerResolver` over SEC's ticker-to-CIK file and submissions record; Task 17A resolves the CIK before research. Ambiguity/conflict goes to administrator review (spec §4.2).
2. **One research-job route scheme** — `POST /research-requests`, `GET /research-jobs/{job_id}`, `GET /research-jobs/{job_id}/preview` in Tasks 21A/22A and all tests; the parallel `/research` and `/securities/{id}` paths are removed (spec §16).
3. **Dispatch classification and uncertain expiry** — Task 04 makes `OpenCodeGoKimi` preserve the underlying `httpx` exception so connect-phase failures are pre-dispatch and release their reservation; uncertain reservations expire with their allocation period as `expired_uncertain` (spec §10.2).
4. **Egress broker has no Redis** — Task 08B/Appendix F remove the Redis ACL user, the `exposure_rate_control` network and `redis-acl.sh`; the research worker serves a grant-scoped pacing RPC over a Unix socket and alone touches Redis/PostgreSQL (spec §9.8).
5. **Minor** — Appendix F.1 adds the backend's read-only evidence mount; Task 27's runbook adds the uid 1000 ownership step for `./data/exposure-evidence`.

**Repository baseline inspected:** `xang1234/stock-screener` at `28c220e4e4ca5afcb5a380678bb2b80dd7f388b7`; main rechecked on 2026-09-26. Repository state is not deployment-state verification.  
**Observed latest migration:** `20260925_0056_drop_economic_theme_embeddings.py`, revision `20260925_0056`, parent `20260922_0055`. This is source inspection, not a claim about the production database's applied head.  
**Suggested plan location:** `docs/superpowers/plans/2026-09-25-company-exposure-map.md`  
**Status:** Implementation plan for agent review. No product implementation, database migration, paid-service enablement, live-provider validation, or production activation was performed in producing this artifact.

## Global Constraints

The paired R2 specification is the execution basis; the user accepted it (R1 amendments as corrected by R2), and the original approved document remains part of the approval record. This plan supplies implementation details; it does not reopen the product choices or silently enlarge the scope.

- Verify existing constituents and discover additional candidates only through bounded requested/enabled-theme investigations.
- Launch markets are `US`, `HK`, `JP`, `TW`; languages are English, Japanese, Traditional Chinese and Simplified Chinese. Preserve the original script and original evidence authority.
- Primary-backed verification is per claim. Search snippets, social posts, model output, analyst questions and hosted third-party reports cannot establish primary verification merely by location.
- A research assessment is not a `ClaimAssignment`, `ThemeObservation`, `ThemeMention`, or development event. Research creates none of those to get around existing foreign keys.
- Preserve `EconomicTheme` UUID and `stock_universe.id`. Use verified issuer links, preserve Social administrator attestations, and do not infer common issuer identity from names/tickers.
- Materiality is disclosed, reproducibly calculated, primary-supported qualitative, or unknown. Never reuse `exposure_strength`/confidence as materiality or a calibrated probability.
- Automatic additions require current primary-backed commercial participation and a reviewed theme-specific role policy. Unknown materiality alone is not disqualifying. Review removals, conflicts and overrides.
- Existing source/Social/manual membership origins and their correction semantics survive. Research may change only its own contribution, subject to global reviewed decisions.
- Stale/disputed claims are blocked from new admission and grounding, not silently removed from existing membership. A blocking-only safety check never substitutes newer unpinned evidence.
- `EXPOSURE_RESEARCH_MODE=disabled` and `EXPOSURE_PAID_SEARCH_ENABLED=false` initially. A key alone cannot enable paid search. Do not enable providers or modes in a migration.
- LLM billing mode is `subscription`: use `OpenCodeGoKimi` / `kimi-k2.6` through `settings.opencode_go_api_key`, local ceilings, no inferred remaining balance, no generic LiteLLM or metered fallback.
- Preserve cumulative root-job limits across retries, pause/resume, changed models and child jobs. Dispatched uncertain requests do not receive an assumed refund.
- Provider/network/browser work is outside database locks. Every authoritative research/hold/membership/identity write uses the existing producer fence and lock order.
- Authoritative research product reads use the existing serving generation and sealed bundle. Early authenticated job/revision previews are explicitly non-authoritative `shadow_preview`; they cannot supply live membership or grounding. Historical generations without the extension remain valid and explicitly unavailable for exposure research.
- Reuse source facts as research leads, never bulk-mark them primary-verified. Do not repeat Idea 1's taxonomy migration or recreate its removed embedding cache.
- No directional fundamental scores, modeled exposure percentages, full corporate graph, general asset master, historical backtesting product or legacy cleanup.

### Test examples and local fixtures

The examples below invoke real services/ORM/API boundaries. A named fixture supplies the stated typed input, seeded database rows, fixed clock or injected MockTransport—not a case-ID handler. Each owning task implements its local fixtures alongside its tests, using Appendix A types and actual model fields. Derive expected effects from returned domain objects, database queries and intercepted calls. `pytest.mark.case` permits many independent tests per requirement; no branch on a case ID executes application behavior. Python examples assume `pytest` and the task’s actual imports; frontend examples use Vitest/Testing Library.

## Review Focus

1. An assessment is assembled from several documents, but every individual proposition, period and scope must survive partial refresh and conflicting captures without last-job-wins replacement. Tasks 01, 03, 14–16 own the tests.
2. Subscription attempts and paid-search reservations must be concurrency-safe and recoverable without hidden retries, fictitious dollar usage, or child-job limit resets. Tasks 02, 04, 13, 17 own the tests.
3. A newly held claim cannot authorize an in-flight addition or grounding even before the next generation publishes; existing approved membership and G1 history must remain intact. Tasks 16, 18–19, 24 own the tests.
4. Research/Social/source origin union and compatibility must preserve administrator decisions, avoid synthetic mentions, and resume after a post-publication crash. Tasks 18–20 own the tests.
5. All four markets need real permitted retrieval paths and original-language evidence locators; a mocked PDF or a configured search key is not a live-coverage or verification result. Tasks 06–13, 25–27 own the tests.

## How to execute this plan

Read the approved spec and this plan before implementation. Create an isolated branch/worktree from freshly fetched `origin/main`, not from an unrelated current feature branch. Do not switch or reset the user's existing checkout. Task 00 pins the actual base and reconciles a newer checkout. Each migration owner allocates and rechecks its own revision immediately before commit; no numeric sequence is reserved in advance. Use separate task commits and independent review; approval of this plan does not authorize a production transition or account purchase.

Do not pretend every task is a five-minute feature. Each task is a reviewable deliverable containing several small test/implementation steps. Split a task into additional commits where useful, retaining its interfaces and acceptance ownership. Work in dependency order and keep activation privileges off until the corresponding release gates pass.

## Current code seams verified for this plan

| Existing path / interface | Verified behavior | Required extension, not assumed existing |
|---|---|---|
| `backend/app/models/economic_taxonomy_runtime_evidence.py` / `ThemeConstituentExposure` | Source-claim-linked fact with security, role/kind and strength | Separate research facts and membership origin projection |
| `backend/app/services/economic_theme_observation_service.py` / `_materialize_constituents` | Copies source confidence to strength | Do not rename historical semantics; add materiality in research domain |
| `backend/app/services/social_company_identity_service.py` / `SocialCompanyIdentityService` | Admin-attested config and Social registry audit, own write transaction | Import bridge then one accepted-link facade; never call its existing transaction-owning replacement inside a publication transaction |
| `backend/app/services/economic_taxonomy_fence.py` / `producer_write`, `exclusive_publication` | Shared/exclusive advisory fence, then authority row | Research writes join this protocol, caller owns commit |
| `backend/app/services/economic_taxonomy_publication_preparation.py` / `PublicationPreparer.prepare` | Builds interpretation, metrics, snapshots, projections, generation | Add typed exposure inputs/selections and safety checks without a new coordinator |
| `backend/app/services/economic_taxonomy_snapshot_builder.py` / `build_snapshot_bundle`, `GenerationSnapshotInputs` | Source-derived constituent list and separate Social memberships | Explicit effective-membership union and exposure snapshot entries |
| `backend/app/services/economic_theme_read_service.py` / `EconomicThemeReader` | Serving-generation-bound sealed reads | Research reader extends this authority, not latest-table joins |
| `backend/app/services/theme_evaluation/public_fetch.py` / `fetch_public` | Per-hop public-IP pinning; fixed UA and 200-only response contract | Extract/reuse tested network policy for official UA, permitted authenticated APIs and conditional checks; do not assume it already meets all acquisition requirements |
| `backend/app/services/theme_evaluation/kimi_client.py` / `OpenCodeGoKimi`, `opencode_go_endpoint` | Actual Go HTTP transport; `provider=opencode-go`, `model=kimi-k2.6`, bounded JSON/failures. Collapses every `httpx.TimeoutException` into `model_timeout` and every other `httpx.HTTPError` into `model_connection_failed`, losing the connect-versus-read distinction | Add compatible optional response metadata, preserve the underlying exception class for dispatch classification, and inject allowance-aware dispatch; do not use generic LiteLLM |
| `theme_evaluation/image_preparation.py` / `OpenCodeGoVision`, `prepare_image`, `validate_image` | Existing original-image vision prompt and image limits | Reuse via an injected budgeted JSON client; add only document page locators |
| `theme_evaluation/kimi_translation.py` / `OpenCodeGoTranslator` | Original-script, quantity-token-preserving translation | Reuse; preserve policy version; do not implement another translator |
| `theme_evaluation/multilingual_preparation.py`, `multilingual_v2.py` | Source segmentation, language decision, finalization and warnings | Adapt exact passage IDs and immutable document references around existing preparation |
| `theme_evaluation/translation_normalization.py`, `translation_quality.py`, `quantity_display.py`, `quantity_review.py` | Existing normalization/quality/quantity modules identified for reuse | Task 00 inventories public signatures and tests; extend documented gaps only |
| `rate_budget_policy.py` / `RateBudgetPolicy`, `rate_limiter.py` / `RedisRateLimiter` | Provider keys and distributed request spacing; SEC global local interval 0.15s | Research uses the same keys and a strict distributed mode, not independent provider limiters |
| `backend/start_celery.sh`, `docker-compose.yml`, `docker-compose.prod.yml` | General, data-fetch, market and Social workers | Add `exposure_research` queue/worker and explicit resource limits |
| `backend/app/config/settings.py` / Tavily and Serper key fields | Existing key configuration surfaces | Not proof of actual keys/adapters; optional Tavily adapter reuses configuration and remains disabled |
| `backend/alembic/versions/20260925_0056_drop_economic_theme_embeddings.py` | Drops unused economic-theme cache | Start after the actual execution head; do not recreate this table |

Pinned code references are in Appendix E. File paths marked **Create** below are proposed new code, not discovered installed functionality.

## Implementation choices made here

These are engineering choices within the approved scope, not new user answers:

- Optional search implementation is **Tavily Search**, reusing `settings.tavily_api_key`; this is a configuration/reuse choice, not a benchmarked superiority claim. Task 00 checks for an actually deployed compatible adapter first. Provider remains `none`, paid enablement false, and bounded charge reservation mandatory. Serper is not an automatic fallback. No Brave account is required.
- Use a private content-addressed filesystem store behind a storage protocol in V1. Existing deployed object storage can implement that protocol; no object-storage vendor is required.
- Use `pypdf` for text/structure, `pypdfium2` for selected page rasterization, and an isolated Playwright Chromium renderer. Task 07/08 resolves and locks versions compatible with the repository and records licensing/security checks; code does not silently install packages at runtime. No OCR dependency is added.
- Keep selection identity, mutable draft construction and immutable published output distinct. Appendix A specifies an acyclic manifest/selection build order and historical hash compatibility.
- “Verified” remains primary-backed under the policy, not a factual guarantee or investment recommendation.

## File layout and migration allocation

New domain files live in `backend/app/domain/company_exposure/`; services live in `backend/app/services/company_exposure/`. Separate immutable evidence and assessment models from leased work. Do not append an unrelated research subsystem to `economic_taxonomy_runtime_evidence.py`.

| Module | Responsibility | Owning task |
|---|---|---:|
| `domain/company_exposure/contracts.py`, `policy.py`, `manifest.py` | Typed keys, enums, pure decisions, explicit versioned payloads | 00 |
| `models/company_exposure_identity.py`, `company_exposure_documents.py` | Issuers, accepted-link history, original documents and derivatives | 01 |
| `models/company_exposure_work.py` | Requests, events, candidates, leases, resource usage, reusable artifacts | 02 |
| `models/company_exposure_assessments.py`, `company_exposure_membership.py` | Claims, dossier revisions, role policies, holds, decisions, sealed selections | 03 |
| `services/company_exposure/resources.py`, `providers.py` | Subscription dispatch and optional search spending | 04 |
| `services/company_exposure/issuer_identity.py` | Attestation import, reviewed links, scope and issuer bridge | 05 |
| `services/company_exposure/acquisition.py`, `storage.py`, `network.py` | Bounded permitted retrieval and private originals | 06 |
| `services/company_exposure/preparation.py`, `passages.py` | HTML/PDF structure and precise original-language locators | 07 |
| `services/company_exposure/rendering.py`, `preparation_adapters.py` | Isolated renderer client and thin reuse adapters for existing language/vision modules | 08 |
| `services/company_exposure/markets/us.py`, `hk.py`, `jp.py`, `tw.py`, `base.py` | Four tested acquisition adapters | 09–12 |
| `services/company_exposure/search.py` | Disabled-by-default Tavily link discovery | 13 |
| `services/company_exposure/claims.py`, `synthesis.py` | Claim-level primary qualification, review and evidence dependencies | 14 |
| `services/company_exposure/materiality.py` | Typed measures and reproducible compatible calculations | 15 |
| `services/company_exposure/assessments.py`, `freshness.py` | Multi-document merge, holds, exact revision selection | 16 |
| `services/company_exposure/research.py` | Bounded resumable investigations and enabled-theme expansion | 17 |
| `services/company_exposure/membership.py`, `decisions.py` | Reviewed policies, origin union and automatic addition proposals | 18 |
| `services/company_exposure/publication.py` | Typed extension to the existing generation coordinator | 19 |
| `services/company_exposure/compatibility.py` | Ordered research-only transport contribution, issuer facade | 20 |
| `schemas/company_exposure.py`, `api/v1/company_exposures.py`, service `reads.py` | Product/operational API separation and authenticated controls | 21 |
| `frontend/src/features/companyExposure/` | Dossier, why-included, candidates and settings/review UI | 22 |
| `tasks/company_exposure_tasks.py` | Bounded triggers, local expiry and scheduling | 23 |
| `services/company_exposure/grounding.py` | Relevant generation-pinned context without circular verification | 24 |
| `services/company_exposure/evaluation.py`, scripts and fixtures | Adjudicated four-market evidence gates | 25 |
| PostgreSQL exact-node runner/manifests, CI | Mechanical release validation | 26 |
| Rollout CLI and runbook | Staged activation, smoke probes and recovery | 27 |

### Migration allocation: just in time, per owner

The inspected baseline ends at `20260925_0056`; it is an observation, **not a reserved sequence**. Each owner below creates the migration from the current execution checkout only when its schema changes are ready. Generate a filename with that day’s `YYYYMMDD`, an unused revision identifier and the listed semantic slug. Record the actual filename, revision, parent, checksum and owning task in `docs/implementation/company-exposure-migrations.json`.

| Owner | Migration slug / schema responsibility |
|---|---|
| 01 | `exposure_identity_documents` |
| 02 | `exposure_research_resources` (typed reservations also support blob bytes) |
| 03 | `exposure_claims_assessments` — core claims/dossiers/holds only in the first slice |
| 18 | `exposure_membership_decisions` — deferred membership/role-policy tables formerly front-loaded in 03 |
| 19 | `exposure_generation_extension` — generation input and sealed selection tables |
| 20 | `exposure_origin_compatibility` |
| 24 | `exposure_grounding_uses` |

Before **every migration commit**, fetch/read the current base changes without resetting the worktree, run `alembic heads`, compare actual parent ancestry, and run the migration graph test. On newly competing unreleased migrations, rebase/reparent and rerun upgrade/downgrade tests; if multiple deployed heads exist, produce the repository’s reviewed merge migration rather than silently guessing a parent. Never rename or renumber a migration already released/applied. Repeat this check immediately before merging the branch; a check in Task 00 is not sufficient for Task 24.

```bash
# In the isolated execution worktree; do not run against production.
git fetch origin
cd backend
./venv/bin/alembic heads
./venv/bin/alembic history --verbose
./venv/bin/pytest tests/unit/test_main_migrations.py -q
```

Readiness uses the actual recorded head/ancestry and the tested capability artifact for that slice, not a prewritten feature revision or string ordering of migration numbers.

### Common test conventions

Use **direct service-level tests** with typed inputs/results and narrow injected network/model/clock boundaries. New unit modules live under `backend/tests/unit/company_exposure/`; PostgreSQL modules under `backend/tests/integration/company_exposure/`. `factory.py` constructs valid rows/value objects only. Each test module owns its real service fixtures and relevant source fixtures; a fixture may not return an expected business conclusion or dispatch on an E/I/R case ID.

Tag a direct test with `@pytest.mark.case("R02")`; use repeated markers when a real test exercises more than one requirement. Register the marker centrally and run with `--strict-markers`. One case can have several independent tests across layers; e.g. network, parser and renderer each own their own R14 tests. The manifest maps requirements to **collected tags plus required layers**, not one guessed canonical function name. Unknown tags, missing required coverage, or an applicable tagged test being uncollected/skipped/xfailed fail that slice’s gate. PostgreSQL concurrency still uses exact real node IDs generated from collection and reviewed with the migration/capability artifact.

No universal `CaseResult`, scenario dispatcher or free-form `value/effects` wrapper is introduced. Assert typed service fields, actual database rows and captured boundary calls. API JSON assertions are appropriate at the HTTP contract boundary. The examples below introduce local pytest fixture names; the owning task constructs each named service fixture from its Interfaces block and recorded inputs, never from the expected outputs. Task 25's corpus evaluator runs the actual pipeline through injected provider recordings; its labels are not pipeline responses.

## Task list

---

### Task 00 — Pin the approved contract, execution baseline and test inventory

**Dependencies:** Original written-spec approval and the user's acceptance of the paired R2 spec, recorded against its hash.  
**Spec coverage:** All D01–D18; all E/I/R case IDs; §18 plan requirements.

**Files**
- Add the accepted R2 spec at its recorded hash, retaining the original and R1 artifact hashes in the approval record, at `docs/superpowers/specs/2026-09-25-company-exposure-map-design.md`.
- Add this plan at `docs/superpowers/plans/2026-09-25-company-exposure-map.md`.
- Create `docs/adr/0006-company-exposure-assessments.md` (next numeric ADR at the inspected baseline; recheck for a collision at execution and allocate the next unused number, never overwrite an existing ADR).
- Create `backend/app/domain/company_exposure/__init__.py`, `contracts.py`, `policy.py`, `manifest.py`.
- Create `backend/tests/fixtures/company_exposure/contract_cases.json`, `factory.py` and `backend/tests/unit/company_exposure/conftest.py`.
- Create `backend/tests/unit/company_exposure/test_contracts.py`.
- Create `docs/implementation/company-exposure-baseline.md`.

**Interfaces**
`collect_case_inventory(items) -> CaseInventory` records actual node IDs, case markers, layers and slice IDs. `validate_case_report(report: CaseRunReport, required: set[CaseLayerRequirement]) -> CaseGateDecision` rejects missing/unknown requirements, omitted required tests, skip/xfail/xpass/failure, or wrong execution backend. It does not execute application scenarios. `CaseGateDecision` contains `passed`, `missing_requirements`, `failed_nodeids` and `execution_mode`. Unit fixture factories remain separate from these metadata contracts.

- [ ] **1. Establish a safe execution workspace and inspect the real baseline.** Use the appropriate isolation skill. Fetch origin, create a separate worktree/branch from updated `origin/main`, and record its SHA. Do not overwrite unrelated changes.

```bash
git status --short
git fetch origin
git rev-parse origin/main
git worktree list
# Run in the isolated worktree, with the repository's configured Python environment:
cd backend
./venv/bin/alembic heads
./venv/bin/pytest tests/unit/test_economic_theme_observations.py -q
```

Record the actual Alembic head, model export/fixture conventions, `OpenCodeGoKimi` transport and `opencode_go_api_key` capability, existing issuer admin facade and publication extension points. Inventory `image_preparation.OpenCodeGoVision`, `kimi_translation.OpenCodeGoTranslator`, `multilingual_preparation`, `multilingual_v2`, `translation_normalization`, `translation_quality`, `quantity_display`, `quantity_review`, their current policy versions and actual regression-test paths. Record reuse/extension responsibility for each; do not create a parallel `language.py`. Inspect `rate_budget_policy`, `rate_limiter`, `start_celery.sh`, `CLAUDE.md` and both Compose files. Record Tavily/Serper adapter code actually found separately from key configuration declarations; never inspect/print secret values. Verify every Modify path in this plan exists. A missing dependency is a concrete baseline finding, not permission to recreate Idea 1. Record unresolved deployment credentials as unavailable; do not read or print secrets. Use `python -m pytest` in the repo's actual environment if its documented invocation differs from `./venv/bin/pytest`.

- [ ] **2. Freeze exact decision vocabulary and operational defaults as failing tests.** Copy spec §17's 41 case descriptions verbatim into a fixture manifest. Add independent safety cases for the implementation details in Appendix A. Create input/row factories with deterministic UUIDs and a fixed UTC clock; use httpx.MockTransport and mocks injected at the real provider/network boundary. Record actual calls and persisted effects, never precomputed outcomes.

```python
from app.domain.company_exposure.contracts import ResearchLimits, ResearchMode
from app.domain.company_exposure.policy import may_dispatch_paid_search

def test_default_configuration_does_not_spend():
    limits = ResearchLimits()
    assert limits.research_mode is ResearchMode.DISABLED
    assert limits.paid_search_enabled is False
    assert limits.llm_billing_mode == "subscription"
    assert not may_dispatch_paid_search(enabled=False, cap=None, key_present=True)

def test_manifest_preserves_all_approved_case_ids(contract_manifest):
    expected = ({f"E{i:02}" for i in range(1, 16)}
                | {f"I{i:02}" for i in range(1, 12)}
                | {f"R{i:02}" for i in range(1, 16)})
    assert set(contract_manifest) == expected
```

Run `cd backend && ./venv/bin/pytest tests/unit/company_exposure/test_contracts.py -q`; observe the intended new-contract failure before implementation.

- [ ] **3. Implement pure contracts, not application behavior.** Use frozen dataclasses/Pydantic strict models and string enums. Define exact values from spec §5/§8/§10; do not overload existing taxonomy support enums. Keys and function signatures in Appendix A are the cross-task interface authority. Define JSON serialization of UUID, UTC datetime and Decimal explicitly. Unknown keys fail validation, while unknown domain facts become a typed unresolved claim rather than invented data.

Implement `ResearchLimits` with spec defaults: investigations/provider concurrency 1; queries 6/root job; candidates 10; documents 12/issuer; download 25 MiB; text pages 300; passages 24; image pages 8; browser navigations 4; provider attempts 24 including retries. The outer root budget has a separately explicit total of 24 provider attempts and six searches by default: ten children do not acquire ten fresh allowances. Operator-approved increases are logged policy revisions. Local daily allocation has no guessed default; absent allocation blocks new paid/provider work.

Fixtures may create typed inputs, persisted rows, mocked HTTP responses and concrete service instances. They do not dispatch by requirement ID and do not return expected-outcome summaries. Each test invokes its owning production interface; case markers are traceability metadata only. Freeze the source-vs-research evidence boundary, manifest order, safety invalidation, Social attestation facade switch and compatibility owner keys in the ADR. Do not replace existing taxonomy ADRs.

- [ ] **4. Record version allocation and provider choices.** Record the current head without reserving future revision numbers. Every migration owner repeats the allocation/commit check in the migration protocol. Select Tavily as the optional disabled-by-default adapter because its configuration surface already exists; do not claim credentials or a deployed adapter have been verified. Recheck its official contract and each selected official-market route at execution; record permitted access and credential needs. Do not enable them. Lock actual new parsing/rendering dependency versions when Tasks 07–08 install them, with a retained dependency-resolution report.

- [ ] **4a. Add case-tag collection, not a scenario execution registry.** Create `backend/tests/company_exposure_case_plugin.py` with `pytest_collection_modifyitems` recording `(nodeid, case_id, layer, slice)`, strict unknown-ID/layer validation, and report hooks recording outcomes. Register `case(id)`, `exposure_layer(name)` and `exposure_slice(name)` in the existing pytest configuration. A release command supplies required `(case, layer)` pairs; collection produces real node IDs. Unit fixtures construct services and recorded source rows only. No callback is selected by case ID. The initial unit gate does not require future-market or renderer tests before their owning slices exist; full admission requires the complete reviewed manifest. Add plugin tests for duplicate IDs on different real tests, unknown IDs, missing layers, skipped/xfailed tests and parameterized node IDs.

Record **the user/product owner as corpus owner**; ask that owner to assign an independent human reviewer before label freeze. An unassigned reviewer is an explicit admission-gate hold, not fabricated authorship. Begin US case collection in the shadow slice.

- [ ] **5. Run the contract tests, check the paired accepted spec hash, and commit only these paths.** Expected: all 41 IDs preserved, defaults match approved spec, paired R2 hash matches the accepted artifact; original and R1 hashes remain preserved, no migration/live write/provider call. Commit `test: freeze company exposure implementation contracts`.

---

### Task 01 — Persist issuer links, original documents and immutable evidence locators

**Dependencies:** 00  
**Approved-spec coverage:** §4, §9.3–9.6, §13; E12–E15, I01–I03, I11, R14

**Files and ownership**
- Create `backend/app/models/company_exposure_identity.py`, `company_exposure_documents.py`.
- Modify `backend/app/models/__init__.py` for exports only.
- Create the just-in-time `exposure_identity_documents` migration under `backend/alembic/versions/` (Task 01; record its actual path/revision in the migration manifest).
- Create `backend/tests/unit/company_exposure/test_evidence_models.py` and `backend/tests/integration/company_exposure/test_evidence_schema_postgres.py`.
- Extend real-row factories and add direct, case-tagged tests in this task’s own module.

**Interfaces**
Produces ORM models `ExposureIssuer`, `IssuerIdentifierRevision`, `IssuerSecurityLinkRevision`, `LegacyIssuerAttestationBridge`, `ExposureDocument`, `ExposureDocumentRevision`, `DocumentCaptureEvent`, `DocumentRelationRevision`, `ExposurePassage`, `PassageDerivative`, `EvidenceTombstoneEvent`. Repositories use existing `StockUniverse` and UUID theme IDs; no new security master.

- [ ] **1. Add the failing tests.** Place these direct tests in the named module; construct real service fixtures locally using the declared Interfaces. Tag applicable requirements and add the named counterexamples below.

```python
@pytest.mark.case("E12")
@pytest.mark.exposure_layer("schema")
def test_capture_does_not_change_document_business_date(db_session, retained_document, capture_factory):
    original_date = retained_document.published_at
    capture_factory(document_revision_id=retained_document.id)
    capture_factory(document_revision_id=retained_document.id)
    db_session.flush()
    db_session.refresh(retained_document)
    assert retained_document.published_at == original_date
    assert db_session.query(DocumentCaptureEvent).count() == 2
    assert db_session.query(ExposureDocumentRevision).count() == 1
```

The test module defines the input/service fixtures named here against this task’s Interfaces; factories only build typed inputs and database rows. Any thin test convenience wrapper calls the real service and returns its real result. Do not add a production method solely to satisfy a fixture or substitute a precomputed effect report. Additional assertions in the implementation steps remain required, even when not repeated in this representative block.

- [ ] **2. Run the focused test before implementation.**

```bash
cd backend && ./venv/bin/pytest tests/unit/company_exposure/test_evidence_models.py -q
```

Expected: the new test fails for its missing interface or intended behavior, not an unrelated environment error; existing regressions remain passing.

- [ ] **3. Implement this task's contract.**

Implement the exact table/uniqueness registry in Appendix B, including `UNIQUE(security_id, revision_number)` for issuer-link revisions and `UNIQUE(document_id, content_hash)` for document revisions. Stable issuer provenance contains no mutable name-based identity rule. Identifier schemes are explicit and issuer/country scoped; uniqueness must not make two unrelated local code schemes collide.

Retain raw bytes in the private content-addressed store through a content-hash reference, not an external URL-only record. Captures/checks are append-only and include retrieval outcome, sanitized URL, observed publication metadata and first-app availability. Provider/correction ordering is evidence, not capture-ID order. Immutable relation revisions express correction, supersession, exact mirror and translation identity without deleting old records. An incomplete/old capture is not automatically effective.

Passages bind document revision/hash, extractor version and canonical locator JSON/hash; a locator can include section, physical PDF page, printed label, text offsets, table headers/cells or image bounding box. Keep original text immutable and derivatives separate. No floating table cells without their required context.

Add PostgreSQL triggers and ORM/bulk guards for immutable payloads. Use foreign keys with RESTRICT on retained evidence; no cascade that erases selected provenance. A legally removed blob creates a tombstone with its hash/reason, not an edited historical passage. Model creation itself must not fetch a document or create an issuer attestation.

```python
# Required logical constraint shapes (actual tables include UUID IDs and provenance).
ISSUER_LINK_UNIQUE = ("security_id", "revision_number")
DOCUMENT_REVISION_UNIQUE = ("document_id", "content_hash")
PASSAGE_UNIQUE = ("document_revision_id", "preparation_policy", "locator_hash")
DERIVATIVE_UNIQUE = ("passage_id", "kind", "policy_version", "model_identity", "input_hash")
```


- [ ] **4. Run the focused command again, then the additional checks.**

Run SQLite upgrade/downgrade in the repository migration harness and the PostgreSQL schema test. Verify raw SQL and ORM updates to immutable rows both fail, a revision can reference only its own document, links can retain accepted/proposed history, and timestamps remain timezone-aware. Run `tests/unit/test_main_migrations.py`. Schema downgrade tests are disposable-only; production rollback retains evidence.

Expected: focused tests/regressions pass. Record actual execution mode; mocked retrieval is not a live probe.

- [ ] **5. Review the diff, stage only this task's files, and commit.**

```bash
git diff --check
git diff --stat
# Stage the exact changed paths from this task after reviewing them.
git commit -m "feat: persist company exposure issuer and document evidence"
```



---

### Task 02 — Persist resumable work, attempts and typed resource accounting

**Dependencies:** 00, 01  
**Approved-spec coverage:** §8, §10, §13; R02–R06

**Files and ownership**
- Create `backend/app/models/company_exposure_work.py` and `backend/app/infra/db/repositories/company_exposure_work_repo.py`.
- Modify model exports.
- Create the just-in-time `exposure_research_resources` migration under `backend/alembic/versions/` (Task 02; record its actual path/revision in the migration manifest).
- Create `backend/tests/unit/company_exposure/test_work_models.py`, `test_work_repo.py` and integration `test_work_resources_postgres.py`.

**Interfaces**
Produces `ExposureRuntimePolicyRevision`, `ExposureResearchRequest`, `ResearchEvent`, `ResearchCandidate`, `ResearchWorkLease`, `ResearchInputManifest`, `ResearchReservation`, `ResearchReservationEvent`, `ResearchProviderAttempt`, `ResearchProviderResult`, `ResearchArtifact`, `ResearchCoverageItem` and a work repository with `enqueue`, `claim_next`, `heartbeat`, `pause`, `cancel`, `complete_step`. Request key is immutable; lease/counters are operational only.

- [ ] **1. Add the failing tests.** Place these direct tests in the named module; construct real service fixtures locally using the declared Interfaces. Tag applicable requirements and add the named counterexamples below.

```python
@pytest.mark.case("R05")
@pytest.mark.exposure_layer("schema")
def test_children_reference_one_root_budget(db_session, root_request, child_request_factory):
    children = [child_request_factory(root_request=root_request) for _ in range(10)]
    db_session.flush()
    assert {child.root_request_id for child in children} == {root_request.id}
    assert root_request.limits.provider_attempts == 24

@pytest.mark.case("R04")
@pytest.mark.exposure_layer("unit")
def test_cancellation_keeps_dispatched_attempt_immutable(work_repo, dispatched_work, db_session):
    attempt_id = dispatched_work.provider_attempt_id
    work_repo.cancel(dispatched_work.request_id, reason="user_cancelled")
    db_session.expire_all()
    assert db_session.get(ResearchProviderAttempt, attempt_id) is not None
    assert dispatched_work.reservation_state() == "uncertain"
```

The test module defines the input/service fixtures named here against this task’s Interfaces; factories only build typed inputs and database rows. Any thin test convenience wrapper calls the real service and returns its real result. Do not add a production method solely to satisfy a fixture or substitute a precomputed effect report. Additional assertions in the implementation steps remain required, even when not repeated in this representative block.

- [ ] **2. Run the focused test before implementation.**

```bash
cd backend && ./venv/bin/pytest tests/unit/company_exposure/test_work_models.py tests/unit/company_exposure/test_work_repo.py -q
```

Expected: the new test fails for its missing interface or intended behavior, not an unrelated environment error; existing regressions remain passing.

- [ ] **3. Implement this task's contract.**

`ExposureRuntimePolicyRevision` persists settings, allowance allocations, route capabilities/permissions, theme-research enablement and search costing policy. It is unique by `(policy_namespace, subject_key, revision_number)`, with immutable payload/hash, parent revision, trusted principal, reason and approval state referenced by append-only decision events. Policy namespaces are `research_limits`, `subscription_route`, `search_cost`, `acquisition_permission`, `theme_research` and `feature_stage`. Every request/attempt pins the policies it used. Secrets remain external references. Operational enablement may stop or authorize future queued work only through the relevant release gate; it cannot rewrite a selected assessment or role policy.

Requests are immutable envelopes for verify/refresh/discover, with `root_request_id`, optional parent, requester principal, requested issuer/theme, trigger signature, enabled-theme policy and accepted budget-policy revision. A new intentional investigation gets a new request ID; duplicate handling of the same trigger/idempotency token reuses it. Jobs cannot turn a fresh HTTP retry into a new allowance allocation. Snapshot the current semantic/issuer inputs per assessment attempt, not into a mutable all-purpose request result.

Append events for queued/researching/evidence_ready/assessing/ready_for_publication, partial, holds, pauses, retries, cancelled and published-generation reference. Distinguish budget exhaustion from failure and complete-empty acquisition from business exit. Keep candidate origin, original seed reference, verification status and no-source-found coverage durable.

Store typed reservations with resource unit, configured pool, root owner, period, reserved maximum, known actual/unknown flag, cost currency where applicable, policy and attempt ID. Lock resource-pool rows in canonical order then root budget rows for compare-and-reserve; never dispatch while holding a database transaction. All provider result events are append-only. Successful artifact uniqueness contains input hash, operation, all relevant policies and actual route/model; failed attempts are not artifacts.

```python
# Leases are fencing tokens, not proof a provider request did not execute.
WORK_KEY = ("request_id", "stage", "input_hash", "policy_bundle_version")
SUCCESS_ARTIFACT_KEY = ("operation", "input_hash", "policy_hash", "model_identity")
ATTEMPT_KEY = ("logical_operation_key", "attempt_number")
```

Claim bounded work with SKIP LOCKED; use a UUID lease and five-minute expiry with heartbeat for approved longer operations. Before completion verify the token and current source/assessment inputs. Lease loss after dispatch preserves uncertain attempt state. Do not reset root usage on a new lease. Seal input manifests; stage transitions cannot mutate prior frozen inputs.

- [ ] **4. Run the focused command again, then the additional checks.**

Run real PostgreSQL contention tests with two workers requesting the last allowance unit, stale leases, cancellation and late provider results. Require one dispatch, truthful usage and no duplicate successful artifact. Reuse existing fence for authoritative completion; document when quota-only short reservations use their own ordered resource locks and never enter the taxonomy fence in reverse order.

Expected: focused tests/regressions pass. Record actual execution mode; mocked retrieval is not a live probe.

- [ ] **5. Review the diff, stage only this task's files, and commit.**

```bash
git diff --check
git diff --stat
# Stage the exact changed paths from this task after reviewing them.
git commit -m "feat: add durable exposure research work and resource history"
```



---

### Task 03 — Persist claim-level assessments and safety history

**Dependencies:** 00–02  
**Approved-spec coverage:** §5–7, §11, §13–14; E04–E06,E15,I04–I05,I10–I11,R07,R11–R12

**Files and ownership**
- Create `backend/app/models/company_exposure_assessments.py` and model invariant helpers. Membership models are allocated by Task 18; serving-selection models by Task 19.
- Modify model exports; create the just-in-time `exposure_claims_assessments` migration under `backend/alembic/versions/` (Task 03; record its actual path/revision in the migration manifest).
- Create unit `test_assessment_models.py` and integration `test_assessment_schema_postgres.py`.

**Interfaces**
Produces the core assessment Appendix B objects: `ExposureClaim`, `ExposureClaimRevision`, `ClaimEvidenceLink`, `MaterialityMeasure`, `IssuerThemeAssessment`, `AssessmentRevision`, `AssessmentClaimSelection` and `ExposureUseHoldRevision`. Task 18 owns role policies and membership/review records; Task 19 owns `ExposureSelectionSet` and selected-input rows; Task 24 owns grounding uses. Do not create optional later-stage tables just to make a preallocated migration sequence convenient.

- [ ] **1. Add the failing tests.** Place these direct tests in the named module; construct real service fixtures locally using the declared Interfaces. Tag applicable requirements and add the named counterexamples below.

```python
@pytest.mark.case("E15")
@pytest.mark.exposure_layer("schema")
def test_dossier_revision_can_select_old_role_and_new_measure(db_session, assessment_rows):
    # Fixture builds real claim/selection rows with different source periods.
    db_session.add_all(assessment_rows.new_revision_rows)
    db_session.flush()
    assert {row.claim_revision_id for row in assessment_rows.new_selections} == {
        assessment_rows.old_role.id, assessment_rows.new_materiality.id,
    }
    assert assessment_rows.old_role.supported_as_of == assessment_rows.original_role_date
```

The test module defines the input/service fixtures named here against this task’s Interfaces; factories only build typed inputs and database rows. Any thin test convenience wrapper calls the real service and returns its real result. Do not add a production method solely to satisfy a fixture or substitute a precomputed effect report. Additional assertions in the implementation steps remain required, even when not repeated in this representative block.

- [ ] **2. Run the focused test before implementation.**

```bash
cd backend && ./venv/bin/pytest tests/unit/company_exposure/test_assessment_models.py -q
```

Expected: the new test fails for its missing interface or intended behavior, not an unrelated environment error; existing regressions remain passing.

- [ ] **3. Implement this task's contract.**

Use stable proposition identity for issuer/theme/kind/product-or-activity/reporting-scope. A materially different proposition is a new claim; revisions retain explicit correction/supersession relationships. Store all separate axes from spec §5. Decimal materiality values serialize as strings and are never floats.

One stable dossier is unique by `(issuer_id, economic_theme_id)`. Assessment revision is unique by `(assessment_id, revision_number)` and independently by `(assessment_id, input_manifest_hash)` for idempotent replay. `input_manifest_hash` includes selected document/claim revisions, issuer/theme/policy fingerprints and prior selection, not job completion time. Selection rows cannot silently cross issuer or theme scopes.

Assessment revision identity is `(assessment_id, revision_number)`; replay identity is `(assessment_id, input_manifest_hash)`. `ExposureUseHoldRevision` has append-only subject-scoped apply/lift records, accountable actor/reason, and the evidence/review revision permitting a lift. A disputed customer relationship must not hold unrelated roles.

Role-policy, membership-association/decision, and reviewed-operation persistence is owned by Task 18. Serving-selection persistence is owned by Task 19. Their tables/tests are not created in this core migration. This keeps the first shadow slice useful without speculative later-stage schema.

Enforce immutable payloads in PostgreSQL and ORM, including evidence-edge children after their parent revision is sealed. Guard same-scope references and DAG validation at seal. Selection identity/status transitions must be explicit, never pair-only history collisions.

- [ ] **4. Run the focused command again, then the additional checks.**

Verify duplicate replay returns the identical revision, concurrent dossier revisions cannot both claim the same number, successful partial assembly retains old valid claims, scope conflicts fail, and raw SQL cannot mutate any selected sealed payload. Migration includes all model registration and rollback-harness checks.

Expected: focused tests/regressions pass. Record actual execution mode; mocked retrieval is not a live probe.

- [ ] **5. Review the diff, stage only this task's files, and commit.**

```bash
git diff --check
git diff --stat
# Stage the exact changed paths from this task after reviewing them.
git commit -m "feat: persist exposure assessments and safety revisions"
```



---

### Task 04 — Enforce subscription allowance and one-dispatch provider accounting

**Dependencies:** 02 (and its transitive contracts); no dependency on Task 03, any market adapter, or search
**Approved-spec coverage:** D12–D13; §8.3, §10; R01–R05,R13

**Files and ownership**
- Create `services/company_exposure/resources.py`, `providers.py`, `config.py` under `backend/app/`.
- Reuse/extend `backend/app/services/theme_evaluation/kimi_client.py` compatibly; do not change generic `LLMService` behavior or funding semantics.
- Modify `backend/app/celery_app.py`, `backend/start_celery.sh`, `docker-compose.yml`, `docker-compose.prod.yml`, `CLAUDE.md`; add dedicated worker bootstrap from Appendix F. Task 23 later adds automatic schedules.
- Add `backend/app/tasks/company_exposure_tasks.py` with a bounded `process_exposure_work` entrypoint initially returning `stage_not_installed` until Task 17A supplies its implementation; it must not accept work in that state. Create the worker-route contract test now.
- Create unit `test_resources.py`, `test_provider_attempts.py`, `test_config.py`; integration `test_resource_reservations_postgres.py`.
- Update relevant environment examples/documentation with disabled defaults, no secrets.

**Interfaces**
`ResearchResources.reserve(DispatchRequest) -> ReservationTicket`; `ResearchResources.finish(ticket, DispatchOutcome) -> None`; `.read(ticket_id) -> ReservationUsage`; `.close_period(allocation_id, period) -> PeriodCloseReport` (R2: expires that period's uncertain reservations); `.available(allocation_id, period) -> ResourceAmounts`; `SubscriptionProvider.call_once(ProviderInput, ticket) -> ProviderOutput`. Add `SubscriptionArtifactRunner.run(provider_input) -> ArtifactRunResult` to coordinate cache lookup, one-dispatch reservation and result persistence; result fields are `artifact_id`, `ticket_id`, `retryable`, `pause_reason`. Tests instantiate this real runner, not a case executor. Exact types in Appendix A. The wrapper uses the permitted configured LLM route, not a new provider account. Request/token allocations and currency charges are distinct.

- [ ] **1. Add the failing tests.** Place these direct tests in the named module; construct real service fixtures locally using the declared Interfaces. Tag applicable requirements and add the named counterexamples below.

```python
@pytest.mark.case("R02")
@pytest.mark.exposure_layer("unit")
def test_subscription_pause_never_falls_back_to_metered(resources, exhausted_dispatch, go_transport, metered_spy):
    ticket = resources.reserve(exhausted_dispatch)
    assert ticket.state == "paused_allowance"
    assert go_transport.requests == []
    metered_spy.assert_not_called()

@pytest.mark.case("R03")
@pytest.mark.exposure_layer("unit")
def test_retry_records_two_go_attempts_and_reuses_success(subscription_runner, provider_input, go_transport, db_session):
    go_transport.queue_status(503)
    go_transport.queue_json({"claims": []})
    first = subscription_runner.run(provider_input)
    assert first.retryable is True
    success = subscription_runner.run(provider_input)
    repeated = subscription_runner.run(provider_input)
    assert repeated.artifact_id == success.artifact_id
    assert len(go_transport.requests) == 2
    assert db_session.query(ResearchProviderAttempt).count() == 2
    assert {request.json["model"] for request in go_transport.requests} == {"kimi-k2.6"}
    assert all(request.url.endswith("/chat/completions") for request in go_transport.requests)

@pytest.mark.case("R04")
@pytest.mark.exposure_layer("unit")
def test_ambiguous_timeout_is_not_refunded(resources, dispatched_ticket, uncertain_timeout):
    resources.finish(dispatched_ticket, uncertain_timeout)
    assert resources.read(dispatched_ticket.id).state == "uncertain"
    assert resources.read(dispatched_ticket.id).actual_dollar_cost is None

@pytest.mark.case("R04")
@pytest.mark.exposure_layer("unit")
@pytest.mark.parametrize("error", [httpx.ConnectError("refused"), httpx.ConnectTimeout("connect"), httpx.PoolTimeout("pool")])
def test_connect_phase_failure_is_pre_dispatch_and_released(subscription_runner, provider_input, go_transport, resources, error):
    go_transport.queue_exception(error)
    result = subscription_runner.run(provider_input)
    usage = resources.read(result.ticket_id)
    assert usage.dispatch_state == "pre_dispatch"
    assert usage.state == "released"

@pytest.mark.case("R04")
@pytest.mark.exposure_layer("unit")
@pytest.mark.parametrize("error", [httpx.ReadTimeout("read"), httpx.WriteTimeout("write"), httpx.ReadError("reset"), httpx.RemoteProtocolError("eof")])
def test_post_send_failure_stays_uncertain(subscription_runner, provider_input, go_transport, resources, error):
    go_transport.queue_exception(error)
    result = subscription_runner.run(provider_input)
    assert resources.read(result.ticket_id).state == "uncertain"

@pytest.mark.case("R04")
@pytest.mark.exposure_layer("unit")
def test_uncertain_reservation_expires_with_its_period(resources, dispatched_ticket, uncertain_timeout, clock):
    resources.finish(dispatched_ticket, uncertain_timeout)
    clock.advance_to(dispatched_ticket.period_end)
    resources.close_period(dispatched_ticket.allocation_id, dispatched_ticket.period)
    usage = resources.read(dispatched_ticket.id)
    assert usage.state == "expired_uncertain"
    assert resources.available(dispatched_ticket.allocation_id, period=clock.next_period()).requests == resources.limit(dispatched_ticket.allocation_id).requests
```

The test module defines the input/service fixtures named here against this task’s Interfaces; factories only build typed inputs and database rows. Any thin test convenience wrapper calls the real service and returns its real result. Do not add a production method solely to satisfy a fixture or substitute a precomputed effect report. Additional assertions in the implementation steps remain required, even when not repeated in this representative block.

- [ ] **2. Run the focused test before implementation.**

```bash
cd backend && ./venv/bin/pytest tests/unit/company_exposure/test_resources.py tests/unit/company_exposure/test_provider_attempts.py tests/unit/company_exposure/test_config.py -q
```

Expected: the new test fails for its missing interface or intended behavior, not an unrelated environment error; existing regressions remain passing.

- [ ] **3. Implement this task's contract.**

Bind funding mode to an operator-approved route configuration. A subscription claim from a model or caller is not entitlement verification. Missing route capabilities, local daily request/token ceilings, or permitted-use configuration stops provider dispatch with a typed reason. Text and vision routes are independently capability-checked; both must remain allowed subscription routes. No fallback to another model unless it is explicitly included in the same approved subscription policy and treated as a new accounted attempt; V1 defaults to no fallback.

Use **the actual Go transport**: `OpenCodeGoKimi` from `theme_evaluation/kimi_client.py`, `model="kimi-k2.6"`, `provider="opencode-go"`, configured by `settings.opencode_go_api_key` and `opencode_go_endpoint()`. Do not call `LLMService.completion`, `EconomicTaxonomyLLMProvider` or Social's dollar-ledger wrapper for this feature. A `metered` keyword is not subscription entitlement, and a model name alone is not an approved route.

Add a compatible `complete_json_response(messages, *, max_tokens, read_timeout=20.0) -> KimiJSONResponse` method to the existing client. `KimiJSONResponse` contains validated JSON data, provider request ID, reported token usage or unknown, bounded response hash and completion metadata. Existing `complete_json()` remains a dictionary-returning wrapper with unchanged validation/failure behavior; existing vision/translation callers must pass their regression tests. Keep its actual HTTP streaming implementation, no hidden SDK/provider retry loop. Expose a narrow injected `JSONCompletionClient` for reuse adapters; no generic provider router.

`SubscriptionProvider.call_once(provider_input, ticket)` checks approved route/model/capability and exact ticket, invokes that client once, and maps `PreparationFailure` to typed outcomes. Missing credentials is pre-dispatch. Once dispatch may have occurred, timeout/connection ambiguity or quota rejection does not create an assumed zero-usage/zero-dollar result. Preserve `Retry-After`; account-specific limits and reset balance remain unknown unless actually reported. The client session ID is provenance, not a claimed provider idempotency guarantee.

**Dispatch classification (R2, spec §10.2).** The existing client collapses `httpx.TimeoutException` into `model_timeout` and every other `httpx.HTTPError` into `model_connection_failed`. Compatibly extend `PreparationFailure` with an optional `dispatch_phase` attribute (`pre_dispatch` | `dispatched` | `uncertain`) and set it inside the client from the actual exception class; existing callers that only read `code` are unaffected and keep passing their regression tests. Classify:

| Exception / outcome | `dispatch_phase` | Reservation |
|---|---|---|
| Missing key/route; `httpx.ConnectError` (DNS failure, refused, unreachable, TLS handshake failure); `httpx.ConnectTimeout`; `httpx.PoolTimeout` | `pre_dispatch` | released |
| HTTP response with any status (`2xx` with invalid body, `429`, `4xx`, `5xx`) | `dispatched` | kept; usage from response if reported, otherwise unknown |
| `httpx.WriteTimeout`, `httpx.ReadTimeout`, `httpx.WriteError`, `httpx.ReadError`, `httpx.RemoteProtocolError`, response-size abort | `uncertain` | kept as uncertain |

Do not infer the phase from the collapsed `code` string. Add unit tests (new `backend/tests/unit/theme_evaluation/test_kimi_client_dispatch_phase.py`, alongside the existing `test_kimi_translation.py` and `test_extraction_runtime.py`) proving each exception class maps to its phase and that `complete_json()` still raises the same `code` values as before.

**Uncertain expiry (R2).** OpenCode Go exposes no reconciliation source beyond usage in a successful response. Uncertain reservations are charged to the allocation period in which they were dispatched. `close_period` runs from the provider-free maintenance task (Task 23; until then, on the first reservation of the next period), moves every still-uncertain reservation of the closed period to terminal `expired_uncertain`, and never refunds or carries it forward. The next period starts from its full configured allocation.

```python
client = OpenCodeGoKimi(
    api_key=settings.opencode_go_api_key,
    session_id=str(ticket.attempt_id),
    transport=transport,  # injected MockTransport in tests; normal HTTP in deployment
)
response = client.complete_json_response(
    provider_input.messages,
    max_tokens=provider_input.max_output_tokens,
    read_timeout=provider_input.read_timeout_seconds,
)
# Persist response.reported_usage or unknown; never infer dollar cost from tokens.
```

Create `ResearchJSONClient` implementing `complete_json` for the existing translation/vision adapters. It obtains and reconciles a fresh reservation for every actual segment/image call via `SubscriptionProvider`; it must not reserve once for a multi-call preparation batch. Dependency injection into the existing adapters preserves their prompt/normalization policies and default behavior for non-research callers.

Reserve a maximum before dispatch. Mark dispatched durably, perform I/O with no transaction held, append the immutable result, then reconcile actual reported usage. Unknown dispatched consumption stays reserved/uncertain until reported usage or period expiry. A `pre_dispatch` failure (classification table above) releases its bound; an ambiguous timeout/cancel cannot. A permitted retry has a new attempt number under the same logical operation and root quota. Success reuse is keyed by actual input, policy and model identity; retryable failures never occupy a success key.

Protect API keys, query terms marked sensitive and original documents from unbounded debug logs. Retain hashes, provider request IDs and bounded redacted diagnostic metadata. Reserve root and daily ceilings atomically; configured limits are capacity, not claims of remaining provider allowance. Daily timezone comes from app configuration and is recorded with the allocation period.

- [ ] **3a. Bring up only the dedicated worker infrastructure.** Register `exposure_research` and implement Appendix F.1's opt-in worker/profile, 1 CPU/2 GiB cap, one slot/prefetch one, no subscription to price queues, and narrow environment. Do not enable research or add scheduled acquisition here. `start_celery.sh` launches the worker only when `EXPOSURE_WORKER_ENABLED=true`. Update `CLAUDE.md` with the intentional queue exception while retaining provider-wide rate-budget requirements. Queue tests must prove all new research tasks use this queue and existing `data_fetch_*` consumers remain unchanged.

- [ ] **4. Run the focused command again, then the additional checks.**

Run PostgreSQL last-unit contention/cancel/retry tests. Test that credentials with disabled search do not call any transport; daily limit absence is a no-dispatch outcome; unknown usage cannot be zeroed by cancelling. Run existing LLM-service and Social-budget contract tests discovered in Task 00 without altering their funding semantics.

Expected: focused tests/regressions pass. Record actual execution mode; mocked retrieval is not a live probe.

- [ ] **5. Review the diff, stage only this task's files, and commit.**

```bash
git diff --check
git diff --stat
# Stage the exact changed paths from this task after reviewing them.
git commit -m "feat: enforce subscription research resource limits"
```



---

### Task 05 — Import issuer attestations and resolve listing-specific research scope

**Dependencies:** 01,02; no LLM call is required to import existing administrator attestations
**Approved-spec coverage:** D10; §4, §13; I01–I03,I10,R13

**Files and ownership**
- Create `backend/app/services/company_exposure/issuer_identity.py` and repository `company_exposure_identity_repo.py`.
- Read/integrate `backend/app/services/social_company_identity_service.py` and `security_master_service.py`; defer facade write cutover to Task 20.
- Create unit `test_issuer_identity.py`, integration `test_issuer_links_postgres.py`.
- Add admin-import support to the future rollout CLI contract, not an automatic migration side effect.

**Interfaces**
`IssuerIdentityAdapter.resolve_security(security_id, selection=None) -> IssuerResolution`; `import_attestations(snapshot, principal) -> ImportReport`; `propose_link(LinkProposal) -> ProposalRef`; `apply_link(preview_id, principal, expected_hash) -> IssuerLinkRef`; `accept_registry_match(match: RegistryMatch, service_principal) -> IssuerLinkRef | ProposalRef` (R2). Snapshot read is generation-pinned after publication.

- [ ] **1. Add the failing tests.** Place these direct tests in the named module; construct real service fixtures locally using the declared Interfaces. Tag applicable requirements and add the named counterexamples below.

```python
@pytest.mark.case("I01")
@pytest.mark.exposure_layer("unit")
def test_verified_cross_listings_resolve_one_issuer(identity_adapter, verified_cross_listings):
    left = identity_adapter.resolve_security(verified_cross_listings.left_security_id)
    right = identity_adapter.resolve_security(verified_cross_listings.right_security_id)
    assert left.issuer_id == right.issuer_id
    assert left.security_id != right.security_id

@pytest.mark.case("I02")
@pytest.mark.exposure_layer("unit")
def test_conflicting_link_stays_a_proposal(identity_adapter, conflicting_link, attestation_reader):
    before = attestation_reader.read()
    proposal = identity_adapter.propose_link(conflicting_link)
    assert proposal.state == "review_required"
    assert attestation_reader.read() == before

@pytest.mark.case("I02")
@pytest.mark.exposure_layer("unit")
def test_unambiguous_registry_cik_is_accepted_for_single_listing(identity_adapter, service_principal, single_us_match):
    ref = identity_adapter.accept_registry_match(single_us_match, service_principal)
    link = identity_adapter.resolve_security(single_us_match.security_id)
    assert ref.state == "accepted"
    assert ref.acceptance_policy == "official_registry_single_listing"
    assert link.identifiers[("US", "cik")] == single_us_match.cik

@pytest.mark.case("I02")
@pytest.mark.exposure_layer("unit")
@pytest.mark.parametrize("variant", ["multiple_ciks", "ticker_not_in_submissions", "cik_linked_to_other_issuer", "security_already_linked_elsewhere", "cross_listed_issuer", "ticker_changed_since_prior_link"])
def test_ambiguous_registry_cik_requires_review(identity_adapter, service_principal, registry_match_variant, variant):
    ref = identity_adapter.accept_registry_match(registry_match_variant(variant), service_principal)
    assert ref.state == "review_required"
```

The test module defines the input/service fixtures named here against this task’s Interfaces; factories only build typed inputs and database rows. Any thin test convenience wrapper calls the real service and returns its real result. Do not add a production method solely to satisfy a fixture or substitute a precomputed effect report. Additional assertions in the implementation steps remain required, even when not repeated in this representative block.

- [ ] **2. Run the focused test before implementation.**

```bash
cd backend && ./venv/bin/pytest tests/unit/company_exposure/test_issuer_identity.py -q
```

Expected: the new test fails for its missing interface or intended behavior, not an unrelated environment error; existing regressions remain passing.

- [ ] **3. Implement this task's contract.**

Import the exact existing `social_company_identities` configuration and its policy/version/verification references. Preserve original `company_id` as a namespace alias/bridge, not a coerced UUID. Idempotency is original configuration version plus canonical payload hash; a repeated import does not create new issuer/link history. New shared issuer links require retained official identity evidence and administrator acceptance. Model or string similarity may propose a link only.

Resolve security through existing StockUniverse/market normalization and typed identifiers. Do not insert unknown securities into the universe. A supported market with unresolved issuer identity may have an isolated candidate dossier, with no automatic cross-listing sharing or membership admission. Parent/subsidiary scope remains an attributed relationship, not the same issuer by default. An ADR issuer/depositary receipt relationship must be evidenced; do not infer identity from ticker stems.

Link revision snapshots must contain enough legal name/identifier/corporate-action scope to reproduce historical company counts without rereading mutable current StockUniverse names. Securities remain separately eligible for basket rules. Cross-listings count as one verified issuer in new research coverage only; existing weight policy stays unchanged.

```python
# Collision rules, independent of the display name.
assert normalized_identifier("US", "cik", "0000123456") != normalized_identifier(
    "TW", "company_code", "123456")
# `normalized_identifier` is implemented here and returns a typed key tuple.
```

**Registry-resolved single-listing links (R2, spec §4.2).** `accept_registry_match` takes a `RegistryMatch` produced by a market resolver (Task 09 supplies the US one). It accepts automatically, as the service principal, only when **all** hold: the security is a supported active `stock_universe` listing; the registry file yielded exactly one identifier candidate for its ticker/exchange; the identifier's own official record (for US, the SEC submissions record) lists the same ticker; no accepted link or identifier for that security or identifier names a different issuer; neither the security nor the issuer participates in a shared-issuer/cross-listing mapping; and no prior link revision for the security shows a different ticker or identifier. Otherwise it returns a `review_required` proposal carrying the failed condition. The accepted revision records `acceptance_policy="official_registry_single_listing"`, the retained registry and official-record capture revision IDs, entity title, matched ticker/exchange and resolver policy version. Names never participate. The service principal can create only this link kind; it cannot accept cross-listing links, override an administrator link or lift a review. A second listing joining that issuer goes through `propose_link`/`apply_link` with administrator acceptance. An administrator-supplied CIK in a research request is a `LinkProposal` through the same reviewed path.

A correction, merger, disposal, split or ticker reuse is a new proposed/accepted link revision with retained prior state. Research cannot overwrite an administrator link. On affected changes, queue assessment compatibility checks and blocking-only holds before further automatic use. Use the existing fence before new authoritative mapping commit, never nest the old transaction-owning Social replacement inside it.

- [ ] **4. Run the focused command again, then the additional checks.**

Assert old configuration bytes/audit rows survive, same configuration import is idempotent, similar company names stay distinct, multi-listing counts and issuer-scoped segment claims remain correct. Race a reviewed link change with assessment commit: old-input work remains audit-only until reevaluated.

Expected: focused tests/regressions pass. Record actual execution mode; mocked retrieval is not a live probe.

- [ ] **5. Review the diff, stage only this task's files, and commit.**

```bash
git diff --check
git diff --stat
# Stage the exact changed paths from this task after reviewing them.
git commit -m "feat: bridge verified issuer identity into exposure research"
```



---

### Task 06 — Build safe public acquisition and private original-document retention

**Dependencies:** 01,02,05; provider-independent transport/storage, with Task 04 only for optional model-derived output
**Approved-spec coverage:** §9.3, §9.6; E10,E12–E14,I06,R14

**Files and ownership**
- Create `backend/app/services/company_exposure/network.py`, `storage.py`, `acquisition.py`.
- Reuse/extract only compatible primitives from `backend/app/services/theme_evaluation/public_fetch.py`; keep its public wrapper contract intact.
- Create unit `test_network.py`, `test_document_store.py`, `test_acquisition.py` and integration `test_document_retention.py`.
- Create versioned permitted-origin/route configuration fixtures.

**Interfaces**
`DocumentAcquisitionRegistry.fetch(target: DocumentTarget, budget: JobBudgetRef) -> CaptureResult`; `OriginalStore.put(bytes, media_type, reservation: StorageTicket) -> BlobRef`; `OriginalStore.open_authorized(blob, principal)`; `PublicDocumentTransport.fetch_once(FetchRequest) -> FetchResponse`. Each result contains sanitized provenance and a typed coverage outcome.

- [ ] **1. Add the failing tests.** Place these direct tests in the named module; construct real service fixtures locally using the declared Interfaces. Tag applicable requirements and add the named counterexamples below.

```python
@pytest.mark.case("R14")
@pytest.mark.exposure_layer("unit")
def test_private_redirect_is_rejected_before_connection(public_transport, public_request, network_spy):
    network_spy.respond_redirect("http://169.254.169.254/latest/meta-data/")
    result = public_transport.fetch_once(public_request)
    assert result.failure_code == "blocked_destination"
    assert network_spy.connected_hosts == [public_request.hostname]

@pytest.mark.case("I06")
@pytest.mark.exposure_layer("unit")
def test_404_records_gap_not_exposure_end(acquisition, document_target, root_budget, http_mock, db_session):
    http_mock.respond(status=404)
    result = acquisition.fetch(document_target, root_budget)
    assert result.coverage.reason == "http_status_404"
    assert result.revision_id is None
    assert db_session.query(ExposureClaimRevision).count() == 0

@pytest.mark.exposure_layer("postgres")
def test_storage_last_capacity_is_reserved_once(storage_contenders):
    # Two real database sessions contend for the one remaining blob allocation.
    outcomes = storage_contenders.reserve_concurrently()
    assert sum(result.allowed for result in outcomes) == 1
    assert {result.reason for result in outcomes if not result.allowed} == {"paused_storage"}
```

The test module defines the input/service fixtures named here against this task’s Interfaces; factories only build typed inputs and database rows. Any thin test convenience wrapper calls the real service and returns its real result. Do not add a production method solely to satisfy a fixture or substitute a precomputed effect report. Additional assertions in the implementation steps remain required, even when not repeated in this representative block.

- [ ] **2. Run the focused test before implementation.**

```bash
cd backend && ./venv/bin/pytest tests/unit/company_exposure/test_network.py tests/unit/company_exposure/test_document_store.py tests/unit/company_exposure/test_acquisition.py -q
```

Expected: the new test fails for its missing interface or intended behavior, not an unrelated environment error; existing regressions remain passing.

- [ ] **3. Implement this task's contract.**

Reuse the existing checked-IP/per-hop/TLS-SNI pattern, but give the new transport an explicit official-service User-Agent, safe conditional GET behavior and safe status/result shape. Keep original `fetch_public` tests passing. Credentialed official APIs are separate narrowly configured targets; never forward credentials to a redirect host. Provider secrets and retrieval cookies are absent from citation URLs and logs.

Verify permitted adapter/origin, protocol and port, resolve all destinations, reject non-global networks, pin the checked destination through connection establishment, and validate every redirect. Enforce byte, MIME sniff, timeout and decompression bounds, including a streaming response with a dishonest Content-Length. Fail closed on DNS rebinding, userinfo URL credentials and internal/metadata/link-local/loopback destinations. API redirects require reauthorization; do not expose an arbitrary proxy endpoint to the browser.

Store original bytes privately by SHA-256, with an atomic temporary-write→fsync→rename operation and a manifest carrying media type, capture outcome, publisher and document ID. Database commit references only a completed blob; an orphan blob after rollback is unreferenced storage, not evidence. Cleanup can remove only proven unreferenced blobs after the configured retention delay. Record 304/identical-byte checks separately and do not update substantive support time.

```python
# A capture updates retrieval coverage, never the document's reported period.
if capture.content_hash == retained_revision.content_hash:
    append_capture_check(retained_revision.id, capture)
    return CaptureResult(revision_id=retained_revision.id, changed=False)
```

The snippet's helper is part of the acquisition repository implemented here. Correction/equivalence links require provider identity or explicit official link evidence; unknown mirror relationships remain annotated, not called independent. Public hosting is not authorship. Retention permission is recorded per adapter/document type; lacking permission holds retention/use. Authorized evidence downloads check the existing app read boundary and return short-lived bounded responses, not a public document mirror. Tombstones prevent a unavailable blob being described as reproducible.

- [ ] **3a. Enforce a shared storage reservation and retention policy.** Implement spec §9.7: defaults `EXPOSURE_STORAGE_MAX_BYTES=5368709120`, `EXPOSURE_STORAGE_MIN_FREE_BYTES=1073741824`, scratch 512 MiB. Add typed `blob_bytes` reservations to Task 02’s existing reservation ledger; use a common store/pool identity so concurrent jobs share the cap. Include unique original/derivative bytes, staging reservations and unreconciled remnants. Reserve before download/model generation; identical blobs settle without double charging. Return `paused_storage` before I/O when full, retaining requested/available counts. Do not add a generic independent quota subsystem.

`collect_unreferenced_blobs(as_of, dry_run=True) -> StorageGCReport` marks and rechecks unreferenced 30-day-old blobs and abandoned 24-hour scratch under a lease; any current/historical selection, review or legal pin prevents deletion. A selection cannot pin a blob already being deleted. Append `EvidenceTombstoneEvent` and reconcile reclaimed bytes; crash recovery is conservative. No automatic deletion of published history. Add last-byte contention, duplicate-content accounting, late-reference-versus-GC, stale temp recovery, no-room-with-all-pinned, and visible hold tests.

- [ ] **3b. Reuse shared per-provider pacing.** `ResearchRateGate.acquire(provider, market, timeout_s) -> RateTicket` adapts existing `RateBudgetPolicy.provider_key`/interval lookup and `RedisRateLimiter.wait`. SEC uses existing `sec_edgar` configuration, not `exposure_sec` or a second bucket. Inventory actual caller keys and consolidate aliases at the wrapper if necessary; an extra aggregate gate must be shared by all affected callers, not research-only. Add a compatible strict-distributed option to `RedisRateLimiter` for research calls; legacy default fallback behavior stays unchanged. With Redis unavailable, required research pacing returns a typed hold before HTTP. Count retries/redirect HTTP attempts too. Approved official-host additions become central provider-policy entries. Provider pacing and storage waits occur outside the taxonomy/DB write fence.

- [ ] **4. Run the focused command again, then the additional checks.**

Run attack fixtures for DNS rebinding, alternate IP forms, redirect credential leakage, oversized streams/archives, MIME mismatch, invalid TLS and truncated content. Verify old paper redownload does not advance business dates, identical bytes add a capture only, and legal-removal tombstone leaves historical hash while reads disclose unavailability.

Expected: focused tests/regressions pass. Record actual execution mode; mocked retrieval is not a live probe.

- [ ] **5. Review the diff, stage only this task's files, and commit.**

```bash
git diff --check
git diff --stat
# Stage the exact changed paths from this task after reviewing them.
git commit -m "feat: retain exposure documents through bounded public acquisition"
```



---

### Task 07 — Extract structured HTML and text-PDF passages with reliable locators

**Dependencies:** 01,06; Task 04 is used only by optional translated/model-derived preparation, not by text extraction or saved fixtures
**Approved-spec coverage:** D14; §9.4–9.6; E04,E06,E10–E12,R14

**Files and ownership**
- Create `backend/app/services/company_exposure/preparation.py`, `passages.py`.
- Create `backend/requirements-exposure.in` and a resolved `backend/requirements-exposure.txt` with exact hashes/versions selected at execution.
- Create unit `test_preparation.py`, `test_passage_locators.py`; saved HTML/PDF fixtures under `backend/tests/fixtures/company_exposure/documents/` with provenance manifest.
- Update the controlled worker dependency/build configuration used for this feature only.

**Interfaces**
`ExposureEvidencePreparer.prepare(DocumentRevisionRef, questions, limits) -> PreparedEvidence`; `select_passages(PreparedEvidence, QuestionSet, limit=24) -> PassageSelection`. Deterministic parser output references immutable original revision and parser version; English/source-preserving preparation works without Task 08. Task 08 adds optional adapters around existing multilingual/vision modules; saved `PreparedEvidence` fixtures unblock claims immediately. Define `MarketDocumentAdapter`/`DocumentQuery`/`DiscoveryResult` and `markets/base.py` here, not inside the US adapter.

- [ ] **1. Add the failing tests.** Place these direct tests in the named module; construct real service fixtures locally using the declared Interfaces. Tag applicable requirements and add the named counterexamples below.

```python
@pytest.mark.case("E06")
@pytest.mark.exposure_layer("unit")
def test_pdf_table_keeps_header_period_and_unit(evidence_preparer, table_document, questions, limits):
    prepared = evidence_preparer.prepare(table_document, questions, limits)
    table = prepared.passages[0]
    assert table.locator.table_header == ["FY2025", "Revenue (USD million)"]
    assert table.locator.footnote_refs
    assert table.original_revision_id == table_document.id

@pytest.mark.case("R14")
@pytest.mark.exposure_layer("unit")
def test_page_bound_is_reported_not_hidden(evidence_preparer, long_document, questions, limits):
    prepared = evidence_preparer.prepare(long_document, questions, limits)
    assert prepared.coverage.processed_pages <= 300
    assert prepared.omitted_ranges
```

The test module defines the input/service fixtures named here against this task’s Interfaces; factories only build typed inputs and database rows. Any thin test convenience wrapper calls the real service and returns its real result. Do not add a production method solely to satisfy a fixture or substitute a precomputed effect report. Additional assertions in the implementation steps remain required, even when not repeated in this representative block.

- [ ] **2. Run the focused test before implementation.**

```bash
cd backend && ./venv/bin/pytest tests/unit/company_exposure/test_preparation.py tests/unit/company_exposure/test_passage_locators.py -q
```

Expected: the new test fails for its missing interface or intended behavior, not an unrelated environment error; existing regressions remain passing.

- [ ] **3. Implement this task's contract.**

Resolve compatible, licensed, security-reviewed `pypdf` and `pypdfium2` releases into the isolated exposure dependency lock; save resolver output and hashes. Use the repo's dependency management where one exists. No process runs pip in production and no broad existing lock update is hidden in this task. Test actual installed parser versions; record them in artifact keys.

Parse HTML DOM headings, management speaker attribution, paragraphs, table headers/cells and linked footnotes while excluding executable instructions. PDFs retain physical page index and printed page label separately. Extract text layout where reliable; otherwise mark table/image structure unverified rather than guessing cell associations. Preserve original Unicode text and source offsets before normalized matching; normalization carries offset maps.

Run parsing in a resource-limited subprocess: explicit byte/page/time/memory/pixel limits; network disabled; no PDF JavaScript/actions/embedded executables; archives are rejected unless a documented official API needs a bounded permitted format. Every processed/omitted page range is in coverage. Exceeding a limit yields partial preparation with usable independently complete passages, not a fabricated complete report.

Retrieve relevant sections using lexical/product/issuer matching and document structure, then include necessary neighbors, units, captions and footnotes. Use deterministic fallback when semantic retrieval is unavailable; do not add the removed theme embedding table. Bound passages24, parser pages300 and configured input size. If a single table cannot fit safely, hold it. The model is not fed only the first fixed document characters.

```python
# Persist both text identity and surrounding context.
locator = {
    "revision_hash": revision.content_hash,
    "preparation_policy": policy.version,
    "page_index": page.index,
    "section_path": section.path,
    "start": span.start, "end": span.end,
    "table_context_ids": tuple(required_context_ids),
}
```

Implement `PreparedEvidence`/`PassageSelection` in Task 00 contract additions owned here without changing their caller shape. Passage hashes include locator and original text; a derivative never replaces them.

- [ ] **3a. Reuse existing preparation/quantity semantics.** Use `multilingual_v2.assess_language` and the existing source-preserving segment records for language identity and offsets. Reuse translation normalization/quality and `quantity_display`/`quantity_review` when normalizing displayed units; record their current policy identities. A research locator adapter adds page/table/period provenance without translating or recalculating quantities itself. Pin original unit tokens and report calendar normalization separately; model-derived interpretation remains a later optional preparation stage. Do not create `company_exposure/language.py`. Keep PDF parsers isolated and add document-only dependencies to the dedicated worker image, not every service image.

- [ ] **4. Run the focused command again, then the additional checks.**

Verify extraction against retained native-text PDFs/HTML, complex table headings/footnotes, non-English text, malformed/encrypted documents, huge page counts, truncated parsing and text-offset integrity. Assert selected passage can reconstruct exact evidence from retained original and parser output. Run existing attachment/public-fetch regressions.

Expected: focused tests/regressions pass. Record actual execution mode; mocked retrieval is not a live probe.

- [ ] **5. Review the diff, stage only this task's files, and commit.**

```bash
git diff --check
git diff --stat
# Stage the exact changed paths from this task after reviewing them.
git commit -m "feat: prepare structural exposure evidence from HTML and PDFs"
```



---

### Task 08 — Add isolated browser rendering, selected-page vision and language derivatives

**Dependencies:** 04,06,07  
**Approved-spec coverage:** D09,D14; §9.4–9.6; E10–E13,R02,R14

**Files and ownership**
- Create `backend/app/services/company_exposure/rendering.py` and `preparation_adapters.py` (provenance/reservation glue only).
- Reuse/compatibly extend `theme_evaluation/image_preparation.py`, `kimi_translation.py`, `multilingual_preparation.py`, `multilingual_v2.py`, `translation_normalization.py`, `translation_quality.py`, `quantity_display.py`, `quantity_review.py`; retain each existing regression suite.
- Create `ops/exposure-renderer/Dockerfile`, `server.py`, `requirements.lock`, `seccomp.json`; `ops/exposure-egress/Dockerfile`, `server.py`, `requirements.lock`; `docker-compose.exposure-browser.yml`, deployment lock manifest and explicit socket/network controls in Appendix F.2–F.3. (R2: no `redis-acl.sh`; the broker has no Redis access.)
- Create `backend/app/services/company_exposure/render_pacing.py` (R2): the grant-scoped pacing RPC served by the research worker during a render call (Appendix F.3).
- Lock Playwright/Chromium/package/base-image inputs in `ops/exposure-runtime.lock.json`; build local `stockscreener/exposure-renderer:r1` and `stockscreener/exposure-egress:r1` images, record resolved image digests and run Appendix F.2 isolation probes before enabling the profile.
- Create unit `test_preparation_adapters.py`, `test_rendering_contract.py`, integration `test_renderer_security.py` and original-language/table fixtures.

**Interfaces**
`PublicRenderer.render(DocumentTarget, RenderPolicy) -> RenderCapture`; `interpret_page(PageImageRef, question, ticket) -> PassageDerivative`; `translate_passage(PassageRef, target="en", policy) -> PassageDerivative`. Every derivative references original hash and provider attempt; none is independent evidence.

- [ ] **1. Add the failing tests.** Place these direct tests in the named module; construct real service fixtures locally using the declared Interfaces. Tag applicable requirements and add the named counterexamples below.

```python
@pytest.mark.case("E11")
@pytest.mark.exposure_layer("unit")
def test_translation_preserves_original_negation(preparation_adapter, japanese_passage, translated_reply):
    derivative = preparation_adapter.translate_passage(japanese_passage, target="en", policy="translation-v3")
    assert japanese_passage.text == "量産出荷は開始していない。"
    assert derivative.original_passage_id == japanese_passage.id
    assert "not" in derivative.text.lower()
    assert derivative.provider_attempt_id is not None

@pytest.mark.case("R14")
@pytest.mark.exposure_layer("deployment")
def test_renderer_has_no_native_network_and_private_proxy_fetch_fails(renderer_probe):
    assert renderer_probe.native_socket_to_external_host().connected is False
    result = renderer_probe.load_page_with_private_subresource()
    assert result.coverage.reason == "blocked_destination"
    assert renderer_probe.private_connections == 0

@pytest.mark.case("R14")
@pytest.mark.exposure_layer("deployment")
def test_egress_broker_cannot_reach_application_services(egress_probe):
    for target in ("redis:6379", "postgres:5432", "backend:8000", "host-gateway:80"):
        assert egress_probe.tcp_connect(target).connected is False
    assert egress_probe.environment_secret_names() == {"EGRESS_SIGNING_KEY_FILE"}

@pytest.mark.case("R05")
@pytest.mark.exposure_layer("unit")
def test_pacing_session_charges_root_budget_and_rejects_other_grants(pacing_session, grant, other_grant, rate_spy, root_budget):
    ticket = pacing_session.acquire(grant.id, nonce="n1", host="www.example-issuer.com", method="GET")
    assert ticket.allowed and rate_spy.keys
    assert root_budget.refresh().browser_requests_used == 1
    assert pacing_session.acquire(other_grant.id, nonce="n2", host="www.example-issuer.com", method="GET").allowed is False
    assert pacing_session.acquire(grant.id, nonce="n1", host="www.example-issuer.com", method="GET").allowed is False  # replay

@pytest.mark.case("R02")
@pytest.mark.exposure_layer("unit")
def test_missing_vision_capability_does_not_dispatch(preparation_without_vision, image_ref, question, ticket, provider_spy):
    result = preparation_without_vision.interpret_page(image_ref, question, ticket)
    assert result.failure_code == "unavailable_capability"
    provider_spy.assert_not_called()
```

The test module defines the input/service fixtures named here against this task’s Interfaces; factories only build typed inputs and database rows. Any thin test convenience wrapper calls the real service and returns its real result. Do not add a production method solely to satisfy a fixture or substitute a precomputed effect report. Additional assertions in the implementation steps remain required, even when not repeated in this representative block.

- [ ] **2. Run the focused test before implementation.**

```bash
cd backend && ./venv/bin/pytest tests/unit/company_exposure/test_preparation_adapters.py tests/unit/company_exposure/test_rendering_contract.py -q
```

Expected: the new test fails for its missing interface or intended behavior, not an unrelated environment error; existing regressions remain passing.

- [ ] **3. Implement this task's contract.**

**08A (reusable language/vision):** compatibly add an optional `json_client: JSONCompletionClient | None` injection to the existing OpenCodeGoTranslator/OpenCodeGoVision constructors (or a shared base hook). Default None preserves their current Go behavior for existing callers; research must supply its reservation-aware ResearchJSONClient. Do not change their public callable/describe_image result shapes. Wrap existing source-preserving preparation and `OpenCodeGoTranslator`/`OpenCodeGoVision` through Task 04’s injected `ResearchJSONClient`. Preserve their prompts, original-language/quantity checks and policy versions; add exact page/passages, original hashes, attempt references and document-specific uncertainty only. Any required behavioral fix extends the existing module with its own regression and policy revision. There is no second language engine.

**08B (optional isolated browser):** implement the concrete images, services, Unix-socket RPC and deny-network renderer in Appendix F.2–F.3. The egress broker has no Redis, database or application-network access; per-request pacing and grant accounting go through the research worker's `RenderPacingSession` (R2). Chromium runs non-root with its sandbox enabled in a container with `network_mode: none`. Every permitted HTTP(S) request is fulfilled through a separate pinned-destination egress fetch service. Native socket/network bypass therefore fails even if Playwright interception misses a request. The proxy enforces approved origins and per-request budgets; disable browser workers, service workers and WebSockets unless a separately tested bounded policy supports them. No app/DB/LLM/search credentials, Docker socket or data directory is mounted into the renderer. No login/forms/CAPTCHA/paywall bypass. A missing isolation proof disables 08B without blocking HTML/text-PDF or 08A.

Render at most four navigations/issuer and rasterize at most eight relevant pages/issuer within root limits. Retain page-image hash, dimensions and bounded region locator. Selective vision uses the same subscription reservation and one-dispatch wrapper, not an unbudgeted helper. Missing vision/browser capability is a coverage gap. Do not silently install OCR or charge an alternate model.

Translate relevant passages rather than the whole report. Preserve original actor, negation, modality, unit scales, dates, ranges and role direction. For Chinese test Traditional/Simplified scripts, 萬/万 and 億/亿 magnitude units; for Japanese test qualification/planned versus shipped and supplier/customer reversal. Retain original script, exact translation/model/policy/input hashes and passage linkage. A discrepancy between official language versions is a conflict, not an invitation to choose a convenient value.

Quantitative image cells require verifiable row/column/period/unit associations and review-policy eligibility. Approximate chart readings remain approximate/review-only and never exact shares. Unsupported or ambiguous claim is held individually; unrelated explicit textual claims proceed.

```python
# Translation identity cannot increase source corroboration.
assert derivative.original_document_revision_id == passage.document_revision_id
assert derivative.evidence_role == "derivative_not_independent_source"
```

Register derivative evidence roles in the claim schema so they cannot be converted to primary leaf authority merely by serialization.

- [ ] **4. Run the focused command again, then the additional checks.**

Run real isolated browser security tests with a controlled test network; prove both navigation and subresource restrictions. Use recorded model fixtures for normal CI plus an explicitly opted-in capable-route probe in Task 27. Test missing capability, ambiguous tables, bilingual conflicts, budget limits and secret-free logs. No network test may reach cloud metadata or a real private service.

Expected: focused tests/regressions pass. Record actual execution mode; mocked retrieval is not a live probe.

- [ ] **5. Review the diff, stage only this task's files, and commit.**

```bash
git diff --check
git diff --stat
# Stage the exact changed paths from this task after reviewing them.
git commit -m "feat: support bounded multilingual and visual exposure evidence"
```



---

### Task 09 — Implement US filing and verified issuer-document acquisition

**Dependencies:** 05,06,07; Task 08 is optional for sources requiring visual/language preparation, not an adapter implementation prerequisite
**Approved-spec coverage:** D08; §9.2–9.6; E10,E12–E15,I02,R14; US release coverage

**Files and ownership**
- Create `backend/app/services/company_exposure/markets/us.py`; consume the shared `markets/base.py` contract from Task 07.
- Create unit `test_market_us.py` and fixtures `documents/us/`, `routes/us.json`.
- Add US route to `DocumentAcquisitionRegistry`; no live-enable side effect.

**Interfaces**
`MarketDocumentAdapter.discover(issuer: IssuerResolution, query: DocumentQuery, limits: AcquisitionLimits) -> DiscoveryResult`; `.resolve_target(raw_metadata) -> DocumentTarget`; `.fetch(target, budget) -> CaptureResult`. `USDocumentAdapter` implements these; common verified-IR enumeration is shared by all markets. **R2:** `USIssuerResolver.resolve_cik(security_id, budget) -> RegistryMatch | CoverageItem` produces the input to Task 05's `accept_registry_match`.

- [ ] **1. Add the failing tests.** Place these direct tests in the named module; construct real service fixtures locally using the declared Interfaces. Tag applicable requirements and add the named counterexamples below.

```python
@pytest.mark.case("E14")
@pytest.mark.exposure_layer("unit")
def test_us_target_preserves_accession_and_amendment(us_adapter, amended_sec_metadata):
    target = us_adapter.resolve_target(amended_sec_metadata)
    assert target.provider_document_id == amended_sec_metadata["accessionNumber"]
    assert target.is_amendment is True
    assert target.reporting_period == "2025-12-31"

@pytest.mark.case("R14")
@pytest.mark.exposure_layer("unit")
def test_us_fetch_uses_existing_sec_rate_key(us_adapter, us_target, budget, rate_spy):
    us_adapter.fetch(us_target, budget)
    assert rate_spy.provider_names == ["sec_edgar"]
    assert all("exposure_sec" not in key for key in rate_spy.keys)

@pytest.mark.case("I02")
@pytest.mark.exposure_layer("unit")
def test_cik_resolution_retains_registry_and_submissions_evidence(us_resolver, us_security, budget, sec_mock, rate_spy):
    sec_mock.serve_company_tickers({"0": {"cik_str": 1234567, "ticker": us_security.symbol, "title": "Example Corp"}})
    sec_mock.serve_submissions("0001234567", tickers=[us_security.symbol], exchanges=["Nasdaq"])
    match = us_resolver.resolve_cik(us_security.id, budget)
    assert match.cik == "0001234567"
    assert match.registry_capture_revision_id and match.official_record_capture_revision_id
    assert match.candidate_count == 1 and match.ticker_confirmed is True
    assert set(rate_spy.provider_names) == {"sec_edgar"}

@pytest.mark.case("I02")
@pytest.mark.exposure_layer("unit")
def test_submissions_without_ticker_is_not_confirmed(us_resolver, us_security, budget, sec_mock):
    sec_mock.serve_company_tickers({"0": {"cik_str": 1234567, "ticker": us_security.symbol, "title": "Example Corp"}})
    sec_mock.serve_submissions("0001234567", tickers=["OTHER"], exchanges=["NYSE"])
    assert us_resolver.resolve_cik(us_security.id, budget).ticker_confirmed is False
```

The test module defines the input/service fixtures named here against this task’s Interfaces; factories only build typed inputs and database rows. Any thin test convenience wrapper calls the real service and returns its real result. Do not add a production method solely to satisfy a fixture or substitute a precomputed effect report. Additional assertions in the implementation steps remain required, even when not repeated in this representative block.

- [ ] **2. Run the focused test before implementation.**

```bash
cd backend && ./venv/bin/pytest tests/unit/company_exposure/test_market_us.py -q
```

Expected: the new test fails for its missing interface or intended behavior, not an unrelated environment error; existing regressions remain passing.

- [ ] **3. Implement this task's contract.**

Use the SEC's documented submissions interface and filing documents, plus independently verified issuer IR/product pages. SEC metadata is discovery/identity input, not itself proof of a product exposure. Bind filings to accepted CIK and accession; validate parallel arrays in `filings.recent` without silently shifting mismatched rows. Additional history files are followed only inside approved caps. Use an operator-configured identifying SEC User-Agent and permitted-access/rate policy; a missing required declaration is a capability gap.

```python
# Explicit official index target; filing-document targets come from its metadata.
def sec_submission_url(cik: str) -> str:
    if not cik.isdigit() or len(cik) > 10:
        raise ValueError("invalid_cik")
    return f"https://data.sec.gov/submissions/CIK{cik.zfill(10)}.json"
```

**CIK resolution (R2, spec §4.2).** The repository has no CIK data; this adapter supplies it. `USIssuerResolver.resolve_cik` normalizes the `stock_universe` symbol for SEC lookup (e.g. share-class separators: record the exact normalization policy and test `BRK.B`/`BRK-B` style cases), fetches `https://www.sec.gov/files/company_tickers_exchange.json` (falling back to `company_tickers.json` when the exchange-qualified file is unavailable) through `DocumentAcquisitionRegistry` so it is retained as a captured document revision under `sec_edgar` pacing, and collects every candidate CIK for the ticker (and exchange when available). For each candidate it fetches the submissions record and checks that its `tickers` list contains the ticker. It returns a `RegistryMatch` with the security ID, candidate count, chosen CIK (zero-padded to 10 digits) when exactly one candidate is confirmed, both capture revision IDs, SEC entity title, matched ticker/exchange and resolver policy version; otherwise it returns the match with `ticker_confirmed=False` or `candidate_count != 1` so Task 05 opens review. Cache the retained ticker file per capture and reuse it within a root job; a new capture creates a check event, not a new business fact. Record the ticker-file URLs and their observed schema in `routes/us.json`. The registry is identity input only, never exposure evidence.

Retain form type, amendment status, accession, issuer identity, publication date, report period and original filing URL. Dates are not all business-effective dates. An amendment replaces only the scope it actually corrects; an unrelated periodic filing cannot erase every older role claim. Do not interpret SEC hosting as independent verification of the issuer's assertions.

Verified IR domains can provide primary product docs, releases, presentations and management transcript sections when permitted. Third-party analyst questions and redistributed research remain attributed to their actual speaker/author. The generic IR enumerator follows a bounded approved link list, no unrestricted web crawler.

Fixtures must include a real retained/publicly permitted submissions response and filing/IR document, a retained excerpt of the ticker-to-CIK file plus synthetic duplicate-ticker and ticker-missing variants, a synthetic redacted mismatch array, a known amendment chain, a repeated capture and inaccessible target. Synthetic documents are explicitly labeled synthetic and do not count toward the adjudicated corpus. Record official policy/contract URLs and retrieval checks in `routes/us.json`.

One common adapter contract carries `not_configured`, `permission_unavailable`, `rate_limited`, `no_matching_document`, `partial` and `complete_for_requested_scope` separately. No claim of exhaustive US issuer coverage follows from this adapter.

**Shared SEC pacing:** all submissions, filing, history and redirect requests acquire Task 06’s existing `sec_edgar` provider budget. At the inspected base the configured fallback global interval is 0.15 seconds; do not describe this number as a provider guarantee or create another limiter. Test concurrent legacy/research SEC calls against a shared fake Redis timeline and fail closed for research when distributed pacing is unavailable.

- [ ] **4. Run the focused command again, then the additional checks.**

Run offline adapter fixtures with fake transport and verify no network by default. Task 27 performs an explicitly opted-in US probe from a disposable database. Acceptance requires at least one permitted filing route and one issuer route fixture with exact passages, not only XBRL metadata.

Expected: focused tests/regressions pass. Record actual execution mode; mocked retrieval is not a live probe.

- [ ] **5. Review the diff, stage only this task's files, and commit.**

```bash
git diff --check
git diff --stat
# Stage the exact changed paths from this task after reviewing them.
git commit -m "feat: acquire US primary exposure documents"
```



---

### Task 10 — Implement Hong Kong issuer and permitted HKEXnews routes

**Dependencies:** 05,06,07; Task 08 is optional for sources requiring visual/language preparation, not an adapter implementation prerequisite
**Approved-spec coverage:** D08–D09; §9.2; E10–E14,I01–I03; HK release coverage

**Files and ownership**
- Create `backend/app/services/company_exposure/markets/hk.py`.
- Create unit `test_market_hk.py`, bilingual fixtures `documents/hk/`, route manifest `routes/hk.json`.
- Register adapter and supported/blocked route reporting.

**Interfaces**
`HKDocumentAdapter` implements the Task 07 adapter protocol; targets bind official issuer/security references and original disclosure IDs/URLs. It shares safe IR enumeration, text/PDF preparation and resource limits.

- [ ] **1. Add the failing tests.** Place these direct tests in the named module; construct real service fixtures locally using the declared Interfaces. Tag applicable requirements and add the named counterexamples below.

```python
@pytest.mark.case("E13")
@pytest.mark.exposure_layer("unit")
def test_hk_language_versions_keep_one_official_disclosure(hk_adapter, bilingual_hk_metadata):
    english = hk_adapter.resolve_target(bilingual_hk_metadata.english)
    chinese = hk_adapter.resolve_target(bilingual_hk_metadata.chinese)
    assert english.origin_disclosure_id == chinese.origin_disclosure_id
    assert english.language != chinese.language

@pytest.mark.case("I06")
@pytest.mark.exposure_layer("unit")
def test_hk_unavailable_access_remains_coverage_gap(hk_adapter, issuer, query, limits):
    result = hk_adapter.discover(issuer, query, limits)
    assert "permission_unavailable" in {gap.reason for gap in result.coverage}
    assert result.exposure_conclusion is None
```

The test module defines the input/service fixtures named here against this task’s Interfaces; factories only build typed inputs and database rows. Any thin test convenience wrapper calls the real service and returns its real result. Do not add a production method solely to satisfy a fixture or substitute a precomputed effect report. Additional assertions in the implementation steps remain required, even when not repeated in this representative block.

- [ ] **2. Run the focused test before implementation.**

```bash
cd backend && ./venv/bin/pytest tests/unit/company_exposure/test_market_hk.py -q
```

Expected: the new test fails for its missing interface or intended behavior, not an unrelated environment error; existing regressions remain passing.

- [ ] **3. Implement this task's contract.**

Implement direct public official disclosure targets and verified issuer IR/product links. Add permitted HKEXnews title/listing-page parsing only after its access and retention policy has been documented for the deployed route. Do not automate an undisclosed backend API, bypass a form/login restriction or assume the publicly documented title-search page grants blanket scraping/reproduction rights. Where automated title enumeration is not permitted, that capability returns a gap while approved direct links and issuer IR discovery remain operational.

Normalize display/security codes through existing HK market identity; preserve the original security/issuer reference and publication locale. Do not match a prospectus or circular to a listed parent purely by similar company name. Retain issuer-origin authorship, disclosure type and scope. Official exchange-hosted copies and issuer-hosted copies can be related as the same disclosure only with established document identity; dedup unknowns are not independent confirmations.

```python
# Common result shape; absence of permission is not a semantic conclusion.
result = DiscoveryResult(
    targets=tuple(verified_ir_targets),
    coverage=(CoverageItem(route="hkex_title_search",
                           outcome="permission_unavailable"),),
)
```

The target list is produced by the implemented permitted-IR enumerator, not fixture injection in production. Test an annual report, an announcement/prospectus distinction, Traditional Chinese units/negation, optional English counterpart and missing/contradictory bilingual content. Non-English evidence uses original-language review rather than requiring English equivalence. Current roles can proceed from independent complete claims even when a materiality table is held.

Record exactly which HK acquisition modes are implemented, credential/permission requirements and their saved-fixture hashes. A functional issuer/official-link route plus truthful unavailable title-search capability satisfies the approved conditional-coverage contract; a stub returning all unavailable does not.

- [ ] **4. Run the focused command again, then the additional checks.**

Run bilingual and document-identity fixtures plus forbidden automation/retention cases. Task 27 executes a permitted HK live probe without paid search. Keep copyright-retention limitations visible and provide excerpt-only UI when full-original retention is not approved.

Expected: focused tests/regressions pass. Record actual execution mode; mocked retrieval is not a live probe.

- [ ] **5. Review the diff, stage only this task's files, and commit.**

```bash
git diff --check
git diff --stat
# Stage the exact changed paths from this task after reviewing them.
git commit -m "feat: acquire Hong Kong primary exposure documents"
```



---

### Task 11 — Implement Japan issuer routes and configured EDINET capability

**Dependencies:** 05,06,07; Task 08 is optional for sources requiring visual/language preparation, not an adapter implementation prerequisite
**Approved-spec coverage:** D08–D09; §9.2–9.5; E09–E14,I02–I03; JP release coverage

**Files and ownership**
- Create `backend/app/services/company_exposure/markets/jp.py`.
- Create unit `test_market_jp.py`, Japanese fixtures `documents/jp/`, route manifest `routes/jp.json`.
- Add narrowly scoped EDINET configuration separate from paid-search settings.

**Interfaces**
`JPDocumentAdapter` implements the common protocol. An EDINET adapter verifies registration/configuration, typed issuer/doc IDs, documented metadata/body contracts and permissions; verified public issuer/JPX/TDnet links remain independently available when permitted.

- [ ] **1. Add the failing tests.** Place these direct tests in the named module; construct real service fixtures locally using the declared Interfaces. Tag applicable requirements and add the named counterexamples below.

```python
@pytest.mark.case("R01")
@pytest.mark.exposure_layer("unit")
def test_missing_edinet_key_does_not_trigger_paid_search(jp_adapter, issuer, query, limits, search_spy):
    result = jp_adapter.discover(issuer, query, limits)
    assert result.route_status["edinet"] == "unavailable_capability"
    assert "issuer_ir" in result.attempted_routes
    search_spy.assert_not_called()

@pytest.mark.case("E11")
@pytest.mark.exposure_layer("unit")
def test_jp_metadata_preserves_original_identity(jp_adapter, japanese_metadata):
    target = jp_adapter.resolve_target(japanese_metadata)
    assert target.original_title == japanese_metadata["original_title"]
    assert target.provider_document_id == japanese_metadata["docID"]
```

The test module defines the input/service fixtures named here against this task’s Interfaces; factories only build typed inputs and database rows. Any thin test convenience wrapper calls the real service and returns its real result. Do not add a production method solely to satisfy a fixture or substitute a precomputed effect report. Additional assertions in the implementation steps remain required, even when not repeated in this representative block.

- [ ] **2. Run the focused test before implementation.**

```bash
cd backend && ./venv/bin/pytest tests/unit/company_exposure/test_market_jp.py -q
```

Expected: the new test fails for its missing interface or intended behavior, not an unrelated environment error; existing regressions remain passing.

- [ ] **3. Implement this task's contract.**

Use the official EDINET API v2 specification linked from the FSA portal to implement metadata and body retrieval, not a third-party scraper's undocumented contract. During this task retain a source-verified API-contract fixture with parameter, response/error, withdrawal/amendment and authentication fields. The portal confirms registration/key requirements; the exact transport contract must be pinned from that official specification before merging the adapter. This is implementation verification of an external interface, not an open product decision.

No key or approved API policy returns `unavailable_capability` for EDINET alone; public verified issuer pages and permitted JPX/TDnet document links can still be researched. Do not subscribe to a paid historical disclosure service. Never treat an English convenience summary as complete proof of everything in the Japanese filing.

Keep issuer EDINET code, filing/document identity, reporting period and correction/withdrawal provenance. Redact any authentication parameter/header before logs/citations, and do not follow authenticated redirects to other origins. Validate body type as well as HTTP status; a successful HTTP envelope carrying an API error is not a PDF. If the official body is an archive, use only its documented permitted format in a sandbox with per-entry count/path/decompression limits; otherwise prefer direct PDF/HTML.

```python
# Credentials gate one route, not the entire market or the search budget.
if not configuration.edinet_key_ref:
    coverage.append(CoverageItem(route="edinet", outcome="unavailable_capability"))
# Independently enumerate approved official issuer links using the shared registry.
```

Fixtures include original Japanese annual/business evidence, qualification-versus-shipment language, a segment/consolidated distinction, amendment/withdrawal metadata and English derivative mismatch. Scope/version parsing must not rely on job completion dates. Save route-contract provenance and an opt-in credentialed probe definition; do not store API keys in fixtures.

- [ ] **4. Run the focused command again, then the additional checks.**

Run offline JP and shared network tests. Require at least one actual allowed issuer/official-document acquisition implementation even without an EDINET key. In Task 27 test configured EDINET if deployment enables it and separately test the keyless route; report untested EDINET as untested rather than verified.

Expected: focused tests/regressions pass. Record actual execution mode; mocked retrieval is not a live probe.

- [ ] **5. Review the diff, stage only this task's files, and commit.**

```bash
git diff --check
git diff --stat
# Stage the exact changed paths from this task after reviewing them.
git commit -m "feat: acquire Japanese primary exposure documents"
```



---

### Task 12 — Implement Taiwan issuer/MOPS routes and listing-aware metadata

**Dependencies:** 05,06,07; Task 08 is optional for sources requiring visual/language preparation, not an adapter implementation prerequisite
**Approved-spec coverage:** D08–D09; §9.2–9.5; E04,E06,E10–E14,I02–I03; TW release coverage

**Files and ownership**
- Create `backend/app/services/company_exposure/markets/tw.py`.
- Create unit `test_market_tw.py`, Traditional Chinese fixtures `documents/tw/`, route manifest `routes/tw.json`.
- Register listing-aware TWSE/TPEx handling without modifying StockUniverse identity authority.

**Interfaces**
`TWDocumentAdapter` implements the same discovery/target/fetch protocol; issuer reports and permitted MOPS links are evidence sources, appropriate TWSE metadata assists identity only. The route manifest pins actual official metadata contracts used.

- [ ] **1. Add the failing tests.** Place these direct tests in the named module; construct real service fixtures locally using the declared Interfaces. Tag applicable requirements and add the named counterexamples below.

```python
@pytest.mark.case("E11")
@pytest.mark.exposure_layer("unit")
def test_taiwan_target_keeps_raw_calendar_and_normalization(tw_adapter, taiwan_metadata):
    target = tw_adapter.resolve_target(taiwan_metadata)
    assert target.original_date == taiwan_metadata["original_date"]
    assert target.normalization_policy
    assert target.language == "zh-Hant"

@pytest.mark.case("E04")
@pytest.mark.exposure_layer("unit")
def test_taiwan_segment_table_stays_scoped(evidence_preparer, taiwan_segment_document, questions, limits):
    prepared = evidence_preparer.prepare(taiwan_segment_document, questions, limits)
    assert "30%" in prepared.passages[0].original_text
    assert prepared.passages[0].locator.row_label == "Server segment"
```

The test module defines the input/service fixtures named here against this task’s Interfaces; factories only build typed inputs and database rows. Any thin test convenience wrapper calls the real service and returns its real result. Do not add a production method solely to satisfy a fixture or substitute a precomputed effect report. Additional assertions in the implementation steps remain required, even when not repeated in this representative block.

- [ ] **2. Run the focused test before implementation.**

```bash
cd backend && ./venv/bin/pytest tests/unit/company_exposure/test_market_tw.py -q
```

Expected: the new test fails for its missing interface or intended behavior, not an unrelated environment error; existing regressions remain passing.

- [ ] **3. Implement this task's contract.**

Use verified official issuer report links and permitted MOPS documents, with the TWSE OpenAPI catalog supplying only appropriate metadata assistance. Pin the actual endpoint/schema in the route manifest from the official catalog at implementation time; do not invent a generic MOPS annual-report API or assume identical TWSE/TPEx listing coverage. The security resolver and reviewed issuer mapping determine the listing/issuer; an issuer code is not globally unique outside its scheme.

Preserve Republic-of-China and Gregorian date text with explicit normalization policy and conversion provenance, calendar units, currency and magnitude units. A unit conversion normalizes the disclosed number; it cannot establish that the number is theme-specific. For bilingual versions preserve the original and document whether an English version is a complete official counterpart or a partial presentation.

```python
# Pure date normalization tested with synthetic examples, not issuer facts.
def roc_year_to_gregorian(year: int) -> int:
    if year < 1:
        raise ValueError("invalid_roc_year")
    return year + 1911
```

Production parsing must first identify the declared source calendar; never apply this conversion to already Gregorian dates merely because a year token is small. Keep raw source date, parsed date and policy together.

A public issuer report page with a MOPS link can establish a specific supported route, not blanket MOPS enumeration permission. Implement an approved bounded issuer index enumerator and direct disclosed links; if portal enumeration is unavailable, report the gap rather than resorting to covert scraping or paid fallback. Saved fixtures include TWSE-listed and TPEx-referenced identity distinctions, an annual report table with footnotes, commercial stage modal language, a new reporting period and a stale copied capture.

- [ ] **4. Run the focused command again, then the additional checks.**

Run TW plus common adapter contract tests proving only compatible date/period/unit normalization. Task 27 performs an opted-in allowed route smoke test. Gate unsupported listing subtypes separately; never silently classify their absence as lack of company exposure.

Expected: focused tests/regressions pass. Record actual execution mode; mocked retrieval is not a live probe.

- [ ] **5. Review the diff, stage only this task's files, and commit.**

```bash
git diff --check
git diff --stat
# Stage the exact changed paths from this task after reviewing them.
git commit -m "feat: acquire Taiwan primary exposure documents"
```



---

### Task 13 — Add optional Tavily search as budgeted lead discovery only

**Dependencies:** 04,06,07; independent of every market adapter, claims, and membership
**Approved-spec coverage:** D11–D13; §9.1, §10; E10,R01,R04–R05,R13

**Files and ownership**
- Create `backend/app/services/company_exposure/search.py`.
- Extend versioned research configuration/usage records and `routes/search_tavily.json` contract fixture.
- Create unit `test_search.py`, `test_search_costs.py` and response fixtures without secrets. No deployed ChatGPT connector is a backend dependency.

**Interfaces**
`SearchAdapter.search(query: SearchQuery, ticket: ReservationTicket | None) -> SearchResult` (None is valid only for DisabledSearchAdapter; enabled adapters reject it before HTTP); implementations `DisabledSearchAdapter`, `TavilySearchAdapter`; factory `configured_search_adapter(config, resources)`. SearchResult contains title, sanitized URL, snippet, provenance and coverage, never a verification claim.

- [ ] **1. Add the failing tests.** Place these direct tests in the named module; construct real service fixtures locally using the declared Interfaces. Tag applicable requirements and add the named counterexamples below.

```python
@pytest.mark.case("R01")
@pytest.mark.exposure_layer("unit")
@pytest.mark.parametrize(("enabled", "cap"), [(False, "5"), (True, None)])
def test_tavily_key_without_enable_and_cap_makes_no_call(search_config, resources, query, http_mock, enabled, cap):
    config = search_config.with_changes(enabled=enabled, cap=cap, key_present=True)
    adapter = configured_search_adapter(config, resources)
    result = adapter.search(query, ticket=None)
    assert result.entries == ()
    assert result.coverage.reason in {"paid_search_disabled", "search_cap_required"}
    assert http_mock.requests == []

@pytest.mark.case("E10")
@pytest.mark.exposure_layer("unit")
def test_search_output_is_lead_only(tavily_adapter, query, ticket, tavily_response):
    result = tavily_adapter.search(query, ticket)
    assert result.entries[0].source_kind == "search_snippet"
    assert result.entries[0].verification_eligible is False
```

The test module defines the input/service fixtures named here against this task’s Interfaces; factories only build typed inputs and database rows. Any thin test convenience wrapper calls the real service and returns its real result. Do not add a production method solely to satisfy a fixture or substitute a precomputed effect report. Additional assertions in the implementation steps remain required, even when not repeated in this representative block.

- [ ] **2. Run the focused test before implementation.**

```bash
cd backend && ./venv/bin/pytest tests/unit/company_exposure/test_search.py tests/unit/company_exposure/test_search_costs.py -q
```

Expected: the new test fails for its missing interface or intended behavior, not an unrelated environment error; existing regressions remain passing.

- [ ] **3. Implement this task's contract.**

Implement **Tavily Search** only after Task 00 checks for an actually deployed compatible search adapter. Reuse `settings.tavily_api_key`; mere presence of Tavily/Serper settings is not evidence that credentials or a working adapter exist. The original Brave choice had no demonstrated advantage justifying another account and is removed. No automatic switch to Serper or another provider occurs.

The selected bounded HTTP contract is `POST https://api.tavily.com/search`, bearer authentication, and `results[]` discovery entries. Use `search_depth="basic"`, `auto_parameters=False`, `include_answer=False`, `include_raw_content=False`, `include_images=False`, `include_usage=True`, `max_results<=20`. These parameters prevent an automatically broader endpoint or generated answer from becoming the research evidence source. Map content snippets as unverified leads only; fetch primary originals using Task 06. Responses and usage credits are retained with request ID; do not guess account dollar costs or hardcode a free tier. Official contract reference is in Appendix E.

Every attempted query counts against the six-query root cap; retries count as new attempts, and a query cannot silently fan out to crawl/extract/research endpoints. Topic/domain/date options are explicit frozen query inputs. Search-result dates are discovery metadata, not source publication truth.

Provider default is `none`, not “Tavily enabled if key exists.” Require explicit operator enablement, selected provider, safe credential reference, account currency, configured daily/monthly ceilings and a maximum-charge costing policy. Do not hardcode a current public price or assume a free tier. Reserve the provider-defined worst permitted call charge and reconcile actual billing evidence when available; otherwise retain reserved/unknown actual cost. Unknown maximum cost blocks dispatch. Credential errors and uncertain timeouts do not authorize another provider.

Do not log secrets or raw credentialed URLs. Treat the endpoint as a fixed provider destination separate from official-document retrieval; no arbitrary URL supplied in settings without explicit trusted-origin configuration. Validate permitted query length/language/country values from the retained contract. Deduplicate link candidates and record repeated/upstream source dependencies; two search hits are not two corroborations.

When disabled, return a clear discovery coverage item and continue permitted official/issuer/user-link retrieval. No paid search fallback from HK/JP/TW failures. Unit tests prove both the enabled bounded adapter and the default-disabled behavior; an actual paid probe is separately opted in and never necessary for ordinary CI.

- [ ] **4. Run the focused command again, then the additional checks.**

Verify enable=false, missing cap, missing key, unknown cost and exhausted ledger all produce zero transport calls. Test pagination caps, timeout uncertainty, retry charging and malformed results. The output is never accepted as an ExposurePassage original or primary leaf. Provider account remains unprovisioned/off after the task.

Expected: focused tests/regressions pass. Record actual execution mode; mocked retrieval is not a live probe.

- [ ] **5. Review the diff, stage only this task's files, and commit.**

```bash
git diff --check
git diff --stat
# Stage the exact changed paths from this task after reviewing them.
git commit -m "feat: add disabled-by-default budgeted exposure search"
```



---

### Task 14 — Verify atomic claims and bounded primary-supported synthesis

**Dependencies:** 03,04,07; use retained PreparedEvidence fixtures, no market/search dependency
**Approved-spec coverage:** D02,D17; §5, §9, §12.3; E01–E03,E07–E11,E13,I03,I09

**Files and ownership**
- Create `backend/app/services/company_exposure/claims.py`, `synthesis.py` and repository `company_exposure_assessment_repo.py`.
- Create unit `test_claim_verification.py`, `test_synthesis.py`, `test_provenance_dag.py`.
- Add validated prompt/schema versions and saved provider responses to the fixture inventory.

**Interfaces**
`verify_claims(evidence: PreparedEvidence, scope: AssessmentScope, policy: VerificationPolicy) -> ClaimReviewBatch`; `validate_synthesis(premises, links, target) -> SynthesisDecision`; `verify_evidence_dag(edges) -> None`. All automated judgments use the one-dispatch resource wrapper; deterministic post-validation cannot upgrade unsupported model output to verified.

- [ ] **1. Add the failing tests.** Place these direct tests in the named module; construct real service fixtures locally using the declared Interfaces. Tag applicable requirements and add the named counterexamples below.

```python
@pytest.mark.case("E01")
@pytest.mark.exposure_layer("unit")
def test_cooccurrence_does_not_verify_compound(verifier, independent_ai_memory, scope, policy):
    batch = verifier.verify_claims(independent_ai_memory, scope, policy)
    assert not any(claim.verified for claim in batch.claims if claim.proposition == "ai_memory_exposure")

@pytest.mark.case("E02")
@pytest.mark.exposure_layer("unit")
def test_product_join_is_narrowly_worded(verifier, primary_product_join, scope, policy):
    batch = verifier.verify_claims(primary_product_join, scope, policy)
    assert batch.claim("hbm_capable_equipment").support_basis == "primary_synthesis"
    assert not batch.has_verified_claim("hbm_sales")
    assert not batch.has_verified_claim("named_customer")

@pytest.mark.case("E03")
@pytest.mark.exposure_layer("unit")
def test_customer_chain_lacks_application_link(verifier, customer_chain, scope, policy):
    batch = verifier.verify_claims(customer_chain, scope, policy)
    assert batch.claim("hbm_application").verified is False

@pytest.mark.case("E10")
@pytest.mark.exposure_layer("unit")
def test_hosting_cannot_upgrade_authorship(verifier, hosted_analyst_question, scope, policy):
    batch = verifier.verify_claims(hosted_analyst_question, scope, policy)
    assert all(claim.support_basis != "primary_explicit" for claim in batch.claims)

@pytest.mark.case("I09")
@pytest.mark.exposure_layer("unit")
def test_generated_research_cannot_be_primary_leaf(circular_evidence_edges):
    with pytest.raises(EvidenceDependencyError):
        verify_evidence_dag(circular_evidence_edges)
```

The test module defines the input/service fixtures named here against this task’s Interfaces; factories only build typed inputs and database rows. Any thin test convenience wrapper calls the real service and returns its real result. Do not add a production method solely to satisfy a fixture or substitute a precomputed effect report. Additional assertions in the implementation steps remain required, even when not repeated in this representative block.

- [ ] **2. Run the focused test before implementation.**

```bash
cd backend && ./venv/bin/pytest tests/unit/company_exposure/test_claim_verification.py tests/unit/company_exposure/test_synthesis.py tests/unit/company_exposure/test_provenance_dag.py -q
```

Expected: the new test fails for its missing interface or intended behavior, not an unrelated environment error; existing regressions remain passing.

- [ ] **3. Implement this task's contract.**

Extract candidate propositions from retained passages with strict fields for issuer/theme/product/scope/kind, quote locator, commercial status, evidence basis, temporal anchor and uncertainty. For every proposed verified claim validate the original document hash, exact cited text/region, speaker/author, reporting entity, period and publisher qualification. “Published on issuer domain” is insufficient when the statement belongs to an analyst, a questioner or a redistributed report. Primary backing is a policy-qualified issuer/official assertion, not independent truth certification.

Review role, application, named customer, status and materiality independently. Co-occurrence alone cannot establish an economic link. A broad memory product cannot narrow to HBM without explicit support. Marketing claims may establish available capability but not shipments, realized customer usage or revenue. Preserve contrary passages and uncertainty. Absent evidence is unresolved; empty search creates no end claim.

Permit synthesis with at most three original primary premises and two explicit joining relationships. Normalize only evidenced issuer-scoped product identities and compatible periods/scopes. Each join records the exact premise establishing it. Unsupported synthesis may be recorded as inferred/unverified, never automatic-primary. No graph traversal through generic supplier/customer knowledge.

```python
# Pure admissibility bound; semantic entailment and scope checks remain mandatory.
def within_synthesis_bound(primary_leaf_ids, explicit_links):
    return len(set(primary_leaf_ids)) <= 3 and len(explicit_links) <= 2
```

Implement DFS/color or equivalent cycle detection on typed evidence dependencies. A verified result must terminate at permitted original-primary document revisions. Derivative text/vision links preserve their original parents. Assessments, summaries, source classifications and search outputs cannot serve as primary leaves. An old assessment can supply retained original passages, but not count itself as corroboration.

Store successful extraction/review artifacts with exact policy/model/evidence hashes; provider failures and input/schema failures retain auditable candidates/attempts and cannot replace accepted claims. Avoid using the existing source extractor's successful-empty correction semantics as dossier-wide deletion. Claim post-validation records rule-specific failures and never rewrites original quotes. Hostile passage instructions are data, not tool/policy directives.

- [ ] **4. Run the focused command again, then the additional checks.**

Run all E01–E03/E07–E11/I09 fixtures, a three-premise accepted join and a four-premise hold, role reversal, conflicting official-language versions, missing original locator, forged primary URL and escaped prompt-injection cases. Verify only the affected claim is held and no source attention row is written.

Expected: focused tests/regressions pass. Record actual execution mode; mocked retrieval is not a live probe.

- [ ] **5. Review the diff, stage only this task's files, and commit.**

```bash
git diff --check
git diff --stat
# Stage the exact changed paths from this task after reviewing them.
git commit -m "feat: verify atomic exposure claims and bounded primary synthesis"
```



---

### Task 15 — Compute disclosed and compatible calculated materiality without proxies

**Dependencies:** 03,04,07; pure materiality contracts can run in parallel with 14; verified-claim integration joins at 16
**Approved-spec coverage:** D04–D05; §6; E04–E06,E08,I03

**Files and ownership**
- Create `backend/app/services/company_exposure/materiality.py`.
- Create unit `test_materiality.py` with property/parameterized period, scope, currency and denominator cases.
- Add normalized measure serialization to the research schemas, not legacy exposure_strength.

**Interfaces**
`validate_measure(MaterialityInput) -> MaterialityMeasureResult`; `calculate_materiality(FormulaInput) -> MaterialityMeasureResult`. Formula kinds in V1: explicit ratio and explicit non-overlapping sum. Return typed held/unknown outcomes instead of coercing invalid arithmetic.

- [ ] **1. Add the failing tests.** Place these direct tests in the named module; construct real service fixtures locally using the declared Interfaces. Tag applicable requirements and add the named counterexamples below.

```python
from decimal import Decimal

@pytest.mark.case("E05")
@pytest.mark.exposure_layer("unit")
def test_ratio_uses_compatible_disclosed_operands(disclosed_ratio_input):
    # Input has cited 200 and 1000 USD-million revenue operands, same FY/scope/basis.
    result = calculate_materiality(disclosed_ratio_input)
    assert result.basis == "calculated"
    assert result.value == Decimal("0.2")
    assert len(result.operand_revision_ids) == 2

@pytest.mark.case("E04")
@pytest.mark.exposure_layer("unit")
def test_server_segment_share_is_not_theme_share(server_segment_for_ai_memory):
    result = validate_measure(server_segment_for_ai_memory)
    assert result.value is None
    assert "scope_mismatch" in result.hold_reasons

@pytest.mark.case("E06")
@pytest.mark.exposure_layer("unit")
@pytest.mark.parametrize("variant", ["period_mismatch", "currency_mismatch", "overlap", "zero_profit", "negative_profit"])
def test_incalculate_materiality_is_held(formula_input_factory, variant):
    result = calculate_materiality(formula_input_factory(variant=variant))
    assert result.basis == "unknown"
    assert result.hold_reasons

@pytest.mark.case("I03")
@pytest.mark.exposure_layer("unit")
def test_subsidiary_operand_cannot_become_parent_share(subsidiary_parent_ratio):
    result = calculate_materiality(subsidiary_parent_ratio)
    assert "scope_mismatch" in result.hold_reasons
```

The test module defines the input/service fixtures named here against this task’s Interfaces; factories only build typed inputs and database rows. Any thin test convenience wrapper calls the real service and returns its real result. Do not add a production method solely to satisfy a fixture or substitute a precomputed effect report. Additional assertions in the implementation steps remain required, even when not repeated in this representative block.

- [ ] **2. Run the focused test before implementation.**

```bash
cd backend && ./venv/bin/pytest tests/unit/company_exposure/test_materiality.py -q
```

Expected: the new test fails for its missing interface or intended behavior, not an unrelated environment error; existing regressions remain passing.

- [ ] **3. Implement this task's contract.**

Keep metric, raw disclosed numbers/ranges, original unit/magnitude, normalized Decimal, currency, period, consolidated/standalone/segment scope, precision, numerator/denominator and original operand citations. `calculate_materiality` is the arithmetic helper implemented in this task; its public result includes value, basis, formula, operands and hold reasons. Production compatibility is independently validated from structured operand provenance, never a boolean supplied by an HTTP caller.

```python
# Minimal exact arithmetic AFTER provenance compatibility checks.
from decimal import Decimal

def ratio_value(numerator: Decimal, denominator: Decimal) -> Decimal:
    if denominator <= 0:
        raise ValueError("nonpositive_denominator_review_required")
    return numerator / denominator
```

`revenue_share`, `profit_share`, `capacity_share` and `backlog_share` remain distinct metrics. Do not invent a composite exposure-strength score. Mixing currencies, periods, numerator/reporting scope or accounting basis holds the calculation. No implicit FX conversion. Sum only segments explicitly shown non-overlapping; overlapping product and region segments cannot be added. Negative/zero profit denominator and out-of-range share require review, not misleading percentage output. Retain the original numbers even when the derived figure is held.

Qualitative labels are `core_business`, `explicitly_material`, `explicitly_limited`, `unknown` and require actual primary wording or a directly justified equivalent. No threshold converts a role confidence into core business. Unknown output says “not separately disclosed in reviewed evidence” with search/coverage scope. A past figure stays labeled for its actual reporting period; unchanged rereads do not turn it current.

A disclosed subsidiary percentage cannot use a parent denominator unless compatible disclosed consolidation inputs and an explicit supported calculation exist. Actual-versus-forecast and commercial capability remain separate. Formula/input hashes include revisions and exact denominator scope; changing any operand yields a new immutable measure/claim revision. Old-period values can coexist with a newer role claim.

- [ ] **4. Run the focused command again, then the additional checks.**

Parameterize all period/unit/currency/denominator/overlap/forecast mismatches, Decimal round trips and ranges. E05 must reproduce the expected result and operand citations; E06 must hold without dropping original data; E08 must preserve unknown materiality without vetoing otherwise eligible membership. Verify source confidence never enters materiality arithmetic.

Expected: focused tests/regressions pass. Record actual execution mode; mocked retrieval is not a live probe.

- [ ] **5. Review the diff, stage only this task's files, and commit.**

```bash
git diff --check
git diff --stat
# Stage the exact changed paths from this task after reviewing them.
git commit -m "feat: represent disclosed and reproducible exposure materiality"
```



---

### Task 16 — Assemble multi-document assessments and enforce claim-specific safety holds

**Dependencies:** 03,04,07,14,15; no live market/search dependency
**Approved-spec coverage:** D16; §5.2, §8, §11; E12–E15,I03–I06,I10,R06,R11–R12

**Files and ownership**
- Create `backend/app/services/company_exposure/assessments.py`, `freshness.py`.
- Extend the assessment repository for CAS revision selection and append-only safety events.
- Create unit `test_assessments.py`, `test_freshness.py`, `test_document_precedence.py`; integration `test_assessment_conflicts_postgres.py`.

**Interfaces**
`ExposureAssessmentService.assess(AssessmentAttemptInput) -> AssessmentResult`; `persist_assessment(result, expected_prior_revision, principal) -> AssessmentRevisionRef`; `ExposureSafety.evaluate(dependencies, at) -> SafetyDecision`; `refresh_due_holds(at) -> HoldReport`. Safety can block/defer only, never substitute an unpinned accepted result.

- [ ] **1. Add the failing tests.** Place these direct tests in the named module; construct real service fixtures locally using the declared Interfaces. Tag applicable requirements and add the named counterexamples below.

```python
@pytest.mark.case("E15")
@pytest.mark.exposure_layer("unit")
def test_partial_refresh_preserves_role_date(assessment_service, partial_refresh_input):
    result = assessment_service.assess(partial_refresh_input)
    assert result.claim("role").supported_as_of == partial_refresh_input.prior_role.supported_as_of
    assert result.claim("materiality").period == "FY2025"

@pytest.mark.case("E12")
@pytest.mark.exposure_layer("unit")
def test_redownload_does_not_reaffirm_business_evidence(assessment_service, redownload_input):
    result = assessment_service.assess(redownload_input)
    assert result.claim("role").supported_as_of == redownload_input.original_support_date

@pytest.mark.case("E14")
@pytest.mark.exposure_layer("unit")
def test_late_old_capture_cannot_restore_ended_exposure(assessment_service, late_archived_input):
    result = assessment_service.assess(late_archived_input)
    assert result.claim("commercial_participation").conclusion == "ended"

@pytest.mark.case("I05")
@pytest.mark.case("R12")
@pytest.mark.exposure_layer("unit")
def test_expiry_blocks_new_use_without_research(safety, expired_dependencies, fixed_clock, provider_spy):
    result = safety.evaluate(expired_dependencies, at=fixed_clock.now())
    assert result.allowed is False
    assert "stale" in result.hold_reasons
    provider_spy.assert_not_called()
```

The test module defines the input/service fixtures named here against this task’s Interfaces; factories only build typed inputs and database rows. Any thin test convenience wrapper calls the real service and returns its real result. Do not add a production method solely to satisfy a fixture or substitute a precomputed effect report. Additional assertions in the implementation steps remain required, even when not repeated in this representative block.

- [ ] **2. Run the focused test before implementation.**

```bash
cd backend && ./venv/bin/pytest tests/unit/company_exposure/test_assessments.py tests/unit/company_exposure/test_freshness.py tests/unit/company_exposure/test_document_precedence.py -q
```

Expected: the new test fails for its missing interface or intended behavior, not an unrelated environment error; existing regressions remain passing.

- [ ] **3. Implement this task's contract.**

Freeze original passage revisions, retained previous claim selection, exact issuer mapping, relevant theme-definition fingerprint, role/verification/freshness policies and model artifacts per assessment attempt. Never include generation ID alone as semantic identity or choose newest job completion. Build new immutable claim revisions and a dossier revision selecting justified replacements plus still-valid carried claims.

A newer report for one metric does not delete an older role. Empty search/404/quota timeout worsens coverage without ending an exposure. An explicit business exit can create exposure_end and a use hold, but removal remains reviewed. Late archived/partial copies are classified by provider correction sequence, original publication/reporting scope and explicit supersession evidence. Capture order is audit sequence only. Unknown contradictory precedence holds the affected proposition; it does not restore superseded claims. Record duplicate/mirror/source dependencies without statistical-independence claims.

Compute role/product freshness450 days and customer/commercial-transition freshness180 days from substantive dated support. Undated support is automatic-use held. Quantitative measures retain real period and newer compatible period selection; do not erase historical facts at timer expiry. A repeated download/translation leaves original substantive time intact.

```python
from datetime import timedelta

def freshness_deadline(substantive_at, claim_kind):
    days = 180 if claim_kind in {"customer_relationship", "commercial_status"} else 450
    return None if substantive_at is None else substantive_at + timedelta(days=days)
```

Use the exact spec policy for actual deployed kinds; raw `participation` facts requiring a commercial transition depend on that status claim's stricter deadline. `freshness_deadline` is implemented/tested here, not applied indiscriminately to materiality or identity.

Persist under `producer_write` with expected dossier revision, issuer-link/theme/policy dependencies and current hold token. If another assessment finished meanwhile, retain this result for audit and rebuild deterministic selection from both manifests or queue review; never overwrite the winner by completion time. Increment append-only claim-use safety revisions on substantive conflict, mapping disqualification, relevant policy change or expiry. The current safety query can only block. Record `checked_at`, exact dependency revisions and reason; affected automatic actions must recheck immediately before persistence and publication. Local hold production remains active even if network research is disabled.

- [ ] **4. Run the focused command again, then the additional checks.**

Race an assessment against primary correction, reviewed issuer mapping, expiry and another job. Verify CAS conflicts do not lose evidence, no provider runs under locks, holds affect only dependent claims, and unrelated generation progress is not blocked. Test undated support, periodic-report coexistence, failure coverage and clock boundary exactly at fresh_until.

Expected: focused tests/regressions pass. Record actual execution mode; mocked retrieval is not a live probe.

- [ ] **5. Review the diff, stage only this task's files, and commit.**

```bash
git diff --check
git diff --stat
# Stage the exact changed paths from this task after reviewing them.
git commit -m "feat: assemble versioned exposure assessments and use holds"
```



---

### Task 17 — Orchestrate bounded verification, refresh and candidate discovery

**Dependencies:** Verification slice 17A: 04,05,06,07,09,14,15,16. Discovery slice 17B: 17A plus the selected enabled market adapter and optional 13; not all adapters are prerequisites.
**Approved-spec coverage:** D01,D07; §8–10; E10,I01,I06,R01–R06,R15

**Files and ownership**
- Create `backend/app/services/company_exposure/research.py`.
- Extend work repository/candidate scenarios, not source-classification work tables.
- Create unit `test_research_workflow.py`, `test_discovery_scope.py`, integration `test_research_jobs_postgres.py`.

**Interfaces**
`ExposureResearchCoordinator.request(ResearchRequestInput, principal, idempotency_key) -> ResearchRequestRef`; `.run_step(work_id, lease_token) -> ResearchStepResult`; `.discover_candidates(theme_id, policy, budget) -> CandidateBatch`. Stages dispatch only their bounded unit, retain artifacts and append progress events.

- [ ] **1. Add the failing tests.** Place these direct tests in the named module; construct real service fixtures locally using the declared Interfaces. Tag applicable requirements and add the named counterexamples below.

```python
@pytest.mark.case("R05")
@pytest.mark.exposure_layer("unit")
def test_repeat_request_reuses_job(coordinator, research_input, principal):
    first = coordinator.request(research_input, principal, idempotency_key="verify-1")
    repeated = coordinator.request(research_input, principal, idempotency_key="verify-1")
    assert first.id == repeated.id

@pytest.mark.case("R15")
@pytest.mark.exposure_layer("unit")
def test_discovery_requires_enabled_theme(coordinator, theme_id, disabled_theme_policy, root_budget, provider_spy):
    result = coordinator.discover_candidates(theme_id, disabled_theme_policy, root_budget)
    assert result.coverage.reason == "theme_discovery_disabled"
    assert result.candidates == ()
    provider_spy.assert_not_called()
```

The test module defines the input/service fixtures named here against this task’s Interfaces; factories only build typed inputs and database rows. Any thin test convenience wrapper calls the real service and returns its real result. Do not add a production method solely to satisfy a fixture or substitute a precomputed effect report. Additional assertions in the implementation steps remain required, even when not repeated in this representative block.

- [ ] **2. Run the focused test before implementation.**

```bash
cd backend && ./venv/bin/pytest tests/unit/company_exposure/test_research_workflow.py tests/unit/company_exposure/test_discovery_scope.py -q
```

Expected: the new test fails for its missing interface or intended behavior, not an unrelated environment error; existing regressions remain passing.

**17A — verify-only deliverable:** implement `ExposureResearchCoordinator.run_step(work_id, lease_token)` for `verify`/`refresh` with retained/supplied/US-adapter documents and typed paused/coverage states. **R2:** the first stage of a US request without an accepted issuer link calls `USIssuerResolver.resolve_cik` then `IssuerIdentityAdapter.accept_registry_match`. An accepted registry link continues the job; a `review_required` result pauses the job as `review_required` with the failed condition visible in job status, and resumes after an administrator applies a link. Add a test for each path. Add the on-demand `process_exposure_work` task body and request entrypoint used by 21A. The default `DisabledSearchAdapter` and absent renderer do not block this workflow. No `discover` job, membership mutation or classifier context is allowed in this slice.

**17B — expansion:** extend the same work state machine for explicitly enabled-theme discovery after that particular market/search capability is installed. It shares root reservations and verification; do not duplicate the assessment engine or make US verification wait for the last acquisition adapter.

- [ ] **3. Implement this task's contract.**

Verification is one issuer–theme pair. Resolve identities and relevant existing assessments before research; reuse fresh accepted answers when inputs/policies are unchanged. Build question-specific document targets through the registry in this order: retained originals, approved official enumerators, verified issuer domains, supplied links, optional budgeted search. The workflow records disabled routes and searched-unresolved questions.

Discovery is permitted only by an authenticated explicit request or an enabled-theme policy. Candidate generation can use admitted source leads, official product/customer links, retained docs and optional search. It does not recursively traverse every supplier/customer graph, infer membership from ETF lists, or enqueue the entire universe. Maximum10 new issuers and6 search attempts/root, with all children sharing root provider budget and constraints. Selection of candidates must retain its retrieval reasons and original lead. Cross-listings reuse one verified issuer investigation.

```python
# Local priority order; all levels remain bounded by the same resource owner.
PRIORITY = {
    "material_conflict": 0, "requested_verification": 0,
    "new_membership": 1, "due_refresh": 2, "enabled_theme_expansion": 3,
}
```

Enforce fairness between queued roots, stable resume cursors, artifacts and coverage. A saved job is not an always-running autonomous tool agent: one stage has a finite target/attempt/page set and no uncontrolled tool recursion. Every substage has an input hash and result state, including no finding and partial finding; new information creates a new assessment attempt, not a changed old result.

Cancelled/paused jobs release only pre-dispatch reservations. A provider operation already sent retains accounting and may finish for audit; it cannot publish automatic use after cancellation unless a separately accepted decision requests it. Retry queues use bounded backoff, no model/provider fallback escalation. Lease failure cannot lose the previous accepted dossier.

Jobs become ready_for_publication after claims are independently complete/held and membership proposals evaluated. They never call the live-pointer setter directly. A partial job may supply valid claims with explicit omitted documents/pages, without claiming comprehensive research. Dirty references trigger the existing generation schedule later; an originating source's observation timestamps remain unchanged.

- [ ] **4. Run the focused command again, then the additional checks.**

Run a full offline verify path and discovery→child verify path with default paid search disabled. Prove root counters span children/retries/models, candidate count bounded, unrelated source ingestion continues after research pause and identical current assessments produce no new research spend. Operational job states must not be presented as accepted product state.

Expected: focused tests/regressions pass. Record actual execution mode; mocked retrieval is not a live probe.

- [ ] **5. Review the diff, stage only this task's files, and commit.**

```bash
git diff --check
git diff --stat
# Stage the exact changed paths from this task after reviewing them.
git commit -m "feat: orchestrate bounded issuer exposure research"
```



---

### Task 18 — Evaluate reviewed role policies and origin-aware listing membership

**Dependencies:** 03,05,14,15,16; integration joins 17A as needed; no candidate-discovery dependency
**Approved-spec coverage:** D03,D05–D06; §7, §11; E07–E09,I01,I04–I05,I10–I11,R08,R10–R13

**Files and ownership**
- Create `backend/app/services/company_exposure/membership.py`, `decisions.py`.
- Create unit `test_membership.py`, `test_role_policies.py`, `test_decisions.py`; integration `test_membership_decisions_postgres.py`.
- Add draft role-policy templates for memory, cybersecurity, refining, tankers and copper without approving them in migration. Register an explicit effective-membership adapter for the generation builder.

- Create the deferred `backend/app/models/company_exposure_membership.py` tables: reviewed role policies, decision requests/previews/events, association identities and decision revisions. Allocate the just-in-time `exposure_membership_decisions` migration in this task. Task 03 does not pre-create these tables for shadow verification.

**Interfaces**
`ExposureMembershipEvaluator.evaluate(MembershipInputs) -> MembershipEvaluation`; `effective_membership(OriginSelections, GlobalDecisionRef) -> MembershipProjection`; `preview_decision(DecisionRequest, principal) -> DecisionPreview`; `apply_decision(preview_id, hash, principal) -> DecisionRevisionRef`. MembershipInputs pins claims, role policy, issuer/listing eligibility, all origin decisions, safety and prior accepted research contribution.

- [ ] **1. Add the failing tests.** Place these direct tests in the named module; construct real service fixtures locally using the declared Interfaces. Tag applicable requirements and add the named counterexamples below.

```python
@pytest.mark.case("E07")
@pytest.mark.exposure_layer("unit")
def test_customer_remains_outside_producer_basket(membership_evaluator, customer_in_producer_policy):
    result = membership_evaluator.evaluate(customer_in_producer_policy)
    assert result.outcome == "candidate"
    assert "role_not_eligible" in result.reasons

@pytest.mark.case("E08")
@pytest.mark.exposure_layer("unit")
def test_unknown_materiality_can_qualify(membership_evaluator, commercial_unknown_materiality):
    result = membership_evaluator.evaluate(commercial_unknown_materiality)
    assert result.outcome == "eligible_addition"

@pytest.mark.case("E09")
@pytest.mark.exposure_layer("unit")
def test_qualification_is_not_commercial_participation(membership_evaluator, qualification_only):
    assert membership_evaluator.evaluate(qualification_only).outcome == "candidate"

@pytest.mark.case("I04")
@pytest.mark.exposure_layer("unit")
def test_admin_veto_survives_new_primary_evidence(membership_evaluator, verified_but_admin_rejected):
    result = membership_evaluator.evaluate(verified_but_admin_rejected)
    assert result.outcome == "rejected"
    assert result.review_proposal_required is True

@pytest.mark.case("I10")
@pytest.mark.exposure_layer("unit")
def test_split_does_not_copy_membership(membership_evaluator, split_target_inputs):
    assert all(membership_evaluator.evaluate(value).outcome == "held" for value in split_target_inputs)

@pytest.mark.case("I11")
@pytest.mark.exposure_layer("schema")
def test_one_association_keeps_two_decision_revisions(db_session, membership_revision_rows):
    db_session.add_all(membership_revision_rows)
    db_session.flush()
    assert db_session.query(ExposureMembershipAssociation).count() == 1
    assert db_session.query(ExposureMembershipDecisionRevision).count() == 2
```

The test module defines the input/service fixtures named here against this task’s Interfaces; factories only build typed inputs and database rows. Any thin test convenience wrapper calls the real service and returns its real result. Do not add a production method solely to satisfy a fixture or substitute a precomputed effect report. Additional assertions in the implementation steps remain required, even when not repeated in this representative block.

- [ ] **2. Run the focused test before implementation.**

```bash
cd backend && ./venv/bin/pytest tests/unit/company_exposure/test_membership.py tests/unit/company_exposure/test_role_policies.py tests/unit/company_exposure/test_decisions.py -q
```

Expected: the new test fails for its missing interface or intended behavior, not an unrelated environment error; existing regressions remain passing.

- [ ] **3. Implement this task's contract.**

Persist the reviewed-role, decision and membership objects deferred from Task 03. `ExposureMembershipAssociation` has `UNIQUE(economic_theme_id, security_id)`; `ExposureMembershipDecisionRevision` has `UNIQUE(association_id, revision_number)`. Accepted and rejected revisions coexist. Pin exact assessment/claim/link/policy and other-origin decision IDs. No synthetic source assignment is required. Allocate/recheck this migration at commit time. Run direct I11 model tests here; raw-SQL mutation of a historical revision must fail.


Implement the ordered gate in spec §7, using current claim facts from accepted selections and current blocking-only safety checks. Missing/unknown theme role policy means review, not inferred acceptance. Policies pin the defining theme fingerprint, allowed roles/stages/scopes and required evidence. Templates are drafts; an authorized administrator approves them explicitly. A policy change cannot silently broaden admission to accommodate a newly discovered company.

Producer/operator basket membership needs supported current production/operation. An equipment-role policy may explicitly accept commercially available theme-specific capability, while displaying that actual sales/materiality remain unknown. Generic equipment, plans or qualification cannot pass a shipping requirement. Separate map visibility from basket membership: customers may be verified but excluded.

Effective membership is the union of eligible source_extraction/social/research/manual contributions, with explicit global reviewed rejection/removal/conflict precedence. Preserve source/Social decisions and source correction behavior. Do not turn absence of research into removal of existing membership. A research claim becoming stale/disputed flags existing research membership for review and suppresses new admissions/grounding; it does not automatically retract its already accepted contribution. A reviewed removal explicitly changes the research origin and, when the reviewed request is global, applies global precedence. It cannot erase another origin through an unscoped delete.

```python
# Persist one proposal/decision per changed input fingerprint, not per poll.
decision_key = (
    association.id, selected_assessment_revision.id,
    role_policy_revision.id, issuer_link_revision.id,
    global_decision_revision.id if global_decision_revision else None,
    safety.semantic_token,
)
```

Include input hashes in preview. Apply under shared fence using trusted principal, expected versions and reason, rejecting stale previews before side effects. Auto service principal may accept only gate-approved additions; it cannot remove, lift admin rejections, approve role policies or rewrite issuer links. Claim verification can remain true even when membership is globally rejected.

New research counters count verified issuers once and expose listing duplication, unknown materiality and review-required members separately. Do not modify attention scoring, statistical-independence claims or basket weights. Theme split/mechanism change holds affected research assessment; no blind clone into every child. Source term rename that preserves mechanism need not trigger new LLM analysis.

The effective-membership function consumes explicit frozen origin inputs, not mutable latest rows. It is integrated into all basket readers in Task 21/22, not just appended to the constituent array.

- [ ] **4. Run the focused command again, then the additional checks.**

Verify all positive/negative role-policy tests, retained existing stale membership, rejection/removal, conflict, global versus origin scope, two listings of one issuer, missing policy and prospect-only exposure. Race reviewer apply with new evidence and safety hold; automatic addition must not commit from stale support. Confirm unchanged source roots/events.

Expected: focused tests/regressions pass. Record actual execution mode; mocked retrieval is not a live probe.

- [ ] **5. Review the diff, stage only this task's files, and commit.**

```bash
git diff --check
git diff --stat
# Stage the exact changed paths from this task after reviewing them.
git commit -m "feat: govern research-backed membership with origin precedence"
```



---

### Task 19 — Extend existing generation capture, selection and snapshot publication

**Dependencies:** Assessment-only 19A: 03,16. Membership 19B: 18 plus 19A; 20 supplies compatibility before membership activation.
**Approved-spec coverage:** D18; §13–15; I05,I10–I11,R06–R09,R11–R12,R15

**Files and ownership**
- Create `backend/app/services/company_exposure/publication.py`.
- Modify `economic_taxonomy_publication.py`, `economic_taxonomy_publication_preparation.py`, `economic_taxonomy_publication_contracts.py`, `economic_taxonomy_publication_validation.py` and `economic_taxonomy_snapshot_builder.py` only at typed extension points.
- Modify `backend/app/infra/db/repositories/economic_taxonomy_publication_repo.py` and runtime publication models.
- Create the just-in-time `exposure_generation_extension` migration under `backend/alembic/versions/` (Task 19; record its actual path/revision in the migration manifest).
- Create unit `test_publication_extension.py`, `test_generation_history.py`; integration `test_exposure_publication_postgres.py`.

- Own `ExposureSelectionSet` and selection child-table persistence deferred from Task 03, in this task’s just-in-time migration. 19A can pin assessments/holds with an empty membership-change list; 19B adds membership selection/validation without inventing source assignments.

**Interfaces**
`ExposurePublicationAdapter.capture_inputs(db, authority, as_of) -> CapturedExposureInputs`; `.prepare_selection(db, captured, manifest_id) -> ExposureSelectionSet`; `.build_entries(db, selection, generation_inputs) -> tuple[SnapshotEntryPayload,...]`; `.validate_safety(db, dependencies, at) -> SafetyDecision`. Existing EconomicTaxonomyPublicationCoordinator remains the only pointer writer.

- [ ] **1. Add the failing tests.** Place these direct tests in the named module; construct real service fixtures locally using the declared Interfaces. Tag applicable requirements and add the named counterexamples below.

```python
@pytest.mark.case("R07")
@pytest.mark.exposure_layer("postgres")
def test_generation_pins_old_assessment(publication, reader, published_g1, accepted_assessment_g2):
    before = reader.read_for_security(42, generation_id=published_g1.id)
    publication.publish_with_exposure(accepted_assessment_g2)
    assert reader.read_for_security(42, generation_id=published_g1.id) == before

@pytest.mark.case("R08")
@pytest.mark.exposure_layer("postgres")
def test_assessment_publication_has_no_source_evidence_effect(publication, assessment, db_session):
    counts = (db_session.query(ThemeObservation).count(), db_session.query(ThemeDevelopmentEvent).count())
    publication.publish_with_exposure(assessment)
    assert (db_session.query(ThemeObservation).count(), db_session.query(ThemeDevelopmentEvent).count()) == counts

@pytest.mark.case("R06")
@pytest.mark.exposure_layer("postgres")
def test_research_and_source_writers_share_fence(pg_writer_race, provider_spy):
    # Fixture runs independent actual sessions through the existing producer fence.
    pg_writer_race.run_source_research_and_publication()
    assert pg_writer_race.commits_after_stale_epoch == 0
    assert pg_writer_race.provider_calls_inside_fence == 0
    assert pg_writer_race.all_commits_have_revision_log is True

@pytest.mark.case("R11")
@pytest.mark.exposure_layer("postgres")
def test_new_hold_rejects_dependent_addition_only(publication_race):
    result = publication_race.publish_after_hold()
    assert result.unsafe_security_id not in result.newly_admitted_security_ids
    assert result.unrelated_generation_published is True
```

The test module defines the input/service fixtures named here against this task’s Interfaces; factories only build typed inputs and database rows. Any thin test convenience wrapper calls the real service and returns its real result. Do not add a production method solely to satisfy a fixture or substitute a precomputed effect report. Additional assertions in the implementation steps remain required, even when not repeated in this representative block.

- [ ] **2. Run the focused test before implementation.**

```bash
cd backend && ./venv/bin/pytest tests/unit/company_exposure/test_publication_extension.py tests/unit/company_exposure/test_generation_history.py -q
```

Expected: the new test fails for its missing interface or intended behavior, not an unrelated environment error; existing regressions remain passing.

- [ ] **3. Implement this task's contract.**

19A creates/seals `ExposureSelectionSet` and its child references here, not in the core assessment migration. Add `test_selection_models.py` and raw-SQL immutability/concurrency tests. Assessment-only selections have no membership changes and do not wait for Task 18. 19B adds membership selection; live additions stay disabled until Task 20 compatibility and full scope gates pass.


Implement Appendix A's acyclic build sequence. At the existing short capture fence, record exact eligible assessment/link/policy/hold/membership revisions and relevant source/Social decisions. Reserve the exposure selection UUID and its immutable input fingerprint; capture that reference in a new nullable `GenerationInputManifest.exposure_inputs` field using a typed versioned `exposure_v1` envelope. Do not alter the meaning of existing source-lineage entries or insert research into `InterpretationSelection` just to reuse its one-source logic.

The reserved exposure selection is a draft artifact, not a live pointer. Preparation fills and seals its deterministic selected rows from captured inputs outside the exclusive publisher lock. The set can reference the captured parent manifest ID; its semantic hash excludes parent IDs, preventing circular hash definitions. The outer immutable input manifest contains the reserved ID and input fingerprint, not an as-yet-unknown output hash. The prepared generation separately pins the sealed selection ID and resulting integrity/semantic hashes. Generation validation rejects any mismatched input fingerprint, unsealed set or current-row substitution.

Add nullable `ServingGeneration.exposure_selection_set_id` and a versioned extension payload/hash contract (or equivalent explicit immutable binding included in generation validation). Keep old generation serializer/hash algorithm unchanged when no extension exists. New reader capability version covers all changed API/basket consumers. Do not relabel old hashes or assert `0055` is the new required migration; the final release records the verified new head.

```python
# A planned extension envelope, not a second authority.
exposure_input = {
    "kind": "exposure_v1",
    "selection_set_id": str(reserved_selection_id),
    "input_fingerprint": frozen_research_inputs_hash,
    "assessment_refs": assessment_refs,
    "issuer_link_refs": issuer_link_refs,
    "role_policy_refs": role_policy_refs,
    "hold_refs": hold_refs,
    "membership_refs": membership_refs,
    "freshness_as_of": as_of.isoformat(),
}
```

All referenced lists have the exact strict shapes in Appendix A. Keep the existing source-selection list unchanged. Validate the new exposure_inputs field through its strict versioned parser; reject malformed or unknown exposure schemas while allowing old manifests without the extension.

Add sealed snapshot entries for exposure dossiers and theme membership projections. Use Task 18's frozen origin union rather than appending raw research rows to source constituents. Preserve attention counts/metrics; research-only generation may recompute current metrics as-of, but at fixed as-of and fixed source evidence its attention values and roots are identical. It can change coverage/basket composition only through selected membership.

Before committing an automatic addition, and again at final publication, validate required claim fresh_until and current dependency-specific safety/link/policy disqualifications. A held result is retained audit/review; existing accepted membership remains flagged. Reprepare dependent proposals outside the lock, leaving unrelated post-cutoff evidence as backlog. No network, model, replay or delivery wait inside capture/final fence. Reference-only current safety checks cannot replace old-generation evidence.

Expose extension only when economic authority, required migrations and all readers understand it. In shadow, build comparison artifacts with no accepted membership or grounding. Keep exposure product snapshots versioned even when acquisition is later disabled.

- [ ] **4. Run the focused command again, then the additional checks.**

Run PostgreSQL races: source/Social/research writers versus capture; unrelated research during preparation; relevant hold/policy/link change; expiry before final commit; old/new serializer preservation; crash before and after pointer switch. Verify one coherent generation, current gating and byte-preserved G1. Run existing publication, snapshot, fence and reader regression suites. Do not enable live mode in this task.

Expected: focused tests/regressions pass. Record actual execution mode; mocked retrieval is not a live probe.

- [ ] **5. Review the diff, stage only this task's files, and commit.**

```bash
git diff --check
git diff --stat
# Stage the exact changed paths from this task after reviewing them.
git commit -m "feat: publish exposure assessments through economic serving generations"
```



---

### Task 20 — Add ordered research-only compatibility and one issuer-attestation facade

**Dependencies:** 05,18–19  
**Approved-spec coverage:** D10,D18; §7.4, §14.3; I01–I04,I11,R07–R10,R15

**Files and ownership**
- Create `backend/app/services/company_exposure/compatibility.py`.
- Modify existing `economic_taxonomy_publication_compatibility.py`, `economic_taxonomy_runtime.py`, rollback recovery/consumer adapters and `social_company_identity_service.py` at scoped seams.
- Create the just-in-time `exposure_origin_compatibility` migration under `backend/alembic/versions/` (Task 20; record its actual path/revision in the migration manifest) for typed legacy research contribution/projection and attestation bridge selection metadata.
- Create unit `test_compatibility.py`, `test_issuer_facade.py`; integration `test_exposure_outbox_postgres.py`.

**Interfaces**
`build_research_projections(selection, previous_selection) -> tuple[CompatibilityProjection,...]`; `apply_research_membership_projection(db, event) -> DeliveryResult`; `accepted_issuer_configuration(generation) -> CompanyIdentityConfiguration`. Existing stage/claim/delivery/checkpoint infrastructure owns events; no new standalone outbox daemon.

- [ ] **1. Add the failing tests.** Place these direct tests in the named module; construct real service fixtures locally using the declared Interfaces. Tag applicable requirements and add the named counterexamples below.

```python
@pytest.mark.case("R09")
@pytest.mark.exposure_layer("postgres")
def test_delivery_resumes_after_commit_before_notify(coordinator, prepared_exposure_generation, principal, delivery_worker, db_session):
    coordinator.publish_generation(prepared_exposure_generation.id, principal=principal, notify_delivery_workers=None)
    delivery_worker.run_once()
    assert db_session.query(ProjectionCheckpoint).filter_by(projection_kind="research_membership").count() > 0

@pytest.mark.case("R10")
@pytest.mark.exposure_layer("postgres")
def test_older_research_delivery_preserves_other_origin(projection_service, revision_two, revision_one, membership_reader):
    projection_service.apply(revision_two)
    projection_service.apply(revision_one)
    current = membership_reader.read(revision_two.security_id)
    assert current.research_projection_revision == 2
    assert "source" in current.origins
```

The test module defines the input/service fixtures named here against this task’s Interfaces; factories only build typed inputs and database rows. Any thin test convenience wrapper calls the real service and returns its real result. Do not add a production method solely to satisfy a fixture or substitute a precomputed effect report. Additional assertions in the implementation steps remain required, even when not repeated in this representative block.

- [ ] **2. Run the focused test before implementation.**

```bash
cd backend && ./venv/bin/pytest tests/unit/company_exposure/test_compatibility.py tests/unit/company_exposure/test_issuer_facade.py -q
```

Expected: the new test fails for its missing interface or intended behavior, not an unrelated environment error; existing regressions remain passing.

- [ ] **3. Implement this task's contract.**

Use the existing transport `source_lineage` string as a logical owner, not a document. The owner is `research-membership:<issuer_uuid>:<theme_uuid>`, projection kind `research_membership`, origin `exposure_research`. It creates no SourceFamily/ThemeMention/ClaimAssignment. Keep a monotonic accepted projection revision when the selected research contribution changes, independent of document bytes, generation ID and authority epoch. Include per-security membership revisions, issuer link, applicable global decision refs and supporting assessment IDs.

```python
owner = f"research-membership:{issuer_id}:{theme_id}"
logical_key = (owner, accepted_projection_revision,
               "research_membership", projection_policy_version, target)
```

Payload is complete replacement of that owner's research contribution only. Removal/empty state occurs only through a reviewed authorized decision or appropriate explicit operational rollback policy, not because a fresh query returned no research. Other owners/origins survive. A global reviewed rejection remains a veto when a stale delivery arrives. Extend legacy-shaped readers to consult an explicit research-origin contribution store; do not insert fake ThemeConstituent facts whose source labels/mention counts imply observed news.

Stage with generation preparation. Deliverability derives from committed successful publication history, not a post-commit ephemeral release flag. The already published event remains deliverable if superseded; target checkpoints reject obsolete revision effects. Abandoned-generation events never apply. Post-commit notification is optional wake-up only. Mark origin to suppress reverse extraction/research triggers. Existing checkpoint ordering and durable failure recovery apply unchanged.

Before switching issuer facade behavior, preserve/import every current admin attestation and prove equivalent current grouping. Under upgraded economic generations, `SocialCompanyIdentityService` reads the generation's accepted issuer-link selection; legacy configuration becomes retained historical provenance, not a competing write authority. Old administrative replacement commands route to trusted reviewed link revisions with pending-publication response semantics. Do not call the old `replace` method's transaction-owning helper from the new fenced write. Update its callers/tests so pending and published version IDs are not confused. Historical generations read their original mapping. Unresearched listings retain compatible attested mappings rather than disappearing.

Feature rollback/disable stops new research additions/spend but retains accepted contributions and history according to the approved spec. Returning to an older binary is safe only if its compatibility readers understand the research contribution store; otherwise rollback remains explicitly unavailable pending the scoped recovery path. No unsupported claim of immediate rollback or destructive legacy cleanup.

- [ ] **4. Run the focused command again, then the additional checks.**

Run out-of-order, duplicate, cancelled/abandoned generation, post-commit crash, global rejection and source-owner preservation tests. Test issuer facade changes across generation G1/G2 without mutating prior config/audit. Rehearse failure recovery and confirm no mirrored event loops or source-attention growth.

Expected: focused tests/regressions pass. Record actual execution mode; mocked retrieval is not a live probe.

- [ ] **5. Review the diff, stage only this task's files, and commit.**

```bash
git diff --check
git diff --stat
# Stage the exact changed paths from this task after reviewing them.
git commit -m "feat: mirror research membership without fabricating source evidence"
```



---

### Task 21 — Expose generation-bound research reads and authenticated operations

**Dependencies:** 21A shadow verification/read preview: 05,16,17A. 21B generation product reads: 19A. 21C reviewed membership/admin operations: 18–20. Keep these independent commits and gates.
**Approved-spec coverage:** §15–16; I04–I08,R07,R08,R13,R15

**Files and ownership**
- Create `backend/app/schemas/company_exposure.py`, `backend/app/api/v1/company_exposures.py`, `backend/app/services/company_exposure/reads.py`.
- Modify `backend/app/api/v1/router.py`, `economic_themes.py` for routes and generation metadata only.
- Update research-aware stock/Social/digest/MCP basket consumers where they read the changed constituent projection.
- Create unit `test_api_reads.py`, `test_api_admin.py`, `test_consumer_contract.py`.

**Interfaces**
`CompanyExposureReader.read_for_security(security_id, generation_id=None)`, `.read_assessment(assessment_id,generation_id=None)`, `.read_for_theme(theme_id,generation_id=None)` return sealed payloads. `ResearchJobReader.read(job_id)` returns explicitly operational progress. Admin endpoints call Task 18 decision service and Task 17 request service.

- [ ] **1. Add the failing tests.** Place these direct tests in the named module; construct real service fixtures locally using the declared Interfaces. Tag applicable requirements and add the named counterexamples below.

```python
@pytest.mark.case("R07")
@pytest.mark.exposure_layer("api")
def test_old_generation_returns_its_pinned_payload(api_client, published_g1, g1_expected, publish_g2):
    publish_g2()
    response = api_client.get(f"/api/v1/company-exposures?security_id=42&generation_id={published_g1.id}")
    assert response.status_code == 200
    assert response.json() == g1_expected

@pytest.mark.case("R13")
@pytest.mark.exposure_layer("api")
def test_body_actor_does_not_authorize_research(api_client, db_session, provider_spy):
    response = api_client.post("/api/v1/company-exposures/research-requests", json={"actor": "admin", "kind": "verify", "security_id": 42})
    assert response.status_code in {401, 403}
    assert db_session.query(ExposureResearchRequest).count() == 0
    provider_spy.assert_not_called()

@pytest.mark.exposure_layer("api")
def test_shadow_preview_route_is_labeled_and_job_scoped(admin_client, completed_shadow_job):
    response = admin_client.get(f"/api/v1/company-exposures/research-jobs/{completed_shadow_job.id}/preview")
    assert response.status_code == 200
    body = response.json()
    assert body["view_kind"] == "shadow_preview"
    assert body["authoritative_membership"] is False
    assert body["assessment_revision_id"] == str(completed_shadow_job.assessment_revision_id)

def test_no_parallel_research_routes_are_registered(app_routes):
    paths = {route.path for route in app_routes}
    assert "/api/v1/company-exposures/research" not in paths
    assert not any(path.startswith("/api/v1/company-exposures/research/") for path in paths)
    assert not any(path.startswith("/api/v1/company-exposures/securities") for path in paths)
```

The test module defines the input/service fixtures named here against this task’s Interfaces; factories only build typed inputs and database rows. Any thin test convenience wrapper calls the real service and returns its real result. Do not add a production method solely to satisfy a fixture or substitute a precomputed effect report. Additional assertions in the implementation steps remain required, even when not repeated in this representative block.

- [ ] **2. Run the focused test before implementation.**

```bash
cd backend && ./venv/bin/pytest tests/unit/company_exposure/test_api_reads.py tests/unit/company_exposure/test_api_admin.py tests/unit/company_exposure/test_consumer_contract.py -q
```

Expected: the new test fails for its missing interface or intended behavior, not an unrelated environment error; existing regressions remain passing.

**21A usable shadow API:** add authenticated `POST /api/v1/company-exposures/research-requests` for US `verify|refresh` with supplied/retained/SEC sources (an optional administrator-supplied CIK becomes a Task 05 `LinkProposal`, never a trusted identifier); `GET /api/v1/company-exposures/research-jobs/{job_id}`; and `GET /api/v1/company-exposures/research-jobs/{job_id}/preview`. These are the spec §16 routes (R2); do not register any `/research` or `/securities/{id}` variant. The preview returns exact immutable assessment/evidence IDs and `view_kind="shadow_preview", authoritative_membership=False`. It never reads as a live serving generation or supplies classifier grounding, and no global pointer is created. Queries use job/revision identity, not a mutable “current issuer profile.” This explicitly operational preview is permitted by spec §18 before 19/20. Return a typed unavailable response for not-yet-installed discovery/decision actions.

**21B product reads** are separate generation-bound endpoints using 19A and its capability; **21C writes** join reviewed decision/compatibility services. Only their installed routes are registered for a slice; unfinished paths are not advertised as ready.

- [ ] **3. Implement this task's contract.**

Implement spec §16's route names, registering static `research-jobs`/admin routes before dynamic `/{assessment_id}` matching. Validate UUID/int/query bounds strictly. Default product reads resolve one serving generation once through EconomicThemeReader, then the exposure selection/snapshot belonging to it. An assessment ID cannot bypass selection and reveal a newer unselected revision. Historical generations without the extension return a typed unavailable envelope, not latest data or zero exposure.

```text
GET  /api/v1/company-exposures?security_id=&generation_id=
GET  /api/v1/company-exposures/{assessment_id}?generation_id=
GET  /api/v1/economic-themes/{theme_id}/exposures?generation_id=
GET  /api/v1/company-exposures/research-jobs/{job_id}
GET  /api/v1/company-exposures/research-jobs/{job_id}/preview
POST /api/v1/company-exposures/research-requests
POST /api/v1/company-exposures/admin/decisions/preview
POST /api/v1/company-exposures/admin/decisions/apply
PUT  /api/v1/company-exposures/admin/theme-research-policy/{theme_id}
GET  /api/v1/company-exposures/admin/settings
PUT  /api/v1/company-exposures/admin/settings
```

Add permission-controlled evidence excerpt/original access via opaque document IDs; no endpoint fetches arbitrary caller URLs synchronously. Research request URLs become safe queued targets. Initiating research, changing spend/settings/policy/membership/issuer/holds and decision apply require existing `require_admin`→trusted AdminPrincipal. An actor body/header never sets stored identity. The additional admin settings routes are the implementation surface for approved subscription/local-limit/search/feature controls; they persist `ExposureRuntimePolicyRevision` and expose secret-presence/capability metadata, never secret values. They do not grant unattended issuer/role-policy approval. Add idempotency keys and expected preview revision hashes, authorization before writes/spend, bounded request/candidate/page values and per-root limits.

Product payload includes issuer/link provenance, theme fingerprint, exact assessment/claim/evidence revisions, support basis, role/stage, metric/period/scope, as-of/fresh_until/holds, membership origins/state and generation. Operational status may be newer but is labeled unaccepted/research progress and does not substitute for live assessment. Include searched routes and omitted evidence with unknown versus none distinguished.

Update consumers of effective membership deliberately. Do not overwrite old stock confidence fields with new materiality. New research coverage fields are separate. Historical source-derived rows remain readable in old generation serializers; new consumer contract rejects any latest-research join. Admin conflicts/source group policy cannot be bypassed through a stock-page variant or MCP route. Documentation describes original short excerpts, private originals and tombstones.

- [ ] **4. Run the focused command again, then the additional checks.**

Run route ordering, idempotency, unauthorized/forged actor, missing admin config, invalid/mismatched generation, old snapshot hash, pagination, document authorization, operational/live distinction and source/Social regression tests. Register every changed consumer in the capability test inventory; no unresolved bypass allowed before feature activation.

Expected: focused tests/regressions pass. Record actual execution mode; mocked retrieval is not a live probe.

- [ ] **5. Review the diff, stage only this task's files, and commit.**

```bash
git diff --check
git diff --stat
# Stage the exact changed paths from this task after reviewing them.
git commit -m "feat: expose generation-scoped company exposure APIs"
```



---

### Task 22 — Add why-included dossiers, candidates and evidence review UI

**Dependencies:** 22A US shadow research panel and evidence preview: 21A. 22B generation dossiers/review UI: 21B/21C; no need to wait for all markets to show a usable preview.
**Approved-spec coverage:** §16; D02–D06,D08–D16; I04–I08,R01,R07,R13

**Files and ownership**
- Create `frontend/src/api/companyExposures.js` and tests.
- Create `frontend/src/features/companyExposure/ExposureDossier.jsx`, `ExposureClaims.jsx`, `ExposureEvidence.jsx`, `ExposureResearchPanel.jsx`, `ExposureReview.jsx` and colocated tests.
- Modify `frontend/src/features/themes/components/EconomicThemeDetailModal.jsx` and stock/theme consumer components identified in Task 00.
- Update existing API contracts/types and query-key helpers without replacing the global theme UI.

**Interfaces**
Generation-aware client functions `getCompanyExposures`, `getExposureAssessment`, `getThemeExposures`, `getResearchJob`, `getResearchJobPreview`, `requestExposureResearch`, `previewExposureDecision`, `applyExposureDecision`, `updateThemeResearchPolicy`. They match Task 21 endpoint shapes; query keys include generation IDs for product data and distinct job keys for operations.

- [ ] **1. Add the failing tests.** Place these direct tests in the named module; construct real service fixtures locally using the declared Interfaces. Tag applicable requirements and add the named counterexamples below.

```javascript
it('shows a usable shadow preview without claiming live membership', async () => {
  render(<ExposureResearchPanel job={shadowCompletedJob} preview={shadowPreview} />);
  expect(screen.getByText(/shadow preview/i)).toBeVisible();
  expect(screen.getByText(/not separately disclosed/i)).toBeVisible();
  expect(screen.queryByText(/added to live basket/i)).not.toBeInTheDocument();
});

it('keeps an accepted dossier pinned when a newer job completes', async () => {
  render(<ExposureDossier assessment={generationOneAssessment} />);
  await publishJobEvent(completedUnpublishedJob);
  expect(screen.getByText(generationOneAssessment.claims[0].statement)).toBeVisible();
  expect(screen.queryByText(completedUnpublishedJob.newClaim)).not.toBeInTheDocument();
});
```

Define the named JS response fixtures in the colocated fixture file using the Task 21 schema; `publishJobEvent` updates the mocked job query only. Use the real components and API adapter. For frontend traceability, generate Vitest test-name metadata mapped to case/layer requirements; do not put pytest decorators on JavaScript. Additional assertions in the implementation steps remain required, even when not repeated in this representative block.

- [ ] **2. Run the focused test before implementation.**

```bash
cd frontend && npm run test:run -- src/api/companyExposures.test.js src/features/companyExposure/ExposureResearchPanel.test.jsx src/features/companyExposure/ExposureDossier.test.jsx
```

Expected: the new test fails for its missing interface or intended behavior, not an unrelated environment error; existing regressions remain passing.

- [ ] **3. Implement this task's contract.**

Build on existing components/style conventions. The theme constituent row's “Why included” opens one issuer–theme dossier. Show source-observed versus research-verified origin, per-claim support badge, original/English passage toggle, commercial stage, materiality with exact period/denominator, current/history dates and explicit coverage. No blanket company verified badge and no confidence-as-percentage materiality display.

The map can show a verified customer outside a producer-only basket. A held/stale member remains visible as review-required with its retained membership; show that it cannot support new automated uses. Existing unresearched source members are unresearched, not rejected. Distinct issuer and listing counts are labeled separately.

Implement candidate queue with unmet gates and job progress/limits. Research request controls respect server-returned allowed capabilities. Show subscription usage quota as local allocation, remaining provider balance unknown when not reported; paid search initially disabled with a separate authenticated explicit enable/cap workflow. UI must not send an enabled flag merely because a key/config entry exists.

```javascript
// Representative actual Vitest assertions to implement with the API fixture.
it('does not relabel unknown materiality or pending research as verified', async () => {
  render(<ExposureDossier assessment={unknownMaterialityFixture} />);
  expect(screen.getByText(/not separately disclosed in reviewed evidence/i)).toBeVisible();
  expect(screen.queryByText(/90% exposure/i)).not.toBeInTheDocument();
});
```

Create `unknownMaterialityFixture` in the colocated fixture file from the real Task 21 schema; the component and test imports belong to this task. Add actual interactions for requested verification, enabled-theme discovery, reviewer reason, stale preview, rejected decision and no generation extension. Avoid JSON-dump-only administration. Evidence links are authorized and do not expose secrets or an unrestricted public original mirror.

Use separate React Query keys for generation-pinned dossier and mutable job status. A completed job invalidates the job query; it does not overwrite live dossier with unpublished data. Refetch the accepted generation after publication, not after every claim extraction. Original-language passages render safely as text, not unsanitized HTML.

- [ ] **4. Run the focused command again, then the additional checks.**

Run `cd frontend && npm run test:run -- src/api/companyExposures.test.js src/features/companyExposure` and existing EconomicThemeDetailModal/stock tests, then `npm run build`. Verify G1 remains pinned across job updates, all relevant language text fits/reads correctly, no secret URLs are rendered and keyboard-accessible review actions require their reason. Record build results, not screenshots alone.

Expected: focused tests/regressions pass. Record actual execution mode; mocked retrieval is not a live probe.

- [ ] **5. Review the diff, stage only this task's files, and commit.**

```bash
git diff --check
git diff --stat
# Stage the exact changed paths from this task after reviewing them.
git commit -m "feat: show evidence-backed company exposure dossiers"
```



---

### Task 23 — Schedule prioritized research and provider-free freshness maintenance

**Dependencies:** 23A dedicated queue was introduced in 04; on-demand work joins 17A/21A. Provider-free holds and due verification: 16,17A. 23B discovery schedules: 17B and enabled adapter. Publication triggers join 19; none depends on full UI.
**Approved-spec coverage:** D01,D07,D12–D13,D16; §8, §11, §14.4, §18; I05–I06,R01–R06,R11–R12,R15

**Files and ownership**
- Create `backend/app/tasks/company_exposure_tasks.py` and `services/company_exposure/triggers.py`.
- Modify `backend/app/celery_app.py`, existing bounded new-constituent trigger seams and economic refresh dependency registration.
- Create unit `test_tasks.py`, `test_triggers.py`, `test_disabled_mode.py` and worker configuration contract tests.

**Interfaces**
Tasks: `discover_exposure_work`, `process_exposure_work`, `refresh_exposure_holds`, `discover_enabled_theme_candidates`. Trigger function `enqueue_exposure_verification(source_reference, issuer, theme, relevant_input_hash)` is idempotent and never synchronous research. Publication remains the existing economic refresh task.

- [ ] **1. Add the failing tests.** Place these direct tests in the named module; construct real service fixtures locally using the declared Interfaces. Tag applicable requirements and add the named counterexamples below.

```python
@pytest.mark.case("R12")
@pytest.mark.exposure_layer("unit")
def test_disabled_acquisition_still_applies_local_hold(task_services, expired_claim, provider_spy):
    task_services.set_acquisition_enabled(False)
    report = refresh_exposure_holds.run()
    assert expired_claim.id in report.held_claim_ids
    provider_spy.assert_not_called()

@pytest.mark.case("R05")
@pytest.mark.exposure_layer("unit")
def test_repeat_trigger_keeps_one_active_request(trigger, source_ref, issuer, theme, db_session):
    trigger(source_ref, issuer, theme, relevant_input_hash="same-input")
    trigger(source_ref, issuer, theme, relevant_input_hash="same-input")
    assert db_session.query(ExposureResearchRequest).count() == 1

@pytest.mark.exposure_layer("deployment")
def test_research_queue_does_not_subscribe_price_worker(compose_config, celery_app):
    assert compose_config.worker_queues("celery-exposure-research") == {"exposure_research"}
    route = celery_app.amqp.router.route({}, "app.tasks.company_exposure_tasks.process_exposure_work")
    assert route["queue"].name == "exposure_research"
```

The test module defines the input/service fixtures named here against this task’s Interfaces; factories only build typed inputs and database rows. Any thin test convenience wrapper calls the real service and returns its real result. Do not add a production method solely to satisfy a fixture or substitute a precomputed effect report. Additional assertions in the implementation steps remain required, even when not repeated in this representative block.

- [ ] **2. Run the focused test before implementation.**

```bash
cd backend && ./venv/bin/pytest tests/unit/company_exposure/test_tasks.py tests/unit/company_exposure/test_triggers.py tests/unit/company_exposure/test_disabled_mode.py -q
```

Expected: the new test fails for its missing interface or intended behavior, not an unrelated environment error; existing regressions remain passing.

- [ ] **3. Implement this task's contract.**

Route all `app.tasks.company_exposure_tasks.*` entrypoints to **`exposure_research`**, consumed only by **`celery-exposure-research`** from Task 04/Appendix F.1. No research task subscribes the worker to `data_fetch_*`, `market_jobs_*` or general `celery`. Default one process, prefetch one, one investigation/provider call. One task advances one bounded stage, not a whole multi-issuer run; return durable continuation IDs. Preserve cumulative budgets across tasks. A delayed network limiter or quota creates rescheduled work rather than occupying the price-fetch worker. No provider I/O or waits occur inside the taxonomy fence.

Keep byte/CPU-heavy PDF stages inside bounded subprocesses with scratch/memory limits; Chromium uses the separate profile from 08B only. Research HTTP uses the **existing shared per-provider rate gate** from Task 06, including `sec_edgar`, independent of queue placement. Update `start_celery.sh`, base/prod Compose and `CLAUDE.md` with queue/resource ownership and regression tests. Heartbeats/leases describe work, not new evidence.

Trigger only when a new relationship or materially relevant product/evidence/policy gap lacks a suitable current assessment. Include the original trigger cause and evidence roots to suppress mirror-origin/self-generated loops. Source ingestion succeeds or fails by its own contracts; research queue failure cannot turn a successful source extraction into a failed source. Add a bounded reconciliation scan for missed research triggers.

Hourly polling schedules due refresh only; local freshness holds can be produced without a model. Enabled-theme discovery is weekly/on demand and disabled by default per theme. Priority is conflicts/requested verification, new memberships, due refresh, optional expansion, with fairness and configured allowance. Do not issue exploratory calls because a timer ran if no work is due.

```python
# Processing permissions are explicit and do not depend on credential presence.
if policy.research_mode == "disabled":
    return TaskOutcome.skipped("research_disabled")
if job.kind == "discover" and not theme_policy.discovery_enabled:
    return TaskOutcome.skipped("theme_discovery_disabled")
```

Local hold evaluation is a separate entry point outside that acquisition skip path. The same provider-free maintenance task calls `ResearchResources.close_period` for every ended allocation period, moving remaining uncertain reservations to `expired_uncertain` (R2, Task 04); it runs even when research mode is disabled. It continues to enforce prior claims' expiry, queues review where required and marks the exposure extension dirty without paid calls. Source/Social behavior and historical reads remain available. When publication is delayed, direct automatic-use checks still reject expired support.

Shadow jobs prepare comparisons but cannot add members or supply new classifier grounding. Live jobs require economic authority and enabled stage capabilities verified by the release gate. Disabling acquisition does not remove accepted membership. Accepted assessment/hold/decision revisions feed the existing coalesced generation schedule; no new publisher or one-generation-per-doc loop is added.

Add operational telemetry for queued/paused/retry/review, allowance/cost unknown state, route coverage, artifact hits and per-stage latency; these are research health metrics, not theme-attention evidence. Redact URLs with secrets and do not log full disclosures routinely.

- [ ] **4. Run the focused command again, then the additional checks.**

Run Celery registration/queue tests, default-off no-network tests, fairness/cumulative budget checks, mirror-loop prevention, source-worker contention and local-expiry-at-disabled tests. Verify weekly expansion happens only for enabled themes and existing source/Social schedules remain unchanged.

Expected: focused tests/regressions pass. Record actual execution mode; mocked retrieval is not a live probe.

- [ ] **5. Review the diff, stage only this task's files, and commit.**

```bash
git diff --check
git diff --stat
# Stage the exact changed paths from this task after reviewing them.
git commit -m "feat: schedule bounded exposure research and local safety maintenance"
```



---

### Task 24 — Ground incoming classification with accepted relevant exposure evidence

**Dependencies:** 14,16,19B,20,21B,23A; activate after full applicable evidence/history/grounding gates. Unrelated market/search implementation is not a code dependency.
**Approved-spec coverage:** D15–D17; §11–12; I05,I07–I10,R02,R06–R08,R11–R12,R15

**Files and ownership**
- Create `backend/app/services/company_exposure/grounding.py` and immutable grounding-use model module.
- Create the just-in-time `exposure_grounding_uses` migration under `backend/alembic/versions/` (Task 24; record its actual path/revision in the migration manifest).
- Modify `backend/app/services/theme_grounding_context.py`, economic source admission/extraction/review adapters and exact source-processing entrypoints identified in Task 00.
- Create unit `test_grounding.py`, `test_grounding_cache.py`, `test_no_circular_support.py`; integration `test_grounding_hold_race_postgres.py`.

**Interfaces**
`ExposureGroundingSelector.select(SourceContext, generation_id, policy) -> ExposureGroundingContext`; `validate_grounding_use(context, at) -> SafetyDecision`; `persist_grounding_use(context, request_id, attempt_id, check) -> ExposureGroundingUse`. Context contains exact accepted claim/evidence/assessment/link revisions and actual generation provenance, not a mutable company profile.

- [ ] **1. Add the failing tests.** Place these direct tests in the named module; construct real service fixtures locally using the declared Interfaces. Tag applicable requirements and add the named counterexamples below.

```python
@pytest.mark.case("I07")
@pytest.mark.exposure_layer("unit")
def test_product_context_does_not_invent_demand_cause(grounding_selector, product_orders_source, generation_id, policy, classifier):
    context = grounding_selector.select(product_orders_source, generation_id, policy)
    result = classifier.classify(product_orders_source, context=context)
    assert result.theme_support == "inferred"
    assert result.has_claim("ai_demand_caused_orders") is False

@pytest.mark.case("I08")
@pytest.mark.exposure_layer("unit")
def test_generic_price_post_has_no_theme_fanout(grounding_selector, price_only_source, generation_id, policy):
    context = grounding_selector.select(price_only_source, generation_id, policy)
    assert context.claim_refs == ()

@pytest.mark.case("I09")
@pytest.mark.exposure_layer("integration")
def test_grounding_output_cannot_be_reintroduced_as_primary(verifier, classifier_generated_evidence, scope, policy):
    batch = verifier.verify_claims(classifier_generated_evidence, scope, policy)
    assert not any(claim.verified for claim in batch.claims)
```

The test module defines the input/service fixtures named here against this task’s Interfaces; factories only build typed inputs and database rows. Any thin test convenience wrapper calls the real service and returns its real result. Do not add a production method solely to satisfy a fixture or substitute a precomputed effect report. Additional assertions in the implementation steps remain required, even when not repeated in this representative block.

- [ ] **2. Run the focused test before implementation.**

```bash
cd backend && ./venv/bin/pytest tests/unit/company_exposure/test_grounding.py tests/unit/company_exposure/test_grounding_cache.py tests/unit/company_exposure/test_no_circular_support.py -q
```

Expected: the new test fails for its missing interface or intended behavior, not an unrelated environment error; existing regressions remain passing.

- [ ] **3. Implement this task's contract.**

Select only claims relevant to explicit issuer/product/activity references in the incoming source. Knowing an issuer alone is not permission to attach all its themes. Select at most5 claims/issuer and10/source, with required configured token/character ceiling. Use accepted serving selection, then current blocking-only safety checks. Missing research queues verification asynchronously and classification proceeds with existing evidence rules; no inline document search, issuer investigation or publication wait.

Keep exposure context in an explicitly versioned namespace separate from original-source evidence. Preserve old GroundingContext v1 serialization and historical hashes. Extend economic packet preparation with a context content hash containing selected claim/evidence revisions and relevant issuer/theme semantics/policies. Serving generation ID is provenance only, not the semantic cache key. Keep it on `ExposureGroundingUse`, outside the research-context subset incorporated into the `EvidencePacket` input hash. Do not embed generation IDs or safety-check timestamps in the hashed `grounding_snapshot` and then accidentally invalidate the cache on every publication. The packet contains semantic content and stable selected-revision refs; the use record separately retains generation/check provenance. Reuse unchanged context after unrelated generation publication. A relevant changed context requires a new preparation/input identity under existing correction/reclassification rules, never mutation of an old frozen packet. Do not proactively reprocess the entire source history whenever a dossier updates.

```python
context_hash_inputs = {
    "claim_revision_ids": sorted(selected_claim_ids),
    "original_evidence_hashes": sorted(original_leaf_hashes),
    "issuer_mapping_semantics": issuer_mapping_fingerprint,
    "theme_definition_semantics": relevant_theme_fingerprints,
    "grounding_policy": grounding_policy.version,
}
# Serving generation, job retry count and retrieval timestamp are audit fields,
# not semantic changes to otherwise identical context.
```

Record generation ID, selection, assessment/claim/original passage refs, context hash, safety revision, retrieval/grounding policy and actual use on `ExposureGroundingUse`. A source request/attempt link may be completed through a separate immutable use-link event rather than updating the frozen context payload. Current context must have been accepted/available before use. Original source timing and original research publication/effective periods remain separate; no attention timestamp reset from dossier maintenance.

Review source citations and context citations separately. Primary research support becomes contextual inference when interpreting another source, not direct evidence that the new source states it. Orders for a supported product can ground a narrowly justified theme link; unstated AI demand, named customers, HBM sales or revenues remain unasserted. Explicit source evidence can still independently support a direct claim under the existing reviewer.

Flatten research verification provenance to original-primary leaves; reject using downstream classifications generated with the same assessment as new primary corroboration. Recheck fresh_until, mapping/policy and hold dependencies before provider use, persisted new classification, and any membership action using that result. An already-running call may finish for audit but cannot commit a disqualified contextual use. Fallback is to existing-source-only processing or retry with approved current context, never silently substitute an unpinned newer assessment.

Feature/mode/capability gates default grounding off and require its separate validation before activation. Local current holds still apply even if new acquisition is disabled.

- [ ] **4. Run the focused command again, then the additional checks.**

Run product-link positive and generic-price negative fixtures, no unstated causation, held/expired context before and during calls, self-support DAG, context-size limits, old/v2 compatibility and cache invariance across unchanged generations. PostgreSQL race tests prove stale contextual decisions cannot commit after a new hold. Existing source-only behavior must be unchanged with feature off.

Expected: focused tests/regressions pass. Record actual execution mode; mocked retrieval is not a live probe.

- [ ] **5. Review the diff, stage only this task's files, and commit.**

```bash
git diff --check
git diff --stat
# Stage the exact changed paths from this task after reviewing them.
git commit -m "feat: ground theme classification with pinned exposure claims"
```



---

### Task 25 — Build the adjudicated corpus and executable quality/capability gates

**Dependencies:** 25A corpus schema/ownership/tag gate: 00; collect labels during US shadow delivery and each market task. 25B final admission evaluation: 09–12,14–20 plus applicable preparation paths. Grounding has a separate validation gate after 24.
**Approved-spec coverage:** §17, §18; all E01–E15,I01–I11,R01–R15

**Files and ownership**
- Create `backend/app/services/company_exposure/evaluation.py`.
- Create `backend/scripts/evaluate_company_exposure.py`.
- Create corpus manifest/schema/adjudication records under `backend/tests/fixtures/company_exposure/corpus/`, retaining permitted originals via hashes/private test assets.
- Create unit `test_evaluation_gate.py`, `test_fixture_integrity.py`, `test_case_tags.py`.
- Complete all 41 case-tag/layer mappings from Appendix D.

**Interfaces**
`evaluate_corpus(corpus_manifest, runner, policies) -> EvaluationReport`; `quality_gate(report, scope) -> GateDecision`. `EvaluationReport` includes exact denominators, errors, recall/recovery, coverage, actual/mock mode, source/policy/model hashes and reviewer adjudications. No model self-score can set an acceptance label.

- [ ] **1. Add the failing tests.** Place these direct tests in the named module; construct real service fixtures locally using the declared Interfaces. Tag applicable requirements and add the named counterexamples below.

```python
from app.services.company_exposure.evaluation import admission_gate

def test_zero_errors_with_no_eligible_recovery_is_not_a_pass():
    result = admission_gate(critical_errors=0, eligible_total=40, recovered=0)
    assert result.allowed is False

def test_required_nontrivial_recovery_threshold():
    assert admission_gate(critical_errors=0, eligible_total=40, recovered=32).allowed
    assert not admission_gate(critical_errors=1, eligible_total=40, recovered=40).allowed
    assert not admission_gate(critical_errors=0, eligible_total=39, recovered=39).allowed
```

- [ ] **2. Run the focused test before implementation.**

```bash
cd backend && ./venv/bin/pytest tests/unit/company_exposure/test_evaluation_gate.py tests/unit/company_exposure/test_fixture_integrity.py tests/unit/company_exposure/test_case_tags.py -q
```

Expected: the new test fails for its missing interface or intended behavior, not an unrelated environment error; existing regressions remain passing.

**Human adjudication milestones (not agent work to silently synthesize):** Task 00 records the product owner/user as accountable corpus owner and the requirement for a second appointed human reviewer. During 17A/21A, obtain the first reviewed US evidence/labels and test failure/report tooling. During 10–12, market-owning implementers collect permitted documents and propose annotation packets; the owner assigns language-capable reviewers, who may use translations but must validate ambiguous original passages. Before the holdout is frozen, two humans sign off every proposed auto-eligible and critical-negative case; disputes retain both labels and the final owner decision. Freeze splits before tuning/evaluation, and record who/when/source hashes. No reviewer is assumed assigned, and no corpus is described as adjudicated until records exist. Missing human work blocks automatic admission, **not** US verify-only shadow delivery.

- [ ] **3. Implement this task's contract.**

Build—not invent an already-adjudicated—minimum80 issuer–theme case corpus with≥20 per launch market and the five original theme families represented overall. Retain exact permitted original revisions, passages, annotations, disputes, final reviewer adjudication and correction/mirror relationships. Require actual negative/ambiguous/stale/synthesis/image/table/cross-listing examples, plus deterministic synthetic unit counterexamples clearly marked synthetic. Synthetic translations or model-generated labels do not count as reviewed primary evidence.

Partition by issuer AND originating disclosure, joining groups when cross-listing/mirror reuse connects them. Do not split near-identical passages across development/holdout. One feasible allocation is16 development and64 heldout cases with40 adjudicated auto-eligible heldout cases, provided the grouping constraints hold; the approved minimum is the gate, not an instruction to force an invalid split. Record actual counts and do not shrink denominators for inconvenient provider failures or held results.

Frozen expected outcomes include required/forbidden claims, basis, role, commercial stage, materiality with operands, membership, freshness/holds, original citation identity and provenance limits. The corpus evaluator must call implemented code with recorded external-provider responses at the boundary; expected labels stay separate from responses. Add a test that deliberately returns the wrong claim/admission and proves the evaluator fails. No expected fixture values may be fed as production outputs or treated as an implementation.

Automatic admission gate: zero observed critical violations in the heldout contract corpus; ≥40 eligible heldout cases across launch markets; recovered≥80%; all mechanical E/I/R safety cases pass. Report scope-specific market/role/category failure as review-only until fixed, without disabling source ingestion or read access. Report sample sizes and uncertainty; zero observed violations does not imply population-perfect accuracy. Measure primary-support accuracy, role/application/status, calculations, false additions, holds, useful coverage and resource use separately.

```bash
cd backend
./venv/bin/python scripts/evaluate_company_exposure.py   --corpus tests/fixtures/company_exposure/corpus/manifest.json   --mode recorded --report /tmp/company-exposure-evaluation.json
```

The script writes a nonzero exit on missing source bytes without declared permission outcome, missing adjudication, leakage, missing required cases, incorrect outputs or failed gates. A live-provider evaluation must be explicit, budgeted and isolated, and its report cannot be mislabeled as the recorded-fixture run. Store report hash/policy/model/source provenance for release capabilities. No scope is declared passed until those reports actually exist.

- [ ] **4. Run the focused command again, then the additional checks.**

Run evaluator tests against passing, held-only, wrong-claim, missing-ID, leaking and failing-market manifests. Record all41 test ownership and actual collection. Run the real retained corpus through recorded providers, then an explicitly enabled bounded model probe for deployment validation where required; credentials/exhausted allowance must remain visible failures, not silent exclusions.

Expected: focused tests/regressions pass. Record actual execution mode; mocked retrieval is not a live probe.

- [ ] **5. Review the diff, stage only this task's files, and commit.**

```bash
git diff --check
git diff --stat
# Stage the exact changed paths from this task after reviewing them.
git commit -m "test: gate exposure verification with an adjudicated corpus"
```



---

### Task 26 — Enforce exact PostgreSQL, reader and no-spend release contracts

**Dependencies:** 26A first-slice mechanical/security/no-spend checks join 00–07,09,14–17A,21A/22A. 26B generation/membership/full-market release checks join their completed tasks and 25B; renderer only after 08B, grounding only after 24.
**Approved-spec coverage:** §17.4, §18; mechanical cases R01–R15 and all changed consumers

**Files and ownership**
- Create `backend/tests/required_company_exposure_cases.json` and `backend/scripts/run_required_company_exposure_postgres.py`.
- Reuse/refactor compatible exact-node gate utilities from `backend/scripts/run_required_economic_taxonomy_postgres.py` without weakening its existing gate.
- Modify `.github/workflows/ci.yml` and consumer capability inventory.
- Create unit `test_required_postgres_gate.py`, `test_capability_gate.py` and integration `test_full_exposure_rehearsal_postgres.py`.

**Interfaces**
`run_required_company_exposure_postgres` verifies actual PostgreSQL and exact collected test IDs; `ExposureCapabilityReport` pins source/code/migration/backend/frontend/evaluation/market-permission artifacts. Existing ReaderCapabilityManifest remains the serving readiness authority with explicit extension support.

- [ ] **1. Add the failing tests.** Place these direct tests in the named module; construct real service fixtures locally using the declared Interfaces. Tag applicable requirements and add the named counterexamples below.

```python
@pytest.mark.exposure_layer("unit")
@pytest.mark.parametrize("outcome", ["missing", "skipped", "xfailed", "xpassed", "zero_collected"])
def test_tag_gate_rejects_incomplete_required_layer(case_gate, collected_report_factory, outcome):
    report = collected_report_factory(case_id="R07", layer="postgres", outcome=outcome)
    assert case_gate.validate(report, required={("R07", "postgres")}).passed is False

@pytest.mark.case("R01")
@pytest.mark.case("R15")
@pytest.mark.exposure_layer("deployment")
def test_credentials_do_not_enable_research(default_boot, provider_spy, search_spy):
    default_boot.run_scheduled_tasks()
    provider_spy.assert_not_called()
    search_spy.assert_not_called()
```

The test module defines the input/service fixtures named here against this task’s Interfaces; factories only build typed inputs and database rows. Any thin test convenience wrapper calls the real service and returns its real result. Do not add a production method solely to satisfy a fixture or substitute a precomputed effect report. Additional assertions in the implementation steps remain required, even when not repeated in this representative block.

- [ ] **2. Run the focused test before implementation.**

```bash
cd backend && ./venv/bin/pytest tests/unit/company_exposure/test_required_postgres_gate.py tests/unit/company_exposure/test_capability_gate.py -q
```

Expected: the new test fails for its missing interface or intended behavior, not an unrelated environment error; existing regressions remain passing.

- [ ] **3. Implement this task's contract.**

Populate a checked-in manifest of required case-tag/layer/scope entries and named additional mechanical invariants for issuer/schema immutability, lease/resource contention, assessment revision CAS, producer lock ordering, membership/admin conflict, selection sealing, cutoff progress, relevant hold races, publication crash recovery, projection ordering, issuer-facade history and grounding hold races. Collect first and materialize exact node IDs into the run artifact; fail on a missing required case/layer or invariant. Every collected required test must pass, not merely one test carrying the tag. Verify the live test connection dialect/server is PostgreSQL and uses the explicitly permitted disposable URL. Fail on any skipped/xfail/xpass/failure or zero collection of a required case. SQLite passing is never a substitute.

Reuse the existing gate's outcome plugin where appropriate, preserving its taxonomy manifest. Test the gate itself with intentionally missing/skipped/xfailed nodes. Parametrized test node IDs are generated from actual collection. Required case/layer coverage cannot be reduced by discovery, and failing/skip/xfail outcomes cannot be hidden by a second passing test with the same tag. Preserve the existing taxonomy exact-node gate unchanged; this feature uses the new tag/layer manifest.

```bash
cd backend
DATABASE_URL=postgresql://ci:ci@localhost:5432/ci STOCKSCANNER_TEST_ALLOW_POSTGRES=1 ./venv/bin/python scripts/run_required_company_exposure_postgres.py
```

Add this step to the PostgreSQL-backed CI job. Normal CI blocks external HTTP/provider dispatch by default and uses saved documents. Renderer/network security tests use a controlled local test environment with explicit fake network policy, never real cloud/internal destinations. Run dependency vulnerability/license checks for the new renderer/parser lock and the full relevant frontend build/tests.

Verify all new member/basket/stock/digest/Social/MCP consumers use the same selected generation and origin union; register test inventory hash and bumped reader extension versions after the tests actually pass. Old generation serializers remain supported. New code must reject a reader capability lacking exposure support before live exposure admission/grounding, without breaking existing economic mode's source-only operation.

Compose the rehearsal with research evidence, source/Social origins, global rejection, two cross-listings, a customer excluded by producer policy, unknown materiality, scope-invalid materiality, late old capture, hold-after-capture, provider pause, out-of-order projection and crash-aftercommit. Record actual generation IDs/hashes, checkpoint revisions and before/after source-count invariants. No report fabricates release success because a test file exists.

- [ ] **4. Run the focused command again, then the additional checks.**

Run full relevant backend regressions including existing taxonomy gate, all new exact PostgreSQL nodes, source/Social/publication/LLM regressions, Vitest and production build. Save explicit failed/unavailable states. Release capability requires the actual final schema head recorded by the last migration-owning task and rechecked at merge, not a guessed literal migration suffix.

Expected: focused tests/regressions pass. Record actual execution mode; mocked retrieval is not a live probe.

- [ ] **5. Review the diff, stage only this task's files, and commit.**

```bash
git diff --check
git diff --stat
# Stage the exact changed paths from this task after reviewing them.
git commit -m "test: enforce company exposure release and concurrency gates"
```



---

### Task 27 — Deliver staged activation, four-market probes and operator recovery runbook

**Dependencies:** 27A US verify-only shadow release: 00–07,09,14–17A,21A,22A,04 worker,26A. 27B final staged launch: all tasks applicable to the privilege being enabled, including four-market admission gate. No blanket 00–26 dependency on first delivery.
**Approved-spec coverage:** §18, all approved decisions, E/I/R gates; R01,R02,R07,R09–R15

**Files and ownership**
- Create `backend/scripts/company_exposure.py` and `docs/runbooks/company-exposure-map.md`.
- Update environment/operations documentation and link approved spec, ADR, plan and report artifacts.
- Create unit `test_runbook.py`, `test_activation.py`; live probes under `backend/tests/live/company_exposure/` opt-in only.
- Preserve completed plan checkboxes only after their actual tests/commits have been verified.

**Interfaces**
CLI commands `status`, `import-identities`, `set-mode`, `set-stage`, `verify`, `discover`, `prepare-preview`, `probe-market`, `disable-acquisition`, `reconcile-compatibility`, `inspect-holds`. Mutations use trusted admin/service authority; publication delegates to the existing coordinator, never direct pointer SQL.

- [ ] **1. Add the failing tests.** Place these direct tests in the named module; construct real service fixtures locally using the declared Interfaces. Tag applicable requirements and add the named counterexamples below.

```python
@pytest.mark.case("R15")
@pytest.mark.exposure_layer("unit")
def test_us_shadow_allowed_while_full_admission_is_held(activation, us_shadow_capability):
    assert activation.evaluate(stage="shadow_verify_us", capability=us_shadow_capability).allowed
    assert not activation.evaluate(stage="automatic_admission", capability=us_shadow_capability).allowed

@pytest.mark.case("R15")
@pytest.mark.exposure_layer("integration")
def test_disable_preserves_existing_history(activation, reader, published_generation, provider_spy):
    before = reader.read_for_security(42, generation_id=published_generation.id)
    activation.disable_acquisition()
    assert reader.read_for_security(42, generation_id=published_generation.id) == before
    provider_spy.assert_not_called()
```

The test module defines the input/service fixtures named here against this task’s Interfaces; factories only build typed inputs and database rows. Any thin test convenience wrapper calls the real service and returns its real result. Do not add a production method solely to satisfy a fixture or substitute a precomputed effect report. Additional assertions in the implementation steps remain required, even when not repeated in this representative block.

- [ ] **2. Run the focused test before implementation.**

```bash
cd backend && ./venv/bin/pytest tests/unit/company_exposure/test_runbook.py tests/unit/company_exposure/test_activation.py -q
```

Expected: the new test fails for its missing interface or intended behavior, not an unrelated environment error; existing regressions remain passing.

- [ ] **3. Implement this task's contract.**

Implement a dry-run-first CLI with bounded arguments, explicit modes, status reasons and request IDs. Ordinary source-only economic mode is not disabled because research is off. New feature permissions are separately gated in the order below; user approval of this document is not the operator's production activation command.

1. Apply additive migrations in disposable rehearsal, import issuer attestations and existing source constituents as unverified research leads; compare grouping/membership before any authority facade switch.
2. Enable research `shadow` with explicit subscription route/local allowance and no paid search. Prove all four market adapters/language/format pipelines and review UX. Nothing changes live membership or grounding.
3. Verify upgraded readers and publish generation-bound assessment reads through the existing coordinator. Role-policy templates remain unapproved until reviewed.
4. Enable eligible membership additions only for approved role-policy and validated market/category scopes. Prove origin preservation, holds and compatibility recovery.
5. Enable bounded discovery for selected themes. Paid search stays false unless separately explicitly enabled with configured costing/caps.
6. Enable targeted grounding last, after all context safety/cache/provenance gates pass.

```bash
cd backend
./venv/bin/python scripts/company_exposure.py status
./venv/bin/python scripts/company_exposure.py import-identities --dry-run
./venv/bin/python scripts/company_exposure.py probe-market --market US --allow-network --max-documents 2
./venv/bin/python scripts/company_exposure.py probe-market --market HK --allow-network --max-documents 2
./venv/bin/python scripts/company_exposure.py probe-market --market JP --allow-network --max-documents 2
./venv/bin/python scripts/company_exposure.py probe-market --market TW --allow-network --max-documents 2
```

These probe commands are implemented by this task and run only on the documented disposable/sandbox store. They never authorize model calls or paid search implicitly. Add separate explicit `--allow-subscription-calls` for bounded real-model validation; existing budgets and mode gates still apply. Provide a fixture-driven dry-run mode for offline operators. No secrets appear in command lines/reports.

A successful live-market probe records the permitted route policy, actual URL/issuer/document hashes, publication/period, locator, parser language/capabilities and source response—not merely HTTP200. Record incomplete/forbidden/credential-missing routes accurately. Launch requires a working tested official/issuer route in each of US/HK/JP/TW; a missing optional EDINET key can coexist with a tested verified issuer route, but cannot be represented as a passed EDINET probe. Paid services remain off unless the operator separately activates them.

Document stop conditions: authorization failure, no local allowance, unknown paid-search maximum cost, scope/issuer conflict, invalid original locator, unacceptable provider result, stale safety token, missing selection extension, reader mismatch, required PostgreSQL skip, failed adjudication gate, incompatible membership origin or projection health. Explain provider-free local holds, partial publication, job cancel/resume, uncertain reservations (including `expired_uncertain` at period close) and tombstone behavior. Explain registry-resolved versus administrator-reviewed issuer links, and how to resolve a job paused on a `review_required` CIK match.

**Evidence directory setup (R2).** Before first starting the `exposure-research` profile, the runbook creates the bind-mounted store and gives it to the backend's non-root user (uid/gid 1000, per `CLAUDE.md`): `mkdir -p ./data/exposure-evidence && sudo chown -R 1000:1000 ./data/exposure-evidence`. `status` reports a typed `storage_not_writable` hold when the worker cannot write there, instead of failing inside a job. The browser profile needs only the egress signing-key secret file; there is no Redis credential to provision.

Disable acquisition via feature policy, not deletion. Stop new admissions and grounding when their gates are revoked; retain old evidence/decisions and source/Social behavior. Reconcile compatibility outside publisher locks before a supported rollback. An older binary that cannot represent accepted research contributions is not a safe rollback target until compatibility is proven. Keep rollback limitation visible rather than synthesizing source rows to hide it.

Record actual source SHA, approved spec hash, final migrations, capability/test/benchmark hashes, model route-policy identity, market probe reports, generation/checkpoint IDs and operator principal in the handoff. Mark production-ready only after these artifacts exist. Do not update approved defaults, enable paid search or claim 100% verification as part of marking tasks complete.

- [ ] **4. Run the focused command again, then the additional checks.**

Run CLI no-spend/default tests and runbook contract checks; execute the exact staged rehearsal against disposable PostgreSQL and controlled documents. Perform only explicitly authorized live probes. The final branch review checks all41 cases, all18 decisions, source preservation, no new publisher, no circular evidence and the actual activation policy state. If external permissions/credentials remain absent, report the precise blocked capability and leave that gate off.

Expected: focused tests/regressions pass. Record actual execution mode; mocked retrieval is not a live probe.

- [ ] **5. Review the diff, stage only this task's files, and commit.**

```bash
git diff --check
git diff --stat
# Stage the exact changed paths from this task after reviewing them.
git commit -m "docs: gate and operate the company exposure map"
```



---

## Appendix A — Cross-task contracts and transaction rules

These R1 decisions make implementation-level boundaries explicit; the amendment record distinguishes original approval from new operational defaults. Task 00 encodes them as strict types, schema assertions and counterexample fixtures. A later task may refine internal code organization, but cannot change these public semantics without a documented spec amendment.

### A1. Immutable keys and typed results

Use UUIDs for new object identities, integers for `stock_universe.id`, UTC datetimes and Decimal values serialized as strings. Do not expose secret-containing URLs as identity strings.

| Type / key | Required identity or payload |
|---|---|
| `ResearchRequestInput` | kind verify/refresh/discover; issuer-or-security and theme; supplied link refs; requester idempotency token; requested root limits; trigger origin |
| `ResearchRequestKey` | authenticated requester namespace + idempotency token; scheduled trigger keys additionally include source relationship/policy/due revision; does not reset with attempt/model changes |
| `ResearchWorkKey` | request, stage, immutable stage input hash, policy bundle |
| `ResearchLimits` | exact approved default limits plus configured daily allocation; mode disabled, paid search false, subscription billing |
| `AssessmentScope` | issuer UUID, theme UUID and defining fingerprint, accepted issuer-link revisions, consolidated/standalone/segment scope |
| `AssessmentAttemptInput` | scope; frozen original evidence/passage/derivative refs; prior dossier revision; relevant policies; exact successful model artifact refs |
| `DocumentTarget` | market adapter, publisher/issuer refs, official document/provider identity or verified-origin URL, permitted-access policy, publication/period metadata, credential reference only when a fixed official API requires it |
| `DocumentQuery` | requested issuer/theme/question, document types and bounded time/scope; not an unrestricted natural-language browser objective |
| `AcquisitionLimits` | byte/page/navigation/document/request ceilings; parent/root budget ID |
| `CaptureResult` | retained revision or typed gap; raw content hash; capture ID; coverage; `changed` only for new content, not new download |
| `BlobRef` | content SHA-256, media type, byte length and private storage key; no public-storage credentials |
| `PreparedEvidence` | original revision refs, exact passages, required neighbors/tables, derivatives, omitted ranges and input/preparation hash |
| `PassageSelection` | bounded passage refs, required context refs, omissions, selection policy, token/character bound |
| `ClaimReviewBatch` | immutable claims/judgments; each has proposition, verified, support_basis, conclusion and citations; `claim(key)`/`has_verified_claim(key)` are deterministic lookups, not model calls; review holds, no dossier-wide deletion |
| `SynthesisDecision` | permitted or held, original primary leaves, ≤2 supported joins, scoped conclusion, forbidden extrapolations and policy |
| `MaterialityInput` / `FormulaInput` | Metric, disclosed value/range, unit/currency/period/reporting scope, original operand revision refs, compatible accounting basis, formula kind; no user-supplied compatibility flag is authoritative |
| `MaterialityMeasureResult` | disclosed/calculated/qualitative/unknown, value/range/metric/unit/currency/period/scope/denominator, formula/operands, hold reasons |
| `MembershipInputs` | candidate security/theme, issuer link, exact assessment/claim/role-policy/global-decision refs, prior research contribution, all frozen other-origin decisions, current safety token |
| `MembershipEvaluation` | outcome `eligible_addition|candidate|held|retained_review_required|rejected`; reasons, exact prerequisites, decision origin and review_proposal_required |
| `DispatchRequest` | root request ID, logical operation, exact route/model, resource units/bounds, fixed provider capability, safe request fingerprint |
| `ReservationTicket` | allocation/root IDs, allocation period and period end, units/bounds, attempt ID and state (`reserved`, `dispatched`, `released`, `reconciled`, `uncertain`, `expired_uncertain`); invalid/uncertain ticket cannot authorize a second dispatch |
| `StorageTicket` | shared store/root reservation ID, reserved blob/staging bytes and expiry; settles unique actual bytes, not provider currency |
| `KimiJSONResponse` | validated data, bounded response hash, provider request ID, reported usage or unknown, finish metadata; old complete_json remains dict-returning |
| `ArtifactRunResult` | reusable artifact_id or null, retryable, pause_reason and attempt reference; errors are never cached successful artifacts |
| `ProviderInput` | permitted Go route/model, messages as evidence-only data, policy/model hash, max_output_tokens, read_timeout_seconds and response format |
| `ProviderOutput` / `DispatchOutcome` | successful result or failure with `dispatch_phase` (`pre_dispatch` / `dispatched` / `uncertain`, classified from the actual `httpx` exception per Task 04), retryable/terminal flag, provider request ID, reported usage, known/unknown actual and response hash |
| `RegistryMatch` (R2) | security ID, market, identifier scheme/value (US: 10-digit CIK), candidate count, `ticker_confirmed`, registry and official-record capture revision IDs, entity title, matched ticker/exchange, resolver policy version |
| `RenderPacingTicket` (R2) | grant ID, nonce, approved host, method, single-use ticket ID, remaining byte allowance, expiry; issued only by the research worker's `RenderPacingSession` |
| `ReservationUsage` | immutable usage view: state, reserved units, reported/unknown consumed units, actual_dollar_cost only for monetary resources (None for subscription usage) |
| `SafetyDecision` | allowed, hold_reasons, evaluated_at, dependency-specific current revision token and minimum expiry; only blocks or defers, never changes evidence selection |
| `ExposureGroundingContext` | selected generation plus relevant issuer/assessment/claim/original passage refs, safety check, source/context citation namespaces, bounded content and semantic hash |

Simple outcomes used in snippets—`CoverageItem`, `DiscoveryResult`, `SearchResult`, `TaskOutcome`—are strict value types exported in `contracts.py`. `SearchResult.disabled(reason)` and `TaskOutcome.skipped(reason)` are named constructors producing empty outputs plus explicit coverage/status, not exceptions masquerading as zero findings. QuestionSet, DocumentRevisionRef, PageImageRef, DecisionRequest/Preview and other `*Ref` names identify the corresponding typed revision UUID/hash with no implicit latest lookup.

### A2. Evidence ordering and assessment refresh

A document capture has a receipt time; the document has its own publication/reporting/correction identity. Repeated identical bytes are checks. An official correction can supersede the corrected proposition/period only; a later periodic report can coexist with older still-valid role evidence. A delayed archived or partial capture is not fresher because its database ID is larger.

All assessment work freezes the previous accepted/proposed dossier selection and dependency hashes. At persistence, CAS the expected dossier/link/policy state. Conflicting concurrent completions are both retained as evidence; reconcile selected claims explicitly rather than last-completion-wins. Missing/failed routes do not remove claims. A positive finding for one question cannot imply coverage of all questions.

Successful artifacts are keyed by actual input/policy/model. Failed attempts are append-only histories, not success-cache occupants. A provider attempt with uncertain remote execution is neither a guaranteed failure nor free allowance; preserve uncertainty until reconciled/explicitly bounded retry.

### A3. Acyclic exposure selection and generation construction

Do not define a content hash recursively over both a parent manifest and its derived selection. Use this exact build order:

1. Inside the existing short cutoff-capture fence, choose the exact available research assessment/link/policy/hold/membership revision IDs and safe projection inputs. Allocate a UUID for `ExposureSelectionSet`, persist its unsealed identity shell with its immutable `input_fingerprint`, and capture the same UUID/fingerprint and raw revision lists in a **new nullable versioned `GenerationInputManifest.exposure_inputs` field**. Existing source-lineage selection entries keep their existing shape.
2. Seal the existing input manifest. Its exposure section describes captured inputs and the reserved output ID; it does not pretend an uncomputed output hash is already known. No provider, model or unbounded assessment work occurs in this capture transaction.
3. Outside the exclusive fence, deterministically fill the reserved selection from only those captured revisions and applicable policies. The set references the parent manifest ID for provenance; this parent ID is excluded from the set's semantic hash. Seal its child rows/payload and compute semantic/integrity hashes.
4. Build membership and exposure snapshots from that sealed set; include exact data needed for reader reproduction. Persist the existing serving generation with a nullable exposure-selection FK and its output hash in the versioned generation hash payload. No generation can publish with an unsealed or differently sourced exposure selection.
5. At final compare-and-set, validate artifact/input hashes, selected dependency safety and current expiry for proposed automatic additions, expected parent and existing reader/compatibility gates. Disqualifying support aborts/rebuilds only the affected unsafe proposal; existing accepted members can remain held-for-review. Ordinary unrelated post-cutoff work is backlog.
6. Publish pointers through the existing coordinator and durable publication event. Post-commit notification is optional. Reads resolve the generation once and cannot consult a new research live pointer.

The selection's reserved shell may be abandoned if preparation fails; it is never served directly. A retry of the same captured inputs reuses the reserved identity or creates a separately auditable abandoned/replacement artifact under a new outer capture, never mutating a sealed set. Selection uniqueness prevents duplicated dossier or membership rows within the set.

Old manifests have `exposure_inputs=NULL`; old generation hashes/serialized payloads omit the new extension rather than inserting a new null key and rehashing. Validation dispatches by an explicit serializer/hash schema version. Old readers supported by a new binary can request G0 and receive “not captured for this generation”; no upgrade backfills invented research into G0.

### A4. Safety versus history

Historical products are generation-pinned. A current safety overlay is permitted only for refusing a new automatic action or explicitly displaying a separately labeled current safety warning; it does not rewrite the historical assessment. Each safety check binds its time, claim expiry and current scoped hold/link/policy revisions. Recheck before provider use, authoritative result commit, and activation of an automatic addition.

Do not globally invalidate every prepared generation when unrelated research changes. Register safety invalidators for exact used dependencies. If an already accepted member becomes stale, retain it review-required under the user's policy. The publisher must not spin forever trying to remove that member automatically; it can publish a coherent held status and review request. New qualifying primary evidence cannot automatically lift a reviewed membership rejection.

### A5. Origins and compatibility ownership

The research membership association has its own immutable decisions and origin. A type-safe union reconciles source, Social, research and manual selections under global administrator precedence. It is not an array concatenation. Source corrections remove only their source contribution; research retraction/removal affects only its scope unless the administrator explicitly made a global decision.

`research-membership:<issuer>:<theme>` is an outbox transport owner, never an evidence family. Accepted projection revisions increment for selected membership/decision changes even with unchanged document bytes. Payloads replace the owner's complete contribution; an empty reviewed state retracts only that owner. Current administrator vetoes cannot be overridden by delayed deliveries. Event identity excludes epoch; attempts contain epoch. Deliverability comes from committed publication history, including superseded-but-published generations, not only the current pointer or an after-commit flag. Reverse mirror triggers are suppressed.

Existing issuer configuration is imported with exact history. After the facade transition, there is one accepted issuer-link selection, exposed through the existing Social attestation API shape. Administrator proposals can be pending publication; APIs/clients must not return them as already-live mapping. Historical configurations and generations are preserved.

### A6. Staged feature controls

`EXPOSURE_RESEARCH_MODE` is disabled/shadow/live. Separate stage capability flags gate accepted assessment reads, automatic additions, enabled-theme discovery and classifier grounding. All privilege-increasing stage transitions require administrator authorization and relevant gate hashes; provider credentials alone grant none. Shadow may build artifacts and display an authenticated job/revision-scoped non-authoritative preview, but cannot change production membership or grounding. It creates no new current product pointer.

Paid search requires explicit enabled flag, selected provider, secret reference, costing policy and caps. LLMs require allowed subscription capability and local request/token allocation. Paid search spending does not consume a fictitious LLM monetary balance. No new paid provider is automatically provisioned or used.

Disable acquisition without deleting published research. Continue local expiry holds and reviewed corrections, and stop new admissions/grounding when their respective capabilities are revoked. Source/Social ingestion keeps its existing behavior. An unsafe binary rollback is a reported blocked operation, not permission to generate fake source claims.

### A7. Authoritative commands and locks

All research evidence/assessment/link/hold/membership authoritative commits take the established order: taxonomy shared fence → authority → applicable issuer/Social registry → dossier/work/domain → outbox. Quota-only reservation transactions lock resource pools/root budgets in canonical order and end before network or taxonomy persistence; they must never enter the taxonomy fence while holding a reverse-order resource lock.

No nested transaction-owning helper from old Social administration is invoked inside that sequence. Provider dispatch is a durable stage boundary. The final publisher alone takes the exclusive fence and switches serving pointers. Authentication precedes business writes; actor is the trusted principal, not an HTTP body/header label.

## Appendix B — Complete persistent-state ownership

All revision/payload history is append-only or one-way sealed, with RESTRICT references and database/ORM protections. Operational queues/leases/checkpoints can update under explicit ownership; they are not historical evidence authority. This table is the schema-review checklist before migrations merge.

| Object | Minimum durable content and constraint | Task |
|---|---|---:|
| ExposureIssuer | Stable UUID, immutable identity provenance; descriptive legal names belong to evidence/revisions | 01 |
| IssuerIdentifierRevision | Issuer, typed scheme/market/id, evidence, status, revision; accepted selection rejects conflicting ownership | 01 |
| IssuerSecurityLinkRevision | `security_id`, issuer, scope, state, `acceptance_policy` (`administrator_reviewed` / `official_registry_single_listing` / `legacy_attestation_import`), actor/reason/evidence (registry and official-record capture revisions for registry links), prior revision; unique security/revision | 01 |
| LegacyIssuerAttestationBridge | Old configuration hash/version, original company_id/symbol/reference, imported issuer/link IDs, audit provenance; exact import idempotency | 01 |
| ExposureDocument | Stable provider document ID or canonical verified-origin identity, publisher/issuer/source-kind | 01 |
| ExposureDocumentRevision | Document/content-hash unique; original blob, publication/reporting metadata, app availability, correction identity | 01 |
| DocumentCaptureEvent | Check/retrieval result, sanitized URL, byte/hash metadata; repeated reads do not revise the business date | 01 |
| DocumentRelationRevision | Original/translation/mirror/correction/supersession relation, scope, evidence and precedence; unknown retained explicitly | 01 |
| ExposurePassage | Original revision, locator+policy hash, source text/context, page/table/region, coverage flags | 01 |
| PassageDerivative | Passage parent, translation/image preparation, model/policy/input hash, output and uncertainty; not primary leaf | 01 |
| EvidenceTombstoneEvent | Removed blob/reference, hash, reason/authority and time; cannot erase earlier provenance | 01 |
| ExposureRuntimePolicyRevision | Namespace/subject/revision unique; immutable settings/route/permission/costing/feature payload, trusted approval events and exact job references; secrets external | 02 |
| ExposureResearchRequest | Stable request envelope/root/parent, kind/scope/trigger, trusted requester/idempotency key and frozen requested limits | 02 |
| ResearchEvent | Per-request ordered append-only stage/outcome/progress/generation events | 02 |
| ResearchCandidate | Root/theme/issuer or unresolved listing, seed provenance, selection/ranking rationale, discovery bounds and candidate history | 02 |
| ResearchWorkLease | Work stage key, cached status, lease/token/expiry, retry availability; no reset of root budgets | 02 |
| ResearchInputManifest | Immutable stage evidence/policy/identity/prior-selection hashes and refs | 02 |
| ResearchReservation | Root/pool/period/unit/currency/bounded amount, policy and attempt; shared-store `blob_bytes` covers storage and staging; operational state via events/checkpoint | 02 |
| ResearchReservationEvent | Reserved/dispatched/reconciled/uncertain/released/expired_uncertain, with `dispatch_phase`; actual reported versus unknown usage | 02 |
| ResearchProviderAttempt | Logical operation/attempt number unique, permitted route/model, parameters/input hash, dispatch provenance | 02 |
| ResearchProviderResult | Immutable result/error/remote request ID/usage state; a successful reusable artifact references actual result | 02 |
| ResearchArtifact | Successful input/policy/model unique result, text/response hash, parent refs; no retryable-failure artifact | 02 |
| ResearchCoverageItem | Requested/searched route/document/page/question and completed/omitted/unavailable outcome, explicitly scoped to attempt | 02 |
| ExposureClaim | Stable issuer/theme/kind/product-or-activity/scope proposition identity | 03 |
| ExposureClaimRevision | Revisioned proposition evidence/conclusion/support/status/time/hold dependencies and exact policies | 03 |
| ClaimEvidenceLink | Supporting/conflicting typed original/derivative/claim dependencies with exact locator and supported join scope; DAG validated | 03 |
| MaterialityMeasure | Claim revision, disclosed/calculated/qualitative/unknown, Decimal/range, metric/unit/currency/period/scope/denominator, formula+operand revisions | 03 |
| IssuerThemeAssessment | Stable dossier; unique issuer/theme | 03 |
| AssessmentRevision | Dossier/revision unique, dossier/input-manifest hash replay unique, prior revision, evidence selections and coverage | 03 |
| AssessmentClaimSelection | Assessment revision/proposition selection unique; retains usable older claims alongside new periods | 03 |
| RoleEligibilityPolicyRevision | Theme, fingerprint, roles/stages/scopes/prerequisites, reviewed state/principal/reason, policy revision | 18 |
| ExposureUseHoldRevision | Typed subject and hold/lift revision, scope/reason/support refs/actor, current safety token derivation | 03 |
| ExposureDecisionRequest/Preview/Event | Immutable request/preview/input hashes, affected scopes, trusted actor; append-only apply/reject/fail transitions | 18 |
| ExposureMembershipAssociation | Stable theme/security pair identity | 18 |
| ExposureMembershipDecisionRevision | Association/revision unique, selected claims/policy/link, origin/global decision refs, safety, accepted/held/rejected/retained-review-required | 18 |
| ExposureSelectionSet/Selection | Reserved UUID+input fingerprint, unsealed→sealed, one selected assessment per dossier, exact issuer/policy/hold/member selections and as-of | 19 |
| GenerationInputManifest exposure extension | Nullable typed exposure_inputs, exact captured refs/reserved set ID/input fingerprint; unchanged old serializer | 19 |
| ServingGeneration exposure binding | Nullable selection-set reference and explicit output-hash version; no second current pointer | 19 |
| Research legacy contribution | Transport owner/security/theme/origin, selected revision payload and checkpoint; never masquerades as a source mention | 20 |
| ExposureGroundingUse/UseLink | Immutable context refs/hash/generation/safety and source processing request/attempt link; append-only link completion | 24 |

Settings, market acquisition permissions, enabled-theme policy, subscription-route capability, costing policy and feature-stage changes are versioned administrator decisions in the research policy/decision store. They are not silently mutable environment-derived assessment facts. Environment values supply secrets and initial disabled configuration; accepted policy records identify the runtime behavior used by each job.

New selected issuer/source/hold revisions cannot be deleted while referenced. An immutable ID is not enough to prevent cross-scope references: validate issuer/theme/document/reporting scope and selected policy coherence before seal. Security corporate-action conflicts are reviewed rather than rebinding historical rows.

## Appendix C — Dependency graph and usable delivery slices

Task numbers are stable review references, **not a compulsory serial order**. A/B substeps are independently testable commits. Claims consume retained `PreparedEvidence`; no external search or international adapter is a prerequisite for building them.

| Work branch | Hard dependencies / permitted parallelism |
|---|---|
| 00 → 01 → 02 → 03 | Core contracts, evidence, work/resource history and assessments only; no speculative membership/serving tables |
| 04 subscription transport + worker bootstrap | 02; independent of 03 and all acquisition/search adapters |
| 05 issuer links | 01–02, no model dispatch |
| 06 storage/network → 07 structured evidence | 01–02/05; saved text/HTML/PDF processing is provider-independent |
| 14 claims and 15 materiality | 03,04,07; may run in parallel using saved evidence; 16 integrates both |
| 09 US, 10 HK, 11 JP, 12 TW, 13 optional search | Parallel from common06–07 contracts; market adapters use05. 13 has no dependency on09–12 |
| 17A verify/refresh | Core verification + **one installed adapter**, US09 for first delivery; retained/supplied documents need no search |
| 21A operational preview → 22A usable shadow panel | 16/17A; no serving-pointer or basket changes; no dependency on18–20 |
| 08A shared translation/vision adapters; 08B renderer deployment | 04,06–07; optional document-specific capabilities, not gates for text-only US verification |
| 18 membership, 19A assessment serving, 19B membership serving, 20 compatibility | 18 needs assessments;19A can precede18;19B joins18;20 required before live additions |
| 21B/21C,22B | Serving reads and reviewed operations after their respective19/20 contracts |
| 17B discovery;23 recurring schedules | Selected enabled adapter and shared verification;13 optional; on-demand worker already usable before recurring schedules |
| 25A corpus ownership/schema | Start at00, collect US labels during17A/21A, expand at10–12; humans adjudicate before25B |
| 25B/26B full admission gates | Four markets, primary verification, membership/generation/compatibility and relevant preparation; grounding validated separately at24 |
| 27A first shadow release | 00–07,09,14–16,17A,21A,22A,26A; worker introduced04 |
| 27B full staged release | All tasks/gates for privileges enabled; no implication that paid search is enabled |

### Slice S1 — Usable US-only verify-only shadow research

Ship an authenticated UI/API that accepts an existing US issuer–theme relationship, resolves its CIK through `USIssuerResolver` + `accept_registry_match` (pausing as `review_required` on any ambiguity), accepts supplied/retained official documents or uses the US adapter, runs bounded primary verification, shows original passages/materiality/coverage, and resumes paused work. `EXPOSURE_RESEARCH_MODE=shadow`; search, discovery, live additions and classifier grounding remain off. It uses exactly `POST /research-requests`, `GET /research-jobs/{job_id}` and `GET /research-jobs/{job_id}/preview`. Preview responses say `view_kind=shadow_preview` and `authoritative_membership=false`; they refer to immutable job/assessment IDs and do not create a new live authority. A minimal usable panel is required; a CLI-only or JSON-only backend is not the stated slice.

S1’s own schema/security/subscription/no-spend/HTTP/storage tests must pass. It does not wait for the 80-case corpus, HK/JP/TW, browser isolation, paid search or complete publication integration. Unavailable capabilities remain visible. Text PDF/HTML are sufficient for this slice, not a claim that they cover every document.

### S2–S5 — Extend capability without weakening gates

S2 adds remaining market and multilingual/visual acquisition plus corpus adjudication. S3 publishes generation-bound assessments, then reviewed origin-aware membership after compatibility. **Automatic admission retains D08’s four-market launch gate and the spec’s quality thresholds**; passing only US does not enable US automatic admission by accident. After the full launch gate passes, a subsequently failing category/market can be held review-only as the spec permits. S4 enables bounded discovery for selected themes; optional Tavily remains separately off/capped. S5 enables targeted classifier grounding after no-circular-support, freshness and provenance tests.

Each slice records its real code/migration/policy/capability hashes and tests; no early slice is labeled full V1 complete. Browser08B is part of full V1 capability delivery but can be disabled for a deployment, with image/browser-specific automatic claims unavailable until its probes pass. It does not hold unrelated text assessments hostage.

One integration owner serializes changes to shared schema exports, rate-budget primitives, publisher and issuer facade. Each migration owner rechecks `alembic heads` at its own commit and the merge gate; parallel branches do not reserve a distant numeric sequence. Independent markets and saved-evidence verification may otherwise progress concurrently.

## Appendix D — Case-tag traceability and required layers

The 41 case descriptions below are preserved verbatim from the approved specification. Requirements are identified by `@pytest.mark.case("ID")`, plus `@pytest.mark.exposure_layer("unit|schema|postgres|api|integration|deployment")` and slice metadata. Multiple direct tests can carry the same ID. **No canonical invented Python function suffix is asserted here.** Task00/26 collection writes exact real node IDs and outcomes to the run artifact and verifies every required case/layer/invariant for the requested slice. Both missing coverage and any failing/skipped required test fail that slice; a passing sibling cannot mask them. Frontend Vitest and deployment probes have an explicit companion mapping/report with their actual test IDs, not Python tags pasted into JavaScript.

| ID | Approved counterexample / expected result | Owner tasks | Required full-scope layers |
|---|---|---|---|
| E01 | AI and memory co-occur without a relationship: no verified AI Memory exposure. | 14 | unit |
| E02 | Official issuer product X is commercially available; another official passage links X to HBM testing: allow the narrowly worded synthesis; do not invent HBM sales or named customers. | 14 | unit |
| E03 | A supplies B and B manufactures HBM: missing application link remains unverified. | 14 | unit |
| E04 | A server segment supplies 30% of revenue: do not assign 30% to AI Memory. | 15 | unit |
| E05 | Compatible disclosed numerator/denominator: reproduce the decimal calculation with exact scope, period and operand citations. | 15 | unit |
| E06 | Conflicting periods, currencies, overlapping segments, or zero/negative profit denominator: hold the derived share; retain original numbers. | 15 | unit |
| E07 | A customer/consumer appears in primary evidence: the map may show the role but a producer-only basket rejects automatic admission. | 18 | unit |
| E08 | Supported current commercial exposure has no materiality breakdown: admission can pass the reviewed role policy and displays materiality unknown. | 18 | unit |
| E09 | Qualification/planned investment without the required commercial status: candidate only. | 18 | unit |
| E10 | Official transcript analyst question, search snippet, generated assessment or third-party report hosted by an issuer: cannot be primary evidence for the issuer claim merely by location. | 14 | unit |
| E11 | Japanese/Chinese negation, modal language, magnitude units, role direction and table context are preserved; material ambiguity holds only affected claims. | 08A,11,12 | unit (original-language market fixtures), frontend |
| E12 | A historical primary PDF is downloaded today: its business-evidence date does not advance. | 16 | unit |
| E13 | Same original report via exchange, issuer site and translation: no multiplied corroboration count. | 06 | unit |
| E14 | An old/partial capture arrives after a corrected source: cannot restore an obsolete exposure. | 16 | unit |
| E15 | An older document's still-valid role claim and a newer period's materiality coexist without source-level wholesale replacement. | 16 | unit |
| I01 | Two verified cross-listings reuse one issuer assessment; securities remain separately eligible and distinct issuer count stays one. | 05 | unit |
| I02 | Similar names or ticker reuse do not merge issuers. An administrator mapping conflict is held, not auto-overwritten. | 05 | unit |
| I03 | Subsidiary materiality cannot silently become consolidated-parent share. | 15 | unit |
| I04 | An administrator-rejected membership stays rejected after new primary evidence; a review proposal may be opened. | 18 | unit |
| I05 | Stale/disputed claims cannot support a new automatic admission or new grounding, but existing research membership remains flagged pending review. | 16 | unit |
| I06 | Research no longer finds a document: coverage worsens, not exposure truth. | 06 | unit |
| I07 | New-source product orders plus accepted product context can support contextual theme inference; it cannot establish an unstated AI-demand cause. | 24 | unit |
| I08 | Generic issuer price movement cannot fan out across all known themes. | 24 | unit |
| I09 | Research → classification → research self-support is rejected; original primary support remains the only verification basis. | 14,24 | unit, integration (grounding slice) |
| I10 | Theme split/mechanism change cannot blindly copy accepted exposure to all destinations. | 18 | unit |
| I11 | Same theme/security can retain accepted revision 1 and rejected revision 2; pair-level identity does not forbid revision history. | 18 | schema, postgres |
| R01 | Paid search configured with credentials but enable flag false: zero paid search calls. Missing spending cap also blocks paid calls. | 13 | unit, deployment |
| R02 | Subscription quota exhausted: durable pause, no metered fallback, no guessed dollar charges. | 04 | unit, deployment |
| R03 | Transient provider failure then success: one logical request, two immutable attempts, one reusable success; another retry issues no call. | 04 | unit, postgres |
| R04 | Ambiguous dispatched timeout/cancellation preserves uncertain allowance/cost reservation. | 04 | unit, postgres |
| R05 | Pause/resume, model change and child investigation cannot reset cumulative job budgets. | 02,17,23 | unit, postgres |
| R06 | Research and source/Social workers contend: one ordered fenced commit, with no provider call under the lock. | 19 | unit, postgres |
| R07 | G1 stays reproducible after G2 assessment, mapping, membership, hold and policy revisions. Old generations lacking the exposure extension remain valid. | 19,21,24,26 | unit, postgres, api, frontend |
| R08 | Assessment publication changes no source-attention root counts or events. Research membership can change basket coverage through its own provenance. | 19 | unit, postgres |
| R09 | Crash after generation commit before worker notification: compatibility delivery resumes from durable published state. | 20 | unit, postgres |
| R10 | New membership projection arrives before old one: old delivery is a no-op; another origin's support is not deleted. | 20 | unit, postgres |
| R11 | Ordinary research arrives during snapshot preparation: a coherent cutoff still publishes; a newly disqualified required support blocks only the unsafe automatic action. | 19 | unit, postgres |
| R12 | Time-based freshness expires without new external documents: holds and a subsequent generation are produced without a provider call; automation checks expiry even if publication lags. | 23 | unit, postgres |
| R13 | Forged actor, unauthenticated mutation, or paid-enable change: rejected before durable decision/spend side effects. | 21 | unit, postgres, api |
| R14 | Private-network redirect, credential leakage, oversized archive/PDF, hostile document instructions: blocked or sandboxed with a typed failure. | 06,07,08B | unit, deployment (browser only when installed) |
| R15 | Feature disabled or rolled back: no new research spending/admissions; historical artifacts remain and existing source/Social behavior is preserved. | 27 | unit, postgres, deployment |

Case-tag presence alone is not correctness. Unit cases call real services with saved evidence/mocked transport; PostgreSQL cases exercise independent sessions and actual constraints; deployment probes enforce OS/network boundaries. Add separately named invariants for storage reservation/GC races, provider-wide pacing, queue isolation, migration-head conflicts, and corpus reviewer completeness. The requirements manifest is reviewed source; collection cannot silently shrink it. S1 excludes future-only membership/generation/browser/grounding requirements but must satisfy its explicitly enumerated subset.

### Requirement-to-task map

| Approved decisions | Implementation ownership |
|---|---|
| D01 verification plus candidates | 13,17,23,25,27 |
| D02 claim-level primary backing | 06–08,14–16,21–22,25 |
| D03 tiered membership/review | 18–23,26–27 |
| D04 disclosed/calculated/qualitative materiality | 03,14–16,21–22,25 |
| D05 unknown materiality permitted | 15,18,22,25 |
| D06 reviewed theme-role eligibility | 18,21–22,27 |
| D07 bounded triggers/refresh | 02,17,23,27 |
| D08 US/HK/JP/TW | 05–12,25–27 |
| D09 original-language authority | 01,07–08,10–12,14,22,25 |
| D10 issuer-centric research, listing decisions | 01,05,18,20–22,25 |
| D11 public retrieval plus optional search | 06,09–13,17,27 |
| D12 subscription allowance | 02,04,08,13,17,23–27 |
| D13 paid search off by default | 00,04,13,21–23,26–27 |
| D14 multi-format retrieval | 06–08,09–12,25–27 |
| D15 targeted grounding | 19,21,24–27 |
| D16 stale/disputed claims held | 03,16,18–20,23–24,26–27 |
| D17 bounded primary synthesis | 03,14–16,24–25 |
| D18 dedicated layer, single publication authority | 00–03,18–24,26–27 |

### Coverage that must not be inferred from the fixture count

The 41 deterministic contract cases do not replace the ≥80 adjudicated issuer–theme corpus, the ≥40 eligible heldout cases/recovery gate, the four-market permission/retrieval probes, the exact-node PostgreSQL concurrency suite or the existing source/Social regression suite. Each has a separate report with its actual execution mode and scope. Do not call synthetic test statements company disclosures or report unexecuted real-provider tests as passing.

## Appendix E — Evidence and reference record

### Approved basis

The approved exposure design is the complete feature authority and travels with this plan. Its SHA-256 is recorded in the header. Prior economic-taxonomy planning attachments are historical background only; this plan extends the implemented code rather than executing Idea 1 again.

### Pinned repository references

These links support the baseline statements in this plan, not proposed new-module existence. Snapshot inspected: `28c220e4e4ca5afcb5a380678bb2b80dd7f388b7`.

- [`backend/app/models/economic_taxonomy_runtime_evidence.py`](https://github.com/xang1234/stock-screener/blob/28c220e4e4ca5afcb5a380678bb2b80dd7f388b7/backend/app/models/economic_taxonomy_runtime_evidence.py)
- [`backend/app/models/economic_taxonomy_runtime_publication.py`](https://github.com/xang1234/stock-screener/blob/28c220e4e4ca5afcb5a380678bb2b80dd7f388b7/backend/app/models/economic_taxonomy_runtime_publication.py)
- [`backend/app/services/economic_theme_observation_service.py`](https://github.com/xang1234/stock-screener/blob/28c220e4e4ca5afcb5a380678bb2b80dd7f388b7/backend/app/services/economic_theme_observation_service.py)
- [`backend/app/services/social_company_identity_service.py`](https://github.com/xang1234/stock-screener/blob/28c220e4e4ca5afcb5a380678bb2b80dd7f388b7/backend/app/services/social_company_identity_service.py)
- [`backend/app/services/economic_taxonomy_fence.py`](https://github.com/xang1234/stock-screener/blob/28c220e4e4ca5afcb5a380678bb2b80dd7f388b7/backend/app/services/economic_taxonomy_fence.py)
- [`backend/app/services/economic_taxonomy_publication.py`](https://github.com/xang1234/stock-screener/blob/28c220e4e4ca5afcb5a380678bb2b80dd7f388b7/backend/app/services/economic_taxonomy_publication.py)
- [`backend/app/services/economic_taxonomy_publication_preparation.py`](https://github.com/xang1234/stock-screener/blob/28c220e4e4ca5afcb5a380678bb2b80dd7f388b7/backend/app/services/economic_taxonomy_publication_preparation.py)
- [`backend/app/services/economic_taxonomy_publication_contracts.py`](https://github.com/xang1234/stock-screener/blob/28c220e4e4ca5afcb5a380678bb2b80dd7f388b7/backend/app/services/economic_taxonomy_publication_contracts.py)
- [`backend/app/services/economic_taxonomy_snapshot_builder.py`](https://github.com/xang1234/stock-screener/blob/28c220e4e4ca5afcb5a380678bb2b80dd7f388b7/backend/app/services/economic_taxonomy_snapshot_builder.py)
- [`backend/app/services/economic_theme_read_service.py`](https://github.com/xang1234/stock-screener/blob/28c220e4e4ca5afcb5a380678bb2b80dd7f388b7/backend/app/services/economic_theme_read_service.py)
- [`backend/app/services/theme_grounding_context.py`](https://github.com/xang1234/stock-screener/blob/28c220e4e4ca5afcb5a380678bb2b80dd7f388b7/backend/app/services/theme_grounding_context.py)
- [`backend/app/services/economic_exposure_extraction.py`](https://github.com/xang1234/stock-screener/blob/28c220e4e4ca5afcb5a380678bb2b80dd7f388b7/backend/app/services/economic_exposure_extraction.py)
- [`backend/app/services/theme_evaluation/public_fetch.py`](https://github.com/xang1234/stock-screener/blob/28c220e4e4ca5afcb5a380678bb2b80dd7f388b7/backend/app/services/theme_evaluation/public_fetch.py)
- [`backend/app/services/llm/llm_service.py`](https://github.com/xang1234/stock-screener/blob/28c220e4e4ca5afcb5a380678bb2b80dd7f388b7/backend/app/services/llm/llm_service.py)
- [`backend/requirements-server.txt`](https://github.com/xang1234/stock-screener/blob/28c220e4e4ca5afcb5a380678bb2b80dd7f388b7/backend/requirements-server.txt)
- [`backend/alembic/versions/20260925_0056_drop_economic_theme_embeddings.py`](https://github.com/xang1234/stock-screener/blob/28c220e4e4ca5afcb5a380678bb2b80dd7f388b7/backend/alembic/versions/20260925_0056_drop_economic_theme_embeddings.py)
- [`backend/tests/unit/test_economic_theme_observations.py`](https://github.com/xang1234/stock-screener/blob/28c220e4e4ca5afcb5a380678bb2b80dd7f388b7/backend/tests/unit/test_economic_theme_observations.py)

### Official external-interface references

- [Tavily Search official API](https://docs.tavily.com/documentation/api-reference/endpoint/search): bounded link discovery using the documented Search endpoint; no account, price or entitlement is assumed.
- [Playwright Python Docker guidance](https://playwright.dev/python/docs/docker) and [Docker Compose network reference](https://docs.docker.com/reference/compose-file/networks/): deployment inputs for the isolated-renderer design, not proof that an image/profile is safe without the specified tests.
- [SEC EDGAR APIs](https://www.sec.gov/search-filings/edgar-application-programming-interfaces) and [SEC access guidance](https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data): official discovery and access contracts, not theme-specific proof in entity-wide metadata.
- [HKEXnews title-search explanatory note](https://www.hkexnews.hk/homelcicontentsearch.html): official disclosure search purpose and relevant access/retention limitations; not blanket automation permission.
- [FSA EDINET portal and API registration guidance](https://disclosure2.edinet-fsa.go.jp/week0020.aspx): official registration/key and Japanese-specification entry point. Task 11 pins the actual v2 request/response contract before implementation, rather than relying on unofficial mirrors.
- [JPX TDnet overview](https://www.jpx.co.jp/english/equities/listing/disclosure/tdnet/): distinguish public disclosure paths from paid historical services.
- [TWSE official OpenAPI catalog](https://openapi.twse.com.tw/): appropriate identity/metadata assistance; does not establish a generic unrestricted MOPS report API.

These interfaces must be checked against official sources again at adapter execution because external access contracts can change. A documented interface is not a claim that the account is configured or a live probe has passed.


## Appendix F — Concrete worker, browser and storage deployment contracts

These are **deployment defaults proposed in R1 and accepted with R2** (the F.3 topology is R2-corrected), not measured capacity guarantees or proof of a running installation. They implement the already selected bounded-document scope without making Chromium or new paid search a prerequisite for the first US verification slice. Task04 owns F.1; Task08B owns F.2–F.4; Task06 owns F.5; Task23/26 own scheduling and deployment verification. The deployer explicitly opts in and records actual image digests/configuration.

### F.1 Dedicated research worker — available in the first usable slice

**Queue:** `exposure_research`. **Compose service:** `celery-exposure-research`. **Worker hostname:** `exposure-research@%h`. Route the bounded `app.tasks.company_exposure_tasks.*` tasks to that queue. Do not put model, document or investigation stages on `celery`, `data_fetch_*`, `market_jobs_*` or `user_scans_*`.

Use the existing backend source image/build mechanism. Add an opt-in `INSTALL_EXPOSURE_DOCUMENTS` build argument and a pinned `backend/requirements-exposure-documents.txt` for the PDF parser/rasterizer dependencies selected by Task07. The base server image/default worker builds remain unchanged. Chromium is **not** installed in this image. Record the parser dependency/license/security check and actual image digest; do not invent a future package or image digest in the plan.

Add the following service contract to `docker-compose.yml`; production resource/health/log overrides belong in `docker-compose.prod.yml`. The implementer expands the actual existing environment aliases **only where they do not import unrelated secrets**. Do not inherit the broad `x-worker-common`/`env_file` secret set wholesale.

```yaml
services:
  celery-exposure-research:
    profiles: [exposure-research]
    build:
      context: .
      dockerfile: backend/Dockerfile
      args:
        INSTALL_THEME_ML: "false"
        INSTALL_EXPOSURE_DOCUMENTS: "true"
    image: stockscreen-exposure-worker:r1
    command:
      - celery
      - -A
      - app.celery_app
      - worker
      - --loglevel=info
      - --pool=prefork
      - --concurrency=1
      - --prefetch-multiplier=1
      - --max-tasks-per-child=25
      - -Q
      - exposure_research
      - -n
      - exposure-research@%h
    environment:
      DATABASE_URL: postgresql://${POSTGRES_USER:-stockscanner}:${POSTGRES_PASSWORD:-stockscanner}@postgres:5432/${POSTGRES_DB:-stockscanner}
      REDIS_HOST: redis
      REDIS_PORT: "6379"
      CELERY_BROKER_URL: redis://redis:6379/0
      CELERY_RESULT_BACKEND: redis://redis:6379/1
      EXPOSURE_RESEARCH_MODE: ${EXPOSURE_RESEARCH_MODE:-disabled}
      EXPOSURE_PAID_SEARCH_ENABLED: ${EXPOSURE_PAID_SEARCH_ENABLED:-false}
      EXPOSURE_SEARCH_PROVIDER: ${EXPOSURE_SEARCH_PROVIDER:-none}
      OPENCODE_GO_API_KEY: ${OPENCODE_GO_API_KEY:-}
      OPENCODE_GO_API_BASE: ${OPENCODE_GO_API_BASE:-https://opencode.ai/zen/go/v1}
      TAVILY_API_KEY: ${TAVILY_API_KEY:-}
      EXPOSURE_DOCUMENT_STORE: /app/exposure-data
      EXPOSURE_STORAGE_MAX_BYTES: ${EXPOSURE_STORAGE_MAX_BYTES:-5368709120}
      EXPOSURE_STORAGE_MIN_FREE_BYTES: ${EXPOSURE_STORAGE_MIN_FREE_BYTES:-1073741824}
    volumes:
      - ./data/exposure-evidence:/app/exposure-data
    tmpfs:
      - /tmp:size=536870912,mode=1777
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
      backend:
        condition: service_healthy
    deploy:
      resources:
        limits:
          cpus: "1"
          memory: 2G
        reservations:
          memory: 256M
    stop_grace_period: 60s
    restart: unless-stopped
    logging:
      driver: json-file
      options:
        max-size: "20m"
        max-file: "3"
    healthcheck:
      test: [CMD-SHELL, "celery -A app.celery_app inspect ping -d exposure-research@$$HOSTNAME"]
      interval: 60s
      timeout: 20s
      retries: 3
      start_period: 30s
```

This is the exact queue/process/resource layout. Task04 fills the narrow remaining allowlist of required application settings from the real backend boot path, plus selected official API credentials and explicit local allowance/search-cost policies. Adding a required setting is not permission to pass all `.env.docker` values. Never mount Docker's socket, the whole host filesystem, SSH keys or a browser profile. The backend API needs the evidence directory **read-only** for authorized downloads; only the research worker/retention service writes blobs. Mount configuration matches `EXPOSURE_DOCUMENT_STORE` in each process. Add this to the existing `backend` service in `docker-compose.yml` (R2):

```yaml
services:
  backend:
    environment:
      EXPOSURE_DOCUMENT_STORE: /app/exposure-data
    volumes:
      - ./data/exposure-evidence:/app/exposure-data:ro
```

The directory must exist and be owned by uid/gid 1000 before either service starts (Task 27 runbook). A missing directory makes evidence downloads return a typed `evidence_store_unavailable` response rather than a 500.

In `backend/start_celery.sh`, add an `EXPOSURE_WORKER_ENABLED` conditional using the script's existing `POOL` convention (`solo` on the local macOS setup, `prefork` in the Linux container):

```bash
if [[ "${EXPOSURE_WORKER_ENABLED:-false}" == "true" ]]; then
  ./venv/bin/celery -A app.celery_app worker \
    --loglevel=info --pool="$POOL" --concurrency=1 \
    --prefetch-multiplier=1 -Q exposure_research -n exposure-research@%h &
fi
```

Each task advances a bounded work step. A provider request has the transport timeout and one actual dispatch per reservation; the worker lease is renewed or the step is durably retried, never allowed to complete with an expired owner. Queue isolation prevents research stages from occupying the price-fetch worker; it does **not** create extra upstream rate entitlement. Task06's central rate gate remains mandatory for every HTTP attempt. Keep shared root limits across stage continuations.

### F.2 Browser images and process isolation — optional later V1 capability

Create:

- `ops/exposure-renderer/Dockerfile`, `requirements.lock`, `server.py`, `seccomp.json` and a render health/probe entrypoint.
- `ops/exposure-egress/Dockerfile`, `requirements.lock`, `server.py`, and its health/probe entrypoint. (R2: no Redis client dependency or ACL script.)
- `backend/app/services/company_exposure/render_pacing.py` — the research worker's grant-scoped pacing RPC (F.3).
- `docker-compose.exposure-browser.yml` and `ops/exposure-runtime.lock.json`.

**Renderer image:** `stockscreener/exposure-renderer:r1`. Build from a locked, compatible Playwright/Chromium runtime; the lock records base image digest, installed Python/Playwright/browser versions, source/build hash and resulting image digest. Run the existing official Playwright sandbox recipe adapted for the supported host: non-root UID/GID10001, reviewed seccomp policy allowing only necessary Chromium user-namespace operations, and a functioning Chromium sandbox. No `--no-sandbox`, privileged container, host PID/network namespace or unconfined seccomp. A host that cannot support the sandbox reports `unavailable_capability`; it does not silently weaken isolation.

**Egress image:** `stockscreener/exposure-egress:r1`. A small Python HTTP-fetch broker using Task06’s safe public transport primitives (not its Redis pacing client), with its own locked dependencies. It is not a general open CONNECT proxy. It accepts only narrowly authorized fetch RPCs over a Unix socket and retrieves permitted public HTTP(S) content. The browser never receives provider API credentials.

### F.3 Enforced traffic path and Compose topology

Use **`network_mode: none` for the renderer**. This is stronger than trusting Chromium proxy flags or route interception alone. The renderer exposes its job RPC on `/run/exposure-render/render.sock`; the research worker shares only that socket directory. The renderer reaches the fetch broker on `/run/exposure-egress/egress.sock` through a second socket volume. It has no TCP connection to the application, PostgreSQL, Redis or the Internet.

**Three sockets, one direction each (R2):**

| Socket volume | Server | Client | Purpose |
|---|---|---|---|
| `exposure_render_rpc` → `/run/exposure-render/render.sock` | renderer | research worker | submit one render job with its signed grant |
| `exposure_egress_rpc` → `/run/exposure-egress/egress.sock` | egress broker | renderer | fulfill one intercepted HTTP(S) request |
| `exposure_pacing_rpc` → `/run/exposure-pacing/pacing.sock` | research worker (`RenderPacingSession`) | egress broker | per-request pacing ticket and byte/outcome report |

The renderer never mounts the pacing socket; the worker never mounts the egress socket.

**Pacing without Redis in the broker (R2).** The application Redis has no authentication, so any container that can reach it could act as the default user with full access to the Celery broker; a restricted ACL user would not contain a compromised broker. The broker is therefore attached only to `exposure_public_egress` and has no Redis, PostgreSQL, model, search or application credential. When the research worker submits a render job it opens a `RenderPacingSession` for that one grant: a context manager that serves `pacing.sock` on a background thread in the worker process only until the render call returns or times out. For each HTTP attempt (navigation, subresource or redirect hop) the broker sends `acquire(grant_id, nonce, host, method)`; the worker verifies the grant signature/expiry, rejects unknown grants and replayed nonces, checks the host against the grant's approved host set, charges one request to the durable root budget, acquires the shared provider key through Task 06's `ResearchRateGate` in strict distributed mode, and returns a single-use ticket bound to that nonce and host with the remaining byte allowance. After the fetch the broker sends `report(ticket, bytes, status)`; the worker records actual bytes and marks any unreported ticket uncertain when the session closes. The broker enforces the returned byte cap locally as well. If the pacing socket is missing, the grant is unknown or pacing is unavailable, the broker refuses the fetch; it has no local fallback limiter. The RPC schema accepts only these two strictly validated messages.

For every browser HTTP(S) navigation/subresource, Playwright request routing delegates a bounded fetch to the egress RPC and fulfills the intercepted response. Do not use `route.continue_()` or `route.fetch()` to bypass that RPC. A missed interception cannot obtain direct network access because the container has none. Block service workers, websockets, downloads, browser extensions, arbitrary file navigation, external protocols, popups and additional navigation contexts. Permit only data needed to render the one authorized public page. Unsupported public pages return a coverage gap; there is no login/CAPTCHA/access-control workaround.

```yaml
# docker-compose.exposure-browser.yml — merged explicitly after base/prod files.
# R2: the egress broker has no Redis/PostgreSQL/app network and no Redis credential.
services:
  celery-exposure-research:
    volumes:
      - exposure_render_rpc:/run/exposure-render
      - exposure_pacing_rpc:/run/exposure-pacing
    environment:
      EXPOSURE_RENDER_SOCKET: /run/exposure-render/render.sock
      EXPOSURE_PACING_SOCKET: /run/exposure-pacing/pacing.sock
      EXPOSURE_RENDERING_ENABLED: ${EXPOSURE_RENDERING_ENABLED:-false}
      EXPOSURE_EGRESS_SIGNING_KEY_FILE: /run/secrets/exposure_egress_signing_key
    secrets:
      - exposure_egress_signing_key

  exposure-renderer:
    profiles: [exposure-browser]
    build:
      context: .
      dockerfile: ops/exposure-renderer/Dockerfile
    image: stockscreener/exposure-renderer:r1
    user: "10001:10001"
    network_mode: none
    read_only: true
    cap_drop: [ALL]
    security_opt:
      - no-new-privileges:true
      - seccomp=./ops/exposure-renderer/seccomp.json
    pids_limit: 128
    tmpfs:
      - /tmp:size=536870912,mode=1777
    volumes:
      - exposure_render_rpc:/run/exposure-render
      - exposure_egress_rpc:/run/exposure-egress
    environment:
      HOME: /tmp
      RENDER_SOCKET: /run/exposure-render/render.sock
      EGRESS_SOCKET: /run/exposure-egress/egress.sock
    deploy:
      resources:
        limits:
          cpus: "1"
          memory: 1G
    restart: unless-stopped

  exposure-egress:
    profiles: [exposure-browser]
    build:
      context: .
      dockerfile: ops/exposure-egress/Dockerfile
    image: stockscreener/exposure-egress:r1
    user: "10001:10001"
    read_only: true
    cap_drop: [ALL]
    security_opt: [no-new-privileges:true]
    pids_limit: 64
    tmpfs:
      - /tmp:size=67108864,mode=1777
    volumes:
      - exposure_egress_rpc:/run/exposure-egress
      - exposure_pacing_rpc:/run/exposure-pacing
    environment:
      EGRESS_SOCKET: /run/exposure-egress/egress.sock
      PACING_SOCKET: /run/exposure-pacing/pacing.sock
      EGRESS_SIGNING_KEY_FILE: /run/secrets/exposure_egress_signing_key
    secrets:
      - exposure_egress_signing_key
    networks: [exposure_public_egress]   # not `default`: no route to redis/postgres/backend
    deploy:
      resources:
        limits:
          cpus: "0.5"
          memory: 256M
    restart: unless-stopped

volumes:
  exposure_render_rpc:
  exposure_egress_rpc:
  exposure_pacing_rpc:

networks:
  exposure_public_egress: {}

secrets:
  exposure_egress_signing_key:
    file: ${EXPOSURE_EGRESS_SIGNING_KEY_PATH:?explicit private key-file path required}
```

No host port is published for either service. The renderer has **no secret mounts, env_file, app data volume or API keys**. Initialize socket directories once with UID/GID10001 and mode0770 using a constrained initialization command; never make sockets world-writable. The worker connects through the dedicated directory with matching group permissions. Broker communication uses per-job grants; possession of a socket is not authority to fetch arbitrary URLs.

The proxy has outbound connectivity, so its trusted authorization and fetch-validation code is part of the security boundary. A signed short-lived grant identifies root job, issuer/document target, exact approved host set, allowed methods (`GET|HEAD`), byte/request limits, expiry and nonce. Do not derive its host permissions from document instructions. The research worker reserves capacity in the durable root budget **before issuing a grant**. Redirects and subresource requests consume it; cannot mint a fresh grant to reset the root budget. Proposed per-navigation cap:128 total requests,25MiB response bytes including subresources and redirects,60 seconds. The existing four-navigation/root-issuer ceiling remains; raise only through an explicit policy revision. Unsupported content is recorded as omitted/blocked.

The egress broker checks each hostname and redirect, resolves all returned addresses, rejects private/link-local/loopback/multicast/metadata destinations, pins the checked connection IP, retains original hostname for TLS/SNI/certificate validation, and strips all supplied authorization/cookie headers. It never retries through an unchecked resolver or alternate public proxy. Provider classification and pacing happen in the research worker's `RenderPacingSession`, which uses the same `ResearchRateGate` provider keys as Task06. Each grant's consumed-request/byte record and nonce replay protection are persisted by the worker in the root budget, not in the broker. Uncertain consumption remains reserved until reconciled.

**No Redis ACL user (R2).** R1's restricted Redis ACL user, `exposure_rate_control` network, `redis-acl.sh` and Redis password secret are removed: an ACL user cannot contain a container that can still connect to the unauthenticated default user. Shared pacing and grant counters live with the research worker (above), whose Redis and database access is unchanged. The broker never receives a database or Redis credential and is not attached to the `default` application network. Do not expose the unauthenticated Redis interface externally; it stays on the `default` Docker network only.

Task08B must validate this actual merged configuration, including inherited settings. Merely writing the YAML or setting Playwright's proxy option does not satisfy the gate. Use read-only test secrets, never production credentials, in the isolation probes.

### F.4 Required isolated-renderer probes and operation

Run Docker/Chromium sandbox and egress tests before setting `EXPOSURE_RENDERING_ENABLED=true`. Pin actual image digests in the deployment record. Required checks:

1. Renderer env/mounts contain no application/provider credentials; direct socket attempts to public, private, DNS and application destinations fail. Its only external communication is authorized Unix RPC.
2. Top-level and script-inserted subresources pass through the broker. Private redirects, mixed public/private DNS answers, rebinding between check/connect and unauthorized origin expansion fail before unsafe connection.
3. Websocket/service-worker/pop-up/download/file navigation cannot bypass mediation. Browser crashes/timeouts leave no retained unchecked partial evidence and release only proven unused reservations.
4. Request, byte, navigation and scratch caps are enforced cumulatively; duplicate grants cannot mint capacity. Shared provider pacing contends correctly with non-research traffic.
5. Pausing or losing renderer/egress leaves HTML/text-PDF verification usable with explicit capability gaps. No model fallback, paid search enablement or privileged sandbox fallback occurs.
6. Actual seccomp/user-namespace/sandbox state is inspected on the supported host; `network_mode:none` and resource limits are confirmed via Docker inspection, not inferred from source text.
7. **(R2) Broker isolation from application state.** From inside the egress container, TCP connections to `redis:6379`, `postgres:5432`, `backend:8000` and every application service name fail (names do not resolve and direct container IPs are unreachable); its environment and mounts contain only the signing-key secret and the two socket directories; no Redis client library is installed in its image. Docker inspection confirms it is attached only to `exposure_public_egress`.
8. **(R2) Broker cannot reach the Docker host.** Connections from the egress container to the Docker host gateway address and to every published host port (at the inspected baseline, the frontend's `${FRONTEND_PORT:-80}`, which proxies `/api`) fail. The broker's own destination checks reject these addresses, but this probe also requires a host-level rule (for example a `DOCKER-USER` iptables/nftables rule dropping traffic from the `exposure_public_egress` subnet to the host and private ranges) so a compromised broker process cannot bypass them. If the host cannot apply such a rule, record the residual risk and leave `EXPOSURE_RENDERING_ENABLED=false`.
9. **(R2) Pacing RPC.** With the research worker stopped or the pacing socket absent, the broker refuses every fetch. A ticket for grant A cannot be used for grant B; replayed nonces are refused; the session socket disappears when the render call ends.

Use the existing enabled-market Compose wrapper with base/prod plus `docker-compose.exposure-browser.yml` and explicit `--profile exposure-research --profile exposure-browser`. Starting the profile does not enable research, paid search or live membership. Task27 records the complete checked command/configuration and all new required secret-file paths without printing their contents.

### F.5 Disk budget and evidence lifetime

The configured 5 GiB store cap counts **unique retained blobs plus staging reservations**, across all issuer jobs and markets. Duplicate content does not multiply the charge. A1’s `StorageTicket` is a typed entry in the resource ledger, not a new generic quota platform. Check the 1 GiB free-space floor on the actual mount before starting a bounded write; reserve space for output before launching processing that must retain it. An external process filling the disk can still produce a typed I/O failure; the app must retain its job state without falsely committing evidence.

Normal GC never evicts evidence selected by any current/historical published generation, adjudication/review/legal pin, or active work lease. Unreferenced blobs become eligible after 30 days; abandoned scratch after 24 hours, with a final reference/lease recheck and tombstone. The referenced-blob pin and GC deletion mark are serialized so a late selection cannot reference a deleted blob. If all capacity is pinned, research visibly pauses until the operator expands storage or approves a separate export/archive/retention operation. No silent historical deletion or claim of unlimited retention fits a bounded disk.

Expose used, reserved, reclaimable, pinned and free bytes in the research health UI. Storage-full holds are resumable and do not imply no company exposure. SQL history, application logs and backups are **not** covered by the blob cap; use bounded container log rotation above and document independent database/backup capacity monitoring. These limits are accepted, editable deployment defaults, not a statement of available space on the user's host.


## Agent handoff and review checklist

Before product changes, record the user's R2 acceptance against the paired spec hash, and confirm that hash and actual branch/migration baseline; do not edit another feature's checkout. Before each task, read its interfaces and upstream dependency outputs. After each task, run red/green and regression checks, inspect the diff and have an independent reviewer check the relevant high-cost failure cases. The agent must not conflate “tests are listed” with “tests ran.”

For the first shadow-slice review attach only that slice’s actual artifacts and unavailable-capability report. Before full-V1/automatic-admission whole-branch review, attach:

- actual task commits and unresolved implementation findings;
- validated migration graph and additive upgrade rehearsal;
- exact accepted-source/corpus manifests, adjudication and evaluation reports;
- PostgreSQL no-skip results and old/new generation snapshots;
- frontend tests/build and updated reader capability;
- four-market probe/permission outcomes and unavailable capabilities;
- default-off/no-spend proof, configured subscription/local allocation behavior and optional search costing controls;
- safety-hold, origin-union, issuer facade and post-commit projection recovery evidence;
- the operator runbook and proposed staged activation state, with paid search still off unless separately authorized.

**Recommended execution:** subagent-driven implementation with review after each task and a whole-branch review before activation. This plan defines implementation and release criteria; it does not assert that any application test, market probe or production rollout has been completed.

