# S3: Generation-bound publication and reviewed membership

**Parent:** [company-exposure-remaining-work.md](company-exposure-remaining-work.md)

| Source | Location |
|---|---|
| Spec | §7 (membership), §13–§16, §18 (steps 3–4) |
| Plan tasks | 18, 19A/19B, 20, 21B/21C, 22B, 23 (publication triggers), 26B, 27B (steps 1, 3, 4) |
| Plan appendices | A3–A7, B |

## Goal

S3 takes research from the job-scoped `shadow_preview` to the product. It has two parts:

1. **Assessment reads (activation step 3).**
   - Accepted assessments are published through the existing economic serving generation, so every read is pinned to one generation.
   - Old generations return exactly what they returned before.
   - Generations without the extension say so explicitly.
2. **Reviewed membership (activation step 4).**
   - Research becomes one origin in a typed union with source, Social and manual membership.
   - Administrator decisions take precedence.
   - Delivery to legacy readers is ordered and research-only.
   - Automatic additions are allowed only for approved role policies and validated scopes, and only after the S2 four-market admission gate (Task 25B) passes.

S3 has no new publisher, no second live pointer and no fabricated source rows.

## Design summary

### Role policies and the automatic-addition gate (spec §7.1–7.2)

- **Policies are immutable and reviewed.**
  - Each theme has a `RoleEligibilityPolicyRevision` pinned to the theme's defining fingerprint. It holds the allowed roles, stages and scopes, and the required evidence.
  - An administrator approves policies explicitly.
  - Templates for memory, cybersecurity, refining, tankers and copper ship as drafts only.
  - With no policy, research can continue, but automatic addition is held.
  - A policy change never quietly widens admission.
- **Producers and operators must be shipping or operating.** Equipment-role policies may accept `commercially_available` plus an explicit product-to-application link, with "sales unknown" shown.
- **What never passes.**
  - Plans and qualification only produce a candidate (E09).
  - A customer is visible on the map but not admitted to a producer basket (E07).
- **Six-part gate. All six must hold:**
  1. an accepted issuer-to-security link, an active listing and a compatible scope;
  2. an accepted theme identity and an approved role policy;
  3. primary-explicit or admissible primary-synthesis support for participation, role, application and commercial status;
  4. no language, identity, integrity, freshness, conflict or automation hold on the required claims;
  5. no overriding administrator rejection or removal, no conflicting decision, and no mapping conflict;
  6. valid recorded revisions, plus compatibility readiness.
- **What the gate does not require, and what it shows.** Unknown materiality alone does not fail the gate (E08). No single model score can pass it. The decision shows both explicit and synthesized support.

### Origins and precedence (spec §7.3; plan A5)

- **Effective membership is a typed union, not an array concatenation.**
  - `effective_membership(OriginSelections, GlobalDecisionRef)` combines `source_extraction`, `social`, `research` and `manual` contributions under global administrator precedence.
  - It consumes frozen origin inputs, never the latest rows.
- **Administrator rejection wins.** It survives new primary evidence (I04). New evidence may open a review proposal but never lifts the rejection automatically.
- **Changes open a review; they never auto-remove.**
  - Triggers: stale or disputed claims, an ended exposure, a later role mismatch, or a negative reassessment.
  - Until an administrator decides, the existing research membership is kept with `review_required`, and those claims are held from new automated use (I05).
  - A failed fetch changes nothing.
- **Removal is always scoped.**
  - A reviewed removal suppresses membership according to its scope.
  - An unscoped delete of another origin is impossible.
  - Re-entry needs an explicit superseding decision.
- **Theme restructuring.** A theme split or mechanism change holds the affected assessment rather than cloning it into every child (I10).
- **Service principal limits.** The auto service principal may only accept gate-approved additions. It cannot remove members, lift rejections, approve policies or rewrite links.
- **Decision history.** Decisions persist once per changed input fingerprint: `(association, assessment revision, policy revision, link revision, global decision revision, safety token)`.
- **Revision history.** `ExposureMembershipAssociation` is unique on `(economic_theme_id, security_id)`, and `ExposureMembershipDecisionRevision` is unique on `(association_id, revision_number)`. Accepted and rejected revisions coexist (I11).
- **Counters.** `research_verified_issuer_count`, `research_candidate_count`, `materiality_unknown_count` and `membership_review_required_count` are reported separately from the source and root counts.
  - Attention scoring and weights are unchanged.
  - Old generations are not rescored.

### Acyclic generation build (plan A3; spec §14)

1. **Capture.** Inside the existing short capture fence:
   - choose the exact assessment, link, policy, hold and membership revision IDs;
   - reserve an `ExposureSelectionSet` UUID with its immutable `input_fingerprint`;
   - write both into the new nullable `GenerationInputManifest.exposure_inputs` as a strict `exposure_v1` envelope.

   The envelope fields are `kind`, `selection_set_id`, `input_fingerprint`, `assessment_refs`, `issuer_link_refs`, `role_policy_refs`, `hold_refs`, `membership_refs` and `freshness_as_of`. No provider call happens here.
2. **Seal the input manifest.** It names the reserved output ID, not an output hash that doesn't exist yet.
3. **Fill the selection outside the fence.** Fill it deterministically from the captured revisions and seal it. The set references its parent manifest ID, but that ID is excluded from the set's semantic hash.
4. **Build snapshots from the sealed set.** Build the exposure snapshot entries and membership projections. `ServingGeneration.exposure_selection_set_id` (nullable) plus a versioned output-hash payload bind them to the generation.
5. **Final compare-and-set.** Recheck the hashes, dependency safety and `fresh_until` of proposed automatic additions.
   - A newly disqualified support aborts only that proposal (R11).
   - Existing members stay, flagged held-for-review.
   - Unrelated post-cutoff work becomes backlog.
6. **Publish** through the existing coordinator and the durable publication event.

Old manifests have `exposure_inputs=NULL`, and their hashes and serializers do not change. Validation dispatches on the serializer version (R07).

### Ordered compatibility delivery (spec §14.3; plan Task 20)

- **Owner and kind.** The transport owner is `research-membership:<issuer_uuid>:<theme_uuid>`, with projection kind `research_membership` and origin `exposure_research`.
  - It is a transport key only: never a `SourceFamily`, `ThemeMention` or `ClaimAssignment`, and never a count of source evidence.
  - The logical key is `(owner, accepted_projection_revision, "research_membership", projection_policy_version, target)`.
- **Payload.** The payload replaces that owner's entire contribution.
  - Revisions only increase, and an older revision is a no-op (R10).
  - Other origins survive.
  - A global rejection still vetoes a late delivery.
- **Deliverability.** It comes from committed publication history, including superseded generations that were published. Post-commit notification only wakes workers up, so a crash between commit and notify still delivers (R09).
- **No loops.** Mirror-origin deliveries never re-trigger research or extraction.
- **Issuer facade.**
  - First import every current admin attestation (the `import-identities` deferral returns here) and prove the grouping is equivalent.
  - Under upgraded generations, `SocialCompanyIdentityService` reads that generation's accepted issuer-link selection. The legacy configuration becomes historical provenance.
  - Old replace commands route to reviewed link revisions with pending-publication semantics.
  - Never call the old transaction-owning `replace` inside a fenced write.
- **Rollback.** Rollback to an older binary is safe only if that binary's readers understand the research contribution store. Otherwise report it as a blocked operation.

### Reads, admin operations and UI (spec §16)

- **Routes.** Register static routes (`research-jobs`, `admin/*`) before dynamic `/{assessment_id}`.

  ```text
  GET  /api/v1/company-exposures?security_id=&generation_id=          (21B)
  GET  /api/v1/company-exposures/{assessment_id}?generation_id=       (21B)
  GET  /api/v1/economic-themes/{theme_id}/exposures?generation_id=    (21B)
  POST /api/v1/company-exposures/admin/decisions/preview              (21C)
  POST /api/v1/company-exposures/admin/decisions/apply                (21C)
  PUT  /api/v1/company-exposures/admin/theme-research-policy/{theme_id} (21C)
  GET/PUT /api/v1/company-exposures/admin/settings                    (21C)
  ```

- **Reads.**
  - Product reads resolve the serving generation once through `EconomicThemeReader`, then read that generation's selection.
  - An assessment ID cannot reveal an unselected newer revision.
- **Writes.**
  - Writes use `require_admin` → `AdminPrincipal`, idempotency keys and expected preview hashes.
  - Apply runs under the shared fence and rejects stale previews before any side effect.
  - An actor string in the body is never trusted (R13).
- **Settings.** Settings persist as `ExposureRuntimePolicyRevision` and expose only whether each secret is present, never its value.
- **Evidence access.** Original and excerpt access works through opaque document IDs with authorization; this is the deferred "authorized original reads".
  - Excerpts are short.
  - There is no public mirror, and tombstones are shown.
- **Consumers.** Every consumer of effective membership reads the same selected generation and origin union: stock, basket, digest, Social and MCP.
  - Register them all in the capability inventory.
  - Bump the reader capability version.
- **UI (22B).** Add:
  - a "Why included" link on theme constituent rows that opens the issuer–theme dossier;
  - per-claim badges (primary explicit, primary synthesis, reported/unverified, held, historical, materiality unknown);
  - an original/English toggle, and materiality with period and denominator;
  - review-required and unresearched states, which are not rejected;
  - separate issuer and listing counts, and a decision review with a required reason.

  A generation-pinned dossier uses a different React Query key from mutable job status. A finished job never overwrites the accepted dossier.

## Implementation plan (in order)

### Task 19A: assessment-only serving (may precede Task 18)

- **Migration.** Create the just-in-time `exposure_generation_extension` migration:
  - `ExposureSelectionSet` and its selection child tables;
  - nullable `GenerationInputManifest.exposure_inputs`;
  - nullable `ServingGeneration.exposure_selection_set_id`.
- **New code:** `services/company_exposure/publication.py`, containing `ExposurePublicationAdapter` with `capture_inputs`, `prepare_selection`, `build_entries` and `validate_safety`.
- **Existing code.** Modify these only at typed extension points:
  - `economic_taxonomy_publication.py`, `_preparation.py`, `_contracts.py`, `_validation.py`;
  - `economic_taxonomy_snapshot_builder.py`;
  - `infra/db/repositories/economic_taxonomy_publication_repo.py`.
- **Tests:**
  - `test_publication_extension.py`, `test_generation_history.py` and `test_selection_models.py` (raw-SQL immutability);
  - `tests/integration/company_exposure/test_exposure_publication_postgres.py`: R06 (shared fence, no provider call inside it), R07 (G1 pinned), R08 (no source-evidence effect), R11 (a hold blocks only the dependent addition), and crashes before and after the pointer switch.
- **Selections are assessment-only.** They carry an empty membership list.

### Task 21B + Task 22B (reads)

- **Product reads** go through `CompanyExposureReader.read_for_security`, `.read_assessment` and `.read_for_theme`. They extend `services/company_exposure/reads.py` and `schemas/company_exposure.py`.
- **Frontend.** Add `ExposureDossier.jsx`, `ExposureClaims.jsx`, `ExposureEvidence.jsx` and `ExposureReview.jsx` under `frontend/src/features/companyExposure/`. Change `EconomicThemeDetailModal.jsx` minimally.
- **New API client functions** in `frontend/src/api/companyExposures.js`: `getCompanyExposures`, `getExposureAssessment` and `getThemeExposures`. Paths start with `/v1/`.
- **Activation step 3 needs these CLI commands:** `set-mode`, `set-stage` and `prepare-preview`.

### Task 18: membership and decisions

- **Migration.** Create the just-in-time `exposure_membership_decisions` migration in `models/company_exposure_membership.py`. It holds:
  - `RoleEligibilityPolicyRevision`;
  - `ExposureDecisionRequest`, `ExposureDecisionPreview` and `ExposureDecisionEvent`;
  - `ExposureMembershipAssociation`;
  - `ExposureMembershipDecisionRevision`.
- **New code:** `services/company_exposure/membership.py` and `decisions.py`:
  - `ExposureMembershipEvaluator.evaluate(MembershipInputs) -> MembershipEvaluation`, where the outcome is one of `eligible_addition`, `candidate`, `held`, `retained_review_required` or `rejected`;
  - `effective_membership`;
  - `preview_decision`;
  - `apply_decision`.
- **Blocking-only safety check.** Reinstate it here. It can only block or defer; it never swaps in newer unpinned evidence.
- **Tests:**
  - `test_membership.py`, `test_role_policies.py`, `test_decisions.py`;
  - `test_membership_decisions_postgres.py`: E07, E08, E09, I04, I10, I11 (schema and Postgres), a reviewer apply racing new evidence or a hold, and two listings of one issuer.

### Task 19B: membership serving

Add membership selection and validation to the sealed set. Build the snapshot from Task 18's frozen origin union, never by appending to source constituents.

Live additions stay disabled.

### Task 20: compatibility and issuer facade

- **Migration.** Create the just-in-time `exposure_origin_compatibility` migration. It holds the research contribution store and the attestation bridge selection metadata.
- **New code:** `services/company_exposure/compatibility.py`:
  - `build_research_projections`;
  - `apply_research_membership_projection`;
  - `accepted_issuer_configuration(generation)`.
- **Existing code.** Modify these at scoped seams:
  - `economic_taxonomy_publication_compatibility.py`;
  - `economic_taxonomy_runtime.py`;
  - `social_company_identity_service.py`.
- **Tests:**
  - `test_compatibility.py`, `test_issuer_facade.py`;
  - `test_exposure_outbox_postgres.py`: R09 and R10, plus duplicate delivery, abandoned generations, global rejection, and the G1/G2 facade.
- **CLI:** add `import-identities` (dry run first) and `reconcile-compatibility`.

### Task 21C + Task 22B: admin operations

- **Endpoints:** decision preview and apply, the theme research policy, and settings.
- **Job controls:** reinstate job pause and cancel. Cancel releases only reservations that were never dispatched.
- **UI:**
  - `previewExposureDecision`, `applyExposureDecision` and `updateThemeResearchPolicy`;
  - a keyboard-accessible review UI with a required reason, a stale-preview error, and a "no generation extension" state.
- **CLI:** add `disable-acquisition`.

### Task 23: publication triggers

- **Dirty marking.** Accepted assessment, hold and decision revisions mark the existing coalesced generation schedule dirty. There is no new publisher and no one-generation-per-document loop.
- **Trigger function:** `enqueue_exposure_verification(source_reference, issuer, theme, relevant_input_hash)`.
  - It is idempotent and never runs research synchronously (R05).
  - It records the cause and roots so it can suppress mirror loops.
  - Add a bounded reconciliation scan for triggers that were missed.

### Task 26B + Task 27B: gates and activation (S3 part)

- **Gate manifest.** Add `S3` to `required_company_exposure_cases.json`:
  - E07, E08, E09, I04, I10 and I11 at the schema and Postgres layers;
  - R06–R11 at the unit and Postgres layers;
  - R07 at the API and frontend layers;
  - R13 at the API layer.
- **Named invariants:**
  - membership and admin conflict;
  - selection sealing;
  - cutoff progress;
  - relevant hold races;
  - publication crash recovery;
  - projection ordering;
  - issuer-facade history.
- **Full rehearsal:** `test_full_exposure_rehearsal_postgres.py`. It covers:
  - source and Social origins, global rejection and two cross-listings;
  - an excluded customer, unknown materiality and scope-invalid materiality;
  - a late old capture and a hold after capture;
  - out-of-order projection and a crash after commit.
- **Activation.** Run activation steps 1, 3 and 4 in a disposable rehearsal first.
- **Step 4 also needs:**
  - S2's Task 25B pass;
  - approved role policies;
  - the reader capability manifest updated with exposure support.

## Stop conditions and things not to do

- **Pointers and hashes.**
  - Never write the serving pointer outside `EconomicTaxonomyPublicationCoordinator`.
  - Never add a second current pointer.
  - Never rehash or re-serialize old generations.
  - Never backfill research into G0.
- **No fabricated source evidence.** Never insert fake `ThemeConstituent`, `ThemeMention` or `ClaimAssignment` rows to satisfy a legacy shape. Report a coverage limitation instead.
- **Review and removal.**
  - Never auto-remove a member because research found nothing.
  - Never lift an administrator rejection automatically.
  - Never move membership to another security after a link correction.
- **Approval and activation.**
  - Never approve role-policy templates in a migration or on the agent's authority.
  - Never enable automatic additions before 25B and 26B pass, whatever the US-only results say.
- **Lock order.** Stop and fix any path that makes a provider call or waits inside the capture or final fence, or that calls a transaction-owning Social helper inside a fenced write.
