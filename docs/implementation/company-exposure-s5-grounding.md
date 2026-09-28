# S5: Classifier grounding

**Parent:** [company-exposure-remaining-work.md](company-exposure-remaining-work.md)

| Source | Location |
|---|---|
| Spec | §11 (freshness and holds), §12 (grounding and acyclicity), §18 step 6 |
| Plan tasks | 24 (grounding), grounding validation within 25 and 26B, 27B step 6 |

## Goal

When the existing economic source pipeline classifies an incoming article, it can use a small set of accepted, generation-pinned exposure claims as context. The claims must be relevant to the issuers, products or activities that the source names explicitly.

This context supports only narrow inferred connections. It is never direct evidence that the new source says something. It can never flow back as primary support for research: no circular verification.

Grounding is enabled last (activation step 6) and is off by default.

**Dependencies:**

- Task 14 (claims) and Task 16 (assessments);
- Task 19B (membership serving);
- Task 20 (compatibility);
- Task 21B (generation reads);
- Task 23A (holds; already in S1).

Market and search tasks are not dependencies.

## Design summary

### Selection (spec §12.1)

- **Selection starts from the source.** Resolve the source's explicit issuer, security, product and activity references first. Knowing the issuer alone does not permit attaching all its themes (I08: generic price moves never fan out).
- **Where claims come from.** Select only from the accepted exposure selection of a serving generation. Then run the current blocking-only safety check (S3), which can only block or defer.
- **Size bounds.**
  - At most 5 claims per issuer and 10 per source.
  - A configured token or character ceiling also applies.
- **No inline research or waiting.** When research is missing, `enqueue_exposure_verification` runs asynchronously, and the classifier continues under its existing evidence rules.
- **What the context contains.** `ExposureGroundingContext` holds:
  - the generation and issuer-mapping revisions;
  - the assessment and claim revisions;
  - original passage locators;
  - support basis and commercial status;
  - the safety check;
  - separate source and context citation namespaces;
  - bounded content and a semantic hash.

### Allowed and forbidden uses (spec §12.2)

- **Allowed.** Orders for a product with an accepted product-to-theme claim may support a narrowly justified, *inferred* theme link.
- **Forbidden.**
  - Unstated AI demand, named customers, HBM sales or revenues stay unasserted (I07).
  - Research `support_basis` is not collapsed into the existing exposure-support enum. Primary research support is contextual inference when interpreting a different source.
- **Citations.** Source citations and context citations are reviewed separately. Explicit source evidence can still independently support a direct claim.

### Cache identity (plan Task 24)

- **Keep the existing namespace.** Exposure context lives in its own versioned namespace. `GroundingContext` v1 serialization and historical hashes are unchanged.
- **What the context hash covers.** The hash that feeds the `EvidencePacket` input hash covers semantic content only:

  ```python
  context_hash_inputs = {
      "claim_revision_ids": sorted(selected_claim_ids),
      "original_evidence_hashes": sorted(original_leaf_hashes),
      "issuer_mapping_semantics": issuer_mapping_fingerprint,
      "theme_definition_semantics": relevant_theme_fingerprints,
      "grounding_policy": grounding_policy.version,
  }
  ```

  The serving generation ID, retry count and timestamps are audit fields on `ExposureGroundingUse`. They are never cache keys. Republishing an unchanged assessment must not re-extract content.
- **Changed context.** A relevant changed context creates a new preparation or input identity under the existing correction rules. It never mutates a frozen packet, and it never triggers bulk reprocessing of history.

### Acyclicity and safety (spec §12.3, §11; plan A4)

- **Primary leaves only.** Flatten verification provenance to original primary leaves.
  - Assessments, summaries, memberships and classifier outputs are never primary premises.
  - A synthesis that depends on its own assessment, or on a downstream classification that used it, is rejected (I09 integration layer).
  - This reinstates the deferred evidence-DAG check.
- **Recheck points.** Check `fresh_until`, mapping and policy, and holds at three points:
  - before the provider call;
  - before persisting the new classification;
  - before any membership action that uses that classification.

  An in-flight call may finish for audit, but it cannot commit a use that has since become disqualified (I05).
- **Fallbacks.** The only fallbacks are source-only processing, or a retry with approved current context. Never substitute an unpinned newer assessment.
- **Timing.** Original source timing and research effective periods stay separate. Dossier maintenance never resets attention timestamps.

## Implementation plan (Task 24)

- **Migration.** Create the just-in-time `exposure_grounding_uses` migration. Add the immutable `ExposureGroundingUse` model, plus an append-only use-link event connecting it to the source processing request and attempt.
- **New code:** `backend/app/services/company_exposure/grounding.py`:
  - `ExposureGroundingSelector.select(SourceContext, generation_id, policy) -> ExposureGroundingContext`;
  - `validate_grounding_use(context, at) -> SafetyDecision`;
  - `persist_grounding_use(context, request_id, attempt_id, check) -> ExposureGroundingUse`.
- **Existing code:**
  - modify `backend/app/services/theme_grounding_context.py`;
  - modify the economic source admission, extraction and review adapters, and the exact source-processing entrypoints;
  - first inventory those entrypoints on current main, since the plan's Task 00 list is from the S1 base.
- **Feature flag.** Grounding sits behind its own stage capability, default off. With the flag off, source-only behavior must be byte-identical.
- **Tests:**
  - `tests/unit/company_exposure/test_grounding.py`:
    - I07: product context does not invent a demand cause;
    - I08: a price-only source yields `claim_refs == ()`;
    - context-size limits;
  - `test_grounding_cache.py`:
    - cache invariance across unrelated generations;
    - v1 and v2 compatibility;
  - `test_no_circular_support.py`:
    - I09 integration: classifier-generated evidence never verifies;
    - the self-support DAG;
  - `tests/integration/company_exposure/test_grounding_hold_race_postgres.py`: a stale contextual decision cannot commit after a new hold.
- **Regressions:** run the existing economic source extraction, grounding and publication suites with the feature off.

## Gate for the slice

Add `S5` to `required_company_exposure_cases.json`:

- I07 and I08 at the unit layer;
- I09 at the unit and integration layers;
- I05 at the unit and Postgres layers, for the grounding path;
- R07 and R12 extended to grounding: pinned context in old generations, and expiry without provider calls.

Add a named invariant, the grounding hold race.

The separate grounding validation runs the Task 25 corpus evaluator in grounding mode, with recorded responses. It must show no unstated causation and no fan-out before activation step 6.

## Stop conditions and things not to do

- **Circular support.** Never let a classification produced with exposure context count as corroboration for the same assessment. Stop if any DAG check can be bypassed.
- **Cache identity.** Never embed generation IDs or check timestamps in the hashed `grounding_snapshot`.
- **No inline work.** Never run document search, issuer investigation or a publication wait inline in classification.
- **Ordering.** Never enable grounding before S3's membership and compatibility gates and this slice's validation pass. It is always the last privilege enabled.
