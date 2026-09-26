# Evidence-Backed Company Exposure Map

## Architecture design — issuer-centric assessments within StockScreener

**Date:** 2026-09-25  
**Revision:** R2 — 2026-09-26, second implementer-review corrections (R1 — 2026-09-26, first implementer-review corrections).  
**Status:** Revised specification. The user approved the original artifact (SHA-256 `a182065bf6b1e5e07026045b50e6ae9124501893b383e3a940efb5f9f13f8249`) and accepted the R2 corrections listed in the change record, including the R1 defaults. Record that acceptance against this file's SHA-256 when installing it. No production changes are authorized.  
**Amendment scope (R1):** Name the subscription transport; reuse existing preparation; specify worker/browser deployment and storage limits; permit a US verify-only shadow delivery; assign human corpus adjudication.  
**Amendment scope (R2):** Define US CIK sourcing and single-listing acceptance; unify research-job API routes; classify connect-phase provider failures as pre-dispatch and expire uncertain reservations with their allocation period; remove all Redis access from the browser egress broker. The evidence and membership standards remain unchanged.  
**Repository inspected:** `xang1234/stock-screener`, `main` at `28c220e4e4ca5afcb5a380678bb2b80dd7f388b7`.  
**Suggested repository location:** `docs/superpowers/specs/2026-09-25-company-exposure-map-design.md`  
**Paired plan:** `2026-09-26-company-exposure-map-implementation-plan-r2.md`, intended for `docs/superpowers/plans/2026-09-25-company-exposure-map.md`. The plan pins the exact R2 spec hash and separately records prior approval.

## 1. Purpose and architectural decision

Build a live-app research capability that answers:

> Why does this issuer participate in this economic theme, in which role, at what commercial stage, with what disclosed materiality, and on what evidence?

Verify existing constituents and conduct bounded discovery of additional candidates. Maintain issuer-level research, security-specific membership, and claim-level evidence rather than treating a company mentioned alongside a theme as an automatically verified constituent.

**Selected approach:** A dedicated issuer-centric exposure-assessment layer inside the existing backend. It feeds the existing membership projection and targeted classification grounding. It is not a second theme catalog, general corporate knowledge graph, independent publication authority, or historical backtesting platform.

Separate six objects throughout the implementation:

1. **Source evidence:** What an original document actually states.
2. **Exposure claim:** One proposition about an issuer, product, activity, theme, or materiality measure.
3. **Assessment:** A versioned synthesis selecting supported claims, unresolved questions, and conflicting evidence for an issuer–theme relationship.
4. **Membership decision:** Whether an individual security belongs in a particular theme basket under its reviewed policy.
5. **Grounding context:** The bounded accepted research supplied to classify a new source.
6. **Source observation/development:** What that new source discusses or reports. Research does not manufacture this object.

A verified exposure is not a prediction, recommendation, guarantee of benefit, or materiality estimate.

## 2. Decisions already agreed with the user

| ID | Agreed requirement |
|---|---|
| D01 | Verify existing constituents and discover additional candidates; do not proactively research the entire universe. |
| D02 | Broad discovery; primary-backed verification, assessed separately for each claim. Secondary-only support remains reported, inferred, or unresolved. |
| D03 | Tiered membership automation: eligible additions may be automatic; removals, conflicts, and overrides of reviewed decisions require review. |
| D04 | Materiality supports disclosed figures, transparent calculations from compatible disclosed inputs, and primary-supported qualitative assessments. No invented numerical estimates. |
| D05 | Unknown materiality does not alone prevent admission of verified, current commercial exposure in an eligible role. |
| D06 | Basket role eligibility is governed per theme. The map can contain verified relationships excluded from the default basket. |
| D07 | On-demand research plus prioritized automatic verification/refresh. Additional-company discovery is limited to explicitly enabled themes or requested investigations. |
| D08 | Full-launch automated research markets: US, Hong Kong, Japan, and Taiwan. A US-only, verify-only shadow preview is an allowed intermediate delivery; it grants no automatic-admission or classifier-grounding privilege. |
| D09 | English, Japanese, Traditional Chinese, and Simplified Chinese; original passages remain authoritative, with linked English explanations. |
| D10 | Research is issuer-centric; sharing across listings requires verified issuer relationships. Membership remains listing-specific. |
| D11 | Budgeted web discovery plus public primary-document retrieval. No paid document subscription or authenticated publisher-portal integration in V1. |
| D12 | LLM research uses OpenCode Go / Kimi k2.6 through the existing `OpenCodeGoKimi` transport and `settings.opencode_go_api_key`. Account-specific rate/remaining-allowance limits are not established here: enforce local ceilings and honor provider throttling. Do not invent dollar charges or fall back to metered LLM providers. |
| D13 | Paid search is disabled by default. Credentials alone cannot activate it. Explicit enablement and a configured spending cap are required. |
| D14 | Direct HTML, text-based PDFs, bounded public-page browser rendering, and selective page-image interpretation are in scope. |
| D15 | Accepted, current, relevant exposure facts can ground incoming theme classification. Exact assessment revisions and original support must be retained. |
| D16 | Stale or materially disputed claims are held from new automated admissions and grounding. Existing membership is flagged for review, not automatically removed by this feature. |
| D17 | Bounded synthesis across primary documents is allowed when every connecting link is supported; missing relationships cannot be invented. |
| D18 | Use a dedicated assessment layer integrated into the existing app, rather than extending source claims into a mutable all-purpose company profile. |

The previous document’s defaults remain the approved baseline. R1 adds **proposed deployment defaults** for storage and worker/browser resources, plus an intermediate delivery boundary. They are called out for review and are not represented as additional user answers. All limits are engineering controls, not measured provider capacity.

## 3. Inspected repository baseline and integration implications

The existing implementation, not earlier planning attachments, is the baseline. This inspection did not run the application, apply migrations, exercise provider accounts, or establish deployment/cutover state.

| Inspected component | Existing behavior | Design implication |
|---|---|---|
| `economic_taxonomy_runtime_evidence.py` | `ThemeConstituentExposure` is attached to a `ClaimAssignment` and `stock_universe.id`, with kind, strength, policy, and payload. [R1] | Keep it as an immutable source-derived fact. Do not repurpose it into the multi-document assessment authority. |
| `economic_theme_observation_service.py` | Constituent materialization stores role, directness and rationale; `exposure_strength` is populated from extracted `confidence`. [R2] | Do not use that field as revenue exposure, economic materiality, or a calibrated probability. New materiality values need their own typed representation. |
| `social_company_identity_service.py` | Administrator-attested company IDs and verification references are stored through a versioned configuration with Social registry/audit integration. [R3] | Import and preserve existing attestation. Do not silently replace verified company grouping with name matching or model output. |
| `economic_taxonomy_publication.py` and its preparation/contracts modules | One coordinator captures a cutoff, prepares artifacts, and publishes a coherent generation. [R4] | Extend its inputs and validation; do not create a separate live exposure pointer. |
| `economic_taxonomy_snapshot_builder.py` | Builds and seals catalog/review snapshots from an explicit taxonomy, interpretation, manifest and metrics combination. [R5] | Add assessment and membership selections to the same generation and reader capability. |
| `economic_theme_read_service.py` | Economic-mode product reads use a selected serving generation and its sealed bundle. [R6] | Research-product reads and historical assessment displays follow this authority; operational job status remains separate. |
| `theme_grounding_context.py` | Bounded reference context distinguishes company background from new source events or demand evidence. [R7] | Add a versioned research-grounding adapter, preserving this distinction. |
| `theme_evaluation/kimi_client.py` | `OpenCodeGoKimi` targets the Go endpoint with model `kimi-k2.6`, returns bounded JSON and typed preparation failures. [R8] | Use this actual subscription transport, not the generic LiteLLM/dollar-ledger route. Preserve existing callers when adding result metadata. |
| `theme_evaluation/image_preparation.py`, `kimi_translation.py` | `OpenCodeGoVision` and `OpenCodeGoTranslator` already subclass the Go transport. [R10] | Inject the research reservation-aware JSON boundary; reuse prompts/validation and retain exact policy versions. |
| `multilingual_preparation.py`, `multilingual_v2.py`, `translation_normalization.py`, `translation_quality.py`, `quantity_display.py`, `quantity_review.py` under `theme_evaluation/` | Existing source-preserving language/quantity preparation modules identified by repository review. [R10] | Extend existing functions only for documented gaps; no parallel language/quantity engine in `company_exposure`. |
| `rate_budget_policy.py`, `rate_limiter.py`, `start_celery.sh`, Compose files | Existing distributed provider pacing and resource-limited queue topology. [R11] | Use dedicated `exposure_research` workers while sharing provider keys/budgets. Browser execution is separately isolated. |

The source read at [R4] currently declares reader contract version 1 and migration readiness `0055`. Those are inspected baseline values, not revision numbers to reuse for the new migrations. The implementation plan must identify the actual Alembic head and allocate each new revision against the execution checkout’s actual Alembic head at that migration-owning task’s commit, rather than reserve a sequence once.

### 3.1 Narrow changes to existing responsibilities

Introduce the research subsystem additively. Existing source-derived membership, Social administrator decisions, theme identities, source correction semantics, and economic attention definitions remain intact unless an explicitly described adapter changes them.

The source pipeline may continue correcting its own source-derived facts according to Idea 1. Research refreshes must not masquerade as source corrections, remove another origin's contribution, or erase a reviewed membership. Membership must record its origins so those paths remain distinguishable.

### 3.2 Non-goals

No directional fundamental score, investment ranking, revenue estimates based on proxies, broad supplier/customer graph propagation, general asset master, exhaustive corporate-ownership tree, automatic theme restructuring, unsupported-market acquisition, or irreversible legacy cleanup. No production performance/accuracy claim follows from this design.

## 4. Domain and identity contracts

### 4.1 Theme identity

Use the existing `EconomicTheme` UUID. Each assessment pins the defining semantic revision or fingerprint it evaluated. A display-only rename does not invalidate otherwise compatible research. A mechanism change, split, role-policy change, or materially different product scope requires reevaluation before new automatic use.

Do not create new themes merely because issuer research encounters a new term. It may submit a proposal through existing taxonomy governance. New names never become identity keys.

### 4.2 Issuer identity and listing links

Use a narrow issuer registry and append-only issuer-to-security mapping revisions. `security_id` remains `stock_universe.id`.

An issuer identity represents the reporting/business entity being assessed, not a ticker string. Preserve official identifiers and their schemes, original/legal names, source references, and mapping scope. A US CIK, Japanese EDINET code, Hong Kong issuer/security reference, and Taiwan company code are typed identifiers, not interchangeable globally unique numbers.

Existing administrator-attested `company_id` values are imported with exact configuration version, verification reference, and actor/audit provenance. Preserve their identifiers as aliases/bridges; do not assume the text itself is a UUID. Matching names alone cannot merge issuers.

**V1 governance default:** New shared-issuer/cross-listing mappings require administrator acceptance of retained official identity evidence. An unresolved listing may be researched in an isolated candidate dossier, but its research is not shared to another listing and cannot produce an automatic cross-listing addition. Existing accepted mappings avoid repeated approval. This is conservative identity governance, not a requirement to review every subsequent exposure claim.

**US CIK sourcing and single-listing acceptance (R2):** The repository has no existing CIK data. A US issuer's CIK is resolved from SEC's official ticker-to-CIK file (`https://www.sec.gov/files/company_tickers.json`, or its exchange-qualified variant `company_tickers_exchange.json`), fetched through the US adapter under the shared `sec_edgar` pacing policy and retained as an immutable captured document revision. The resolver then fetches that CIK's submissions record and requires it to list the same ticker. When a supported active US `stock_universe` listing has exactly one official CIK candidate, that candidate's submissions record confirms the ticker, and no existing accepted link or identifier for the security or CIK conflicts, the service principal may accept the CIK identifier and a **single-listing** issuer link automatically under policy `official_registry_single_listing`. The link revision retains both captured files, the SEC entity title, the matched ticker/exchange and the resolver policy version. The UI labels such links as registry-resolved rather than administrator-reviewed.

Automatic acceptance never applies when: the ticker maps to zero or several CIKs; the submissions record does not list the ticker; an accepted link or identifier for that security or CIK already names a different issuer; the security or CIK already participates in a shared-issuer/cross-listing mapping; or a ticker change/reuse is detected against a prior link revision. Those outcomes create a `review_required` proposal. An administrator may also supply a CIK in a research request; it is applied through the reviewed link path, never trusted from the request body alone. Names never participate in the match. A registry-resolved single-listing link satisfies the accepted-mapping requirement for shadow research and, later, gate 1 of §7.2; linking a second listing to that issuer still requires administrator acceptance.

`SocialCompanyIdentityService` remains the only existing administrator-attestation surface until its adapter is upgraded. After migration it becomes a compatibility facade over the accepted issuer-link selection; there must not be two independently writable issuer-grouping authorities. Preserve the old configuration and audit history for older generations.

### 4.3 Reporting scope and corporate changes

Every claim identifies `issuer_consolidated`, `issuer_standalone`, or a named `segment_or_subsidiary` scope. Segment names are issuer-scoped labels with evidence references; V1 does not need a universal business-unit graph.

A subsidiary product claim does not automatically become a consolidated-parent role or materiality claim. A primary-backed consolidation/ownership link can be retained as a limited synthesis premise, but it cannot establish an undisclosed earnings/revenue share.

Mergers, disposals, spin-offs, renamed/reused securities, and conflicting identifier updates trigger review and new mapping revisions. Do not rewrite old assessments under a successor issuer or transfer membership by ticker continuity.

Report both listing count and verified distinct-issuer count. Cross-listings of one issuer are not independent companies. Do not change existing basket weights in this release; annotate/filter cross-listing duplication and use verified issuer counts in the new exposure-coverage fields.

## 5. Claim-level assessment model

### 5.1 Atomic claim shape

An immutable claim revision contains:

```text
issuer_id, economic_theme_id, evaluated_theme_fingerprint
claim_kind, normalized_proposition, role, product_or_activity_key
reporting_scope, commercial_status
support_basis, conclusion, freshness_state, hold_reasons
supporting_evidence_edges[], conflicting_evidence_edges[]
effective_start/end or reporting_period, source_publication_time
first_available_to_app_at, assessed_at, last_substantive_support_at
fresh_until, verification_policy_version, model_attempt_refs[]
```

`claim_kind` is one of `participation`, `role`, `product_application`, `customer_relationship`, `commercial_status`, `materiality`, `exposure_end`. Product identity is issuer-scoped unless an explicit cross-issuer identity is supported.

Keep these axes separate:

| Axis | V1 values / meaning |
|---|---|
| Support basis | `primary_explicit`, `primary_synthesis`, `secondary_reported`, `inferred_unverified`, `unresolved` |
| Conclusion | `supported`, `contradicted`, `disputed`, `unknown` |
| Freshness | `current`, `stale`, `undated` |
| Commercial status | `research`, `announced`, `qualification`, `commercially_available`, `shipping_or_operating`, `discontinued`, `unknown` |
| Automatic use | Computed `allowed` or `held`, with reasons; not inferred from one confidence score |
| Membership | Separate security-level decision; not a claim-support status |

User-facing “verified” means **primary-backed under the recorded policy**. Display the primary source and whether synthesis was required. It is not independent certification of management's statement. Never give the whole issuer a blanket verification badge because one claim is supported.

### 5.2 Assessment identity and version selection

`IssuerThemeAssessment` is a stable dossier for one issuer–theme relationship. `AssessmentRevision` is an immutable result over a frozen evidence/policy input manifest. It selects claim revisions and records coverage gaps, conflicts, and questions searched without resolution.

A newer completed job is not automatically a better assessment. Selection respects explicit source corrections, reporting periods, scope, identity, and reviewed precedence. A late archived capture cannot revive a superseded claim merely by receiving a later ingestion ID.

A partial investigation may publish usable independently complete claims with explicit coverage limits, but it cannot silently remove previous claims. Carry forward prior claims with their original age and evidence, replace only claims with justified newer support, and publish new holds separately. Empty search results are not exposure-ending evidence.

### 5.3 Primary-source qualification

Qualify primary status per claim and passage. Examples include issuer-filed reports, attributable management statements, official product documentation, and direct statements by a named commercial counterparty about its own relationship.

An analyst question inside an official transcript is not a management assertion. A hosted third-party report is not primary merely because it appears on an issuer website. A regulator/exchange hosting a filing establishes filing provenance, not independent verification of the issuer's claims. Marketing material may support a product capability but not actual customer sales or revenue share.

Search snippets, generated summaries, old assessments, and downstream classifier output cannot supply primary support. They are retrieval aids only.

### 5.4 Bounded synthesis

V1 permits at most three primary premises and two explicit connecting links per synthesized claim. Every premise carries an exact original passage. Entity/product identity, relationship direction, relevant period, reporting scope and qualifications must be compatible. Larger or ambiguous chains require review.

Allowed example: an issuer report establishes commercial sale of product X; official documentation establishes X's HBM-testing capability. The conclusion is commercial availability of HBM-capable testing equipment, not proven HBM revenue or a named customer.

Disallowed inference: A supplies B; B makes HBM; therefore A supplies B's HBM operations. The product/application link is missing.

Documents from the same issuer, a translation, an exchange mirror and an issuer-hosted copy do not become independent confirmations. Distinct document counts remain provenance/coverage measures, not statistical-independence claims.

## 6. Materiality

Materiality is typed evidence, never `exposure_strength` or extraction confidence.

A materiality record retains `basis` (`disclosed`, `calculated`, `qualitative`, `unknown`), metric, value/range, units, currency, denominator definition, reporting scope, period, as-of, original precision, and all supporting passage IDs.

Allowed metrics include revenue, operating profit, production, capacity, backlog and assets, but their shares are not interchangeable. Preserve a subsidiary denominator or segment scope in the display.

Use decimal arithmetic. V1 calculations allow ratios and sums of explicitly non-overlapping compatible disclosed quantities. Inputs must match issuer/reporting scope, relevant period, unit and accounting basis. Preserve the derivation expression and operand revisions. Mixed-currency calculations require a separately approved conversion contract; V1 holds them rather than silently choosing an exchange rate. Reject division by zero, nonsensical share denominators, double-counted segments and unexplained out-of-range shares; retain original reported values for review.

A 30% server-segment revenue share is not a 30% AI Memory share. Product capability and shipping status cannot be converted to a percentage. Negative or zero consolidated profits cannot be turned into a meaningful exposure share through blind division.

Qualitative labels are restricted to `core_business`, `explicitly_material`, `explicitly_limited`, and `unknown`, with primary wording that supports the characterization. Management optimism alone does not establish materiality. Numerical materiality is never required solely because the source does not disclose it.

“Not separately disclosed in the reviewed evidence” is the default qualified wording. Only an explicit disclosure can support “the company does not disclose this measure.” Historical figures remain valid for their stated period even when unsuitable evidence of present scale.

## 7. Membership and role-policy contract

### 7.1 Theme-specific policy

Each theme has an immutable `RoleEligibilityPolicyRevision` containing permitted roles, required theme-specific evidence, acceptable commercial states, scope constraints, and policy rationale. Policies are administrator-reviewed; the model cannot broaden a policy to admit a discovered company. Missing policy means research can continue, but automatic addition is held.

Suggested seed policies are drafts, not automatically approved:

- Copper Miners: producing/mining roles; a downstream copper consumer does not qualify merely because it buys copper.
- Petroleum Refining: refinery owners/operators; metals refiners remain a different exposure.
- Crude/Product Tankers: relevant operators/owners, with vessel and cargo segment support. Rate exposure or spot/fixed-charter details are separate claims.
- AI Memory: theme-specific commercial memory participation and explicitly linked equipment roles where approved. Generic semiconductor exposure is insufficient.
- AI-Powered Cybersecurity versus AI Security: preserve the taxonomy's different mechanisms. Generic AI language cannot establish either role.

Producers/operators normally require `shipping_or_operating`. A reviewed equipment-supplier policy may accept `commercially_available` plus an explicit product/application link, while displaying that actual theme-specific sales remain unknown. Research, plans, qualification and unknown commercial status do not pass automatic admission.

### 7.2 Deterministic automatic-addition gate

All of the following must hold:

1. Accepted issuer-to-security mapping, supported active listing, and compatible reporting scope.
2. Accepted current theme identity and an approved role policy for its defining semantics.
3. Primary-explicit or admissible primary-synthesis support for the required participation, role, application and commercial claims.
4. No material language, identity, source-integrity, freshness, conflict or automation hold on the required claims.
5. No overriding administrator rejection/removal, conflicting accepted decision, or unresolved mapping conflict.
6. Valid recorded policy, evidence and assessment revisions, with compatibility readiness where the existing serving mode requires it.

Unknown materiality alone does not fail this gate. No single LLM score can pass it. Both explicit and synthesized primary support must be visible in the decision.

### 7.3 Membership origins and precedence

Maintain an origin-aware projection: `source_extraction`, `social`, `research`, `manual`. Preserve the original decisions and contribution IDs. A research assessment is not backfilled as a fake source `ClaimAssignment` merely to satisfy a foreign key.

Research additions emit typed membership contributions to the existing generation builder. New membership persistence references either an existing source contribution or an assessment-based decision through explicit foreign keys/discriminated types; it never requires a nonexistent source claim.

An administrator rejection applicable to the security/theme takes precedence over automatic research. Do not convert a rejection into a verified issuer fact or treat it as a denial of the business relationship. A supported exposure may remain excluded from the basket for a recorded membership reason.

Research-based removal, ended exposure, stale claims, role mismatch discovered later, and negative reassessment open a review case. Until decided, retain existing research-owned membership with `review_required` and visible warnings; hold its claims from new automated use. A failed fetch alone changes neither support truth nor membership.

A reviewed removal suppresses the relevant global membership according to its scope, not just one randomly selected supporting document. Re-entry cannot occur automatically until the administrator decision is explicitly superseded. An issuer-link correction never silently moves membership to another security.

### 7.4 Metrics and compatibility

Research may change an approved basket and its derived price/breadth statistics, with the membership revision pinned. It must not add narrative/technical/fundamental roots, create a development, or increase source-attention counts solely because research ran.

Report `research_verified_issuer_count`, `research_candidate_count`, `materiality_unknown_count`, and `membership_review_required_count` separately from existing direct-source/root counts. Do not retroactively rescore old generations.

Compatibility delivery carries typed research-origin contributions and reconciles complete membership state in order. It must not manufacture `ThemeMention` rows or reverse-mirror research into the source pipeline. Reuse the existing outbox framework with an explicitly owned membership projection stream; its key must be based on membership revision, not a fabricated article lineage.

If an old consumer cannot represent research evidence or membership safely, expose a coverage limitation or hold that capability until its adapter is upgraded. Do not silently create false evidence to preserve a legacy response shape.

## 8. Research workflow and bounded defaults

### 8.1 Triggers

- User requests `verify issuer–theme`, `refresh assessment`, or `discover candidates for theme`.
- A newly proposed company–theme relationship lacks a suitable current assessment.
- An enabled theme's bounded discovery schedule is due.
- A relevant new or corrected primary document, issuer mapping, policy change, freshness deadline or material conflict affects existing claims.

Use a deduplicated dirty queue. Repeated posts about an already assessed relationship reuse current research. Changes in relevance/coverage can queue additional work; they do not reset every dossier. A verified relationship reused across accepted cross-listings is researched once.

### 8.2 Work lifecycle

```text
queued → researching → evidence_ready → assessing → ready_for_publication
                      ↘ partial/held ↗
queued/researching/assessing → paused_allowance | paused_search_budget
                           | paused_storage | unavailable_capability | retryable_failure
                           | review_required | terminal_failure | cancelled
ready_for_publication → published (generation reference)
```

These are append-only job events; mutable queue status is an operational cache, not assessment truth. Claim-level publication may succeed while unresolved questions remain; the job must report partial coverage rather than claim comprehensive verification.

Freeze the actual evidence selection, theme semantics, issuer mapping, policy bundle and model result inputs for every assessment attempt. Do not reuse a result under a changed input hash. Fetch/LLM work occurs outside publication or producer locks; final persistence uses the existing shared fence and optimistic revision checks.

### 8.3 Proposed conservative limits

Limits are configurable policy, not estimates of research cost or sufficient coverage:

| Limit | Proposed V1 default after feature enablement |
|---|---:|
| Concurrent research investigations | 1 |
| Concurrent provider requests from research | 1 |
| Verification request | 1 issuer–theme pair |
| New issuers investigated per discovery job | 10; stop earlier at any budget/coverage bound |
| External search queries per job | 6, and zero paid queries unless explicitly enabled |
| Primary documents retained/processed per issuer investigation | 12 |
| Download size | 25 MiB/document; oversized result is a visible hold |
| Durable research evidence store | **Proposed R1:** 5 GiB total content-addressed originals and retained derivatives; reserve bytes before writing |
| Minimum filesystem free space | **Proposed R1:** 1 GiB after reservation; a volume-specific larger operator reserve takes precedence |
| Per-worker scratch | **Proposed R1:** 512 MiB tmpfs; included separately in container-memory sizing, not unlimited disk |
| Text extraction | Up to 300 pages/document, with omitted ranges recorded |
| Model-read passages | 24 per issuer investigation; capped tokens/characters per request |
| Selective image pages | 8 per issuer investigation |
| Browser-rendered navigations | 4 per issuer investigation |
| Paid/provider attempts | 24 per issuer investigation including retries, plus configured daily allowance limit |
| Automatic refresh polling | Hourly for due work; avoid issuing calls when nothing is due |
| Enabled-theme expansion | Weekly or on demand, not every theme-attention change |

The outer job allowance bounds all child investigations; ten candidates do not automatically authorize ten independent full budgets. Pause/resume retains progress without resetting cumulative limits. Query, candidate and page limits must be visible in coverage results.

Prioritize material conflicts and requested verification, then new memberships, then due refreshes, then optional expansion. Enforce fairness and a deployment-configured allowance allocation so the feature does not monopolize a shared subscription.

## 9. Discovery, acquisition, and market coverage

### 9.1 Evidence discovery hierarchy

Reuse retained original documents and accepted issuer metadata; enumerate permitted official documents; inspect approved issuer investor-relations/product pages; follow supplied primary links; optionally invoke configured search. Existing social/news evidence can seed leads, never satisfy primary verification by itself.

Paid-search-disabled is a supported operating mode, not a degraded verification standard. Research can work from official enumerators, verified issuer domains and supplied links. Discovery coverage must say which routes were available and which were disabled.

The repository exposes `tavily_api_key` and `serper_api_key` configuration names; their presence does not prove either service is deployed, paid for, or integrated. Task 00 checks deployed adapter code without exposing credentials. R1 selects **Tavily Search** as the optional concrete adapter because it fits an existing configuration surface, not because a comparative quality advantage was measured. Reuse a compatible deployed adapter if one is actually present; otherwise implement the bounded Tavily adapter. Do not require a Brave account. Provider selection remains `none` and paid enablement remains false by default; do not infer permission from either existing key. Serper is an available configuration surface, not a silently enabled fallback. Chat tools/connectors are not live-app infrastructure. [R11, W8]

### 9.2 Four launch-market adapters

Each market adapter must implement issuer resolution assistance, document discovery, fetching, publication/period parsing, correction identity, permission/rate controls, and explicit unsupported outcomes. No unverified hidden endpoint is an architectural dependency.

| Market | Official acquisition targets | Requirements and limits |
|---|---|---|
| US | SEC ticker-to-CIK file, submissions/filing documents plus verified issuer IR/product pages | Resolve CIK per §4.2 (R2). Use CIK-bound filing identity, accession/amendment links, declared user agent and fair-access limits. SEC submissions/XBRL APIs are documented, but entity-wide XBRL facts alone do not establish a theme-specific business exposure. [W1–W2] |
| HK | HKEXnews issuer-document search and official issuer pages | Preserve stock/issuer and bilingual-document identity; distinguish announcements, reports and prospectuses. Public title-search availability is documented; it does not establish blanket permission to automate every interface. [W3] |
| JP | EDINET, applicable public JPX/TDnet/issuer disclosures | EDINET API use requires registration/key configuration; lack of a key is a capability gap, not paid-search enablement. Public disclosure and paid historical services are different access paths. English summaries must not be presumed complete substitutes for Japanese documents. [W4–W5] |
| TW | MOPS/official issuer reports; TWSE OpenAPI for appropriate identity/metadata assistance | Check TWSE/TPEx listing distinction in the existing security authority. Preserve source calendars, company codes and unit scales. Open-data metadata does not itself verify all exposure claims. [W6–W7] |

The acquisition targets are documented externally, but end-to-end automated retrieval for all four markets has **not** been tested in this design exercise. Each adapter requires saved fixtures and an explicit opt-in live smoke test, including its permitted-access policy. Do not claim a whole market supported because one issuer PDF downloaded.

Full launch and automatic-admission enablement require all four markets to have a tested official/issuer retrieval route and truthful failure modes. This requirement does not block the US-only verify-only shadow preview in §18. Credentials and permitted-access configuration may still be required at deployment. Lack of access to one route is displayed; do not substitute licensed feeds or access-control bypasses.

### 9.3 Document identity, versions, and corrections

Keep a stable source-document identity and immutable content revisions. Prefer filing/accession/provider document IDs; otherwise bind canonical URL to the verified publishing origin. Store raw bytes by content hash with media type and provenance.

Retrieving identical bytes again creates a capture/check event, not another corroborating document or a newer business fact. Changed bytes, amended filings and explicit correction chains create new revisions. Capture order is not semantic precedence. An older saved copy first encountered later remains historical or held when it conflicts with newer accepted evidence.

Preserve document publication time, reported/effective period, first availability to the app, retrieval time and revision identity separately. Official translations and exchange/issuer mirrors link to the same originating disclosure where identity is established. Undetermined duplicates remain annotated and never assumed independent.

### 9.4 Multi-format extraction and evidence locators

Prefer text extraction. Parse document structure and retrieve relevant sections rather than feeding the first fixed number of characters to a model. Preserve neighboring paragraphs, section paths, page labels, table headings, units, footnotes and scope.

A passage locator binds original document revision hash, extractor version, section/page, original text offsets or table-cell references, and—when necessary—page-image bounding region. Retain extracted original-language text and a separate translation derivative.

Use browser rendering only for permitted public content that requires it, in an isolated renderer. Use model-based image interpretation selectively for pages where text extraction cannot recover the needed evidence. Do not silently add OCR/model dependencies or paid routes. When necessary capability is unavailable, persist `unavailable_capability` with the exact affected claim/page.

An image-only quantitative table is eligible for automatic materiality use only if labels, units, period and cell associations can be validated under the image policy. Otherwise it is review-required. Approximate chart readings are not exact disclosed values.

### 9.5 Language contract

Support English, Japanese, Traditional and Simplified Chinese. Retain original script and exact supporting passages. Translation is a derivative, not an additional primary source. Keep original uncertainty, negation, actor/supplier direction, modal verbs, dates, ranges, and magnitude units.

Reuse `OpenCodeGoTranslator`, `OpenCodeGoVision`, source-preserving segmentation/finalization in `multilingual_preparation`/`multilingual_v2`, and the existing translation normalization/quality and quantity display/review modules. Research adds document/page provenance, bounded reservation-aware calls and dossier-specific validation; it does not create a second language detector, translator or CJK magnitude parser. Preserve existing policy identities and increment only a policy whose behavior actually changes. Translate relevant passages for assessment/explanation rather than automatically translating every page. Any material ambiguity holds the affected claim; unrelated claims can proceed. Pin both translation and review policy/model identities. A bilingual document mismatch is a conflict to review, not permission to pick the convenient version.

### 9.6 Retrieval security and retention

Only fetch HTTP(S) through approved adapters and verified domains. Validate each redirect and DNS destination; block private, loopback, link-local, metadata-service and internal-network addresses. Browser subresources need the same network controls. Never send provider/API credentials to a redirected host.

Use media-type sniffing, decompression/page/byte/time limits, a sandbox for PDF/rendering, and no execution of downloaded programs, macros or document-supplied commands. Disable form submission and credential entry. Public-page JavaScript may run only inside the separately isolated, network-denied renderer described in §9.8; never in the app/research process. Source text cannot instruct models to use tools, change policies, reveal secrets or approve itself.

Respect documented terms, automated-access limits and retention rights. Missing permission is a capability hold, not a reason for covert scraping. Do not bypass login, paywalls, captchas or restrictions. Official API authentication such as EDINET configuration is distinct from automating a publisher login.

Store permitted originals and derivatives privately; expose short supporting passages and source links, not an unrestricted public mirror. Redact credentials, signed tokens and secrets from logs/citation URLs. Immutable records retain hashes and audit references; if legally required removal occurs, keep an explicit unavailability/tombstone record and disclose that full reproduction is no longer possible.

### 9.7 Storage capacity, retention and garbage collection — R1

The private store has a 5 GiB default byte ceiling plus a 1 GiB filesystem-free-space floor, both deployment-configurable. Count physical unique blob bytes, retained derivatives, staging reservations and unreconciled crash remnants; a repeated content hash is not charged twice. Before downloading or generating a derivative, reserve its worst permitted byte bound against the **shared store**, not just the current job. Concurrent writers cannot overbook the final space. Crossing either limit returns `paused_storage`, records required/available bytes and performs no download/model step that cannot be retained. Existing evidence reads remain available.

Retain any original/derivative referenced by a historical or current published assessment, an accepted or pending review, or a legal hold. Do not silently evict reproducibility dependencies. Garbage collection may remove only unreferenced blobs after 30 days and abandoned temporary objects after 24 hours; jobs with valid leases are excluded. A reviewed artifact becoming unreferenced does not itself bypass the 30-day grace. Mark/select candidates under a storage lease, recheck reference pins before deletion, and record a tombstone and reclaimed-byte ledger entry. Selecting new evidence and marking blobs for deletion must be mutually exclusive. Crash recovery reconciles filesystem and reservation records; unknown bytes remain charged until resolved.

When retained/pinned history fills the cap, pause new acquisition and show operator choices to raise capacity or perform an explicit verified archive/retention operation. Automatic GC must not delete old published evidence merely to make space. An archive is not considered a successful reproduction store until restoration/hash validation is demonstrated; lawful deletion remains explicitly unreproducible as in §9.6. PostgreSQL metadata, backups and general app logs need deployment monitoring separately; the blob limit is not a claim to cap all app disk use.

### 9.8 Browser deployment boundary — R1

Browser support remains in full V1, but is **off by default and outside the first verify-only slice**. The paired plan specifies an `exposure-browser` Compose profile, a secret-free renderer image, an egress-proxy image, a `network_mode:none` renderer, narrowly shared Unix-socket RPC volumes, a pacing RPC served by the research worker, request/byte limits and negative network tests (paired plan Appendix F). Do not start Chromium inside `celery-general`, `celery-datafetch`, the API container or the research worker.

The renderer uses a non-root Chromium sandbox with no IP network. Intercepted browser requests are fulfilled through its narrowly scoped Unix-socket egress proxy; a missed interception has no native network route. The renderer has no application or provider credentials; a per-job capability authorizes only bounded document fetches. The proxy resolves/checks each destination, rejects non-global or unapproved addresses, pins the checked IP when connecting, and preserves HTTPS hostname validation. Redirects create fresh checked connections; allowed public origins do not exempt subresources. Browser flags/interception are defense in depth, not the egress boundary. No app/database/Redis/model/search credentials or Docker socket are mounted into the renderer. Lack of the tested proxy/firewall/sandbox causes `unavailable_capability`; it does not authorize an unsafe browser fallback. [W9–W10]

**Egress broker has no Redis or database access (R2).** The application Redis runs without authentication, so any container that can reach it has full access to the Celery broker and every cache; a restricted ACL user would not isolate a container that can connect as the unauthenticated default user. The egress broker handles untrusted Internet responses and therefore is not attached to any network that reaches Redis, PostgreSQL, the API or other application services, and holds no Redis, database, model or search credential. Shared provider pacing and grant accounting for browser traffic are performed by the **research worker**, which already holds Redis and database access: for every HTTP attempt the broker asks the worker, over a dedicated Unix-socket pacing RPC, for a single-use fetch ticket; the worker validates the grant, charges the durable root budget, acquires the shared `RedisRateLimiter` key in strict distributed mode, and returns the ticket; the broker reports actual bytes/outcome back on the same socket. The pacing RPC exists only while the worker is waiting on that grant's render call and accepts only that grant. If the worker's pacing RPC is unavailable, the broker refuses the fetch. Compromise of the broker can exhaust the one active grant, not reach application state. The broker must also be unable to reach the Docker host gateway or published host ports (for example the frontend's port 80), verified by the deployment probes.

## 10. Subscription allowance and search-cost controls

### 10.1 Independent controls

- `EXPOSURE_RESEARCH_MODE=disabled` initially; `shadow` and `live` are explicit operator choices.
- `EXPOSURE_PAID_SEARCH_ENABLED=false` initially. Both explicit enablement and a currency-denominated cap are mandatory before a metered search call. Supplying a key is insufficient.
- `EXPOSURE_LLM_BILLING_MODE=subscription`; use `OpenCodeGoKimi` (`provider="opencode-go"`, `model="kimi-k2.6"`) from `theme_evaluation/kimi_client.py`, configured by `settings.opencode_go_api_key` and its existing Go endpoint resolution. A route change requires explicit validated subscription/capability configuration. Do not route research through `LLMService.completion(..., metered=True)`, `EconomicTaxonomyLLMProvider`, or the Social dollar ledger. Text/vision capabilities remain separately checked; no metered fallback.
- Operator-configured daily request/token allocation and concurrency limits are required. The account’s exact requests-per-minute limit, remaining allowance and reset balance are not established by this repository or specification. Show them as unknown unless actually reported; enforce local request/token ceilings, one research provider call at a time, and honor `429`/`Retry-After`. Local quotas are not a promise of available provider capacity, especially when other app features share the account.
- Search has a separate daily/monthly cost ledger and reservation ceiling. This feature's live-app costs are distinct from implementation-agent usage and existing hosting costs.

Reuse the Go transport and its preparation adapters; extend response/attempt metadata compatibly where needed. Do not misuse a dollar-priced Social ledger as a fictitious subscription balance or record an unknown dispatched failure as `actual_cost=0`. A resource reservation has a typed unit (`requests`, `reported_tokens`, `currency_amount`, or retained/staging `blob_bytes`) and purpose. Records must distinguish estimated reserved bounds, actual reported usage, and unknown actual usage.

A request cannot reset its allowance by pausing, retrying, changing model or spawning child investigations. A paid search request with unknown/unbounded pricing is blocked until an operator-approved costing policy can reserve its maximum allowed charge. Cancellation does not automatically refund a dispatched or uncertain request.

### 10.2 Retry and artifact reuse

A logical research request, provider attempts and successful reusable artifacts are separate identities.

- Successful parsing, extraction, translation and review results are reused when exact inputs and policy/model identities match.
- A retryable timeout/quota failure is an attempt event, not a successful cached artifact.
- A permitted retry creates a distinct immutable attempt record; success never overwrites the failure.
- An ambiguous dispatched timeout retains uncertain consumption and follows provider idempotency/reconciliation policy. Do not assume zero charge or zero allowance consumption.
- **Dispatch classification (R2):** a failure is **pre-dispatch** only when no request bytes can have reached the provider: missing credentials or route configuration, DNS resolution failure, connection refused/unreachable, connect timeout, pool-acquisition timeout, and TLS handshake failure before the request is written. Pre-dispatch failures release their reservation. Write timeouts, read timeouts, connection loss or protocol errors after the request started, and received HTTP error responses (including `429` and `5xx`) are **dispatched**: a response with a status code is dispatched with a known outcome and unknown usage unless usage is reported; the rest are uncertain. The transport must preserve the underlying exception class so this classification is exact rather than inferred from a collapsed error code.
- **Uncertain reservations expire with their period (R2):** the OpenCode Go route exposes no usage-reconciliation source beyond usage fields in a successful response. An uncertain reservation is therefore charged to the allocation period in which it was dispatched, is never refunded or carried into the next period, and closes as `expired_uncertain` when that period ends. A later successful response that reports usage for the same attempt, if one ever arrives, may reconcile it before expiry.
- Budget exhaustion pauses work and preserves completed evidence. It never weakens verification or moves to a metered model.
- Authentication/capability/permission failures stop that route visibly. Retries use bounded backoff, not a busy loop.

Actual route, model, parameters, request/response hashes, provider request ID when returned, policy revision and usage metadata must be retained. Tokens reported by a subscription route are usage telemetry, not automatically converted to dollars.

### 10.3 Dedicated research worker and shared provider pacing — R1

The queue is **`exposure_research`** and the worker is **`celery-exposure-research`**, hostname `exposure-research@%h`, initially one execution slot, prefetch one. Its tasks perform staged research/network/model operations; no worker executing price fetch queues consumes this queue. Document the deliberate exception to the existing general external-API queue rule in `CLAUDE.md`, `backend/start_celery.sh` and both Compose files. Default feature mode still prevents any dispatch until configured.

All research HTTP attempts use the existing `RateBudgetPolicy`/`RedisRateLimiter` provider keys. SEC uses the existing `sec_edgar` policy (0.15-second **local global interval** at the inspected baseline), not a newly independent SEC bucket. Reuse the provider/market key selection and add only missing official-provider entries. Where aggregate and market constraints both apply, research must acquire both, using identical aggregate key semantics to other callers. Research fails closed when a required distributed limiter is unavailable instead of using an independent process-local quota. Do not alter legacy fallback behavior as a side effect. Per-job document limits and daily spending/allowance reservations apply **in addition to**, not instead of, shared rate pacing. Browser-originated requests acquire the same provider keys through the research worker's pacing RPC (§9.8); the egress broker never talks to Redis itself.

Proposed R1 worker cap: 1 CPU / 2 GiB memory, 512 MiB bounded scratch, one prefork process on Linux, the documented solo pool locally on macOS. These are initial deployment limits, not measurements; prove resource/safety behavior before enablement. Browser memory is allocated to its separate profile, not hidden inside this worker.

## 11. Freshness, contradictions, and automation holds

### 11.1 Proposed V1 freshness policy

Freshness limits attach to claim type and substantive evidence time, not the job completion or download time:

| Claim type | Proposed automatic-use window |
|---|---|
| Stable business role/product application supported by a dated operational disclosure | 450 days from the evidence's asserted-as-of time, or publication date when the passage unambiguously describes then-current activity |
| Named customer relationship, qualification stage, commercial transition | 180 days from substantive dated support |
| Quantitative materiality | Always displayed for its actual period; never silently relabeled current. Newer compatible reporting supersedes current-period use. |
| Undated capability/relationship without a supported current-as-of anchor | Review-only for automatic-use purposes |
| Issuer mapping | Revision/decision based; corporate-action conflicts hold use. Do not recreate identities merely because a timer elapsed. |

These are conservative defaults to validate, not evidence that a business fact stops being true on its expiry date. A legitimate current dated reaffirmation can refresh a claim; another download or translation of old evidence cannot.

Run freshness evaluation locally at least hourly for due claims and before every automated decision. It can append hold revisions without calling a provider. On-demand grounding also checks expiry at its actual use time.

### 11.2 Conflict handling

Preserve opposing passages. Compare metric, subject, scope, period, modality, and actual-versus-forecast status before declaring contradiction. A later revenue period is usually another observation, not automatically a restatement of the previous period.

A credible material conflict appends a claim-specific automation hold and opens review. Explicit business-exit/disposal evidence may support an exposure-end claim, but removal still follows the reviewed membership workflow. Missing documents or unsuccessful searches create coverage limitations, not negative exposure claims.

If publication lags, a held claim must not slip into a new automated admission or grounding call. Writers perform a current safety check against append-only hold/issuer/policy decisions and record the checked revision. This safety check can only block or defer an action; it cannot replace its evidence with unpinned newer claims.

Historical product reads remain generation-pinned. Current UI shows assessment-as-of and publication-as-of. It must not claim “checked just now” merely because the API request occurred now.

### 11.3 Related in-flight work

A provider call already in flight when its support is held may finish for audit, but cannot commit a new automatic use without revalidation. At persistence/publication, reject or requeue decisions whose required support, issuer mapping or role policy became disqualified. Do not alter prior completed classifications or earlier generations.

## 12. Targeted classification grounding and prevention of circular support

### 12.1 Grounding selection

Resolve the incoming source's explicit issuer, security, product and activity references first. Retrieve only relevant claims from an accepted exposure selection in a serving generation. Freeze:

```text
serving_generation_id
issuer_mapping_revision_ids
assessment_revision_ids, selected_claim_revision_ids
original_evidence_revision_ids and exact passage locators
support_basis, commercial_status, freshness/hold check revision
retrieval_policy_version, grounding_policy_version, context_content_hash
```

Bound the context: proposed V1 maximum five relevant claims per issuer and ten per source, with a configured token ceiling. No network research runs inline in normal classification. Missing research enqueues verification separately; the source classifier continues with its existing evidence rules rather than blocking indefinitely.

The grounding cache semantic key uses the selected claims/evidence, relevant issuer/theme semantics and policies—not the entire global serving generation. Merely republishing an unchanged assessment must not re-extract all incoming content. Retain the generation ID as provenance separately.

### 12.2 Allowed and forbidden uses

An official product-to-HBM-testing claim can interpret a new source reporting orders for that product. The new classification labels the connection as inferred using retained business evidence. It cannot attribute the order increase to AI demand without relationship support.

A generic “Issuer A shares rose” does not justify attaching all known issuer themes or stronger HBM demand. A source mentioning a customer does not automatically describe every business of that customer.

Keep original-source citations and research-context citations separately addressable through claim review. A primary-backed synthesized research claim is still contextual inference when interpreting a different new source. Do not collapse the research `support_basis` into the existing exposure-support enum by equating primary support with direct event support.

### 12.3 Provenance acyclicity

Every verified research claim must ultimately trace to retained original primary passages. Assessments, generated summaries, memberships and classifier outputs cannot appear as new primary premises. A prior assessment may be referenced only by expanding and validating its original support edges.

Reject a synthesis that depends on its own assessment output or on a downstream classification that used it. Record parent/derivation IDs and enforce acyclic evidence dependencies. Reusing one primary document in several assessments is evidence reuse, not additional corroboration.

## 13. Persistence boundaries and complete state ownership

This section defines logical storage ownership; the implementation plan will allocate migrations and exact module splits. These names are proposed new objects unless explicitly identified as existing. Avoid putting unrelated research rows into the large economic-runtime model file.

| Object group | Identity, revisions, and constraints |
|---|---|
| Issuer registry | `ExposureIssuer` stable UUID; immutable identity provenance; `IssuerSecurityLinkRevision` with unique `(security_id, revision_number)` and accepted/proposed/rejected state; identifiers are scheme-scoped; existing Social IDs bridge through preserved attestation. |
| Documents and captures | `ExposureDocument` stable original-source identity; `ExposureDocumentRevision` unique by document/content hash; capture/check events separate from revisions. Correction/equivalence edges retain provenance. |
| Prepared passages | `ExposurePassage` unique by document revision, preparation policy and locator; original text, table context and optional image reference. `PassageDerivative` retains translation/vision output and parent hash. |
| Research orchestration | `ExposureResearchRequest`, append-only `ResearchEvent`, durable `ResearchCandidate`; leased queue state separate from immutable selected inputs. Record root job budgets, progress, retryability, cancellation, and coverage. |
| Provider and search usage | Typed `ResearchReservation`, immutable attempt/result events, reusable successful artifacts. Use existing transport/attempt infrastructure through an adapter; do not overwrite original Social dollar-accounting semantics. |
| Claims and evidence edges | `ExposureClaim` stable proposition identity and immutable `ExposureClaimRevision`; supporting/conflicting `ClaimEvidenceLink` rows; each link has exact scope, direction and locator. Dependency cycles rejected. |
| Materiality | Typed claim payload or normalized child row with metric, value/range, unit, currency, period, denominator, formula and operand revisions. No generic strength percentage. |
| Assessments | Stable `IssuerThemeAssessment`; immutable `AssessmentRevision` unique by dossier and revision, plus input-manifest hash for idempotent reuse. Revision selects exact claim revisions and coverage state. |
| Reviewed policy and holds | `RoleEligibilityPolicyRevision`, `ExposureUseHoldRevision`, reviewed assessment/issuer decisions, with trusted actor/reason and append-only history. Reviews cannot be overwritten by model processing. |
| Membership | Stable research-origin membership association unique by `(economic_theme_id, security_id)`; immutable `ExposureMembershipDecisionRevision` unique by `(association_id, revision_number)`. Existing Social/source origins remain referenced, not deleted. |
| Serving selection | A sealable `ExposureSelectionSet` plus rows selecting one assessment revision per dossier, exact issuer/policy/hold/membership revisions, freshness-as-of, and manifest references. It has no independent live pointer. |
| Grounding uses | Immutable `ExposureGroundingUse` pins each source-processing use to selected claims, evidence, context hash, safety-check revision and policy, linked to the existing classification request/attempt. |

All selected payloads and evidence revisions are append-only/sealed with the repository's database and ORM protections. Draft assembly and leased-work status may be mutable operationally. A pair-only uniqueness constraint must not be placed on a revision table.

Same-issuer/same-theme assessments may contain multiple valid role/product claims. One dossier is not a single undifferentiated exposure edge. A changed proposition gets a new claim identity or explicit revision/supersession, rather than mutating the meaning of an old claim ID.

Research is multi-document: do not force a dossier through the source pipeline's one-selected-attempt-per-source-lineage mechanism. Reuse original document identity and integrity patterns, while the assessment has its own frozen multi-source manifest. Periodic reports can coexist; a later report does not erase every older still-valid claim.

## 14. Publication and membership integration

### 14.1 One serving authority

Extend `GenerationInputManifest` with a typed, versioned exposure-input section. Pin the `ExposureSelectionSet`, issuer mappings, assessment/claim revisions, role-policy revisions, holds, membership decisions, freshness-as-of and corresponding compatibility projection revisions.

The existing `EconomicTaxonomyPublicationCoordinator` remains the only publisher. Research preparation selects sealed inputs and builds new exposure snapshot entries outside the final lock. Publication switches all pointers with the normal generation compare-and-set. The exposed map cannot be newer than the membership and grounding context with which it claims coherence.

Older generations with no exposure section remain valid and return `exposure_research_unavailable_for_generation`, not an invented zero or a join to latest research. Version reader contracts and serializers; never change hashes or payloads of previously sealed artifacts.

### 14.2 Cutoff and safety

Research, issuer, hold and membership producers append typed dirty revisions using the existing shared producer fence and lock order. No provider calls occur while holding these locks.

Ordinary post-cutoff evidence is backlog, preserving publication progress. A new material disqualification of support used for an automatic addition is an applicable safety invalidator; the publisher must not activate that decision from a now-disqualified claim. Reprepare outside the lock. Define invalidation scope using the exact dependencies rather than rejecting all publications for unrelated research.

A mere new research result does not mutate the processing taxonomy or invalidate every classifier. Semantic changes use the existing reviewed taxonomy path.

### 14.3 Ordered compatibility delivery

The inspected `TaxonomyProjectionEvent` and checkpoint use a string `source_lineage` transport key and a projection kind, not a foreign key requiring a document lineage. [R9] Reuse that infrastructure with a namespaced logical owner such as:

```text
source_lineage = research-membership:<issuer_uuid>:<theme_uuid>
projection_kind = research_membership
projection_revision = monotonic accepted membership-projection revision
origin_representation = exposure_research
```

This is explicitly a **transport ownership key**, not a fabricated `SourceFamily` or article. It contributes zero source-evidence counts.

Payloads replace that research origin's complete membership contribution, include reviewed decisions/assessment IDs, and do not remove another origin's independent contribution. Stage alongside the candidate generation; deliverability follows its durable published event. Crashing after publication but before notification cannot strand events. Older revisions are successful no-ops; mirror-origin deliveries do not re-enqueue research or source extraction.

Keep source/Social and research evidence provenance separate even when their accepted memberships converge. A new typed membership adapter must reconcile administrator precedence; merely appending research rows to the catalog's constituent array is insufficient.

### 14.4 Freshness and rollout schedules

Exposure-only generation refreshes reuse the existing coalesced economic publication schedule. Freshness deadlines and accepted assessment changes mark the generation dirty without requiring a provider call. Capability/contract changes require a new verified reader manifest; routine data refreshes do not.

Research jobs can run in shadow mode for comparison, but cannot publish live admissions or grounding there. Live integration requires economic authority and the new reader/compatibility gates. Disabling further research stops acquisition; it does not silently delete retained evidence or previously accepted memberships. Continue local hold enforcement and reviewed correction paths.

## 15. Service interfaces and module ownership

Prefer a small package with internal modules, not another set of unrelated global services. Proposed location: `backend/app/services/company_exposure/`, with pure contracts/policy in `backend/app/domain/company_exposure/` and focused research models outside the existing economic-runtime module.

| Service boundary | Consumes | Produces / prohibition |
|---|---|---|
| `IssuerIdentityAdapter` | Existing attested mappings, `StockUniverse`, official identity references | Pinned accepted issuer/listing mapping or unresolved proposal; never name-only merge |
| `ExposureResearchCoordinator` | Scoped request, theme policy, allowance/search controls | Leased bounded work, evidence manifest, progress/coverage; no direct live membership write |
| `DocumentAcquisitionRegistry` | Approved issuer/domain/document target, market adapter | Immutable original revision/capture or typed gap; no unrestricted browser/search agent |
| `ExposureEvidencePreparer` | Original revision, target questions, format/language policy | Exact passages, linked derivatives and limitations |
| `ExposureAssessmentService` | Frozen original evidence, issuer/theme scope, prior assessment, policies | Immutable claim-level assessment revision and review/hold proposals |
| `ExposureMembershipEvaluator` | Accepted claims, role policy, listing identity, all decision origins | Proposed/accepted/held addition or removal review; never synthetic source mention |
| `ExposureGroundingSelector` | New-source explicit entities/activities, accepted selection, safety check | Bounded frozen research context and provenance |
| `ExposurePublicationAdapter` | Explicit cutoff, approved selection revisions, existing generation preparation | Sealed exposure selection/snapshot entries and typed membership projections |
| `CompanyExposureReader` | One resolved serving generation | Dossiers, claims, membership reasons and historical evidence; no latest-row bypass |

Research-stage transport and generation-stage publication are distinct. Reuse existing utility functions/providers only after verifying their contracts; a shared helper that commits its own transaction cannot be called inside another locked publication transaction.

The existing data model has raw source constituent facts and Social-specific decision references. The plan must explicitly add the assessment-backed membership union and publisher/read adapters. It must not assume a generic global membership API already accepts arbitrary research contributions.

## 16. API and product behavior

Route names below are proposed API contracts to align with repository conventions in the implementation plan:

```text
GET  /api/v1/company-exposures?security_id=...&generation_id=...
GET  /api/v1/company-exposures/{assessment_id}?generation_id=...
GET  /api/v1/economic-themes/{theme_id}/exposures?generation_id=...
GET  /api/v1/company-exposures/research-jobs/{job_id}
GET  /api/v1/company-exposures/research-jobs/{job_id}/preview
POST /api/v1/company-exposures/research-requests
POST /api/v1/company-exposures/admin/decisions/preview
POST /api/v1/company-exposures/admin/decisions/apply
PUT  /api/v1/company-exposures/admin/theme-research-policy/{theme_id}
```

The first three are generation-scoped product reads. Job status is operational and must identify that it is not the live assessment. **R2 route authority:** these are the only research-job paths. `POST /research-requests` creates a request (returning its job ID), `GET /research-jobs/{job_id}` returns operational progress, and `GET /research-jobs/{job_id}/preview` returns the job's immutable assessment/evidence revisions with `view_kind="shadow_preview"` and `authoritative_membership=false`. The US shadow slice (§18) uses exactly these routes; no parallel `/research` or `/securities/{id}` path exists. Security-scoped product reads use the `security_id` query parameter on the first route. A request changing membership, issuer mapping, spending, role policy or review state requires the trusted existing administrator principal. V1 research initiation is administrator-authorized because it consumes shared allowance and may publish eligible additions; read access follows the existing app policy. Never trust a caller-supplied actor string.

Responses include issuer identity and mapping provenance, theme and definition reference, role, commercial status, support basis, materiality by metric/period, source passages, assessment/freshness time, conflicts, reviewed membership state, research origin, and generation ID. Short evidence excerpts link to retained originals under authorized access.

The UI adds:

- **Why included** on a theme's constituent row, distinguishing observed source membership from research verification.
- An issuer–theme dossier with separate claim badges: primary explicit, primary-supported synthesis, reported/unverified, held, historical, and materiality unknown.
- A candidate queue with concrete unmet requirements and bounded investigation coverage.
- Theme discovery enablement and role-policy review controls; no hidden automatic opt-in for all themes.
- Freshness/conflict warnings, review outcomes, original-language/English evidence toggles, and per-job allowance/search status.

Do not display a general “exposure 83%” unless it is a specifically defined disclosed/calculated share. Do not relabel extraction confidence as probability. An undated or unavailable source must not be shown as freshly verified.

## 17. Acceptance tests and evaluation gates

These are required implementation outcomes, not tests performed while drafting this specification.

### 17.1 Semantic and evidence fixtures

| ID | Counterexample / expected result |
|---|---|
| E01 | AI and memory co-occur without a relationship: no verified AI Memory exposure. |
| E02 | Official issuer product X is commercially available; another official passage links X to HBM testing: allow the narrowly worded synthesis; do not invent HBM sales or named customers. |
| E03 | A supplies B and B manufactures HBM: missing application link remains unverified. |
| E04 | A server segment supplies 30% of revenue: do not assign 30% to AI Memory. |
| E05 | Compatible disclosed numerator/denominator: reproduce the decimal calculation with exact scope, period and operand citations. |
| E06 | Conflicting periods, currencies, overlapping segments, or zero/negative profit denominator: hold the derived share; retain original numbers. |
| E07 | A customer/consumer appears in primary evidence: the map may show the role but a producer-only basket rejects automatic admission. |
| E08 | Supported current commercial exposure has no materiality breakdown: admission can pass the reviewed role policy and displays materiality unknown. |
| E09 | Qualification/planned investment without the required commercial status: candidate only. |
| E10 | Official transcript analyst question, search snippet, generated assessment or third-party report hosted by an issuer: cannot be primary evidence for the issuer claim merely by location. |
| E11 | Japanese/Chinese negation, modal language, magnitude units, role direction and table context are preserved; material ambiguity holds only affected claims. |
| E12 | A historical primary PDF is downloaded today: its business-evidence date does not advance. |
| E13 | Same original report via exchange, issuer site and translation: no multiplied corroboration count. |
| E14 | An old/partial capture arrives after a corrected source: cannot restore an obsolete exposure. |
| E15 | An older document's still-valid role claim and a newer period's materiality coexist without source-level wholesale replacement. |

### 17.2 Identity, decisions, and consumer fixtures

| ID | Counterexample / expected result |
|---|---|
| I01 | Two verified cross-listings reuse one issuer assessment; securities remain separately eligible and distinct issuer count stays one. |
| I02 | Similar names or ticker reuse do not merge issuers. An administrator mapping conflict is held, not auto-overwritten. |
| I03 | Subsidiary materiality cannot silently become consolidated-parent share. |
| I04 | An administrator-rejected membership stays rejected after new primary evidence; a review proposal may be opened. |
| I05 | Stale/disputed claims cannot support a new automatic admission or new grounding, but existing research membership remains flagged pending review. |
| I06 | Research no longer finds a document: coverage worsens, not exposure truth. |
| I07 | New-source product orders plus accepted product context can support contextual theme inference; it cannot establish an unstated AI-demand cause. |
| I08 | Generic issuer price movement cannot fan out across all known themes. |
| I09 | Research → classification → research self-support is rejected; original primary support remains the only verification basis. |
| I10 | Theme split/mechanism change cannot blindly copy accepted exposure to all destinations. |
| I11 | Same theme/security can retain accepted revision 1 and rejected revision 2; pair-level identity does not forbid revision history. |

### 17.3 Runtime, generation, and provider fixtures

| ID | Counterexample / expected result |
|---|---|
| R01 | Paid search configured with credentials but enable flag false: zero paid search calls. Missing spending cap also blocks paid calls. |
| R02 | Subscription quota exhausted: durable pause, no metered fallback, no guessed dollar charges. |
| R03 | Transient provider failure then success: one logical request, two immutable attempts, one reusable success; another retry issues no call. |
| R04 | Ambiguous dispatched timeout/cancellation preserves uncertain allowance/cost reservation. |
| R05 | Pause/resume, model change and child investigation cannot reset cumulative job budgets. |
| R06 | Research and source/Social workers contend: one ordered fenced commit, with no provider call under the lock. |
| R07 | G1 stays reproducible after G2 assessment, mapping, membership, hold and policy revisions. Old generations lacking the exposure extension remain valid. |
| R08 | Assessment publication changes no source-attention root counts or events. Research membership can change basket coverage through its own provenance. |
| R09 | Crash after generation commit before worker notification: compatibility delivery resumes from durable published state. |
| R10 | New membership projection arrives before old one: old delivery is a no-op; another origin's support is not deleted. |
| R11 | Ordinary research arrives during snapshot preparation: a coherent cutoff still publishes; a newly disqualified required support blocks only the unsafe automatic action. |
| R12 | Time-based freshness expires without new external documents: holds and a subsequent generation are produced without a provider call; automation checks expiry even if publication lags. |
| R13 | Forged actor, unauthenticated mutation, or paid-enable change: rejected before durable decision/spend side effects. |
| R14 | Private-network redirect, credential leakage, oversized archive/PDF, hostile document instructions: blocked or sandboxed with a typed failure. |
| R15 | Feature disabled or rolled back: no new research spending/admissions; historical artifacts remain and existing source/Social behavior is preserved. |

### 17.4 Evidence-quality release gate

**Human ownership:** the product owner (the user or an explicitly appointed research owner) is accountable for expected exposure/membership labels. A second designated human reviewer checks proposed automatic-admission and critical-negative cases; disagreements are recorded and adjudicated before gate acceptance. The implementing agent prepares documents, manifests and reports but does not approve its own ground truth. Assign reviewer identities when the first shadow slice is prepared; collect US cases then and add HK/JP/TW cases alongside each adapter, not at the end of implementation. No named reviewer or completed adjudication means automatic admission remains off, while verify-only shadow use may continue.

Keep an adjudicated corpus spanning all four launch markets and the five original theme families: memory, cybersecurity, refining, tankers, and copper. Include source documents with negative, ambiguous, stale, synthesis, image/table and cross-listing cases—not just successful examples.

Proposed minimum: 80 issuer–theme cases, at least 20 per launch market, plus deterministic negative/unit fixtures. Freeze exact source revisions and split evaluation by issuer and originating disclosure, not by overlapping passage. Retain reviewer disagreements and final adjudication.

Report per-claim primary-support precision, role/application/commercial-state accuracy, supported-calculation accuracy, false automatic admissions, conflict/staleness handling, coverage and provider use. No zero-error claim follows from a finite sample. Because false admission is the high-cost failure, V1 automatic admission requires zero observed critical violations in the held-out contract corpus; a failing category/market remains review-only until corrected. Report denominators and uncertainty, and do not treat this gate as a population precision guarantee.

The held-out corpus should contain at least 40 adjudicated auto-eligible cases across the launch markets; zero erroneous admission is meaningful only alongside nontrivial successful coverage. Proposed minimum recovery is 80% of those cases. Failure pauses automatic admission for the affected scope, not source ingestion or evidence viewing. Thresholds are product gates, not model self-assessed scores.

Mechanical invariants, migrations, producer fencing, publication recovery and budget reservations require the existing style of exact-node PostgreSQL tests with no skipped required cases. Routine CI uses saved documents and mocked providers. Live retrieval/model probes are explicitly opted in, bounded, credentialed, and do not alter production. Record untested routes honestly.

## 18. Delivery boundaries and rollout

This is one coherent feature delivered in independently reviewable slices, not a requirement to deploy all privileges at once.

**Explicit intermediate deliverable:** US-only, verify-only **shadow** research is allowed before HK/JP/TW adapters, paid search, browser rendering, candidate expansion, membership automation or classifier grounding. It must be usable: submit one existing US issuer–theme pair with retained/supplied/SEC documents, resolve its CIK per §4.2 (or route an ambiguous result to administrator review), run bounded subscription research on `exposure_research`, and inspect a dated read-only assessment/evidence/job preview through `POST /research-requests`, `GET /research-jobs/{job_id}` and `GET /research-jobs/{job_id}/preview` (§16). Its API/UI is labeled `shadow_preview`, has immutable assessment/evidence references, and does not claim to be a serving-generation product result. It does not switch serving pointers, change basket membership, generate theme observations, or ground incoming classification. Existing production readers stay unchanged.

This preview can run on the live deployment as an administrator-only feature without enabling live membership. The four-market D08/evaluation gate applies to **automatic admission/full launch**, not to shipping this preview. A later generation-bound assessment-only release still requires its reader/history/permission gates, but not candidate discovery or grounding. No intermediate release is advertised as four-market automatic coverage.

Delivery sequence:

1. **Contract and evidence core:** Freeze issuer identity, claim types, original evidence, temporal precedence, materiality, policy and membership invariants. Add safe ingestion and fixtures with no live membership change.
2. **Verification workflow:** Ship the US verify-only shadow slice and preview first; then add HK/JP/TW and language/image/browser capabilities in parallel against the shared `PreparedEvidence` contract. All required full-launch routes and budgets must pass before automatic action.
3. **Generation-bound assessment reads:** Add selection and reader capability; publish assessments without inventing mentions or changing existing basket decisions.
4. **Controlled membership integration:** Turn on eligible automatic additions only for reviewed role policies and validated scopes; prove administrator precedence, holds, origin-aware compatibility and rollback.
5. **Enabled-theme candidate discovery:** Add bounded expansion using the same verification path. Disabled paid search remains supported and visible.
6. **Targeted classifier grounding:** Activate only after context provenance, dependency holds, no-circular-support tests and generation pinning pass.

All database changes are additive at first. Existing source facts are imported as research leads with original provenance, not relabeled primary-verified. Current memberships do not disappear at rollout because they have not yet been researched. Existing Social decisions and issuer attestations are preserved before facades are changed.

Do not create a new full-taxonomy migration or duplicate the Idea 1 authority machinery. Register new research revisions as publication inputs, update consumers, and publish the upgraded capability through the existing system. No paid provider is enabled and no production cutover is performed by generating this document.

The implementation plan must explicitly name ownership for each object in section 13 and each integration in section 15. Each migration-owning task must allocate/recheck its migration against the then-current execution head and date immediately before commit, include red/green test steps, and verify all import/API names against current code rather than copying proposed names as if already implemented.

## 19. Principal risks and mitigations

| Risk | Mitigation / required evidence |
|---|---|
| Source coverage creates false confidence | Show searched routes, dates, omitted pages and disabled capabilities; no search-failure inference. |
| Model confuses capability, qualification and commercial use | Separate claim/status axes and reviewed role requirements; adversarial fixtures. |
| Research makes theme attention self-reinforcing | No generated mentions/developments; original evidence DAG and channel separation. |
| Issuer grouping corrupts multiple listings | Preserve administrator mapping authority, immutable links and no name-only merges. |
| Multi-source materiality invents precision | Typed period/denominator/scope, decimal formulas, no undisclosed numerical models. |
| Subscriptions are exhausted by long reports | Section retrieval, artifact reuse, one-worker defaults, shared allowance and resumable bounds. |
| Publication/compatibility bypasses prior reviews | Origin-aware decisions, common fence/coordinator, ordered typed membership projection. |
| Stale evidence continues in new actions | Claim-level holds and last-moment safety checks, without rewriting history or automatic removal. |
| Official sites change or restrict retrieval | Versioned adapters, permitted-access checks, retained fixtures, explicit capability gaps. |
| Too many review cases prevent useful automation | Prioritize conflicts and membership candidates; use validated primary synthesis; measure coverage, not only precision. |

## 20. Traceability and written-spec approval

| Decisions | Implemented in this specification |
|---|---|
| D01, D07 | Sections 1, 8–9, 18: verification plus enabled-theme bounded discovery |
| D02, D17 | Sections 5, 9, 17: claim-level primary evidence and bounded synthesis |
| D03, D05, D06 | Sections 7, 11, 14: tiered admission, unknown materiality, reviewed role policies |
| D04 | Section 6: disclosed/calculated/qualitative materiality and reporting scope |
| D08, D09, D14 | Sections 9 and 17: four markets, languages, multi-format retrieval and validation |
| D10 | Sections 3–4, 13–14: shared issuer research and listing-specific decisions |
| D11–D13 | Sections 9–10: public retrieval, subscription allowance, paid-search default off |
| D15–D16 | Sections 11–12: pinned grounding, stale/conflict holds, no circular evidence |
| D18 | Sections 1, 3, 13–15: dedicated integrated subsystem, not a competing authority |

**Approval record:** The original architecture was approved. The user accepted the R1 amendments/defaults as corrected by R2 (see the R2 review-change record). Record that acceptance against this R2 file's SHA-256 when it is installed; it does not retroactively change the original approval record. Implementation may begin with the plan's Task 00 and the US shadow slice. Do not mark the feature production-ready or enable any privilege on the basis of this document alone.

## 21. Sources and inspection record

Repository references below are pinned to the inspected commit. They establish the baseline only; proposed services and policies in this document are new design decisions. Original official-document-service references were checked on 2026-09-25; R1 transport, queue, rate and settings seams were checked at the same pinned commit on 2026-09-26. Additional search/deployment references are listed separately. They establish documented discovery/access capabilities, not that this app's future adapters have already passed live tests.

### Repository

- **[R1]** Existing evidence and constituent records: [economic_taxonomy_runtime_evidence.py](https://github.com/xang1234/stock-screener/blob/28c220e4e4ca5afcb5a380678bb2b80dd7f388b7/backend/app/models/economic_taxonomy_runtime_evidence.py).
- **[R2]** Source-derived fact/constituent materialization: [economic_theme_observation_service.py](https://github.com/xang1234/stock-screener/blob/28c220e4e4ca5afcb5a380678bb2b80dd7f388b7/backend/app/services/economic_theme_observation_service.py), particularly `_materialize_constituents`.
- **[R3]** Administrator-attested issuer grouping: [social_company_identity_service.py](https://github.com/xang1234/stock-screener/blob/28c220e4e4ca5afcb5a380678bb2b80dd7f388b7/backend/app/services/social_company_identity_service.py).
- **[R4]** Existing publication coordinator and contracts: [economic_taxonomy_publication.py](https://github.com/xang1234/stock-screener/blob/28c220e4e4ca5afcb5a380678bb2b80dd7f388b7/backend/app/services/economic_taxonomy_publication.py) and [economic_taxonomy_publication_contracts.py](https://github.com/xang1234/stock-screener/blob/28c220e4e4ca5afcb5a380678bb2b80dd7f388b7/backend/app/services/economic_taxonomy_publication_contracts.py).
- **[R5]** Sealed reader-bundle construction: [economic_taxonomy_snapshot_builder.py](https://github.com/xang1234/stock-screener/blob/28c220e4e4ca5afcb5a380678bb2b80dd7f388b7/backend/app/services/economic_taxonomy_snapshot_builder.py).
- **[R6]** Generation-scoped product reads: [economic_theme_read_service.py](https://github.com/xang1234/stock-screener/blob/28c220e4e4ca5afcb5a380678bb2b80dd7f388b7/backend/app/services/economic_theme_read_service.py).
- **[R7]** Background/source grounding boundary: [theme_grounding_context.py](https://github.com/xang1234/stock-screener/blob/28c220e4e4ca5afcb5a380678bb2b80dd7f388b7/backend/app/services/theme_grounding_context.py).
- **[R8]** Existing route configuration/preparation: [services/llm/config.py](https://github.com/xang1234/stock-screener/blob/28c220e4e4ca5afcb5a380678bb2b80dd7f388b7/backend/app/services/llm/config.py), [live_attachment_tasks.py](https://github.com/xang1234/stock-screener/blob/28c220e4e4ca5afcb5a380678bb2b80dd7f388b7/backend/app/tasks/live_attachment_tasks.py), and [theme_evaluation/kimi_client.py](https://github.com/xang1234/stock-screener/blob/28c220e4e4ca5afcb5a380678bb2b80dd7f388b7/backend/app/services/theme_evaluation/kimi_client.py). These references do not verify the user's subscription balance or account capabilities.
- **[R9]** Publication/outbox/checkpoint models: [economic_taxonomy_runtime_publication.py](https://github.com/xang1234/stock-screener/blob/28c220e4e4ca5afcb5a380678bb2b80dd7f388b7/backend/app/models/economic_taxonomy_runtime_publication.py).

### Official document services

- **[W1]** SEC: [EDGAR Application Programming Interfaces](https://www.sec.gov/search-filings/edgar-application-programming-interfaces).
- **[W2]** SEC: [Accessing EDGAR Data](https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data).
- **[W3]** HKEXnews: [Listed Company Information search explanatory note](https://www.hkexnews.hk/homelcicontentsearch.html).
- **[W4]** FSA EDINET: [public portal/API registration guidance](https://disclosure2.edinet-fsa.go.jp/week0020.aspx).
- **[W5]** JPX: [Overview of TDnet](https://www.jpx.co.jp/english/equities/listing/disclosure/tdnet/) and [Company Announcements Service](https://www.jpx.co.jp/english/listing/disclosure/).
- **[W6]** TWSE: [official OpenAPI catalog](https://openapi.twse.com.tw/).
- **[W7]** First-party issuer example linking annual reports to MOPS: [Shin Kong Synthetic Fibers investor information](https://www.shinkong.com.tw/en/front/investors). This is evidence of an official issuer/MOPS retrieval route, not proof of a generic MOPS document API.

## R1 review-change record

Original approved artifact SHA-256: `a182065bf6b1e5e07026045b50e6ae9124501893b383e3a940efb5f9f13f8249`. R1 corrects transport and reuse contracts; adds storage/worker/browser resource defaults and human-review ownership; explicitly permits the intermediate US shadow preview. Existing 41 acceptance-case descriptions are unchanged. Additional storage, queue/isolation, direct-test-tag and staged-delivery tests are specified in the paired plan. Approval of the original artifact is not approval of this R1 document.

R1 artifact SHA-256: `a5f407209777c6e48b9a58311b090aafebec418ad370184455cdd2182ed526d9`.

## R2 review-change record

R2 applies the second implementer review, which the user accepted together with the R1 proposed defaults (5 GiB store, 1 GiB free-space floor, 30-day/24-hour GC, 1 CPU / 2 GiB research worker, browser off by default):

1. **US CIK sourcing (§4.2, §9.2, §18).** CIKs come from SEC's official ticker-to-CIK file, confirmed against the CIK's submissions record, both retained as captured evidence. A single unambiguous, non-conflicting match may be auto-accepted as a registry-resolved single-listing link; every ambiguity, conflict, ticker change or cross-listing goes to administrator review.
2. **One research-job route scheme (§16, §18).** `POST /research-requests`, `GET /research-jobs/{job_id}`, and the new `GET /research-jobs/{job_id}/preview`.
3. **Dispatch classification and uncertain expiry (§10.2).** Connect-phase failures are pre-dispatch and release their reservation; uncertain reservations are charged to their dispatch period and expire with it.
4. **Egress broker has no Redis (§9.8, §10.3).** The broker is not on any network that reaches Redis or other app services; the research worker performs pacing and grant accounting through a grant-scoped Unix-socket RPC. The Redis ACL user, rate-control network and `redis-acl.sh` from R1 are removed.

The 41 acceptance-case descriptions are unchanged. The evidence, materiality, membership and publication standards are unchanged.

### Additional repository and deployment references (R1)

[R10] Pinned preparation modules in `backend/app/services/theme_evaluation/`: `image_preparation.py`, `kimi_translation.py`, `multilingual_preparation.py`, `multilingual_v2.py`, `translation_normalization.py`, `translation_quality.py`, `quantity_display.py`, `quantity_review.py`, at commit `28c220e4e4ca5afcb5a380678bb2b80dd7f388b7`. These are reuse inputs, not evidence that full research document coverage is already implemented.

[R11] At the same commit: `backend/app/services/rate_budget_policy.py`, `rate_limiter.py`, `backend/start_celery.sh`, `docker-compose.yml`, `docker-compose.prod.yml`, and `CLAUDE.md`. The inspected general worker is resource-capped; R1 deliberately allocates a separate research queue.

[R12] `backend/app/config/settings.py` (configuration definitions for Tavily and Serper) and corresponding environment documentation. Configuration symbols do not prove configured operator credentials or a deployed adapter.

[W8] Tavily Search API: https://docs.tavily.com/documentation/api-reference/endpoint/search — use bounded search results, with generated answers and raw-content retrieval disabled; account price/allowance is not assumed.

[W9] Playwright Python Docker guidance: https://playwright.dev/python/docs/docker — packaged images alone are not an adequate untrusted-site deployment; non-root sandboxing and the additional network boundary are mandatory here.

[W10] Docker Compose networks: https://docs.docker.com/reference/compose-file/networks/ — internal networks are one layer; R1 also requires explicit egress policy and negative tests, not network naming as proof of isolation.
