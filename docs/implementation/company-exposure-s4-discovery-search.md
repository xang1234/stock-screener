# S4: Discovery and optional Tavily search

**Parent:** [company-exposure-remaining-work.md](company-exposure-remaining-work.md)

| Source | Location |
|---|---|
| Spec | §8.1–8.3 (triggers, lifecycle, limits), §9.1, §10 (allowance and search cost), §18 step 5 |
| Plan tasks | 13 (Tavily), 17B (discovery), 23B (schedules), 21C/22B (candidate queue and settings), 27B step 5 |

## Goal

Allow bounded candidate discovery for themes an administrator has explicitly enabled. Discovery uses the same verification path as S1, and every child shares one root budget.

Add Tavily Search as an optional, off-by-default source of leads. It can suggest links to primary documents but never verifies anything. Operating with paid search disabled is a supported mode, not a degraded one.

Task 13 depends only on Tasks 04, 06 and 07, so it can be built at any time. Task 17B needs 17A plus at least one installed market adapter. It does not need every adapter. Enabling discovery is activation step 5, after S3.

## Design summary

### Discovery scope (spec §8; plan Task 17)

- **When discovery may run.**
  - An authenticated explicit request, or an enabled-theme policy.
  - Discovery is off per theme by default.
  - A `discover` request for a disabled theme returns `theme_discovery_disabled`, with no candidates and no provider call (R15).
- **Where candidates come from:**
  - admitted source leads (imported as unverified leads, never marked verified);
  - official product and customer links;
  - retained documents;
  - optional search.

  Each candidate keeps its retrieval reason and original lead.
- **Traversal limits.**
  - No recursive supplier or customer graph walk.
  - No ETF-list inference.
  - No enqueueing the whole universe.
  - Cross-listings reuse one issuer investigation.
- **Per-root limits:**
  - at most 10 new issuers;
  - at most 6 search attempts, where retries count as new attempts;
  - stop early at any budget or coverage bound.
- **Children and budgets.**
  - Children are ordinary verify stages under the same root reservation. Ten candidates never get ten independent budgets.
  - Pause, resume, a model change or a child job cannot reset cumulative limits (R05).
- **Priority.** Material conflict and requested verification come first (0), then new membership (1), then due refresh (2), then enabled-theme expansion (3). Fairness applies between queued roots.
- **What discovery produces.**
  - Discovery only produces candidates; membership still goes through S3's gate.
  - Jobs never write the live pointer.
- **Cancel and pause** release only reservations that were never dispatched. A sent provider call keeps its accounting and can finish for audit, but cannot publish automatic use after cancellation.

### Optional Tavily adapter (spec §9.1, §10; plan Task 13)

- **Interface:** `SearchAdapter.search(query: SearchQuery, ticket: ReservationTicket | None) -> SearchResult`.
  - Implementations are `DisabledSearchAdapter` and `TavilySearchAdapter`, chosen by the factory `configured_search_adapter(config, resources)`.
  - `ticket=None` is valid only for the disabled adapter. Enabled adapters reject it before any HTTP call.
- **Default.** The provider is `none`. Enabling needs all of the following, none of which is implied by a key being present (R01):
  - an explicit operator enable;
  - a selected provider;
  - a safe credential reference;
  - an account currency;
  - daily and monthly ceilings;
  - a maximum-charge costing policy.

  A missing cap returns `search_cap_required`, and a disabled flag returns `paid_search_disabled`. Both make zero transport calls.
- **HTTP contract** (fixed destination, pinned in `routes/search_tavily.json`): `POST https://api.tavily.com/search` with bearer auth.
  - Parameters are fixed: `search_depth="basic"`, `auto_parameters=False`, `include_answer=False`, `include_raw_content=False`, `include_images=False`, `include_usage=True`, `max_results<=20`.
  - Topic, domain and date are explicit frozen query inputs.
  - Queries never fan out to crawl, extract or research endpoints.
- **Output is a lead only.**
  - `source_kind="search_snippet"` and `verification_eligible=False` (E10).
  - The originals are fetched through Task 06.
  - Result dates are discovery metadata only.
  - Two hits on the same upstream source are one lead, not two corroborations.
- **Cost.**
  - Before dispatch, reserve the provider-defined worst permitted charge.
  - Reconcile against actual billing evidence when it exists. Otherwise keep the amount reserved or unknown.
  - An unknown maximum cost blocks dispatch.
  - Uncertain timeouts keep their reservation (R04).
  - No hardcoded price, and no assumed free tier.
- **No fallbacks.** No automatic Serper or other provider, and no paid fallback when HK, JP or TW routes fail. A credential error never authorizes another provider.
- **Logging.** Never log secrets or credentialed URLs. Keep the response and usage credits with the request ID.

### Schedules (plan Task 23B)

- `discover_enabled_theme_candidates` runs weekly or on demand, only for themes with discovery enabled, on the `exposure_research` queue.
- A timer firing with no due work issues no calls.
- Existing source and Social schedules are unchanged.

## Implementation plan

### Task 13: search adapter

- **Create:**
  - `backend/app/services/company_exposure/search.py`;
  - `tests/fixtures/company_exposure/routes/search_tavily.json` (contract, no secrets);
  - tests `tests/unit/company_exposure/test_search.py` and `test_search_costs.py`.
- **Reinstate the deferred paid-search and dispatch predicates** (removed from S1) as this adapter's reservation checks.
- **Extend** the versioned research configuration and usage records: provider, enable flag, caps, currency and costing policy, stored as `ExposureRuntimePolicyRevision`.
- **Tests:**
  - R01, parametrized over (enabled=False, cap set) and (enabled=True, no cap), each making zero HTTP calls;
  - E10: lead only;
  - missing key, unknown cost and an exhausted ledger, each making zero calls;
  - pagination cap, timeout uncertainty, retry charging and malformed results.
- **Account state:** the provider account stays unprovisioned and off after the task.

### Task 17B: discovery workflow

- **Extend `services/company_exposure/research.py`** with `ExposureResearchCoordinator.discover_candidates(theme_id, policy, budget) -> CandidateBatch`, using the same work state machine and `ResearchCandidate` rows.
- **Lift the refusal.** Remove the `discovery_not_installed` refusal in `research_requests.py`, but only once this path exists and is gated by the theme policy.
- **Reinstate** job pause and cancel for child jobs, if S3 has not already done so.
- **Tests:**
  - `test_discovery_scope.py`: R15, the disabled-theme case;
  - `test_research_workflow.py`: children share the root budget, and the 10-issuer and 6-search caps;
  - `tests/integration/company_exposure/test_research_jobs_postgres.py`: root counters across children, retries and models.
- **Full offline path:** discovery → child verify, with paid search disabled.
- **CLI:** add `discover`.

### Task 23B: schedules

- **Beat entry.** Add `discover_enabled_theme_candidates` to beat, with a message `expires` so it drops harmlessly when the worker is absent (same pattern as the S1 entries).
- **Tests** (in `test_tasks.py`, `test_triggers.py` and `test_disabled_mode.py`):
  - the task runs only for enabled themes;
  - a disabled mode is skipped;
  - fairness between roots.

### Task 21C/22B: candidate queue and paid-search settings UI

- **Candidate queue.** Show unmet gates, job progress and limits.
- **Research controls** respect the capabilities the server says are allowed.
- **Allowance display.** Show subscription usage as a local allocation. When the provider does not report a remaining balance, show it as unknown.
- **Paid search** has a separate authenticated enable-and-cap workflow. The UI never sends an enable flag just because a key exists.
- **Endpoints:** `PUT /admin/theme-research-policy/{theme_id}` and `GET/PUT /admin/settings`, which are shared with S3.

## Gate for the slice

Add `S4` to `required_company_exposure_cases.json`:

- R01 at the unit and deployment layers, for search;
- R04 and R05 at the unit and Postgres layers, for search reservations and root budgets across children;
- R15 at the unit layer, for the disabled theme;
- E10 at the unit layer, for search snippets.

Add a named invariant: concurrent discovery roots cannot overbook the shared allowance.

For activation step 5, the operator enables discovery per selected theme. Paid search stays `false` unless the operator separately supplies the provider, key reference, caps and costing policy.

## Stop conditions and things not to do

- **Enabling search.** Never enable Tavily because `TAVILY_API_KEY` is set, and never treat Serper settings as a fallback.
- **Search results are leads.** Never store a search result as an `ExposurePassage` original or a primary leaf.
- **No unbounded research.** Never run an always-on autonomous agent or recursive tool loop. Every stage has a finite target, attempt and page set.
- **Budgets.** Never reset budgets for children, retries or model changes. Never refund an uncertain dispatched call.
- **Costs.** Never hardcode a price or assume a free tier. An unknown maximum cost stops dispatch.
