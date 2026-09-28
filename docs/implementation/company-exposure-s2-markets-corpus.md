# S2: HK, JP and TW routes, browser and vision acquisition, adjudicated corpus

**Parent:** [company-exposure-remaining-work.md](company-exposure-remaining-work.md)

| Source | Location |
|---|---|
| Spec | §9.1–9.8, §17.4, §18 |
| Plan tasks | 08 (08A/08B), 10, 11, 12, 25 (25A/25B), 27 (probes) |
| Plan appendices | F.2–F.4 |

## Goal

Extend verify-only research from the US to all four launch markets:

- Traditional Chinese, Simplified Chinese and Japanese stay in the original language, and English appears only as a derivative;
- selected-page vision and an optional isolated browser cover the documents that text extraction cannot;
- the human-adjudicated corpus that gates automatic admission gets built.

S2 changes no membership, publication or grounding. Every capability it adds is used in `shadow` mode (activation step 2).

## Design summary

### Market adapters (spec §9.2)

Every adapter implements the S1 protocol `MarketDocumentAdapter` in `services/company_exposure/markets/base.py`:

- `discover(issuer, query, limits, budget) -> DiscoveryResult`;
- `resolve_target(raw_metadata) -> DocumentTarget`;
- `fetch(target, budget) -> CaptureResult`.

Adapters share Task 06's network policy, storage reservations and the rate gate (`RateBudgetPolicy` provider keys). They also share Task 07's passage preparation.

A missing route is a typed `CoverageItem` gap, never a conclusion about exposure (I06). An adapter that returns only "unavailable" does not count as coverage.

| Market | Evidence routes | Identity and metadata | Hard limits |
|---|---|---|---|
| HK | Verified issuer IR and product links; direct official HKEXnews disclosure links | Existing HK market identity; original security and issuer reference; publication locale | Title-search or listing-page parsing only after its access and retention policy is documented. Otherwise report `permission_unavailable` for route `hkex_title_search`. No undisclosed backend API and no form bypass. |
| JP | EDINET API v2 (metadata plus body) when a key is configured; public issuer, JPX and TDnet links | EDINET code, docID, period, and amendment or withdrawal provenance | No key means `unavailable_capability` for EDINET only, with no paid-search fallback. Never a paid historical service. An English summary is not a complete substitute. |
| TW | Verified issuer report links; permitted MOPS documents | TWSE OpenAPI for identity and metadata assistance only; TWSE vs TPEx distinction through the existing security authority | ROC calendar is preserved with a normalization policy and conversion provenance. No invented generic MOPS API. Blanket MOPS enumeration is a gap unless permitted. |

Rules that apply to every market:

- **Document identity.**
  - Exchange-hosted and issuer-hosted copies of one disclosure share one `origin_disclosure_id` only when identity is established (E13).
  - An unresolved duplicate is annotated as such; it is never an independent confirmation.
  - Bilingual versions share the disclosure but keep separate languages.
  - A mismatch between official language versions is a conflict to review.
- **Dates.** Publication, reporting period, first availability and retrieval time stay separate (E12, E14).
  - `roc_year_to_gregorian(year) = year + 1911` runs only after the source calendar is identified as ROC.
  - The raw date, parsed date and policy are kept together.
- **Units.** 萬/万 and 億/亿 magnitudes, currency and calendar units are normalized with provenance. Normalization never makes a number theme-specific (E04).
- **Access and credentials.** Respect terms and retention rights.
  - EDINET authentication is a fixed official API credential, reached through a reference and redacted from logs and citations.
  - Authenticated redirects to other origins are never followed.
  - Validate the body type as well as the HTTP status: an API error inside a 200 response is not a PDF.
  - Archives open only in a sandbox with entry-count, path and decompression limits.

### Language and vision derivatives (Task 08A; spec §9.4–9.5)

- **Reuse the existing code; there is no second language engine.** Reuse:
  - `OpenCodeGoTranslator` (`translation-v3`) and `OpenCodeGoVision` (`image-v1`);
  - `multilingual_preparation` and `multilingual_v2`;
  - the translation normalization and quality modules;
  - the quantity display and review modules.
- **Inject the JSON client.** Add an optional `json_client` constructor parameter:
  - the default `None` keeps current behavior;
  - research passes the reservation-aware `ResearchJSONClient` from Task 04, so every call reserves allowance and dispatches once;
  - prompts, quantity checks and policy versions do not change, except through a policy revision with its own regression test.
- **Interfaces:**
  - `translate_passage(PassageRef, target="en", policy) -> PassageDerivative`;
  - `interpret_page(PageImageRef, question, ticket) -> PassageDerivative`.
- **Derivatives never carry primary authority.** Every derivative records `original_document_revision_id`, the original hash, the provider attempt and `evidence_role="derivative_not_independent_source"`. The claim schema must refuse to promote a derivative to a primary leaf.
- **Translation preserves the source.** Translate only relevant passages. Keep actor, negation, modality, units, dates, ranges and role direction. Ambiguity holds only the affected claim (E11).
- **Vision limits.**
  - At most 8 rasterized pages per issuer.
  - Image table cells need verified row, column, period and unit associations.
  - Chart readings stay approximate and review-only.
  - No OCR dependency is added.
  - Missing vision is `unavailable_capability` with no dispatch (R02).

### Isolated browser (Task 08B; spec §9.8; plan Appendix F.2–F.4)

The browser is optional and off by default (`EXPOSURE_RENDERING_ENABLED=false`). Leaving it off must never block HTML, text-PDF or 08A work.

- **Renderer.** `stockscreener/exposure-renderer:r1`:
  - runs Playwright Chromium as uid 10001 with its sandbox on and a reviewed seccomp profile;
  - uses `network_mode: none`, a read-only root, `cap_drop: ALL` and `pids_limit` 128;
  - has no secrets, env_file, data volume or API keys.
- **Egress broker.** `stockscreener/exposure-egress:r1`:
  - is not an open proxy; it only handles authorized fetch RPCs;
  - is attached only to `exposure_public_egress`;
  - has no Redis, PostgreSQL or application network and no credentials other than the signing-key file;
  - checks the destination on every hop, rejects private, link-local and metadata addresses, pins the checked IP and keeps TLS hostname validation;
  - strips auth and cookie headers.
- **Three single-direction Unix sockets:**
  - `render.sock`: worker → renderer;
  - `egress.sock`: renderer → broker;
  - `pacing.sock`: broker → worker.
- **Pacing.** The worker's `RenderPacingSession` (`render_pacing.py`) handles pacing on `pacing.sock`:
  - it serves only one grant, for the duration of one render call;
  - `acquire(grant_id, nonce, host, method)` verifies the signature, expiry, nonce replay and approved host;
  - it charges the durable root budget and takes the shared `ResearchRateGate` key in strict distributed mode;
  - it returns a single-use ticket, and `report(ticket, bytes, status)` settles it;
  - unreported tickets become uncertain;
  - without the pacing socket, the broker refuses every fetch.
- **Limits.**
  - At most 4 navigations per issuer.
  - Per navigation: 128 requests, 25 MiB and 60 s.
  - No service workers, websockets, downloads, popups, logins, forms, CAPTCHA or paywall workarounds.
- **Enablement gate.** All nine probes in F.4 must pass on the actual merged Compose configuration. They include:
  - the broker cannot reach `redis:6379`, `postgres:5432`, `backend:8000` or the host gateway and published ports (a host `DOCKER-USER` rule is required);
  - replay and cross-grant use are refused.

  If the host cannot meet this, record the residual risk and leave rendering off.

### Adjudicated corpus (Task 25; spec §17.4)

- **Size.**
  - At least 80 issuer–theme cases, with at least 20 per market.
  - They cover five theme families: memory, cybersecurity, refining, tankers and copper.
  - They include negative, ambiguous, stale, synthesis, image/table and cross-listing cases.
- **Splits.** Split by issuer and by originating disclosure; cross-listing and mirror reuse join groups. A feasible split is 16 development and 64 held-out cases, with at least 40 auto-eligible held-out cases.
- **Adjudication.** Two humans sign off every auto-eligible and critical-negative case:
  - disputes keep both labels plus the owner's final decision;
  - freeze splits before tuning;
  - record who signed off, when, and the source hashes.
- **Automatic admission gate.** All of these must hold:
  - 0 observed critical violations in the held-out corpus;
  - at least 40 eligible held-out cases;
  - at least 80% recovery;
  - every mechanical E/I/R case passes.

  A failing scope becomes review-only. Report denominators and uncertainty, not a population-accuracy claim.
- **The evaluator runs the real pipeline with recorded provider responses.** Labels never feed outputs. A test that returns a deliberately wrong claim must make the evaluator fail.

## Implementation plan

The market tasks (10, 11, 12) can run in parallel after S1. 08A and 08B can too.

### Task 10: Hong Kong adapter

- **Create:**
  - `backend/app/services/company_exposure/markets/hk.py` (`HKDocumentAdapter`);
  - `tests/unit/company_exposure/test_market_hk.py`;
  - fixtures `tests/fixtures/company_exposure/documents/hk/` and `routes/hk.json`.
- **Register:** add `HK` to `INSTALLED_MARKETS` and the adapter registry, but only once a working permitted route exists.
- **Tests:**
  - E13: bilingual versions share one `origin_disclosure_id` with different languages;
  - I06: a missing title-search permission yields `permission_unavailable` and `exposure_conclusion is None`;
  - an annual report, an announcement vs a prospectus, Traditional Chinese units and negation, and a missing or contradictory bilingual counterpart.
- **Record:** the implemented modes, the permission requirements and the fixture hashes.
- **Excerpt-only UI** if full-original retention is not approved.

### Task 11: Japan adapter

- **Create:**
  - `markets/jp.py` (`JPDocumentAdapter` plus an EDINET sub-route);
  - `test_market_jp.py`;
  - `documents/jp/` and `routes/jp.json`.
- **Configuration:** EDINET settings stay separate from paid-search settings, e.g. an `edinet_key_ref`.
- **Pin the official contract first.** Before merging, pin the EDINET v2 request, response, error, withdrawal and authentication contract from the FSA specification into a source-verified contract fixture. Don't use third-party mirrors.
- **Tests:**
  - R01: a missing key yields `route_status["edinet"] == "unavailable_capability"`, `"issuer_ir"` is still attempted, and search is never called;
  - E11: `original_title` and the docID are preserved;
  - qualification vs shipment wording, segment vs consolidated scope, and amendment or withdrawal;
  - an English derivative that doesn't match.
- **Keyless route:** at least one keyless issuer or official route must work even without EDINET.

### Task 12: Taiwan adapter

- **Create:**
  - `markets/tw.py` (`TWDocumentAdapter`);
  - `test_market_tw.py`;
  - `documents/tw/` and `routes/tw.json`, with the TWSE OpenAPI endpoint and schema pinned from the official catalog.
- **Tests:**
  - E11: the raw calendar is kept plus the normalization policy, with `zh-Hant`;
  - E04: a segment table keeps its row scope (`row_label == "Server segment"`);
  - TWSE vs TPEx identity, footnoted tables, commercial-stage modal language, a new period and a stale copied capture.
- **Don't** change `StockUniverse` identity authority.

### Task 08A: language and vision adapters

- **Create:**
  - `services/company_exposure/preparation_adapters.py`;
  - `test_preparation_adapters.py`.
- **Extend:** add the `json_client` parameter to the `theme_evaluation` translator and vision classes, keeping their existing suites green.
- **Tests:**
  - E11: Japanese negation (量産出荷は開始していない) survives into the derivative, with `provider_attempt_id` set;
  - R02: missing vision means no dispatch;
  - derivative evidence roles;
  - Traditional and Simplified magnitude units;
  - supplier/customer reversal.
- **E11 frontend:** add a companion Vitest for the original/English toggle.

### Task 08B: isolated renderer (optional)

- **Create:**
  - `services/company_exposure/rendering.py` (`PublicRenderer.render(DocumentTarget, RenderPolicy) -> RenderCapture`);
  - `render_pacing.py`;
  - `ops/exposure-renderer/{Dockerfile,server.py,requirements.lock,seccomp.json}`;
  - `ops/exposure-egress/{Dockerfile,server.py,requirements.lock}`;
  - `docker-compose.exposure-browser.yml`;
  - `ops/exposure-runtime.lock.json`, recording digests and versions.
- **Tests:**
  - `test_rendering_contract.py`: R05, pacing charges the root budget and refuses other grants and replays;
  - `tests/integration/company_exposure/test_renderer_security.py`: R14 at the deployment layer, with no native network, a blocked private subresource, and a broker that cannot reach application services.
- **Before enabling:** run license and vulnerability checks on the new locks.

### Task 25A/25B: corpus and evaluator

- **Create:**
  - `services/company_exposure/evaluation.py`, with `evaluate_corpus(manifest, runner, policies) -> EvaluationReport` and `admission_gate(critical_errors, eligible_total, recovered)`;
  - `backend/scripts/evaluate_company_exposure.py`, using `--mode recorded` and exiting nonzero on any gap;
  - `tests/fixtures/company_exposure/corpus/`, holding the manifest, schema and adjudication records;
  - tests `test_evaluation_gate.py`, `test_fixture_integrity.py` and `test_case_tags.py`.
- **Case mappings:** complete all 41 case/layer mappings from plan Appendix D.
- **Collect cases during the market tasks, not at the end.** Implementers propose annotation packets, and the owner assigns reviewers.

### Task 27 (S2 part): `probe-market`

- **Add the command:** `scripts/company_exposure.py probe-market --market {US,HK,JP,TW} --allow-network --max-documents 2`. It needs a dry-run fixture mode, and never makes model calls unless `--allow-subscription-calls` is given.
- **Record in the report:** the route policy, URL, issuer and document hashes, period, locator, parser language, and response. HTTP 200 alone is not a pass.
- **What launch needs:** one working official or issuer route per market. An untested EDINET route is reported as untested.

## Gate for the slice

Add an `S2` entry to `backend/tests/required_company_exposure_cases.json`:

- E11 at the unit layer (market fixtures), plus the frontend companion;
- E13 and I06 for each new adapter;
- R14 at the deployment layer, only when 08B is installed.

Keep every S1 requirement.

The slice is complete when all of the following hold:

- the gate passes on PostgreSQL;
- probe reports exist for every market an operator has credentials and egress for;
- corpus collection has started in every market.

The corpus admission decision (25B) can finish later, but S3 step 4 (automatic additions) cannot start without it.

## Stop conditions and things not to do

- **Access and permissions.**
  - Never bypass login, paywalls, CAPTCHA or form restrictions.
  - Never automate an undocumented endpoint.
  - Never switch to paid search when a market route fails.
  - Never subscribe to a paid historical feed.
- **Language fidelity.** Never treat an English summary as complete, and never pick the convenient version of a bilingual mismatch.
- **Browser isolation.** Never weaken the sandbox. That means no `--no-sandbox`, no privileged container and no unconfined seccomp. If isolation fails, the result is `unavailable_capability`.
- **Honest reporting.**
  - Never label synthetic fixtures or model output as adjudicated corpus data.
  - Never describe the corpus as adjudicated until the signed records exist.
  - A mocked download is not a live probe.
