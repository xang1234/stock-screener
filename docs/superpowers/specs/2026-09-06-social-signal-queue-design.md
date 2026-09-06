# Social Signal Queue Design

**Date:** 2026-09-06

**Status:** Approved in design review

**Source intent:** Implement the Minervini X-list research and Social Signal
Queue mockup as a live-only, provider-neutral feature. The July 2026 research
brief and SVG supplied during design review update the checked-in June research
artifacts at `docs/research/minervini_x_feature_report_2026-06-20.md` and
`docs/research/minervini_social_signal_queue_mockup.svg`.

## Context

The application already has most of the confirmation system this feature
needs: content sources and items, technical and fundamental theme extraction,
canonical theme identity, SecurityMaster normalization, multi-Market scan
features, Setup Engine evidence, RS ratings, group rankings, liquidity,
Market-health exposure, watchlists, and stock-detail views.

The missing product is a ticker-first translation layer over selected X lists.
It must answer: "Which socially discussed securities are also technically and
contextually worth reviewing?" It must not become a raw X feed, an AI stock
ranker, an automated trading system, or a replacement for the existing Themes
page.

The existing X provider seam is useful but the private adapter is not usable as
written. `PrivateXUIFetcher` expects `xui.read_source`; the current proprietary
package exposes `xui_reader` and documents its `xui ... --json` CLI as the
stable consumer boundary for scheduled reads. The implementation will replace
that assumption with a provider-neutral record contract and a CLI-based private
adapter.

X's current rules prohibit non-API website automation and warn that it can lead
to account suspension. A six-hour schedule lowers request volume but does not
make UI automation compliant or risk-free. The deployment therefore treats
restriction or suspension as an explicitly accepted operator risk, stays
read-only, and fails closed on authentication or challenge signals. The
official X API remains the compliant provider option.

## Goals

- Combine two installation-wide X lists into one ranked research queue:
  - `1522014550211457024`
  - `1986290701492232693`
- Support resolved securities in the US, HK, CN, JP, and TW Markets.
- Rank candidates with a deterministic, versioned 60% social / 40%
  confirmation formula.
- Offer a Pure Social ranking mode without removing confirmation evidence from
  the display.
- Reuse existing ingestion, theme identity, SecurityMaster, scan-feature,
  Market-health, stock-detail, and watchlist capabilities.
- Preserve X-derived theme evidence separately from existing Theme rankings.
- Support `disabled`, official X API, and proprietary `xui` provider modes
  without changing downstream contracts.
- Keep the public application buildable, testable, and runnable without the
  proprietary package or any X credential.
- Run six-hour collection safely in local Docker and offer a restricted admin
  refresh path.
- Keep all social data out of static-site bundles.

## Non-goals

- User-configured X sources or user-specific queues.
- Static-site Social Signals.
- Embedded X images, video, or article bodies.
- Posting, liking, following, bookmarking, direct messaging, or any other X
  mutation.
- Notifications, trade alerts, or automatic order execution.
- AI-generated candidate scores or opaque AI ranking judgments.
- Treating social options-flow claims as verified market data.
- Historical strategy backtesting.
- Silently falling back between providers.
- Making social evidence alter existing Theme rank or momentum calculations.

## Domain language

Implementation will add the following terms to `CONTEXT.md`:

**Social Source**: One administrator-controlled X list supplying read-only
evidence to the shared Social Signal Queue. A Social Source is represented by a
`ContentSource` but does not contribute to existing Theme rankings.

**Social Post**: One normalized X post stored as the canonical `ContentItem`,
with separate source-membership, engagement, and ticker-resolution records.

**Social Signal Run**: One immutable attempt to read every configured Social
Source, normalize new observations, resolve securities, and calculate a
versioned queue snapshot.

**Published Social Signal Run**: The latest complete Social Signal Run selected
for live reads. Partial and failed attempts cannot replace it.

**Social Score**: A deterministic 0-100 measure derived only from mention
acceleration, independent authors, engagement, recency, and cross-list
confirmation.

**Confirmation Score**: A deterministic 0-100 measure derived from existing
Setup Engine, RS, group-strength, and non-social theme-price evidence.

**Queue Score**: The versioned `0.60 * Social Score + 0.40 * Confirmation Score`
used by the default Blended ranking mode.

**Signal State**: The deterministic presentation category `actionable`,
`watch`, `risk_off`, `context`, or `unresolved` assigned independently of the
numeric Social Score.

## Architectural approaches considered

### 1. Separate ingestion worker — selected

A dedicated Celery worker consumes only the `social_ingestion` queue. It shares
PostgreSQL and Redis with the public application but the web process never
loads or exposes `xui-reader`. The public repository owns all provider-neutral
models, use cases, APIs, UI, and tests. A private image adds the proprietary
package and Playwright runtime.

This provides the strongest public/private boundary while preserving the
existing application pipeline.

### 2. Private replacement backend image — rejected

The entire backend could be rebuilt with `xui-reader` installed. This reduces
the number of container types but unnecessarily exposes X session material to
the API process and creates avoidable drift between public and private backend
deployments.

### 3. External feeder API — rejected

An independent reader could POST normalized items to a new authenticated API.
This isolates the reader but adds another credential, network-facing write
endpoint, retry protocol, and attack surface without a current cross-host
requirement.

## Deployment and provider modes

`SOCIAL_INGEST_PROVIDER` has exactly three values and defaults to `disabled`:

| Mode | Reader | Required secret | Image |
| --- | --- | --- | --- |
| `disabled` | None | None | Public application images |
| `official` | X API v2 | Developer-app token and prepaid API credits | Public social worker image |
| `xui` | `xui read --json` | Local authenticated automation profile | Private social worker image |

The application never changes provider automatically. In particular, an `xui`
failure cannot trigger billable official API requests.

The official adapter reads `GET /2/lists/:id/tweets` and requests author,
timestamp, and available public-engagement fields. It observes configured
per-run and daily resource ceilings and X rate-limit reset headers.

The private adapter invokes the installed `xui` executable directly without a
shell, uses the dedicated `automation` profile with non-interactive login
policy, and parses only validated JSON output. The application adapter does not
import `xui_reader` modules.

### Local Docker

Normal local use pulls the private multi-architecture worker image from GHCR.
The operator authenticates Docker to GHCR with package-read-only access. The
image supports `linux/arm64` for Apple Silicon and `linux/amd64` for common
Docker hosts.

GHCR is not required for development. A documented Compose override can build
the same worker locally from a checked-out private repository or with a
BuildKit SSH secret. Neither repository credentials nor package source are
copied into the public repository, Docker build context, image configuration,
or image layers.

The X automation profile is created by a human login on the Mac and mounted
from a dedicated, access-restricted Docker volume. It is never built into the
image. The volume is writable only by the non-root worker so legitimate session
state can be refreshed; no other service mounts it.

### Scheduling

Celery Beat dispatches a social refresh every six hours only when a provider is
enabled. A provider-specific worker consumes the `social_ingestion` queue. A
singleton Redis lease prevents overlapping Social Signal Runs, and stale queued
refresh messages collapse into one current run instead of replaying a backlog.

An authenticated administrator may request a manual refresh no more than once
per hour. A manual request uses the same queue, lease, limits, and publication
policy as a scheduled run.

## Provider-neutral record contract

Both readers map provider output to the same typed record before persistence:

- provider post ID;
- text and canonical source URL;
- author handle when available;
- creation time;
- likes, reposts, replies, quotes, bookmarks, and views when available;
- media count and quote/repost flags, without downloaded media;
- source-quality tier and score when available;
- Social Source ID;
- provider observation time.

Fields distinguish missing from observed zero. Provider-specific data that is
not in the contract cannot leak into scoring. Invalid records are rejected with
stable reason codes. X content is untrusted input: it is stored, classified,
and shown as a short excerpt, but cannot be executed or interpreted
as operator instruction.

## Sources and collection policy

The two approved list URLs are seeded as shared Social Sources assigned to the
existing technical extraction pipeline. They are not editable by ordinary
users. Administrators may enable or disable a source but v1 does not allow
adding arbitrary X lists through the Social Signals UI.

Disabling either approved source prevents new publication and marks collection
partial; it does not turn a one-list result into a complete run. This makes
source maintenance reversible without changing the agreed two-source product.

The first successful run attempts a fourteen-day backfill capped at 1,000 posts
per list. A cap-truncated source is incomplete and cannot silently claim full
coverage. Later runs use provider checkpoints and overlapping incremental
reads, retaining overlap so recent engagement observations can be refreshed.

The scorer uses 1-day, 7-day, and 14-day windows based on post publication time
in UTC. Market-session dates are used only for joining existing Market data;
social windows are continuous clock windows because X posts are not limited to
trading sessions.

## Persistence model

### Existing records

`ContentSource` remains the configured-source record. It gains an explicit
`contributes_to_theme_rankings` boolean, defaulting to true. The two
Social Sources set it to false. Existing source behavior therefore remains
unchanged unless explicitly opted out.

`ContentItem` remains the canonical post record and retains the existing
`(source_type, external_id)` uniqueness behavior. Social extensions do not add
engagement JSON to this table.

`ThemeMention` remains the canonical extracted-theme evidence. Technical theme
extraction processes Social Posts and resolves the same pipeline-scoped
`ThemeCluster` identities, but Theme metric calculations must exclude mentions
whose source does not contribute to Theme rankings. Queue theme confirmation
uses price/RS/breadth fields, not mention velocity or a momentum score that
mixes social evidence back into confirmation.

### New records

**SocialPostSource** is the many-to-many membership between a ContentItem and a
Social Source. It records first-seen and last-seen times and has a unique key on
`(content_item_id, content_source_id)`. This preserves both list memberships
after cross-list post deduplication.

**SocialContentMetrics** is the current engagement observation for one Social
Post. It stores nullable provider metrics, source-quality fields,
`observed_at`, and the provider name. A newer valid observation replaces the
current values; missing fields never erase a previously observed value.

**SocialPostTicker** stores each distinct post/ticker association. It includes
the raw token, canonical symbol when resolved, Market, MIC, local code,
resolution method, resolver-policy version, status, and reason. It references
the active `StockUniverse` row when one exists but retains unresolved evidence
without fabricating a security.

**SocialSignalRun** records immutable run identity, provider, formula version,
start/end times, per-source outcome and coverage, counts, failure reason, and
publication state.

**SocialSignalSnapshot** records the per-security inputs, component scores,
Queue Score, Signal State, evidence counts, freshness dates, and stable
tie-break fields for one Social Signal Run. Inputs needed to explain or replay
the result are pinned to the run rather than read from mutable latest rows.

A single published-run pointer identifies the latest Published Social Signal
Run atomically. Live APIs never assemble a queue from a mixture of different
runs.

## Security and listing resolution

Ticker resolution reuses the existing deterministic multi-Market ticker
validator, CJK alias resolver, SecurityMaster rules, and active Universe check.
Resolution follows this precedence:

1. An explicit canonical ticker or cashtag selects that listing.
2. A company-name-only mention selects its home-Market primary listing when a
   deterministic alias exists.
3. ADRs and alternate listings are returned as related instruments, not
   duplicate queue entries.
4. Multiple listings become separate queue entries only when the source
   evidence explicitly mentions each listing.
5. Ambiguous or unknown text remains unresolved and appears only in All
   Signals evidence.

Broad-Market ETFs such as SPY and QQQ become `context` evidence and cannot enter
the candidate ranking. Industry and thematic ETFs such as SMH and REMX remain
eligible and display an ETF badge.

## Scoring policy v1

Scoring is pure, deterministic, and versioned. No LLM participates in numeric
ranking. AI may perform existing theme and evidence classification, but the
stored source records and component inputs determine every numeric score.

### Anti-manipulation preprocessing

- A provider post ID contributes at most once per ticker.
- Duplicate reposts, quotes with no new thesis text, and repeated canonical URLs
  do not add independent mention weight.
- One author contributes at most three weighted posts per ticker in any rolling
  24-hour period.
- Invalid timestamps and future-dated observations do not enter a score.
- Engagement is log-scaled and winsorized before percentile conversion so a
  viral outlier cannot dominate the queue.
- Each component records its input coverage; missing is never converted to
  observed zero.

### Social Score

| Component | Queue weight | Definition |
| --- | ---: | --- |
| Mention acceleration | 20% | Percentile of `(mentions_1d + 1) / (mentions_previous_13d / 13 + 1)` |
| Unique authors | 15% | Percentile of capped distinct authors in the selected 1D, 7D, or 14D window |
| Engagement | 15% | Percentile of the summed, winsorized per-post engagement value defined below |
| Recency | 5% | Exponential decay from the latest qualifying mention, with a 48-hour half-life |
| Cross-list confirmation | 5% | Full component credit when both approved lists independently mention the ticker; zero when only one does |

The Social Score renormalizes these five Queue weights to 0-100. Component
percentiles are calculated within the selected Market. When fewer than 20
resolved candidates exist in a Market/window cohort, the scorer uses the global
supported-Market cohort and records `normalization_scope=global_fallback`.

Per-post engagement value is:

```text
log1p(
    likes
  + 2.0 * reposts
  + 1.5 * replies
  + 1.5 * quotes
  + 2.0 * bookmarks
  + 0.001 * views
)
```

The engagement component is unavailable unless likes, reposts, and replies are
all observed. Quotes, bookmarks, and views contribute only when observed. The
per-post value is winsorized at the 95th percentile of the normalization cohort
before ticker-level summation. Cross-list confirmation requires qualifying
posts with different provider post IDs in both approved lists; one duplicated
post present in both lists does not earn the bonus.

### Confirmation Score

| Component | Queue weight | Definition |
| --- | ---: | --- |
| Setup quality | 20% | Existing 0-100 `se_setup_score`; `se_setup_ready` controls Signal State rather than adding points |
| Relative strength | 10% | `0.40 * rs_rating_1m + 0.60 * rs_rating_3m` when both exist; available-input normalization otherwise |
| Group strength | 5% | Current group rank converted to a 0-100 percentile within its Market |
| Theme confirmation | 5% | Price/RS/breadth evidence from linked themes, explicitly excluding social mention metrics |

The Confirmation Score renormalizes these four Queue weights to 0-100. Setup
and current Market data are required for `actionable`; missing optional
components reduce coverage but are not treated as zero.

Group-strength conversion is
`100 * (1 - (group_rank - 1) / (market_group_count - 1))`, bounded to 0-100;
it is unavailable when fewer than two ranked groups exist. For each linked
theme, Theme confirmation is the arithmetic mean of available
`basket_rs_vs_spy`, `avg_rs_rating`, and `pct_above_50ma`, which are already
0-100 fields. The ticker receives the highest linked-theme value, with
canonical theme key ascending as the stable tie-breaker. No `mentions_*`,
`mention_velocity`, sentiment, or composite `momentum_score` value enters this
component.

### Queue Score and ordering

The Blended ranking uses:

```text
queue_score = 0.60 * social_score + 0.40 * confirmation_score
```

Stable ordering is Queue Score descending, Social Score descending, most recent
qualifying mention descending, then canonical symbol ascending. The Pure Social
mode orders by Social Score, recency, then canonical symbol and does not use
confirmation fields in ordering.

The ranking window selector supports 1D, 7D, and 14D. View and rank mode are
independent, so an operator may rank only Actionable candidates by Pure Social
strength or rank All Signals by the Blended score.

## Signal State policy

`actionable` requires all of the following:

- deterministic resolution to an active supported-Market security;
- non-stale Market and scan-feature inputs;
- the existing Market-specific liquidity eligibility gate;
- `se_setup_ready` equal to true; and
- a current Market exposure score greater than or equal to 50, covering Power
  Trend, Confirmed Uptrend, and Uptrend Under Pressure.

`watch` is a resolved candidate that lacks current setup readiness or another
required confirmation but remains research-worthy.

`risk_off` meets security-level confirmation but its Market posture currently
does not permit new exposure. Market posture changes Signal State, not Social or
Queue Score.

`context` is a broad-Market ETF or macro instrument intentionally excluded from
candidate ranking.

`unresolved` retains source evidence that could not be mapped deterministically.
It never receives fabricated Market or technical data.

The Actionable view includes only `actionable`. All Signals includes every
state with clear missing-data and exclusion reasons.

## Run and publication flow

1. Acquire the singleton social-ingestion lease.
2. Create an immutable Social Signal Run and pin provider, source set, source
   checkpoints, formula version, and current time.
3. Read each Social Source independently through the selected adapter.
4. Validate and normalize provider records.
5. Upsert ContentItems, source memberships, and newer engagement observations
   transactionally per source.
6. Seed existing technical extraction state and run theme/ticker extraction.
7. Resolve ticker identities using SecurityMaster and the active Universe.
8. Aggregate 1D/7D/14D evidence with anti-manipulation rules.
9. Join one coherent current feature snapshot per Market plus Market posture,
   group ranks, liquidity, and non-social theme confirmation.
10. Calculate and persist immutable Social Signal Snapshots.
11. Evaluate publication quality and atomically advance the published pointer
    only after both sources have complete valid outcomes.
12. Release the lease and expose the run outcome to Operations.

A partial or failed attempt is retained for diagnosis but cannot replace the
last Published Social Signal Run. If no published run exists, the API returns a
typed unavailable state rather than partial candidates.

## API and authorization

All Social Signal read endpoints require an authenticated app user. Read
responses expose only the Published Social Signal Run.

The provider-neutral live API supplies:

- queue summary and freshness;
- supported windows, views, ranking modes, filters, and formula version;
- paginated rows for a selected Market;
- per-row component explanations and Signal State reasons;
- top source evidence limited to short excerpts and canonical X links;
- source coverage and degraded-state metadata; and
- related listing and unresolved evidence where applicable.

Provider configuration, authentication health, run history, source enablement,
and manual refresh require administrator authorization. API responses and logs
must not reveal bearer tokens, filesystem paths to session state, cookies,
headers, raw provider debug payloads, or private package installation details.

Runtime capabilities expose whether Social Signals is supported and enabled.
Ordinary users do not see the Social tab when disabled. Administrators receive
an explicit disabled/setup state in Operations.

## User interface

The supplied SVG is adapted to the existing dense MUI theme rather than copied
as an isolated visual system.

### Daily Snapshot card

The compact card shows the top five Published Social Signal candidates for the
selected Market, dominant linked themes, current Market posture, two-source
coverage, last successful refresh time, and stale/degraded status. It links to
the full Social Signals tab.

### Social Signals tab

The existing Daily page's left navigation gains `Social Signals`. Controls
include:

- Actionable or All Signals view;
- Blended or Pure Social ordering;
- 1D, 7D, or 14D window;
- source list, theme, instrument type, and Signal State filters; and
- ticker search.

The table includes symbol, Market, Queue Score, Social Score, mention counts,
unique authors, Setup Engine score/readiness, RS, group rank, linked theme,
Signal State, and evidence freshness. The current global Market selector scopes
the queue; US volume does not crowd out Asian Markets.

Clicking a row opens a social evidence drawer first. The drawer explains every
score component and state decision and shows at most three supporting posts,
each with author, timestamp, short excerpt, engagement, source-list badge, and
`Open on X` link. It does not embed X media.

Drawer actions reuse existing application workflows:

- open the stock chart and Setup Engine view;
- start a Scan over the currently visible resolved symbols; and
- add the selected security to an existing watchlist.

### Themes and administration

The existing Themes page, rankings, and terminology remain intact. It shows
a separate `Social Pulse` field derived from the Published Social Signal Run,
but that field cannot affect Theme ordering or stored Theme momentum. Queue
theme links navigate into the existing Theme detail experience.

Provider health, last run, source coverage, reauthentication requirements,
rate-limit state, and the admin refresh control live in Operations/settings,
not in the ordinary user's queue controls.

Static mode does not render the Daily card, Social tab, social API calls, or
Social Pulse. Static exporters contain no social rows, excerpts, metrics, or
run metadata.

## Failure handling and observability

Health states are `disabled`, `healthy`, `partial`, `stale`,
`reauthentication_required`, `rate_limited`, and `provider_error`.

- Missing provider configuration yields `disabled` and schedules no reads.
- Missing or expired `xui` authentication yields
  `reauthentication_required`; the worker stops without interactive login.
- Authentication challenge, selector drift, or suspected blocking stops the
  run immediately and triggers a provider cooldown.
- A temporary network error receives at most one delayed retry.
- Official API 429 responses respect the returned reset time and do not switch
  provider.
- Invalid provider JSON fails that source with a stable schema reason.
- Partial source coverage cannot replace a published complete snapshot.
- A published snapshot older than seven hours is `stale`, while remaining
  readable as last-known-good data.
- Logs record run IDs, provider names, source IDs, counts, durations, and stable
  error codes. They redact session, token, cookie, response-header, and raw
  debug-payload material.

Existing Market Workload concepts remain authoritative for Market-data jobs.
Social Signal Run health is a separate source-collection lifecycle and cannot
claim that a Market itself is unhealthy.

## GitHub Actions and image security

Public pull-request CI runs backend unit/integration tests, frontend component
tests, and Playwright UI tests against fixtures. It neither installs the
proprietary package nor reads live X.

The private worker workflow runs only from trusted branches, release tags, or a
manual dispatch. It never runs with private-package credentials on forked or
untrusted pull requests.

Access to the separate private package repository uses a narrowly scoped
GitHub App token, deploy key, or fine-grained token delivered as a BuildKit
secret. Package credentials are not Docker `ARG` or `ENV` values and do not
survive in layers. GHCR publication uses repository-scoped package-write
authorization. The resulting package remains private.

The workflow builds and tests both `linux/arm64` and `linux/amd64` images. It
publishes immutable commit-SHA tags and an explicitly promoted release tag;
deployments do not rely solely on mutable `latest`.

Docker hosts pull with package-read-only credentials. Source-repository read
credentials are not needed to run a prebuilt image.

## Testing strategy

### Domain and service tests

- Exact component formula, normalization, missing-value, capping, decay, and
  tie-break tests.
- Deterministic replay: identical normalized inputs produce byte-equivalent
  scoring payloads for the same formula version.
- Cross-list deduplication retains both source memberships.
- One-author caps and repost/quote deduplication prevent inflated scores.
- Listing resolution covers explicit US/HK/CN/JP/TW symbols, company aliases,
  ADR relationships, ambiguity, and unresolved evidence.
- Broad-Market and thematic ETF classification follows the stated policy.
- Signal State tests cover stale data, liquidity, setup readiness, and
  Market-posture changes without score mutation.
- Theme metric regression tests prove that opted-out Social Sources cannot
  change existing Theme rankings.

### Provider contract tests

- Sanitized official and `xui` fixtures normalize to the same typed records.
- Missing engagement remains null while observed zero remains zero.
- CLI argument construction never invokes a shell and JSON validation rejects
  unknown or malformed structures.
- Provider caps, 429 handling, authentication failures, challenge signals, and
  non-fallback behavior are deterministic.
- Tests contain no session state, cookies, bearer tokens, or live X reads.

### Persistence and publication tests

- ContentItem deduplication, source membership, engagement update, ticker
  mapping, and run snapshots satisfy their uniqueness rules.
- Partial and failed runs cannot advance the published pointer.
- A complete run advances publication atomically.
- Concurrent or repeated Celery deliveries collapse under the singleton lease.
- API queries read one published run and do not mix mutable latest data.

### Frontend and end-to-end tests

- Disabled, loading, empty, healthy, partial, stale, and reauthentication
  states render correctly by role.
- View and ranking controls remain independent.
- Pure Social sorting ignores confirmation fields.
- Evidence drawers show no more than three excerpts and open canonical links.
- Existing chart/setup, scan-symbol, and watchlist actions receive canonical
  Market-aware symbols.
- Static application tests assert that no Social Signals UI or network request
  exists.
- Playwright covers a signed-in user queue workflow and an administrator health
  and manual-refresh workflow using fixtures only.

## Rollout

1. Merge persistence and provider-neutral contracts with the feature disabled.
2. Add official provider and fixture-backed contract tests.
3. Add scoring, immutable run publication, and APIs behind the disabled runtime
   capability.
4. Add Daily, queue, evidence, Theme Social Pulse, and Operations UI.
5. Build the private multi-architecture worker and verify local Docker with a
   dedicated automation profile.
6. Run a fourteen-day two-list backfill without publication and compare source
   coverage, deduplication, mappings, and rankings with the research artifacts.
7. Publish only after the quality gate passes, then enable Social Signals for
   authenticated users.

Rollback disables scheduling and the runtime capability and leaves the last
published rows intact for audit. No rollback requires deleting ContentItems or
changing existing Theme identities.

## Acceptance criteria

- The public repository builds and tests without `xui-reader`, GHCR access, or
  X credentials.
- A disabled installation schedules no X traffic and renders no ordinary-user
  Social Signals surface.
- Both enabled providers satisfy the same normalized record contract.
- Identical pinned inputs and formula version produce identical scores and
  ordering.
- Cross-list posts are deduplicated without losing list attribution.
- Existing Theme rankings are unchanged by Social Sources.
- Healthy installations publish a complete two-source snapshot at least every
  seven hours.
- Failed and partial runs preserve the last complete published snapshot and
  expose an accurate health state.
- Pure Social ordering excludes technical inputs while continuing to display
  them as evidence.
- Security resolution is deterministic across the five supported Markets and
  never fabricates an unresolved listing.
- Broad-Market ETFs remain context-only and thematic ETFs are visibly labelled.
- Only signed-in users can read the queue and only administrators can configure
  or refresh it.
- No credential, session state, raw provider debug payload, or social content
  enters image metadata, CI artifacts, logs, or static exports.
