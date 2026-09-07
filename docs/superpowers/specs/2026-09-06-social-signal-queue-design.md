# Social Signal Queue Design

**Date:** 2026-09-06

**Status:** Product decisions approved; consolidated after nine-finding review on 2026-09-07. Application implementation has not started.

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

- Combine installation-managed X lists into one ranked research queue, seeded
  with:
  - `1522014550211457024`
  - `1986290701492232693`
- Support resolved securities in the US, HK, CN, JP, and TW Markets.
- Rank candidates with a deterministic, versioned 60% social / 40%
  confirmation formula.
- Offer a Pure Social ranking mode without removing confirmation evidence from
  the display.
- Reuse existing ingestion, theme identity, SecurityMaster, scan-feature,
  Market-health, stock-detail, and watchlist capabilities.
- Share theme discovery across sources while evaluating social attention and market confirmation separately.
- Support `disabled`, official X API, and proprietary `xui` provider modes
  without changing downstream contracts.
- Keep the public application buildable, testable, and runnable without the
  proprietary package or any X credential.
- Run six-hour collection safely in local Docker and offer a restricted admin
  refresh path.
- Keep all social data out of static-site bundles.

## Non-goals

- Ordinary-user-configured X sources or user-specific queues. Administrators
  manage the installation-wide source set.
- Static-site Social Signals.
- Embedded X images, video, or article bodies.
- Posting, liking, following, bookmarking, direct messaging, or any other X
  mutation.
- Notifications, trade alerts, or automatic order execution.
- AI-generated candidate scores or opaque AI ranking judgments.
- Treating social options-flow claims as verified market data.
- Historical strategy backtesting.
- Silently falling back between providers.
- Silently treating social engagement as market confirmation or blending it into legacy attention scores.

## Domain language

The following terms form the shared implementation vocabulary:

**Social Source**: One administrator-controlled X list supplying read-only
evidence to the shared Social Signal Queue. A Social Source is represented by a
`ContentSource`; it can discover themes and propose company associations without
feeding social engagement into legacy Theme attention scores.

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
Setup Engine, RS, group-strength, and theme market-price evidence independent of engagement.

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

Collection ownership is explicit: a `ContentSource` with a
`SocialSourceConfiguration` belongs exclusively to Social collection, regardless
of lifecycle state. Existing Theme scheduled polling, bulk/manual ingestion,
and direct-source ingestion must skip or reject these sources before any
provider call. Only the dedicated social worker may collect them, including
administrator-requested tests and refreshes. `SOCIAL_INGEST_PROVIDER=disabled`
prevents all collection of these sources; the legacy Theme provider setting
cannot override it. Existing non-social sources retain their current behavior.
Collected posts may reuse Theme extraction, subject to the separate
shared-discovery and score-separation requirements below.

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
the same worker locally using BuildKit SSH access to the pinned private package.
Keep the private checkout outside the public repository and ordinary build
context. Private package code must not enter public images, public caches, or
public build artifacts. The private worker image necessarily contains installed
xui-reader code; anyone allowed to pull it can extract that code. Restrict private
image and private build-cache access to people trusted with the package itself.
Repository credentials must not persist in any image layer or configuration;
authentication/session data is mounted only at runtime, never baked into images.

The public image supports the application and official X reader without xui.
The private build installs the pinned reader, its matching Playwright browser,
and required browser system libraries. Supply build-only Git/SSH tooling and a
verified GitHub host key in the private build stage, using temporary BuildKit
credentials. Perform privileged installation during the build, then run as the
existing non-root worker with browser binaries in a readable, fixed location
outside the mounted session/data directories. Keep the public/default target
free of private dependencies even when the Dockerfile includes private targets.
Verify non-root browser launch against local content, without X access or session
credentials, on both linux/arm64 and linux/amd64. Package import alone is not a
sufficient runtime smoke test.

The X automation profile is created by a human login on the Mac and mounted
from a dedicated, access-restricted Docker volume. It is never built into the
image. The volume is writable only by the non-root worker so legitimate session
state can be refreshed; no other service mounts it.

### Scheduling

**Approved operating modes:** `SOCIAL_SIGNALS_MODE` is `off|validation|live`,
defaulting to `off`, and replaces the previously proposed enabled boolean.
Off schedules no collection or LLM processing. Validation performs real bounded
collection and analysis, but results are administrator-only: no published Social
pointer changes and no changes to user-facing themes, constituents, taxonomy,
lifecycle, or rankings. Stage discovery/association proposals separately from
the shared live catalog, using existing identities for read-only matching.
Validation consumes X resources and the same US$2/day Social LLM allowance.
Retain observations and extraction results for reuse by a newly evaluated live
run; switching mode never blindly publishes an old validation snapshot.

Live applies approved publication checks and can publish or apply eligible staged
theme changes. Gate every user-visible mutation, not just the Social pointer.
Pin mode/version per run and recheck before user-visible writes so changing to
off/validation cannot be bypassed by an in-flight live run. Ordinary users see
Social Signals only in live mode with a configured provider. Operations remains
available to administrators for setup and validation. Explicit Test List remains
a separately authorized bounded diagnostic when the provider is configured;
provider `disabled` prevents reads in every operating mode. Never auto-promote
validation to live based on elapsed time.

Deployment environment settings initialize an empty installation's shared runtime
policy. After initialization, administrators apply mode/provider changes through
the versioned runtime control; workers read the same database policy and never
overwrite it from stale startup environment values. Record runtime changes as
redacted registry-scoped audit events. Credentials remain deployment secrets,
not fields in this API. Keep deployment defaults aligned for disaster recovery.

Celery Beat dispatches a social refresh every six hours only in validation/live
mode with a configured provider. A provider-specific worker consumes the `social_ingestion` queue. A
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

The two initial list URLs are seeded as shared Social Sources assigned to the
existing technical extraction pipeline. They are not editable by ordinary
users. Administrators manage the installation-wide source registry from
Operations without rebuilding the application or private worker image.

Social-owned sources may be changed only through the administrator-only
Social Sources service and Operations panel. Legacy Theme source mutation
endpoints reject changes to these sources, even for administrators, and direct
them to Operations → Social Sources. Legacy controls show these sources as
managed elsewhere rather than offering edit/disable/delete actions. Ownership
checks apply in every lifecycle state, preserving testing requirements,
minimum-enabled-source checks, and audit history. Ordinary Theme sources
retain their existing management behavior.

Adding a source requires a 1-100-character human-readable display name plus an
X list URL or a 1-32-digit numeric list ID. The list ID is the immutable
external identity; the local display name may be edited later. Saving performs
syntax and duplicate validation only, stores the canonical
`https://x.com/i/lists/{id}` URL, and creates the source in `pending` state. It
does not contact X automatically.

An explicit `Test List` action asynchronously dispatches a provider read of at
most five posts through the configured social worker. The source shows
`queued`, `running`, and completed test status while the UI polls the admin
projection. This action may consume official API
credits or create X UI traffic, so the UI states that consequence before the
administrator runs it. A successful test records the provider, result time,
sample count, and redacted outcome, after which the source may be enabled. A
pending or disabled source's test pass is valid only when it matches the
provider selected at enable time. Already-enabled sources are validated by the
next complete refresh after a deployment changes provider. The test neither
stores queue evidence nor publishes a Social Signal Run.

At least two Social Sources must remain enabled. An administrator cannot
disable or archive a source if that would leave fewer than two enabled sources;
a replacement must be tested and enabled first. Every enabled source must
complete successfully for a refresh to publish. Pending, disabled, and
archived sources are not part of the pinned run source set.

**Approved concurrent administration protection:** Serialize source-registry
mutations with a shared database transaction lock. After acquiring it, read the
current enabled count, validate the requested change and source version, then
persist the change and audit event in that same transaction. Per-source versions
alone cannot protect a rule spanning multiple sources. Concurrent attempts to
disable different sources must recheck the updated registry; reject any change
that would leave fewer than two enabled with guidance to enable a replacement
first. Keep provider reads outside this short transaction and retain stale-edit
checks. All registry mutation entry points use this same protection.

Removing a source archives it rather than deleting it. Archival stops future
collection and hides the source from normal controls while preserving source
membership, evidence badges, run inputs, and audit history. All add, rename,
test, enable, disable, and archive operations require administrator
authorization and create immutable audit events for both request and
completion.

Each source's first enabled run attempts a fourteen-day backfill capped at
1,000 posts. A newly enabled source joins the pinned set after its explicit test.
Limited initial history, including a successful read reaching the cap, does not
by itself block publication. Later source reads use bounded overlapping reads of recent posts,
retaining previously observed posts so engagement observations can be refreshed.

**Approved warming-up publication:** Separate successful participation and
processing from historical coverage. Every enabled source must return a valid,
successful bounded read, and collected posts included in the run must finish
analysis before publication. A failed source or unfinished analysis still blocks
replacement of the previous snapshot. Successfully processed bounded reads may
publish with explicit per-source/window warming-up or limited-history metadata;
they must not claim complete historical coverage. This applies both at startup
and when an additional list joins.

Record observed time bounds, known gaps, and limit/coverage reasons without
inferring continuity from the oldest post or assuming a short response proves
exhaustion. Missing history is not zero activity. Acceleration is unavailable
where its comparison history cannot be supported; expose component coverage and
do not silently report a normal full-history score. Subsequent six-hour reads
build history, but elapsed time alone never clears a coverage limitation. Ongoing
gaps and capped reads remain visible. Unknown provider outcomes are not successful
limited-history reads. In this document a complete published run means complete
processing of all pinned sources' declared inputs, not exhaustive X history.

**Approved repeatable collection:** Do not use xui's new-only filtering or its
automatic checkpoint advancement. The application owns durable collection
progress, advancing it only after observations are committed to its database.
Retries may return the same posts; idempotent post/source upserts prevent duplicate
evidence while updating engagement. Unchanged content reuses persisted LLM
extraction. Interrupted reads can retry the same bounded request without a
reader-side checkpoint hiding previously returned items; this is not a guarantee
of complete history if posts disappear or fall outside provider limits.
The six-hour schedule is unchanged. Test List never advances production collection
progress or inserts application evidence; private reader-local storage is not
used as the application's progress authority. A production read must still return
a post previously seen during a test. Official pagination/progress is likewise
committed only with durable observations, retaining the approved overlap policy.

The scorer uses 1-day, 7-day, and 14-day windows based on post publication time
in UTC. Market-session dates are used only for joining existing Market data;
social windows are continuous clock windows because X posts are not limited to
trading sessions.

## Persistence model

### Existing records

`ContentSource` remains the configured-source record. Social evidence remains
separate from legacy Theme attention inputs; this separation must not prohibit
theme discovery or accepted constituent contributions to market measurements.
Use explicit content eligibility and association provenance instead of a blanket
source ranking flag. A canonical post may have both Social and independently
ingested legacy evidence. Record eligibility per content/pipeline/channel so the
first source to insert a deduplicated ContentItem does not determine its use.
Social collection alone must not enable legacy attention, ingestion-day counts,
legacy extraction jobs, or static exports; an independent legacy observation may
enable those uses once, without importing Social engagement.

`ContentItem` remains the canonical post record and retains the existing
`(source_type, external_id)` uniqueness behavior. Social extensions do not add
engagement JSON to this table.

`ThemeMention` remains the canonical extracted-theme evidence. Technical theme
extraction processes Social Posts and resolves the same pipeline-scoped
`ThemeCluster` identities and can create new candidates from an empty database.
Queue theme confirmation uses price/RS/breadth fields, not mention velocity or
a momentum score that mixes social evidence back into confirmation.

### Shared discovery, separate evaluation — approved review decision

One shared Theme Catalog serves Social Signals and Themes. Social posts can
create candidate themes immediately and propose company associations, retaining
source/post/author provenance. Non-social coverage is helpful but not mandatory.
Reuse existing theme matching and candidate lifecycle concepts; do not create a
parallel social-only catalog.

Distinguish discovery evidence, social strength, and market strength. Deduplicate
posts and count independent authors rather than treating duplicate list
membership as independent evidence. Company associations are proposed or
accepted; only accepted associations enter the measured stock basket. Acceptance
assesses whether a company belongs, not whether its stock is rising.

Association acceptance is automatic when the agreed evidence requirements are
met, with administrator review reserved for exceptions. Evidence must explain
the company's business connection to the theme; a bare ticker mention or price
co-movement alone is insufficient. Reposts and duplicate list appearances do
not provide corroboration. Ambiguous company identities and weak connections
remain proposed and visible. Administrators can accept or reject associations
with a recorded reason. Price performance never determines membership.

The approved initial automatic-acceptance threshold is two qualifying posts
from two distinct authors within a rolling fourteen-day window, each explaining
the same resolved company's business connection to the same theme. Copied
claims, reposts, and duplicate list appearances count once. Distinct authors
are a practical corroboration check, not proof of independence. No minimum
engagement, price strength, or presence in both lists is required. One useful
post can create a visible proposal that an administrator may accept immediately.

### Explicit LLM dependency

Automated theme discovery and business-connection assessment require a configured
LLM extraction provider. Reuse the existing `LLMService` extraction integration
and configured extraction model; extend its structured output for relationship
assessment rather than assuming the current theme/ticker output is sufficient.
The X reader (`official` or `xui`) only collects posts and does not replace this
dependency. Both reader modes use the same extraction contract.

The LLM identifies candidate themes, company mentions, and explicit business
relationships supported by the supplied text, with source excerpts and evidence
references. It can flag copied/paraphrased claims as potential duplicates;
uncertain corroboration stays proposed rather than counting as independent
support. It must not invent business facts or certify a claim as externally
verified. Securities are resolved against the application's security records.

Application rules validate structured output and supporting excerpts, perform
post/list deduplication, apply the two-author/fourteen-day threshold, and record
acceptance decisions. The LLM cannot directly change baskets, publish runs,
override administrator decisions, or compute numeric scores. Its semantic
judgments influence eligibility, but scoring and acceptance-rule evaluation are
deterministic for the same persisted extraction results and policy version.
Fresh LLM calls are not assumed to reproduce identical judgments.

Persist extraction results with actual model/provider and prompt/schema versions
plus input identity, allowing reuse for unchanged content. Engagement-only updates
must not trigger re-extraction. Missing configuration, quota exhaustion, invalid
output, or provider failure leaves affected extraction pending/failed; it cannot
silently mean no themes or successful qualification. Incomplete processing keeps
the previous published Social snapshot under the existing publication policy.

Setup and Operations must disclose the separate LLM configuration, that post text
is sent to the configured provider (which may be external), possible usage charges,
and extraction health. Credentials/session state and private reader code are not
LLM inputs. Treat all post text as untrusted data, not instructions. Tests use
stored/synthetic extraction fixtures, not live paid model calls.

**Approved budget:** Start with a configurable US$2/day installation-wide budget
for Social LLM processing, separate from other application LLM use and X API
charges. Reserve estimated request cost before dispatch and reconcile reported
usage afterward. Concurrent workers, retries, and any permitted fallback must
share the same budget ledger; hidden retries cannot bypass it. A request that
does not fit the remaining allowance is deferred, not sent. This is an
application spending guardrail, not a guaranteed provider billing cap. If actual
charges exceed the estimate, record them and stop further calls rather than
hiding the excess or automatically increasing the allowance.

**Approved budget-deferral workflow:** Keep collected
posts and unfinished extraction in a durable database backlog, resume bounded
batches after the daily reset, and reuse completed results. Six-hour X collection
continues under its separate limits; deferred LLM processing does not re-read X.
Keep the previous complete Social snapshot visible, marking it stale when due;
a fresh installation shows processing pending until a complete run can publish.
Operations shows budget usage, waiting-post count, oldest pending age, and next
reset. Deferred batches do not imply a provider-specific Batch API or a discount.
Sustained arrivals above daily processing capacity cause growing delay; expose
that delay rather than silently dropping evidence or publishing incomplete results.

**Approved backlog ordering and aging:** Process the oldest waiting posts still
within the rolling fourteen-day signal window first, using post publication time
rather than collection time. Posts older than fourteen days leave automatic LLM
processing but remain stored and explicitly marked not analysed/outside signal
window. An administrator can request later analysis under the same budget; this
does not make an old post eligible for a current signal window. Completed results
are retained. Aging out is not successful extraction and cannot satisfy a pinned
historical run's completeness checks. Some historical theme discoveries may
therefore remain unanalysed; expose skipped counts and do not claim full historical
coverage.

**Daily budget reset:** The shared Social allowance resets at midnight in the
configurable IANA timezone `Asia/Singapore` by default. Every Mac/Docker worker
uses the same installation setting and database ledger, not its host timezone.
Store reservation/usage timestamps in UTC and derive the budget date and next
reset from that setting. Reserve in the dispatch day's bucket; completion after
midnight reconciles against that original bucket. Unused allowance does not
accumulate. Preserve prior usage/reservations when configuration changes; changing
timezone or worker restart must not grant a second allowance for the same period.

Candidates appear in Social Signals and a Discovering view on Themes. When
accepted constituents have sufficient market data, calculate market strength;
otherwise display insufficient data, not zero. Socially strong but technically
weak themes remain discoverable without being labeled market-confirmed.

**Approved social-led candidate promotion:** Promote a candidate to active when
it has at least three accepted companies and qualifying discussion on at least
three distinct UTC calendar dates within the rolling fourteen-day window. Count
different listings of one company once; duplicate posts, copied claims, and
reposts do not add qualifying discussion. Neither rising prices nor non-social
coverage is required. Before promotion the theme remains visible in Discovering.
Active denotes an established research theme, not a buy signal or market
confirmation. Apply this as an explicit social-led promotion policy rather than
silently replacing the existing non-social lifecycle rules.

**Approved market-strength coverage:** Evaluate each selected Market separately.
Require usable, sufficiently fresh data for at least three distinct accepted
companies and at least 70% of the theme's accepted companies in that Market before
showing a market-strength score. Count multiple listings of one company once in
both numerator and denominator. Show measured/accepted coverage explicitly, for
example four of five companies. Do not pool US, HK, JP, or other Market baskets
or benchmarks into one score. Below either threshold, show insufficient market
data while retaining social strength and discovery evidence. Apply the session
freshness policy below and reuse the existing feature engine's indicator-history
validity rules; never synthesize unavailable moving averages or RS inputs.

New themes and accepted associations may legitimately change Theme rankings.
Social engagement must not silently alter legacy attention scoring or masquerade
as technical confirmation. Preserve separately identifiable legacy scoring.
Regression tests must cover discovery from an empty database, shared identities,
provenance, proposed-versus-accepted basket membership, and score separation.

### Persistence and projection boundaries

Store extraction work/results separately from live Theme mutations. Work is keyed
by content revision, prompt/schema version, and selected model configuration.
A run pins work/result IDs and copied numeric inputs for reproducible scoring;
provider/model provenance records the actual model used. Durable work status
distinguishes pending, running, waiting_budget, succeeded, failed_retryable,
failed_terminal, and outside_window. Successful empty extraction is distinct
from failed extraction. Retain per-post completion and errors if batching calls.

Budget-day and request-attempt rows hold atomic cost reservations, usage, status,
and idempotency keys. Use a conservative input/output-token reservation and a
configured pricing version. Unknown pricing pauses billable dispatch; ambiguous
provider completion retains the reservation pending reconciliation rather than
freeing it for duplicate spending. Do not auto-switch extraction models.

Stage theme keys and supported company relations in extraction results. Only
live publication materializes shared ThemeCluster/ThemeMention identities and
proposed/accepted associations. Record association provenance, state/version, and
immutable admin/system decisions. Existing constituents remain legacy-accepted;
Social cannot erase independently supported legacy membership. Rejected admin
decisions persist until explicitly changed by an admin. Association acceptance
does not expire merely because the qualifying posts age out of scoring windows.

Keep legacy attention/source-diversity/ingestion-day metrics on legacy-eligible
evidence. Social discovery uses its own versioned lifecycle evidence without
feeding engagement into those metrics. Social-only candidates must not be
immediately demoted by legacy policies that cannot see their evidence. Apply
existing lifecycle timing rules to the corresponding evidence channel; audit the
policy and inputs. Shared accepted constituents can affect live market baskets,
but proposed relations cannot. Static projections use legacy-eligible evidence
and membership only, excluding social-only themes and additions.

Theme market measurements are keyed by shared theme identity, Market, session,
and basket version; reuse existing price calculations behind a Market-scoped
projection and Market Benchmark Registry selection, not a global SPY basket.
Run explanations freeze the selected basket, coverage, benchmark and component
values. New social-discovered themes need not have a legacy ThemeMetrics row to
become measurable. These are technical implementation boundaries for the
approved shared-catalog policy, not additional scoring weights.

### New records

**SocialSourceRegistry** is the singleton database lock/version record protecting
cross-source administration invariants and shared runtime policy changes.

**SocialSourceConfiguration** is the one-to-one social lifecycle extension for
a ContentSource. It stores the immutable X list ID, lifecycle state
`pending|enabled|disabled|archived`, test execution state, last tested provider,
last test outcome and time, archive time, and optimistic version. The ContentSource retains the
human-readable name and canonical list URL. The two original lists are seeded
as enabled with `system_seed` provenance; later sources must pass the explicit
test-before-enable flow.

**SocialSourceAuditEvent** is an immutable administrator action record. It
stores the Social Source, action, actor, timestamp, and redacted before/after
state. It never stores provider credentials, session paths, post content, raw
provider responses, or response headers.

Audit scope distinguishes source actions from runtime-policy actions: source
events require a ContentSource reference; runtime events reference the shared
registry and carry no fabricated source ID.

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

Scoring is pure, deterministic, and versioned. No LLM generates numeric
scores or selects the final ordering. LLM extraction affects evidence eligibility;
the persisted validated judgments and numeric inputs determine every score.
Replay uses these saved judgments, not an assumption that a new LLM call is identical.

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

**Approved missing-data behavior:** Preserve a usable Social Score when technical
confirmation is unavailable, allowing ordinary Pure Social ordering. If all
confirmation components are missing, Confirmation Score and Blended Queue Score
are null and displayed as an em dash, never zero. In Blended mode, scored rows
sort first; unscored rows follow in descending Social Score with the normal
deterministic tie-breakers. Do not renormalize the top-level 60% social/40%
confirmation blend to 100% social.

When only some components are unavailable, renormalize available weights within
their own social or confirmation portion and expose reduced component coverage.
Keep absent fields/reasons in explanations. Missing required technical checks
preclude Actionable, independently of rank mode. Reduced-coverage ranking is not
evidence that all technical checks passed; Pure Social changes ordering, not
eligibility. Missing social components during history warm-up follow the same
within-portion rule, without treating missing history as observed zero.

| Component | Queue weight | Definition |
| --- | ---: | --- |
| Mention acceleration | 20% | Percentile of `(mentions_1d + 1) / (mentions_previous_13d / 13 + 1)` |
| Unique authors | 15% | Percentile of capped distinct authors in the selected 1D, 7D, or 14D window |
| Engagement | 15% | Percentile of the summed, winsorized per-post engagement value defined below |
| Recency | 5% | Exponential decay from the latest qualifying mention, with a 48-hour half-life |
| Cross-list confirmation | 5% | Full component credit when distinct qualifying posts from at least two enabled lists mention the ticker; zero when fewer than two do |

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
posts with different provider post IDs from at least two enabled lists; one
duplicated post present in multiple lists does not earn the bonus. The score is
binary and does not increase above full credit when three or more lists mention
the ticker. The UI still shows the observed list count, such as `3 of 5`.

### Confirmation Score

**Approved daily-data freshness:** Evaluate against the relevant exchange/MIC
calendar, including holidays, weekends, and early closes. During trading, the
previous completed session is the required daily-data session. After a session
closes, allow a configurable two-hour update grace period; once it expires,
require that newly completed session. Equivalently, the required session is the
latest session whose close plus grace has passed. Show actual input session dates
explicitly; this is daily snapshot freshness, not a claim of live quotes.
Required technical inputs older than the required session, or with unverifiable
freshness, preclude Actionable while social evidence stays visible. Calendar
uncertainty must not silently assume a trading day. Apply the relevant Market
calendar to Market-level inputs and the security's MIC calendar to listing-level
inputs. Preserve coherent snapshot joins. Social snapshot staleness remains a
separate seven-hour clock-time rule.

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
0-100 values when supplied by the existing feature/metric engine. For non-US
Markets, use the equivalent benchmark-relative component from the Market-specific
projection rather than treating the legacy `basket_rs_vs_spy` name as permission
to use SPY globally. Enforce the three-company/70% threshold per component before
including it; absent valid history means that component is missing. The ticker
receives the highest linked-theme value, with
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

**Approved All Signals presentation:** Use three clearly separated sections.
Ranked candidates contain resolved stocks and thematic ETFs in the selected
Market, including those with missing technical data under the approved null-score
rules. Market context contains broad-Market ETFs and macro evidence, unranked.
Needs resolution contains ambiguous company mentions with their original post
evidence, unranked and excluded from candidate score cohorts and top-five cards.
When an unresolved item's Market cannot be established, expose it in an explicitly
global Market unknown group accessible from every Market view. Never infer Market
from the currently selected tab or duplicate the item into Market-specific scoring.
Only scope evidence by Market when supported by its stored provenance; preserve
unscoped macro context separately rather than assigning a fabricated Market.
API responses and pagination must distinguish candidate rows from context and
resolution groups so clients do not mix global evidence into ranked totals.

## Run and publication flow

1. Acquire the singleton social-ingestion lease.
2. Create an immutable Social Signal Run and pin provider, the complete enabled
   source set, application-owned committed collection progress, operating mode/version,
   formula/extraction policy versions, and current time.
3. Read each Social Source independently through the selected adapter.
4. Validate and normalize provider records.
5. Upsert ContentItems, source memberships, and newer engagement observations
   transactionally per source.
6. Reuse or enqueue durable, budgeted Social extraction; retain staged discovery
   and company-association evidence without calling live Theme mutation paths.
7. Resolve ticker identities using SecurityMaster and the active Universe.
8. Aggregate 1D/7D/14D evidence with anti-manipulation rules.
9. Join one coherent current feature snapshot per Market plus Market posture,
   group ranks, liquidity, and Market-scoped theme price confirmation.
10. Calculate and persist immutable Social Signal Snapshots.
11. Evaluate quality against every pinned source's successful bounded read and
    completed processing, carrying explicit historical-coverage limits. Validation
    saves administrator-only output. Live rechecks operating mode/version and
    atomically materializes eligible Theme changes plus advances the pointer.
    Prepare heavy extraction/price work outside that short transaction.
12. Release the lease and expose the run outcome to Operations.

A partial or failed attempt is retained for diagnosis but cannot replace the
last Published Social Signal Run. If no published run exists, the API returns a
typed unavailable state rather than partial candidates.

## API and authorization

Ordinary Social Signal read endpoints require an authenticated app user and live
mode. Their responses expose only the Published Social Signal Run. Separate
administrator-only projections expose validation results without publishing them.

The provider-neutral live API supplies:

- queue summary and freshness;
- supported windows, views, ranking modes, filters, and formula version;
- paginated rows for a selected Market;
- per-row component explanations and Signal State reasons;
- top source evidence limited to short excerpts and canonical X links;
- source coverage and degraded-state metadata; and
- related listing and unresolved evidence where applicable.

Provider configuration, authentication health, run history, source creation,
rename, test, enablement, disablement, archival, and manual refresh require
administrator authorization. API responses and logs
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
selected Market, dominant linked themes, current Market posture, enabled-source
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

The Themes page remains separate from Social Signals and gains a Discovering
view for candidate themes plus a `Social Pulse` field derived from the Published
Social Signal Run. Social Pulse does not directly feed legacy attention scoring.
Shared discovery and accepted basket changes can affect rankings; social and
market evaluations remain distinguishable. Queue theme links navigate into the
same Theme detail experience.

Provider health, last run, source coverage, reauthentication requirements,
rate-limit state, source management, and the admin refresh control live in
Operations/settings, not in the ordinary user's queue controls. The source
panel shows required display name, canonical list ID/URL, lifecycle state,
tested provider/time/outcome, last successful collection, and archive history.

Static mode does not render the Daily card, Social tab, social API calls, or
Social Pulse. Static exporters contain no social rows, excerpts, metrics, or
run metadata.

## Failure handling and observability

Expose independent status dimensions: operating mode; collection health
(`disabled|healthy|partial|reauthentication_required|rate_limited|provider_error`);
processing (`pending|running|waiting_budget|failed|complete`); history coverage
(`warming_up|limited|observed_window`); and publication freshness (timestamp,
stale boolean). A fresh limited-history snapshot and a budget-paused stale
snapshot must both be representable. Observed-window coverage is not a promise
that all X posts were available.

- Missing provider configuration yields `disabled` and schedules no reads.
- Missing or expired `xui` authentication yields
  `reauthentication_required`; the worker stops without interactive login.
- Authentication challenge, selector drift, or suspected blocking stops the
  run immediately and triggers a provider cooldown.
- A temporary network error receives at most one delayed retry.
- Official API 429 responses respect the returned reset time and do not switch
  provider.
- Invalid provider JSON fails that source with a stable schema reason.
- Missing/failed source participation or unfinished processing cannot replace a
  published snapshot; successful all-source processing with limited history can
  publish with explicit warming-up/limited-history labels.
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
- Cross-list deduplication retains every source membership.
- One-author caps and repost/quote deduplication prevent inflated scores.
- Listing resolution covers explicit US/HK/CN/JP/TW symbols, company aliases,
  ADR relationships, ambiguity, and unresolved evidence.
- Broad-Market and thematic ETF classification follows the stated policy.
- Signal State tests cover stale data, liquidity, setup readiness, and
  Market-posture changes without score mutation.
- Theme regressions prove empty-database discovery, shared identities, supported
  basket changes, and separation of social attention from market confirmation.

### Provider contract tests

- Sanitized official and `xui` fixtures normalize to the same typed records.
- Missing engagement remains null while observed zero remains zero.
- CLI argument construction never invokes a shell and JSON validation rejects
  unknown or malformed structures.
- Provider caps, 429 handling, authentication failures, challenge signals, and
  non-fallback behavior are deterministic.
- Source-management tests cover duplicate IDs, required names, pending creation,
  explicit five-post tests, provider-change invalidation, the two-enabled
  minimum, all-enabled publication, optimistic concurrent edits, archival, and
  immutable redacted audit events.
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
6. Set `SOCIAL_SIGNALS_MODE=validation`, attempt the bounded fourteen-day
   backfill and review admin-only coverage, deduplication, mappings, and rankings.
   Confirm live Themes and the Social pointer stay unchanged; budgets still apply.
7. Explicitly set `SOCIAL_SIGNALS_MODE=live`; a newly evaluated run reuses saved
   results and publishes only after the agreed checks pass. No automatic mode
   transition or blind publication of a validation snapshot.

Rollback disables scheduling and the runtime capability and leaves the last
published rows intact for audit. No rollback requires deleting ContentItems or
changing existing Theme identities.

## Acceptance criteria

- The public repository builds and tests without `xui-reader`, GHCR access, or
  X credentials.
- A disabled installation schedules no X traffic and renders no ordinary-user
  Social Signals surface.
- Both enabled providers satisfy the same normalized record contract.
- Administrators can add, rename, test, enable, disable, and archive lists
  without an application or worker-image rebuild; ordinary users cannot.
- New lists remain pending until an explicit five-post-or-fewer test succeeds
  for the currently selected provider.
- At least two lists remain enabled and publication requires every enabled list
  to complete successfully.
- Identical pinned inputs and formula version produce identical scores and
  ordering.
- Cross-list posts are deduplicated without losing list attribution; distinct
  evidence from any two enabled lists earns the same full binary credit.
- Social can discover new themes and support basket changes without social
  engagement being counted as market confirmation.
- With healthy collection, completed analysis, and available budgets, six-hour
  runs target a published all-enabled-source snapshot less than seven hours old;
  this is not guaranteed while processing is deferred or providers are failing.
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
