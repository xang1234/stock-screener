# Options Analytics and Command Center Design

**Date:** 2026-09-04

**Status:** Approved in design review

**Source intent:** GitHub PR #339, used only to identify product intent and failure modes

## Context

PR #339 attempted to introduce ticker-level options analytics and a market-wide
Options Command Center. It also mixed in unrelated operations controls,
deployment changes, stock-news endpoints, worker topology changes, and several
independent persistence models. The branch is far behind current `main`, fails
CI, and predates important repository architecture such as the Market Catalog,
explicit domain/use-case/infrastructure boundaries, published feature runs,
current Market Workload coordination, and static-site parity contracts.

This feature will therefore be implemented cleanly from current `origin/main`.
No PR #339 code, migrations, fixtures, or file structure will be copied,
cherry-picked, or rebased. The PR is evidence of desired behavior and known
failure modes only.

The design deliberately limits Yahoo Finance traffic. The Command Center is a
ranked view over a small cohort selected from already-published equity results;
it is not a broad options-market scanner and it never fetches data in response
to opening the Command Center page.

## Goals

- Provide a US Options Command Center sourced from existing high-quality equity
  candidates and leaders.
- Provide ticker-level drill-down for every current Command Center member.
- Fetch one bounded, comparable option chain per tracked symbol each trading
  day and calculate all supported metrics from that observation.
- Preserve ticker history across changes in daily leadership without requiring
  consecutive membership.
- Publish new rankings atomically only when at least 90% of the current cohort
  has valid observations.
- Expose the same versioned read contract to the live app and the read-only
  static site.
- Make metric assumptions, freshness, data coverage, and unavailable states
  explicit.
- Integrate with current Market Catalog, Celery queues, Market Workload,
  Operations, authentication, and static-export patterns.

## Non-goals

- Scanning every optionable US security or every daily-scan result.
- Fetching every expiration for Command Center members.
- Supporting non-US options in the first release.
- Claiming observed dealer positioning or exchange-provided Greeks.
- Inferring net premium inflow, trade aggressor, bullish buying, or bearish
  selling from Yahoo quotes.
- Trading recommendations, strategy recommendations, or fabricated market
  conclusions.
- Technical/options correlated-signal endpoints or unrelated stock-news APIs.
- Print/PDF-specific layouts.
- New Celery worker families, subprocess batch scripts, pgAdmin, Docker resource
  changes, or new force-release operations.
- Backfilling historical option chains that Yahoo does not provide.

## Domain language

**Options Analytics Run**: One immutable attempt to observe and calculate
options metrics for a pinned Candidate Cohort using one provider policy and one
calculation version.

**Candidate Cohort**: The deduplicated set of Current Candidates and Continuity
Candidates pinned when an Options Analytics Run starts.

**Current Candidate**: A security selected from the Top Scan Candidates or
Leaders in Leading Groups for the pinned published US equity run. Current
Candidates may appear in Command Center rankings.

**Continuity Candidate**: A recently dropped Current Candidate retained only to
keep its options history continuous. A Continuity Candidate is observed but is
excluded from current rankings.

**Chain Observation**: The normalized calls and puts for one security, one
selected expiration, one provider fetch time, and one source equity run.

**Published Options Run**: The single Options Analytics Run selected by the
market-specific published-run pointer and served to live and static clients.

**Model Estimate**: A metric calculated using declared assumptions rather than
an exchange- or provider-observed value. Estimated GEX, gamma flip, and gamma
walls are Model Estimates.

Implementation will add these terms to `CONTEXT.md` so later code and
documentation use consistent language.

## Architecture

Options analytics will be a dedicated bounded context following the
repository's existing run/stage/quality/publish pattern:

1. A candidate selector reads one published US equity feature run.
2. A provider port supplies normalized expiration metadata and one normalized
   Chain Observation per selected symbol.
3. Pure domain calculators derive every current metric and its availability.
4. A use case stages run items and bounded strike points, evaluates the run
   publication policy, and advances the published pointer atomically.
5. Query use cases read only the Published Options Run.
6. Protected live APIs and the static exporter serialize the same versioned
   response contracts.
7. Shared React presentation components render either the live or static data
   client.

The bounded context owns options-specific types and policies. It consumes
existing SecurityMaster identity, published equity-run queries, price history,
Market Catalog capabilities, task dispatch, and unit-of-work interfaces. Domain
calculators do not import Yahoo, SQLAlchemy, FastAPI, Celery, Redis, or React.

### Market capability and runtime enablement

The Market Catalog gains an `options_analytics` capability. It is true only for
the US in the first release. Capability means the product is supported for the
Market; it does not mean the local deployment has enabled collection.

One runtime setting, `OPTIONS_ANALYTICS_ENABLED`, controls scheduled and manual
collection. It defaults to false for self-hosted deployments so merging the
feature cannot silently introduce Yahoo traffic. The live navigation combines
the Market Capability and runtime setting. Static navigation follows the
exported manifest.

No other candidate caps, thresholds, formulas, or UI labels become environment
variables. They are versioned domain policy.

## Candidate selection

Candidate selection is deterministic and pinned to the source equity `run_id`
and `as_of_date`.

### Current Candidates

The selector takes the union of:

1. The first 40 Top Scan Candidates after requiring daily dollar volume
   strictly greater than USD 100,000,000.
2. The first 40 Leaders in Leading Groups after requiring the same daily dollar
   volume floor.

Top Scan Candidate ordering must reuse the current Daily Snapshot selection
policy. Leaders in Leading Groups must reuse its current shared policy,
including group rank at most 40 and RS at least 80. Implementation extracts or
calls the authoritative selectors rather than copying their conditions into a
new options module.

Security identity is deduplicated using the current canonical SecurityMaster
identity. Each selected item preserves both source ranks and both provenance
flags when it qualifies through both lists. Stable symbol ordering breaks exact
ranking ties.

If either source has fewer than 40 qualifying securities, the selector takes
all of that source's qualifying securities without filling the unused places
from another list. The Current Candidate cohort therefore contains at most 80
securities.

### Continuity Candidates

When a Current Candidate drops from both source lists, it enters a continuity
window for the first through fifth US trading sessions after its last Current
Candidate membership. It leaves the window before the sixth session.
Continuity collection is capped at 20 securities. If more than 20 qualify,
selection is ordered by:

1. most recent Current Candidate membership;
2. best prior source rank;
3. canonical symbol as a stable tie-breaker.

Returning to either source list immediately restores Current Candidate status.
Continuity Candidates receive the same Chain Observation and history storage as
Current Candidates, but are excluded from all current rankings and the static
Command Center summary.

The complete nightly cohort is therefore capped at 100 securities.

## Expiration and strike policy

Each cohort member can contribute at most one successful option-chain payload
and one Chain Observation per run.

The expiration selector chooses the nearest provider-listed standard monthly
expiration with 14 through 45 calendar days to expiration, inclusive. A listed
Thursday or Friday in the standard third-Friday expiration week is considered
monthly so exchange holidays do not invalidate the selection. Trading-day and
holiday decisions use the current US Market/MIC calendar facts.

The selector does not silently fall back to a weekly, same-day, or more distant
expiration. A symbol without an eligible monthly expiration receives an
explicit `expiration_unavailable` result.

Calculations consume every normalized contract with a finite positive strike
from the selected-expiration payload in memory. Persistence retains the
closest-to-spot strike, then at most the 30 distinct nearest strikes below it
and the 30 distinct nearest strikes above it. Equal-distance choices use lower
strike first. This bounds storage and static payloads without narrowing the
inputs to aggregate calculations.

## Provider boundary and request budget

The provider port exposes normalized contracts, not Yahoo-specific dataframes.
The Yahoo adapter maps provider fields into typed call and put records and
distinguishes absent, null, zero, invalid, and stale values.

The use case supplies the source equity run's spot price. Provider quote values
are recorded for diagnostics but cannot silently replace the pinned source
price. An absolute percentage difference greater than 2% produces a quality
warning but does not replace either value.

Expiration discovery metadata may be cached within the run. The collector runs
at most two symbols concurrently. Each symbol permits at most three total
attempts to fetch its selected expiration, using the existing US data-fetch
queue, external-fetch coordination, and rate-budget/backoff conventions. Only
the first successful payload is consumed; retries never create multiple Chain
Observations, and already-successful items are not restarted.

The provider adapter is replaceable without changing domain calculators,
persistence, APIs, static artifacts, or React components.

## Metric definitions

All calculations carry a schema version and calculation version. Changing a
formula or material assumption creates a new calculation version; historical
observations from incompatible versions are not combined.

### Max Pain

For each distinct usable settlement strike `S` in the full normalized payload,
calculate the total option holder intrinsic payoff at expiration:

- calls: `max(S - strike, 0) * call_open_interest * multiplier`;
- puts: `max(strike - S, 0) * put_open_interest * multiplier`.

Max Pain is the candidate settlement strike with the minimum combined payoff.
Open interest must be present and positive in aggregate. Stable lower-strike
ordering resolves an exact payoff tie. Max Pain is an open-interest-derived
estimate, not a price forecast.

### Estimated gamma exposure

Per-contract gamma is calculated using the Black-Scholes gamma formula from
spot, strike, time to expiration, Yahoo implied volatility, and the run's
recorded market assumptions. Contract multipliers use normalized provider data
and default to 100 only for explicitly regular US equity options.

The risk-free input is one run-level annualized rate resolved from the most
recent available `^IRX` close on or before the source equity date. If it cannot
be resolved, GEX-family metrics are unavailable. The dividend input uses the
pinned security fundamental when it is finite and non-negative; otherwise v1
uses zero and records an explicit `zero_dividend_assumption` warning. Neither
input is silently replaced by an unrecorded constant.

Dollar gamma exposure per 1% underlying move is:

`unit_gamma * open_interest * multiplier * spot^2 * 0.01`.

Calls receive a positive and puts a negative sign under a documented dealer
positioning proxy. This sign is not observed dealer inventory. The run records
the risk-free-rate source/value, dividend assumption/source, valuation time,
and sign convention. Missing required assumptions make the affected metric
unavailable rather than fabricated.

### Gamma flip and walls

Gamma flip is found by recalculating total estimated GEX across hypothetical
spot prices from the greater of the lowest usable strike and 80% of pinned spot
through the lesser of the highest usable strike and 120% of pinned spot. The
grid includes one-percent-of-spot increments and every usable strike in that
range. A genuine adjacent sign crossing is linearly interpolated. It is not
calculated by cumulatively summing per-strike GEX. If the range is empty or no
crossing exists, gamma flip is unavailable; the nearest low-magnitude point is
not relabelled as a crossing.

Call wall and put wall are the strikes with the greatest absolute estimated
call and put gamma exposure under the same calculation version. They are
labelled Estimated Call Wall and Estimated Put Wall.

### Implied volatility and skew

ATM IV is the mean of call and put IV at the usable strike closest to pinned
spot and requires a finite positive IV on both sides. Skew is 25-delta put IV
minus 25-delta call IV, using the contracts nearest absolute delta 0.25 only
when each falls within 0.20 through 0.30 under the same model assumptions.
Missing one side makes the metric unavailable. The UI never turns a missing IV
into zero.

Once history is eligible, IV percentile is the fraction of compatible ATM-IV
observations in the trailing 30 US trading sessions less than or equal to the
current ATM IV. IV rank is `(current - minimum) / (maximum - minimum)` over the
same observations; it is unavailable when the range is zero.

### Volatility risk premium

Twenty-session realized volatility comes from 21 valid closes in the existing
pinned price-history path, using 20 explicit no-fill log returns and square-root
of-252 annualization. Volatility risk premium is `ATM IV - realized
volatility`, expressed on the same annualized scale. Missing or insufficient
underlying history makes the metric unavailable.

### Unusual Options Activity

This replaces PR #339's unsupported Net Premium Inflow concept. It reports:

- total call and put volume;
- call/put volume ratio when the denominator is valid;
- total call and put open interest;
- volume/open-interest ratios with explicit zero-denominator handling;
- concentration of volume and open interest within plus or minus 5% of spot;
- the highest qualifying contract activity ratios subject to an absolute
  volume floor of 100 contracts;
- activity intensity, defined as total option volume divided by total option
  open interest when aggregate open interest is positive; and
- descending cross-sectional activity-intensity rank among Current Candidates
  with valid intensity, with canonical symbol as the stable tie-breaker.

It may say call-heavy or put-heavy activity. It cannot infer buying, selling,
trade aggressor, or directional intent.

### Historical metrics

Yahoo supplies current chains, not historical chains. No historical option
observation is synthesized or backfilled.

History is ticker-centric and separate from current leadership membership. A
ticker dropping from the cohort does not delete or reset its observations. If
it later returns, the new observation appends to its prior history. Dates with
no observation remain gaps and are never forward-filled.

Eligibility depends on recent observation density, not lifetime or consecutive
membership:

- Short-term change metrics require at least five compatible valid observations
  within the trailing seven US trading sessions, including the current
  observation. A five-observation change compares the current observation with
  the fifth-most-recent observation; missing dates remain visible gaps.
- IV percentile/rank requires at least 20 compatible valid observations within
  the trailing 30 US trading sessions, including the current observation.
- Observations outside the rolling window age out naturally.
- If the recent requirement is no longer met, the metric returns to
  `building_history` even though older observations remain stored.

The UI reports the actual recent count, such as "18 valid observations in the
last 30 trading sessions," rather than implying consecutive coverage.

## Quality contract

Quality is represented by explicit evidence and reason codes, not a single
opaque score. Every run item records:

- provider and provider fetch time;
- source equity run, source spot, and options observation date/time;
- selected expiration and days to expiration;
- normalized call, put, and strike counts;
- open-interest, IV, volume, and two-sided-quote coverage;
- latest relevant contract trade time;
- source/provider spot disagreement when present;
- calculation assumptions and version;
- retry count and terminal failure or warning reasons.

A core-valid observation requires all of the following:

- a finite positive pinned spot and an eligible expiration;
- a successful provider fetch begun after the source equity run published and
  completed during the current options run;
- at least five normalized calls and five normalized puts with finite positive
  strikes;
- positive aggregate open interest on both call and put sides; and
- at least three distinct strikes with usable open interest across the two
  sides.

Contract `lastTradeDate` values are coverage evidence, not a chain-wide
freshness gate because many valid contracts do not trade every session. If no
retained contract traded in either of the prior two completed US sessions, the
item receives a stale-trades warning. Individual secondary metrics may still
be unavailable when their narrower IV, quote, volume, price-history, or model
inputs are insufficient.

The item states exposed to clients are:

- `available`;
- `building_history` for a valid current observation lacking historical depth;
- `insufficient_quality` with reasons;
- `unavailable` with reasons;
- `stale` only when serving a previously Published Options Run.

Zeros remain zeros. Null, invalid, unavailable, insufficient-quality, and stale
states remain distinct.

## Persistence and publication

Persistence uses four focused concepts rather than independent Max Pain, GEX,
IV, and command-center snapshot systems.

### Options runs

An Options Analytics Run stores:

- market and provider;
- source equity run ID and date;
- candidate-policy, expiration-policy, schema, and calculation versions;
- deterministic input signature;
- lifecycle status;
- Current and Continuity expected/success/failure counts;
- Current Candidate core-valid coverage;
- start, completion, and publication timestamps;
- run-level failure and diagnostic summaries.

An idempotency constraint prevents duplicate active/successful runs for the
same input signature and versions. A forced administrative rerun must create a
new explicit attempt identity rather than bypass natural keys.

### Run items

There is exactly one item per `(run_id, security_identity)`. It stores candidate
kind, both source ranks, both source flags, observation state, selected
expiration, current aggregate metrics, historical-readiness metadata, quality
evidence, and failure reasons. Frequently ranked fields are typed columns;
versioned diagnostic details may use a structured JSON field.

### Strike points

Bounded strike points belong to one successful run item and contain the values
required for open-interest, volume, estimated-GEX, and IV-smile charts. There is
one row per `(item_id, strike)`. Raw Yahoo responses are not persisted.

### Published pointer

One market- and calculation-version-specific pointer identifies the Published
Options Run. Pointer advancement and publication status change occur in one
database transaction after quality evaluation.

The publication denominator is the Current Candidate cohort only. Continuity
success is reported separately and cannot prevent an otherwise valid current
ranking from publishing. A run publishes only when the Current Candidate cohort
is non-empty and `core_valid_current_count / current_candidate_count >= 0.90`.
Below 90%, or for an empty cohort, the run finishes as failed quality, the prior
pointer remains unchanged, and clients explicitly report that the prior
published run is stale relative to the source equity run.

Published aggregate run items are retained for 252 US trading sessions.
Bounded strike rows are retained for the most recent 30 Published Options Runs.
Failed or abandoned staged runs may be removed after 30 calendar days.
Retention deletes only data not referenced by the current published pointer and
never mutates a Published Options Run in place.

## Task orchestration and Operations

The existing US daily pipeline remains authoritative for prices, breadth,
groups, and the published equity feature run. After the equity snapshot guard
succeeds, it dispatches one options-refresh task to the existing
`data_fetch_us` queue. The options task is external-fetch work and participates
in current data-fetch locking, external-fetch coordination, cancellation, and
rate-budget behavior.

The task calls one in-process application use case. It does not launch scripts
or subprocesses and does not split Max Pain, GEX, and options analysis into
separate provider passes.

An options failure cannot roll back or invalidate the already-published equity
run. It records its own failed state and leaves the previous options pointer in
place. The later static export can therefore publish fresh equity data with an
explicitly stale prior options bundle.

The Operations surface and scheduled-task registry expose one logical Daily US
Options Analytics job and one manual refresh action. Progress reports expected,
completed, valid, failed, retried, and current publication coverage counts.
Failed items may be resumed within the same staged run without recalculating
successful items. Final serialized task results contain only JSON-safe values.

## API contracts and security

The live API exposes a small protected surface under one options-analytics
namespace:

- read the Published Options Run manifest and Command Center rows;
- read one current symbol's aggregate, strike, history, and quality detail;
- read run diagnostics for Operations-authorized users;
- trigger or resume a refresh for Operations-authorized users.

Reads never wait on Celery results and never invoke Yahoo. Refresh endpoints
return accepted task/run identities immediately. All live routes use existing
authentication and authorization; there are no anonymously writable or
operational endpoints.

Response models are explicit, strict, and schema-versioned. Summary aggregates
are calculated over the complete filtered Published Options Run before any
pagination. The Command Center has at most 80 rows, so its public read contract
may return the complete current set for client-side ranking and filtering.

Production endpoints never return mock, synthetic, or fabricated market data.

## Static-site contract

Static mode never calls Yahoo or the live API. The exporter reads the Published
Options Run and emits split, versioned artifacts:

```text
options/manifest.json
options/command-center.json
options/symbols/<url-safe-symbol-key>.json
```

The manifest contains schema/calculation versions, published run identity,
source equity run/date, options observation times, coverage, provider, stale
state, and symbol-detail paths. The manifest maps canonical SecurityMaster
identity to its URL-safe symbol-detail key so clients never construct paths by
guessing. The Command Center artifact contains one summary for every Current
Candidate, up to 80, including explicit unavailable states. Continuity-only
candidates are not listed or exported unless they return to current membership.

Each current symbol artifact contains its aggregate metrics, bounded strike
points, quality evidence, and compatible rolling historical observations. If a
ticker returns after an absence, that day's symbol artifact again includes its
preserved compatible history and visible date gaps.

The static exporter must either emit a complete internally consistent options
artifact set or retain/restore the last-good set. It never mixes manifest,
summary, or symbol files from different options runs. Artifact validation
checks schema version, run identity, path coverage, candidate count, duplicate
symbols/strikes, non-finite numbers, and stale metadata before publication.

Equity `as_of_date` and options observation time are displayed separately. A
fresh equity export with a retained prior options run displays a prominent
options-stale banner rather than blocking the entire static site.

## User experience

The feature is one Options workspace:

- `/options` opens the Command Center.
- `/options/:symbol` opens ticker-level options detail.

The Command Center header shows source equity date/run, options observation
time, provider, valid coverage, calculation version, and stale/building-history
status. It displays source badges and ranks so users can see whether a security
came from Top Scan Candidates, Leaders in Leading Groups, or both.

One sortable/filterable table supports focused views for structural gamma,
volatility, skew, and activity instead of duplicating the same cohort across
many independent tables. Core row information includes security identity,
equity source/rank, selected expiration, Max Pain distance, Estimated Gamma Flip
distance, estimated net GEX, put/call OI, ATM IV, volatility risk premium, skew,
activity measures, quality, and freshness. Unavailable values render a neutral
marker with an accessible reason.

Ticker detail shows:

- current summary and explicit model-estimate labels;
- open-interest/volume and estimated-GEX strike views;
- the one-expiration IV smile;
- compatible historical aggregate observations with genuine gaps;
- observation-count/history-readiness information;
- provider, assumptions, freshness, coverage, and warnings.

The live page may offer an authorized refresh action. The static page is
read-only and explains that its data comes from the exported Published Options
Run. Both modes share presentation components and contract adapters while
retaining separate live/static clients in line with current repository
patterns.

Components use the current MUI theme, responsive layout, keyboard-accessible
row navigation, and request identity/cancellation guards so a late response for
one ticker cannot overwrite another ticker's detail.

## Error handling

Provider and calculation failures use stable internal reason codes grouped as:

- unsupported or missing expiration;
- provider throttling/transient transport failure;
- provider schema/incomplete-chain failure;
- stale observation;
- insufficient call/put/open-interest/IV coverage;
- invalid source spot or material source/provider disagreement;
- calculation assumption unavailable;
- persistence conflict;
- cancellation.

Expected per-symbol failures do not throw away other staged results. Unexpected
run-level failures mark the run failed and preserve the prior pointer. Database
errors roll back the current unit of work before retry or failure recording.

No request handler catches an error and returns a successful fabricated
response. No API or test makes an implicit localhost Redis connection.

## Verification strategy

### Domain tests

- Golden synthetic chains for Max Pain, estimated GEX, genuine/no gamma
  crossings, walls, ATM IV, skew, volatility risk premium, and activity.
- Boundary cases for empty sides, absent columns, null versus zero, invalid IV,
  zero denominators, non-regular contracts, stale trades, expiration-day math,
  and non-finite serialization.
- Property/invariant tests for stable ordering, payoff minima, sign-crossing
  interpolation, and deterministic strike truncation.

### Selection and history tests

- Exact Top 40 plus Top 40 selection with the strict greater-than USD 100
  million floor.
- Deduplication and preservation of both ranks/source flags.
- Five-trading-session continuity, deterministic 20-symbol cap, return to
  current membership, and exclusion of continuity rows from rankings.
- A ticker dropping and returning without history deletion or counter reset.
- Rolling recent-observation eligibility, aging observations, gaps, and
  incompatible calculation versions.

### Application and persistence tests

- At most three fetch attempts per selected symbol, with only the first
  successful selected-expiration payload consumed as one observation.
- Per-symbol retry/resume without recomputing successful items.
- Input-signature idempotency and concurrent-run exclusion.
- Exactly-at, above, and below 90% Current Candidate publication coverage.
- Continuity failure not affecting the publication denominator.
- Atomic pointer advancement and prior-pointer retention on every failure path.
- Natural-key uniqueness, transaction rollback, retention safety, and strict
  response serialization.

### Interface tests

- Protected live reads and authorized manual refresh.
- Read endpoints never invoking provider or waiting for Celery.
- Complete-set summary calculations and stable client sorting.
- Static/live contract parity and static last-good fallback.
- Manifest/summary/symbol run-identity validation and corrupted-artifact
  rejection.
- Frontend rendering for available, building-history, insufficient-quality,
  unavailable, and stale states; ticker navigation races; keyboard access; and
  static refresh restrictions.

CI uses newly created synthetic or recorded-normalized fixtures and never calls
Yahoo. A separate manual smoke check queries a very small public ticker set to
verify the adapter contract. Existing backend, frontend, migration, static
workflow, and Compose CI suites must remain green.

## Rollout

1. Merge schema, domain calculations, selection/history policy, and provider
   adapter behind `OPTIONS_ANALYTICS_ENABLED=false`.
2. Run migrations and execute a manual US canary against a small cohort.
3. Inspect request rate, optionability, metric availability, observation
   freshness, and quality diagnostics.
4. Run a full Current plus Continuity cohort canary and require the 90% gate.
5. Enable the single US daily follow-on for the intended deployment.
6. Produce and validate a static artifact set from a Published Options Run.
7. Enable static navigation only after a valid manifest is present.

Rollback disables collection and navigation without deleting history. The
published pointer remains available for diagnosis or a later compatible
release.

## Acceptance criteria

- A published US equity run deterministically produces no more than 80 Current
  Candidates and no more than 20 Continuity Candidates.
- Every selected symbol contributes at most one successful selected-expiration
  chain payload and one Chain Observation to an Options Analytics Run, even
  when retries occur.
- The same Chain Observation feeds all current metrics for that item.
- The Command Center contains every Current Candidate and no Continuity-only
  symbol; each metric ranking excludes rows where that metric is unavailable.
- A ticker can drop out and return without losing compatible stored history;
  history readiness is based on recent valid observations, not consecutive
  membership.
- A run below 90% Current Candidate core-valid coverage cannot move the
  published pointer.
- Live and static Command Centers render the same Published Options Run
  contract, with separate equity and options timestamps.
- Static mode performs no provider or live-API requests and can retain a
  last-good options artifact set with an explicit stale state.
- Yahoo limitations are visible: GEX-family values are model estimates and no
  output claims net premium flow or trade direction.
- The implementation adds no new worker family, subprocess batch pipeline,
  pgAdmin/deployment change, mock production response, or unrelated endpoint.
- All new tests and the repository's existing CI gates pass.
