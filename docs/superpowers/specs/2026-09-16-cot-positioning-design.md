# Commitments of Traders Positioning Design

**Date:** 2026-09-16

**Status:** Approved in design review

## Context

The application has two Daily experiences:

- the live FastAPI/PostgreSQL application, whose Daily page uses vertical tabs;
- the read-only static site, whose Daily page is rendered from exported JSON and
  must not call live APIs.

Both experiences need Commitments of Traders (COT) positioning for a curated
set of physical commodities and financial futures. The feature must retain
official history, expose long, short, and net positions, calculate a three-year
positioning percentile, and add best-effort price context without introducing a
new price provider.

The attached Ariel and table screenshots are visual references only. They do
not define data, calculation, or implementation requirements.

## Goals

- Use official open CFTC data as the canonical positioning source.
- Support both physical commodities and financial futures using the appropriate
  CFTC report family.
- Backfill the complete supported history from 2006 and update it weekly.
- Retain long, short, spreading, open interest, net, weekly changes, and a
  rolling three-year net-position percentile.
- Present one shared, consistent feature in live and static modes.
- Use only the existing Yahoo Finance integration for machine-readable prices,
  with optional TradingView links for instruments Yahoo cannot cover.
- Keep price enrichment best effort: price failure must never suppress valid
  COT data.
- Publish data atomically and retain the last validated snapshot when imports
  fail.

## Non-goals

- Intraday or real-time positioning. CFTC positions are weekly and dated as of
  Tuesday.
- Individual trader positions, transaction-level data, or inferred trade
  direction.
- Trading recommendations or a predictive conviction score.
- Every contract in the CFTC catalogue.
- Regional electricity, emissions, natural-gas basis, specialist spreads,
  duplicate micro contracts, or minor digital-asset perpetuals.
- A new commercial or open price feed.
- Scraping or extracting machine-readable price history from TradingView.
- A full COT table in the static site.

## Official sources and date semantics

Physical commodities use the CFTC **Disaggregated Futures Only** dataset,
dataset ID `72hh-3qpy`. Financial futures use **Traders in Financial Futures
(TFF) Futures Only**, dataset ID `gpe5-46if`.

Authoritative references:

- [CFTC Public Reporting Environment](https://publicreporting.cftc.gov/stories/s/r4w3-av2u)
- [Disaggregated COT reference](https://publicreporting.cftc.gov/stories/s/ubmb-6exi)
- [TFF Futures Only dataset](https://publicreportinghub.cftc.gov/Commitments-of-Traders/TFF-Futures-Only/gpe5-46if)
- Programmatic Socrata resources:
  [`72hh-3qpy`](https://publicreporting.cftc.gov/resource/72hh-3qpy.json) and
  [`gpe5-46if`](https://publicreporting.cftc.gov/resource/gpe5-46if.json)

These public datasets contain the same underlying COT data as the traditional
CFTC reports. Disaggregated and TFF history begins in 2006. Each observation's
`report_date` is the Tuesday position date, not the later publication or import
date.

Only Futures Only data is in scope. Futures-and-options-combined and Legacy COT
must not be mixed into the series.

## Domain language

**COT Instrument**: One curated CFTC contract market identified by its stable
CFTC contract-market code.

**COT Report Week**: One Tuesday-dated CFTC observation for a COT Instrument.

**Participant Position**: Long, short, and spreading contracts for one CFTC
participant category in one COT Report Week.

**Focal Participant**: The comparable speculative category used in the summary
strip and live table: Managed Money for Disaggregated COT and Leveraged Funds
for TFF.

**COT Publication**: The latest complete, validated import selected for live and
static serving.

**Exact Price**: A Yahoo continuous futures series representing the selected
COT market directly.

**Price Proxy**: A Yahoo ETF or index used only as contextual price history and
explicitly labelled as a proxy.

## Curated instrument universe

The registry is ordered. UI consumers preserve this order unless the user
chooses a table sort. A restore action returns to curated order.

### 1. Equity and volatility

| Display name | CFTC contract | CFTC code |
|---|---|---|
| S&P 500 | S&P 500 Consolidated | `13874+` |
| Nasdaq-100 | NASDAQ-100 Consolidated | `20974+` |
| Russell 2000 | RUSSELL E-MINI | `239742` |
| MSCI EAFE | MSCI EAFE | `244041` |
| MSCI Emerging Markets | MSCI EM INDEX | `244042` |
| Nikkei | NIKKEI STOCK AVERAGE YEN DENOM | `240743` |
| VIX | VIX FUTURES | `1170E1` |

### 2. Rates

| Display name | CFTC contract | CFTC code |
|---|---|---|
| 1-Month SOFR | SOFR-1M | `134742` |
| 3-Month SOFR | SOFR-3M, CME | `134741` |
| US Treasury 10-Year | UST 10Y NOTE | `043602` |

The FMX 3-month SOFR duplicate is excluded.

### 3. Energy

| Display name | CFTC contract | CFTC code |
|---|---|---|
| WTI Crude | WTI-PHYSICAL | `067651` |
| Brent Crude | BRENT LAST DAY | `06765T` |
| Henry Hub Natural Gas | NAT GAS NYME | `023651` |
| RBOB Gasoline | GASOLINE RBOB | `111659` |
| ULSD / Heating Oil | NY HARBOR ULSD | `022651` |

### 4. Currencies

| Display name | CFTC contract | CFTC code |
|---|---|---|
| US Dollar Index | USD INDEX | `098662` |
| Japanese Yen | JAPANESE YEN | `097741` |

### 5. Digital assets

| Display name | CFTC contract | CFTC code |
|---|---|---|
| Bitcoin | BITCOIN, CME | `133741` |
| Ether | ETHER CASH SETTLED | `146021` |

### 6. Metals

| Display name | CFTC contract | CFTC code |
|---|---|---|
| Gold | GOLD | `088691` |
| Silver | SILVER | `084691` |
| Copper | COPPER- #1 | `085692` |
| Platinum | PLATINUM | `076651` |
| Palladium | PALLADIUM | `075651` |

### 7. Grains and oilseeds

| Display name | CFTC contract | CFTC code |
|---|---|---|
| Corn | CORN | `002602` |
| Wheat | WHEAT-SRW | `001602` |
| Soybeans | SOYBEANS | `005602` |
| Canola | CANOLA | `135731` |

Only Chicago SRW Wheat and standard Soybeans are used. Other wheat varieties,
soybean meal, soybean oil, and mini soybean contracts are excluded.

### 8. Softs

| Display name | CFTC contract | CFTC code |
|---|---|---|
| Coffee | COFFEE C | `083731` |
| Cocoa | COCOA | `073732` |

Sugar, cotton, and orange juice are excluded.

### 9. Lumber

| Display name | CFTC contract | CFTC code |
|---|---|---|
| Lumber | LUMBER | `058644` |

Livestock is excluded.

## Participant categories

Disaggregated COT stores and charts:

- Producer/Merchant/Processor/User;
- Swap Dealer;
- Managed Money;
- Other Reportables;
- Non-reportables.

TFF stores and charts:

- Dealer/Intermediary;
- Asset Manager/Institutional;
- Leveraged Funds;
- Other Reportables;
- Non-reportables.

The summary strip and live table use Managed Money for physical commodities and
Leveraged Funds for financial futures. The UI labels the focal category by its
actual CFTC name; it does not rename both categories to a synthetic shared name.

## Metric definitions

For each participant and COT Report Week:

- `net = long - short`;
- `delta_long = long - previous_week.long`;
- `delta_short = short - previous_week.short`;
- `delta_net = net - previous_week.net`;
- `net_pct_open_interest = 100 * net / open_interest` when open interest is
  positive.

The three-year percentile ranks current net positioning against the most recent
156 non-null weekly net observations for the same instrument and participant,
including the current observation. It is:

`100 * (count(value < current) + 0.5 * count(value == current)) / 156`.

The displayed percentile is rounded to the nearest whole percentile. Fewer
than 156 usable observations produces `insufficient_history`; the calculation
must not silently shorten the window.

The percentile is descriptive historical context. It is not combined with
price, open interest, or another factor into a proprietary score.

## Price policy and mappings

Price history comes only from the existing Yahoo Finance service and price
cache. Live page requests never fetch Yahoo. The refresh pipeline prefetches
and stores price history before publication.

The refresh pipeline calls the existing internal Yahoo/price-cache boundary
directly. It does not route futures symbols such as `ES=F` through the public
stock-history endpoint or its stock-symbol validation.

| COT market | Yahoo symbol | Classification |
|---|---|---|
| S&P 500 | `ES=F` | Exact future |
| Nasdaq-100 | `NQ=F` | Exact future |
| Russell 2000 | `RTY=F` | Exact future |
| MSCI EAFE | `EFA` | ETF proxy |
| MSCI Emerging Markets | `EEM` | ETF proxy |
| Nikkei | `NKD=F` | Exact future |
| VIX | `^VIX` | Index proxy |
| 1-Month SOFR | `SGOV` | ETF proxy |
| 3-Month SOFR | `BIL` | ETF proxy |
| US Treasury 10-Year | `ZN=F` | Exact future |
| WTI Crude | `CL=F` | Exact future |
| Brent Crude | `BZ=F` | Exact future |
| Henry Hub Natural Gas | `NG=F` | Exact future |
| RBOB Gasoline | `RB=F` | Exact future |
| ULSD / Heating Oil | `HO=F` | Exact future |
| US Dollar Index | `DX-Y.NYB` | Index proxy |
| Japanese Yen | `6J=F` | Exact future |
| Bitcoin | `BTC=F` | Exact future |
| Ether | `ETH=F` | Exact future |
| Gold | `GC=F` | Exact future |
| Silver | `SI=F` | Exact future |
| Copper | `HG=F` | Exact future |
| Platinum | `PL=F` | Exact future |
| Palladium | `PA=F` | Exact future |
| Corn | `ZC=F` | Exact future |
| Wheat | `ZW=F` | Exact future |
| Soybeans | `ZS=F` | Exact future |
| Canola | none | TradingView link only |
| Coffee | `KC=F` | Exact future |
| Cocoa | `CC=F` | Exact future |
| Lumber | `LBR=F` | Exact future with partial history |

Canola may link to TradingView `ICEUS:RS1!`, but no TradingView data is copied
into app payloads. A missing price produces a normal, explicit unavailable
state. Lumber price history begins later than its COT history and is labelled
as partial when the selected chart range predates Yahoo coverage.

For each COT Report Week, price alignment uses the latest Yahoo close on or
before the Tuesday report date. Weekly price change compares that aligned close
with the aligned close for the previous COT Report Week. It is therefore
Tuesday-to-Tuesday rather than latest-close or Friday-to-Friday.

Continuous futures may contain contract-roll discontinuities. The UI describes
the price line and weekly change as context, and price never enters the
positioning percentile.

## Architecture

COT is a dedicated bounded feature with one canonical stored history:

1. CFTC adapters fetch and normalize Disaggregated and TFF Futures Only rows.
2. A versioned registry selects the curated CFTC codes and provides product
   grouping, ordering, participant policy, Yahoo mapping, and TradingView link.
3. An import use case stages and validates the new or corrected report weeks.
4. A calculator derives net, weekly changes, open-interest percentage, and
   rolling percentiles.
5. Price enrichment reads through the existing Yahoo service and persistent
   price cache.
6. A publication transaction advances the served COT snapshot only after COT
   validation succeeds.
7. Live query services and the static exporter serialize the same canonical
   stored observations.
8. Shared React presentation components render live and static data clients.

Domain calculations must not import SQLAlchemy, FastAPI, Celery, Redis,
yfinance, or React.

## Persistence

### `cot_instruments`

- stable slug;
- CFTC contract-market code;
- display name;
- category and position within category;
- report family (`disaggregated_futures_only` or `tff_futures_only`);
- focal participant;
- active flag;
- registry version.

The versioned code registry is the authoring source. Import synchronization
materializes it into the table for referential integrity and historical audit.

### `cot_weekly_positions`

- instrument ID;
- report date;
- participant category;
- long, short, spreading, and open interest;
- net, weekly changes, net/open-interest percentage, and three-year percentile;
- source dataset ID and source row ID;
- import-run ID;
- created and updated timestamps.

The natural uniqueness constraint is instrument, report date, and participant
category.

### `cot_import_runs`

- run ID and status;
- source fetch and completion timestamps;
- source dataset metadata;
- requested and observed report dates;
- coverage counts and validation diagnostics;
- registry and calculation versions;
- published timestamp or failure reason.

Price observations remain in the existing price cache. The COT response
serializer records aligned price values and exact/proxy/partial/unavailable
status; it does not create a second general price-history store.

## Refresh and publication

One Celery task checks both CFTC datasets at 17:00 America/New_York on US
business evenings. If no new or revised rows exist, the task exits successfully
without rewriting the publication.

The initial administrative backfill imports complete available history from
2006. Later runs are incremental but must detect revised historical source rows
using stored source values or deterministic row fingerprints.

For a new or corrected report:

1. fetch both official datasets;
2. select rows by the registry's exact CFTC codes;
3. stage the complete curated snapshot;
4. validate coverage, uniqueness, field values, participant structure, and
   dataset identity;
5. upsert raw positions transactionally;
6. rebuild weekly changes and every percentile affected by a correction;
7. attempt price enrichment independently;
8. advance the COT publication pointer;
9. invalidate live COT caches.

The static exporter consumes only the published pointer. It may reuse the last
good published COT data when a later import attempt fails; it must never export
staged or partial rows.

## Quality gates and failure behavior

Publication requires:

- every active curated instrument expected in its report family;
- exactly one source row per instrument and report date;
- the expected participant categories for that report family;
- nonnegative long, short, spreading, and open-interest values;
- derived net values equal to long minus short;
- dataset-specific reported/non-reportable reconciliation;
- no unexplained source-history truncation;
- no automatic substitution of a new CFTC code for a missing registry code.

If CFTC fetch or validation fails, the previous publication remains active. If
one or all Yahoo mappings fail, valid COT data still publishes and each affected
price is marked unavailable. A corrected CFTC row is audited and causes all
dependent metrics to be rebuilt before publication.

The UI exposes report date, retrieval freshness, price mapping classification,
partial price coverage, and stale-data state. It never renders missing values
as zero.

## Live API

Protected read endpoints follow the Daily page's existing authentication
policy:

- `GET /v1/cot/instruments` returns the ordered catalog, participant labels,
  current availability, registry version, and price-mapping status.
- `GET /v1/cot/instruments/{slug}/history?range=1y|3y|5y` returns weekly
  participant positions, focal summary metrics, open interest, percentile, and
  aligned prices.
- `GET /v1/cot/snapshot` returns one latest focal-participant row per active
  instrument for the live table.

These endpoints read only the published COT data and stored price cache. They
must not call CFTC, Yahoo, or TradingView. Responses include schema version,
calculation version, report date, retrieval metadata, and freshness.

## Static contract

The root static manifest advertises the global optional asset as
`assets.cot.path = "cot/index.json"`. The data is independent of the selected
stock market, so Static Daily reads this root asset rather than a
market-specific asset. CFTC positioning covers US futures, so Static Daily
renders it only when the US market is selected. Static Daily remains valid when
older artifacts do not advertise it.

The export contains:

- `cot/index.json`: schema metadata, ordered catalog, freshness, and per-instrument
  history paths;
- `cot/<slug>.json`: at most five years of weekly COT history and available
  aligned prices for one instrument.

S&P 500 is the default and its file is prefetched. Other files load only when
selected. Static mode never calls `/api` and never fetches CFTC or Yahoo.

## User experience

### Live Daily page

`COT Positioning` is a lazy-loaded vertical tab in the existing left-hand Daily
tab rail. It appears after `Key Markets` and before `Themes`. It does not become
a top-level application route or horizontal tab.

The tab contains:

1. a category-grouped instrument dropdown, defaulting to S&P 500;
2. `Net` and `Long & Short` modes;
3. 1Y, 3Y, and 5Y range controls, defaulting to 1Y;
4. a summary strip for report date, focal long, focal short, focal net,
   three-year percentile, and aligned weekly price change;
5. a diverging positioning bar chart with a zero line;
6. a best-effort dashed price line on a separate right axis;
7. an open-interest mini-chart;
8. the complete latest-positioning table.

In Net mode, all participant net series are available and legend items toggle
visibility. In Long & Short mode, one selected participant is displayed with
long above zero and short below zero; the selected participant defaults to the
instrument's Focal Participant.

The live table shows, in curated category order:

- market;
- focal long and weekly change;
- focal short and weekly change;
- focal net and weekly change;
- focal net as percent of open interest;
- three-year percentile;
- compact net-position trend;
- Tuesday-to-Tuesday price change and price classification.

Columns remain sortable. Asset-class filtering is allowed. A restore action
returns rows to curated category order. Clicking a row selects that instrument
in the chart.

### Static Daily page

The COT section appears immediately after Market Health and before Top Scan
Candidates. It reuses the dropdown, mode toggle, range controls, summary strip,
positioning chart, price line, and open-interest mini-chart. It does not render
the full table.

### Presentation rules

- Exact, proxy, partial, and unavailable price states are visible in text.
- Signs and labels communicate direction independently of colour.
- Tooltips show source report date, participant, long, short, net, percentile,
  open interest, and price mapping.
- Charts include accessible descriptions and keyboard-operable controls.
- Responsive layouts may stack controls and summary metrics but preserve the
  left navigation in the live desktop layout.

## Caching and performance

Live API responses may use the existing Redis and bounded in-process cache
patterns. Cache keys include schema version, calculation version, publication
ID, instrument, and requested range. Publication invalidation removes old COT
keys.

The initial live tab load fetches catalog, S&P 500 history, and snapshot table
in bounded requests. Static mode loads the index and default S&P 500 history,
then lazy-loads other instruments. No page load downloads all five-year
instrument files.

## Operations and observability

Existing operational telemetry will expose:

- latest successful and failed COT import runs;
- source report dates and retrieval timestamps;
- expected versus observed instrument coverage;
- rejected rows and validation reasons;
- price exact/proxy/unavailable coverage;
- duration, retry count, and last published run;
- publication age and stale status.

Logs include dataset ID, import-run ID, registry version, calculation version,
and CFTC report date. They must not log full provider payloads unnecessarily.

## Testing

### Backend unit tests

- Parse representative Disaggregated and TFF source fixtures.
- Resolve exact CFTC codes and preserve curated ordering.
- Validate participant-category mappings.
- Calculate long, short, net, weekly changes, net/open-interest, and percentile
  boundary/tie cases.
- Enforce the 156-observation percentile requirement.
- Align Yahoo closes on or before Tuesday and calculate report-to-report price
  change.
- Label exact, proxy, partial, and unavailable price states.
- Rebuild metrics after historical corrections.

### Backend integration tests

- Initial backfill and incremental idempotency.
- Transactional publication and last-good fallback.
- Missing instrument, duplicate row, wrong dataset, and truncated history
  rejection.
- CFTC success with partial or total Yahoo failure.
- Live API authentication, schemas, range filtering, cache invalidation, and
  freshness.
- Static exporter output and optional-manifest compatibility.

### Parity tests

Given one published database snapshot, live and static serializers must emit
identical COT values, derived metrics, participant labels, price values, and
availability classifications for the same instrument and report weeks.

### Frontend tests

- Live COT appears in the left Daily tab rail after Key Markets.
- Static COT appears after Market Health and does not render the full table.
- S&P 500 and 1Y are the defaults.
- Curated groups and rows use the approved fixed order.
- Net and Long & Short modes render the correct series.
- 3Y and 5Y controls alter the visible range without changing percentile
  methodology.
- Physical and financial participant labels are correct.
- Row selection updates the live chart.
- Price proxy, partial, unavailable, stale, and insufficient-history states are
  visible.
- Keyboard navigation, accessible chart text, and responsive layout work.

## Acceptance criteria

- All 31 curated instruments have complete available CFTC history backfilled
  from 2006 or their later official inception.
- Every published observation passes the quality gates.
- Three consecutive refresh simulations succeed, including no-op and corrected
  source cases.
- Live/static parity tests pass.
- Page requests make no external CFTC, Yahoo, or TradingView data calls.
- COT still publishes when every Yahoo price lookup fails.
- The live left-tab placement and static-after-Market-Health placement match
  the approved mockups.
- Backend and frontend test suites pass.

## Implementation sequencing boundary

Implementation planning should separate work into vertical slices:

1. registry, domain models, source adapters, and calculations;
2. persistence, backfill, incremental refresh, and publication;
3. live APIs and static serialization/parity;
4. shared chart/table components and live/static integration;
5. operations, full verification, and documentation.

No implementation begins until this specification has been reviewed and the
implementation plan is approved.
