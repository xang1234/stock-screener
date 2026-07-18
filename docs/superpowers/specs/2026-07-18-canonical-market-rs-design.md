# Canonical Market RS and Group Rankings

**Date:** 2026-07-18

**Status:** Approved design — awaiting written-spec review

**Formula version:** `balanced-horizon-percentile-v2`

## Problem

Relative Strength (RS) is not currently one canonical fact across the product.
The Scan pipeline normally percentile-ranks stocks within the rows it prepared,
the live Group path linearly scales weighted raw excess return, and the static
Group page independently aggregates RS values serialized from a feature run.
Consequently, the same Market/date can produce different stock and Group RS
values across Scan, live Groups, and static Groups.

The legacy weighted-return formula also allows an extreme long-period return to
dominate the composite. For example, a 1,000% 12-month excess return contributes
150 percentage points when multiplied by a 15% weight, easily overwhelming a
recent 30–50% decline. This conflicts with the desired behavior: recent weakness
must be visible without discarding longer-term leadership context.

## Goals

1. Define one Market-wide RS calculation used by Scan, live Groups, static
   Groups, and all downstream RRG inputs.
2. Normalize each time horizon before weighting so an extreme raw return cannot
   overwhelm the other horizons.
3. Add 1-month and 3-month Group RS columns to both live and static Group tables.
4. Preserve overall Group RS as the sole basis of the main Group Rank.
5. Recalculate all available Group history and rebuild RRG without mixing legacy
   and new formulas.
6. Publish formula/date/universe metadata so RS values can be audited.
7. Update the root README and detailed live-app documentation.

## Non-goals

- Reproduce IBD's undisclosed proprietary formula exactly.
- Add a separate actionable or momentum-adjusted Group Rank.
- Change the RRG transformation mathematics.
- Display separate 6-month or 9-month columns in Scan or Groups.
- Allow RS to change according to an ad hoc scan, index, or watchlist universe.

## Canonical universe and eligibility

RS is calculated independently for each Market and trading date.

The comparison universe is every security that is active in that Market as of
the calculation date and has:

- a valid split-adjusted close for the Market's resolved as-of session;
- valid split-adjusted closes at the 21-, 63-, 126-, 189-, and 252-Market-
  session anchors; and
- a current, valid Market benchmark series at the same anchors.

The canonical price field is `adj_close`; ingestion's existing normalized
fallback to `close` applies when a vendor does not provide a distinct adjusted
value. Lookbacks are resolved from the Market's session calendar, not by taking
an arbitrary number of rows from each stock's series. A missing anchor makes
that stock ineligible for the entire run.

The same eligible set is used for overall, 1M, 3M, 6M, 9M, and 12M ratings. A
stock is never present in only the short-period distributions. This keeps Group
overall/1M/3M averages and their displayed constituent count directly
comparable.

Historical runs use the point-in-time Market universe recorded for that date.
They must not silently substitute today's active universe. Missing point-in-time
universe or benchmark inputs make that historical run unavailable and produce an
explicit coverage failure.

## Stock RS calculation

### Excess-return horizons

For every eligible stock, calculate cumulative excess return at these Market
session horizons:

| Horizon | Sessions | Composite weight |
| --- | ---: | ---: |
| 1M | 21 | 20% |
| 3M | 63 | 30% |
| 6M | 126 | 20% |
| 9M | 189 | 15% |
| 12M | 252 | 15% |

For horizon `h`:

```text
stock_return(h) = current_stock_close / prior_stock_close(h) - 1
benchmark_return(h) = current_benchmark_close / prior_benchmark_close(h) - 1
excess_return(h) = stock_return(h) - benchmark_return(h)
```

Raw excess returns are not combined directly.

Every stock in one Market/date shares the same benchmark return. Subtracting
that constant therefore does not change the cross-sectional order: ranking
excess return produces the same component ratings as ranking stock return. The
benchmark-relative values are retained for semantic consistency and audit, but
the protection against extreme winners comes from percentile normalization.

### Per-horizon percentile normalization

Rank each horizon's excess returns across the eligible Market universe. Ties use
their average ascending rank. Convert an average rank to an integer rating from
1 through 99:

```text
rating(rank, N) =
  1 + floor(98 * (rank - 1) / (N - 1) + 0.5)
```

`N` must be at least two. With unique extrema, the worst observation maps to 1
and the best maps to 99. Tied observations receive the same average-rank
rating; if every observation is tied, all map to 50. This produces `P1`, `P3`,
`P6`, `P9`, and `P12`.

### Balanced composite and final percentile

```text
composite =
    0.20 * P1
  + 0.30 * P3
  + 0.20 * P6
  + 0.15 * P9
  + 0.15 * P12
```

Rank `composite` across the same eligible Market universe with the same average-
tie and 1–99 mapping. That final rating is the stock's overall RS.

Normalizing first bounds every horizon's influence. A 1,000% 12M return and a
500% 12M return can both become `P12 = 99`; neither contributes more than
`0.15 * 99 = 14.85` composite points. Weak 1M and 3M readings jointly control
50% of the composite.

The canonical snapshot stores overall RS plus `P1`, `P3`, `P6`, `P9`, and
`P12`. Scan continues to display overall, 1M, 3M, and 12M only.

## Canonical snapshot architecture

RS becomes a versioned daily Market fact calculated once after price refresh.

### Market RS run

A run records:

- Market and as-of date;
- formula version;
- benchmark symbol and benchmark as-of date;
- point-in-time universe hash;
- eligible, excluded, and expected-symbol counts;
- coverage/freshness diagnostics;
- lifecycle state (`running`, `completed`, or `failed`); and
- creation/completion timestamps.

### Stock RS rows

Each completed run owns one row per eligible symbol containing:

- overall RS;
- RS 1M, 3M, 6M, 9M, and 12M; and
- the pre-final weighted percentile composite for audit/debugging.

Rows for a run and the transition to `completed` are committed atomically.
Consumers only resolve completed runs. A failed or incomplete run can never
replace the last published run.

An active-formula pointer is maintained per Market. Shadow history for
`balanced-horizon-percentile-v2` can therefore coexist with
`legacy-linear-v1` until validation and activation are complete.

## Consumer contract

For a requested Market/date, every RS consumer resolves the same completed run
belonging to the Market's active formula version.

- Full-market, index, and watchlist Scan results use the same stored ratings.
- A smaller requested scan universe never becomes the percentile universe.
- Live Group calculation aggregates the canonical stock rows.
- Static Group export serializes the canonical stored Group snapshot.
- RS-dependent work fails explicitly when an exact required snapshot is absent;
  it never falls back to legacy linear scaling.

Published live and static payloads expose the formula version, RS as-of date,
and eligible-universe size.

## Group calculation

For a Market/date, intersect the canonical stock RS rows with active, mapped
Group constituents. Require at least three eligible constituents.

```text
Group RS    = arithmetic mean of constituent overall RS
Group 1M RS = arithmetic mean of constituent RS 1M
Group 3M RS = arithmetic mean of constituent RS 3M
```

All three metrics use the same constituent set. Existing metrics are retained:

- median and population standard deviation of overall RS;
- market-cap-weighted overall RS;
- count and percentage of constituents with overall RS at least 80; and
- the top constituent by overall RS.

Main Group Rank is determined only by the unrounded Group RS. Exact Group RS
ties are ordered by industry-group name for deterministic display; no secondary
performance metric contributes to the main rank. Top-stock ties are resolved
deterministically by overall RS, RS 1M, market capitalization, and symbol.

The stored Group row gains:

- `avg_rs_rating_1m`;
- `avg_rs_rating_3m`;
- `rs_formula_version`; and
- the canonical Market RS run identifier.

## Live and static Group surfaces

The live table inserts the new columns immediately after overall RS and keeps
its existing metrics:

```text
Rank | Group | RS | 1M RS | 3M RS | Med RS | Wtd RS | ...
```

The static table inserts the new columns immediately after Avg RS and preserves
its existing stock-count, rank-change, and top-stock columns:

```text
Rank | Group | Avg RS | 1M RS | 3M RS | Stocks | 1W | 1M | 3M | 6M | Top Stock
```

Live 1M/3M RS columns follow the existing sortable-column behavior. Static
Groups adds the values without introducing a separate client-side sort model.

The static exporter stops independently recomputing current Group rankings from
feature rows. It requires and serializes the exact canonical Group snapshot for
the export Market/date. Missing snapshots make the Groups section explicitly
unavailable instead of producing divergent values.

The additive fields and changed semantics require a new static schema version.
The static manifest and Groups/RRG artifacts include the RS formula version.

## Daily pipeline

The ordered daily path becomes:

```text
Price refresh
  -> Canonical Market RS snapshot
  -> Group snapshot
  -> Feature/Scan snapshot
  -> Static export and RRG consumers
```

Downstream stages require the exact successful upstream Market/date. Manual
Group refresh invokes or resolves the canonical Market RS snapshot first.

When a new Group snapshot publishes, the live Groups page invalidates and
refetches its bootstrap query before refetching date-pinned rankings. This fixes
the existing behavior where an open page can continue polling an old
`as_of_date`.

## Historical migration and RRG

The legacy and balanced formulas remain isolated throughout migration.

Backfill sequence:

1. Build versioned canonical Market stock RS snapshots for every available
   trading date with valid point-in-time inputs.
2. Build versioned Group overall/1M/3M history from those snapshots.
3. Validate coverage, ranges, deterministic ranks, missing groups, and
   live/static parity.
4. Rebuild RRG using only balanced-formula Group history.
5. Regenerate static artifacts under the new schema version.
6. Atomically activate the balanced formula per Market.

RRG mathematics remains unchanged: weekly bucketing, 5-week RS-Ratio EMA,
26-week temporal z-score, 4-week momentum change, 3-week momentum EMA, and
13-week momentum z-score. RRG providers filter to one formula version. A static
rolling-history bundle whose formula version differs from the requested formula
is invalidated and rebuilt rather than merged.

Legacy history remains available for rollback until the balanced rollout is
accepted. A failed Market backfill leaves that Market on its prior active
formula.

## Failure and quality policy

A canonical run is not published when:

- the Market benchmark is missing or stale for the resolved session;
- the point-in-time universe is unavailable;
- price coverage fails the existing daily Market refresh coverage gate;
- fewer than two eligible stocks remain for percentile calculation;
- output contains values outside 1–99 or non-finite composites; or
- the expected symbol count does not match the atomically written row count.

The previous completed snapshot remains active. Diagnostics record exclusions
by reason so operators can distinguish insufficient 12M history from stale or
missing price data.

## Testing strategy

Implementation follows red-green-refactor TDD.

### Calculation tests

- Verify the balanced weights exactly equal 100%.
- Verify adjusted-close inputs and exact Market-session anchors.
- Verify each horizon is percentile-normalized before weighting.
- Verify average-rank tie handling and deterministic integer 1–99 mapping.
- Verify subtracting the common benchmark return does not change a horizon's
  percentile ratings.
- Verify a 1,000% raw return cannot exceed its percentile horizon's weighted
  contribution.
- Verify severe 1M/3M deterioration materially lowers the composite of a former
  long-term winner.
- Verify Market isolation and the common 12M eligibility set.
- Verify the final overall RS is a percentile of the weighted component score.

### Persistence and service tests

- Verify incomplete/failed runs cannot replace completed runs.
- Verify exact Market/date/formula resolution.
- Verify full-market, index, and watchlist consumers return the same symbol RS.
- Verify Group overall/1M/3M averages share one constituent set.
- Verify deterministic Group and top-stock tie resolution.
- Verify formula versions cannot mix in Group or RRG history.

### Parity, API, and frontend tests

- Golden-test identical live and static Group payload values for a fixed
  Market/date.
- Golden-test identical live and static RRG output for identical versioned
  history.
- Test new live sortable 1M/3M columns and static 1M/3M columns.
- Test missing-data rendering and bootstrap formula/date rollover.
- Test migration upgrade/downgrade and static schema-version enforcement.

## Documentation

Update:

- `README.md`;
- `docs/LIVE_APP_GUIDE.md`;
- Scan field definitions;
- Group and RRG methodology; and
- migration/release notes.

Documentation must explicitly state that overall RS weights per-horizon Market
percentiles, not raw returns, and that overall Group RS averages constituent
overall stock RS while Group 1M/3M average the matching component ratings.

## Acceptance criteria

1. A stock has one identical overall/1M/3M/12M RS for a Market/date across Scan,
   live Groups, static Groups, index scans, and watchlist scans.
2. Every visible stock RS is an integer from 1 through 99; Group averages may be
   decimal.
3. Live and static Group tables show Group 1M RS and Group 3M RS.
4. Main Group Rank remains based only on average overall RS.
5. A synthetic 1,000% former winner cannot dominate through raw magnitude; each
   horizon is bounded by its percentile and configured weight.
6. Balanced and legacy histories never mix in RRG.
7. Static and live Group/RRG golden fixtures match for the same versioned input.
8. All available valid history is rebuilt before formula activation.
9. README and detailed application documentation describe the new formula and
   fields.
