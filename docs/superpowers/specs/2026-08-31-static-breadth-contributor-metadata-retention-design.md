# Static Breadth Contributor Metadata Retention Design

**Date:** 2026-08-31

## Objective

Ensure every static Breadth contributor drilldown contains useful company names
and frozen IBD groups while preserving the existing revision-3 breadth counts,
qualifying values, and live-application behavior.

The static build uses a fresh database on every run. It currently calculates
breadth history before it creates the dated feature snapshot that supplies
contributor metadata. The calculation therefore persists `company_name = null`
and `ibd_industry_group = "No Group"` for every contributor. The later feature
snapshot and IBD enrichment do not repair those already-frozen rows.

## Requirements

- Keep the canonical breadth engine and all aggregate formulas unchanged.
- Continue advertising no more than the latest 20 completed contributor
  sessions.
- Freeze each contributor's company name and IBD group after its first
  successful static publication. Later classification changes must not rewrite
  retained historical sessions.
- Populate a first-time or newly qualifying contributor from the reference
  metadata available in the current static build.
- Prevent a completely unlabeled contributor bundle from being published
  silently.
- Keep live daily and historical breadth paths unchanged.
- Treat contributor metadata as additive state: failure must not corrupt
  aggregate breadth data or remove an otherwise valid last-known-good market
  artifact.

## Selected Architecture

### Rolling metadata state

Add a compact, market-scoped gzip bundle on a dedicated GitHub release, using
the same restore-before-build and publish-after-success lifecycle as rolling RRG
history. The bundle stores only frozen metadata, not formula-derived values:

```text
schema: static-breadth-contributor-metadata-v1
market
generated_at
sessions[] (newest first, at most 20):
  - date: YYYY-MM-DD
    contributors[] (sorted by symbol):
      - symbol: SYMBOL
        company_name
        ibd_industry_group
```

Asset names are deterministic:
`breadth-contributor-metadata-<market>.json.gz`.

The contributor calculation signature remains the authority for symbols and
qualifying values. The rolling state is only an identity-preserving metadata
source and cannot create, delete, or change contributors.

### Finalization order

The static market job retains its existing early breadth calculation because
market exposure and feature-snapshot construction depend on it. After the
feature snapshot, group-rank backfill, and IBD enrichment complete, a static-only
finalizer runs:

1. Query the newest 20 persisted canonical contributor snapshots.
2. Load current company and IBD metadata once for their union of symbols.
3. For every persisted date/symbol, prefer restored frozen metadata from that
   exact date and symbol.
4. When no restored value exists, use the current reference metadata and freeze
   it into the snapshot. Blank groups normalize to `No Group`.
5. Update metadata columns only. Do not rerun formulas or update aggregate rows,
   signals, qualifying values, calculation signatures, or snapshot identity.
6. Validate that any non-empty retained bundle has at least one company name and
   at least one classified IBD group.
7. Serialize the finalized metadata state for publication.

On the first run, all retained dates bootstrap from current reference metadata.
On later runs, overlapping dates retain their published metadata; only the new
session and newly appearing symbols use current metadata.

### Restore and publication safety

The workflow creates a `breadth-contributor-metadata-data` release when absent.
Each market job restores its deterministic canonical asset before running the
static export. If that asset is absent after an interrupted replacement, the
restorer tries the market's deterministic `.previous` asset before classifying
the run as a genuine cold start. Downloaded bytes are validated for gzip,
schema, and market identity before restore is declared safe.

Restore outcomes follow three states:

- `restored`: validated prior state is available and must be preserved.
- `missing`: first publication; bootstrap is allowed.
- `failed`: prior state may exist but could not be validated or downloaded.
  The current market artifact and replacement state must not be published, so
  the combine job retains the last-known-good market artifact.

Publish the finalized metadata asset with bounded retries before uploading the
corresponding market artifact. Before replacing an existing canonical asset,
copy its validated contents to the deterministic `.previous` asset. Do not
touch the canonical asset if preserving that backup fails. The market artifact
may advance only after the new canonical state is durable. If canonical
replacement is interrupted after GitHub deletes the old asset, the `.previous`
asset remains restorable on the next run. If state publication fails, suppress
the current market artifact and let the combine job retain the last-known-good
fallback. If the later market-artifact upload fails, the safely advanced
metadata state is harmless: it already contains the frozen values the next run
must preserve.

## Components

### Metadata-state contract

A dedicated module owns schema validation, deterministic asset naming,
serialization, retention, and gzip I/O. It rejects wrong markets, duplicate or
unordered dates, more than 20 dates, blank symbols, malformed metadata, and
unsupported schemas.

### Release restorer

A small adapter downloads and validates the market asset from the dedicated
release, falling back to the `.previous` asset only when the canonical asset is
missing. It returns a structured restore result with `status`,
`safe_to_publish`, source asset/path, and reason. It follows the existing RRG
release-restorer pattern but does not share the RRG schema or domain types.

### Static metadata finalizer

The finalizer is the only component allowed to merge restored and current
metadata. Its database write is one transaction and touches only
`MarketBreadthContributor.company_name` and
`MarketBreadthContributor.ibd_industry_group`. It emits counts for retained
dates, contributors, restored values, bootstrapped values, named values, and
classified values.

### Workflow integration

The Static Site workflow will:

1. ensure the metadata release exists;
2. restore the selected market's metadata state;
3. pass the state location and restore status to the market export;
4. suppress current market publication after a failed restore;
5. upload finalized state for a successfully produced candidate; and
6. upload the market artifact only after state publication succeeds.

## Error Handling and Observability

- Invalid restored state produces a failed restore result; it is never partially
  applied.
- A database or finalization error quarantines the current market output and
  leaves prior static artifacts available to the combine job.
- A non-empty finalized bundle with zero company names or zero real IBD groups
  fails the metadata coverage gate.
- The daily-refresh report includes a per-market
  `breadth_contributor_metadata` result with source and coverage counts.
- Static logs clearly distinguish `restored`, `bootstrapped`, `failed`, and
  `published` state transitions.

## Testing

- Contract tests cover round trips, deterministic ordering, retention, market
  identity, corrupt gzip, and invalid schemas.
- Finalizer tests prove restored metadata wins, missing values bootstrap from
  current references, `No Group` normalization works, only metadata columns
  change, and fully blank bundles fail.
- Static refresh tests prove finalization runs after IBD enrichment and is
  skipped for unpublishable feature snapshots or failed state restore.
- Workflow contract tests prove restore occurs before export, a failed restore
  suppresses market publication, and successful output publishes the state.
- Static exporter tests continue reconciling contributor symbols and qualifying
  values with aggregate breadth.

## Compatibility and Rollout

There is no database migration, API revision, frontend contract change, or
breadth calculation revision. The live application ignores the new release
asset. Existing static contributor documents remain compatible.

The first successful run after deployment bootstraps metadata for the retained
20 sessions. Subsequent runs preserve those values through rolling state. If no
safe current artifact can be produced, the existing fallback-market mechanism
continues serving the last-known-good static market bundle.
