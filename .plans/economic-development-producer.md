# Economic-native development producer (#513)

## Problem

Developments (event facts such as "X wins order Y") are produced only from
legacy theme data:

- **Discovery** (`theme_development_worker.discover`) finds content items with
  legacy `ThemeMention` rows.
- **Preparation** (`theme_development_preparation.input_bundle`) collects the
  item's legacy theme ids and names. The model is asked to attribute each event
  to "integer IDs of only the supplied themes".
- **Recording** (`record_developments`) writes `ThemeDevelopmentTheme` links.
  It already accepts `economic_theme_ids`, but no caller passes any, and when
  passed they are linked to every observation of the item, not per event.

Under economic authority the legacy extractors skip, so no new developments are
recorded: development history stops growing at cutover. Separately, the
economic snapshot builder (`_development_rows`) still maps legacy
`ThemeDevelopmentTheme` links to economic themes at build time, so the legacy
link table can't be retired.

## What already exists (verified on main)

- Content ingestion admits every polled item as economic evidence (#471):
  `EvidencePacket.source_metadata.content_item_id`, route
  `CONTENT_INGESTION_ROUTE`, one `SourceLineage` per source family.
- An item's economic themes are reachable:
  `EvidencePacket` ← `ProcessingRequest.evidence_packet_id` ←
  `ClassificationAttempt.processing_request_id` → `ClaimAssignment.economic_theme_id`.
- Per-packet lens eligibility: `LensEligibilityRevision.evidence_channels`.
- `EconomicThemeDevelopment(observation_id, economic_theme_id, link_origin)` with
  origins `economic_native | legacy_mapping | compatibility`.
- The snapshot builder already unions `EconomicThemeDevelopment` links with the
  legacy mapping.

## Design

Decisions (owner, 2026-10-09): latest classification drives discovery;
legacy links are backfilled per taxonomy version; developments recorded in
economic mode are projected to legacy links at rollback.

Delivered in two PRs: **PR 1** is the producer (fixes the cutover blocker:
development history keeps growing under economic authority). **PR 2** is the
legacy-link backfill, the builder reading only economic rows, and the rollback
projection.

### PR 1 — discovery, preparation and recording in economic mode (question 1)

- Mode: only `economic` authority takes this path; legacy, shadow and dual keep
  the legacy path unchanged.
- **Discovery:** recent completed `ProcessingRequest`s on effective
  content-ingestion `EvidencePacket`s, whose newest `completed`
  `ClassificationAttempt` has claim assignments. The content item comes from
  `source_metadata.content_item_id` (read in Python, no JSON SQL). The channel
  must be in the packet's lens eligibility. Explicit backfill by item id stays.
- **Bundle:** the item's sources as today, plus the economic themes of the
  newest completed attempt (names from `EconomicThemeRevision.display_name` in
  the authority's processing taxonomy version, else the attempt's output
  version). The revision digest includes the bundle kind, so a mode switch
  between preparation and recording supersedes the work instead of mixing kinds.
- **Prompt:** unchanged. Economic themes are offered as integers `1..n` with
  their names; `normalize_batch` validates them as today, and each event's
  integers map back to UUIDs, so links are **per event** with
  `link_origin=economic_native`. No `ThemeDevelopmentTheme` link is written.
- **Known event identities:** events whose observations link (through
  `EconomicThemeDevelopment`) to the bundle's economic themes.

- **Entry points:** the scheduled `prepare_developments` and
  `POST /themes/developments/backfill` choose the producer by mode; the legacy
  producer keeps the #472 write fence.
- **Not ready is not empty:** economic evidence not yet classified for a
  channel records nothing (recording an empty revision would supersede the
  item's legacy history).
- **Cutover order:** until PR 2's backfill, economic "known event identity"
  hints see no legacy history, so the model may re-word an event legacy history
  already holds. Land PR 1 and PR 2 before a deployment cuts over.

### PR 2 — legacy links → economic rows (question 2)

- Migration: `EconomicThemeDevelopment` gains `taxonomy_version_id`, required for
  `legacy_mapping` rows and null otherwise; uniqueness covers the version.
- A backfill (CLI and task) writes the `legacy_mapping` rows for one taxonomy
  version from `ThemeDevelopmentTheme` with that version's
  `LegacyClaimAllocation` / `LegacyDestinationMapping` (the logic moves out of
  the builder), replacing that version's rows. It runs at cutover and whenever
  the processing version changes, and records a completion marker.
- The builder reads native rows plus `legacy_mapping` rows of the version it
  builds, and fails closed when legacy links exist but that version has no
  completion marker. The `_SNAPSHOT_BUILDER` allowlist entry goes; the backfill
  is a separate entry under the rollback/bridge category until retirement.

### PR 2 — rollback projection (question 3)

- When a deployment rolls back from economic authority, write
  `ThemeDevelopmentTheme` links for observations recorded in the economic
  window whose economic theme maps 1:1 to a legacy theme (version's
  destinations); skip and log ambiguous ones. Nothing extra runs during normal
  economic operation.

## Done when (from the issue)

- Developments keep being recorded under economic authority, with economic links.
- The snapshot builder reads only `EconomicThemeDevelopment`, and the
  `_SNAPSHOT_BUILDER` allowlist entry is removed.
