import { describe, expect, it } from 'vitest';

import { formatSnapshotFreshnessLabel } from './snapshotFreshness';

describe('formatSnapshotFreshnessLabel', () => {
  it('uses the canonical snapshot date for coherent payloads', () => {
    expect(formatSnapshotFreshnessLabel({
      snapshot_as_of_date: '2026-04-24',
      scan_as_of_date: '2026-04-24',
      market_timezone: 'America/New_York',
      breadth_latest_date: '2026-04-24',
      groups_latest_date: '2026-04-24',
      exposure_latest_date: '2026-04-24',
      key_markets_latest_date: '2026-04-24',
      date_coherence_status: 'coherent',
    })).toBe('As of 2026-04-24 · America/New_York');
  });

  it('does not present scan freshness as a snapshot anchor when unanchored', () => {
    expect(formatSnapshotFreshnessLabel({
      scan_as_of_date: '2026-06-12',
      date_coherence_status: 'unanchored',
    })).toBe('Snapshot date unavailable · Scan 2026-06-12 · unanchored');
  });

  it('surfaces section dates that differ from the snapshot date', () => {
    expect(formatSnapshotFreshnessLabel({
      snapshot_as_of_date: '2026-06-11',
      market_timezone: 'America/New_York',
      breadth_latest_date: '2026-06-11',
      groups_latest_date: '2026-06-11',
      exposure_latest_date: '2026-06-11',
      key_markets_latest_date: '2026-06-12',
      date_coherence_status: 'future_section_data',
    })).toBe(
      'As of 2026-06-11 · America/New_York · future_section_data · Key markets 2026-06-12',
    );
  });
});
