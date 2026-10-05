import { describe, expect, it } from 'vitest';

import { canOfferLastPublished, describePublishedSource } from './publishedSource';

describe('canOfferLastPublished', () => {
  it('offers the fallback for conflicts a snapshot read can bypass', () => {
    expect(canOfferLastPublished({ message: 'refreshing' }, null)).toBe(true);
    for (const code of ['market_refresh_active', 'scan_already_active', 'market_data_stale']) {
      expect(canOfferLastPublished(null, { detail: { code } })).toBe(true);
    }
  });

  it('does not offer it for other failures or a failed last-published request', () => {
    expect(canOfferLastPublished(null, null)).toBe(false);
    expect(canOfferLastPublished(null, { detail: { code: 'snapshot_unavailable' } })).toBe(false);
    // Even while a refresh still blocks scanning, a failed snapshot read is final.
    expect(canOfferLastPublished({ message: 'refreshing' }, { detail: { code: 'snapshot_unavailable' } })).toBe(false);
    expect(canOfferLastPublished(null, { message: 'boom', detail: null })).toBe(false);
  });
});

describe('describePublishedSource', () => {
  it('returns null for computed scans', () => {
    expect(describePublishedSource(null)).toBeNull();
    expect(describePublishedSource({})).toBeNull();
  });

  it('warns when the publication is older than the latest session', () => {
    expect(describePublishedSource({
      data_mode: 'last_published',
      as_of_date: '2026-10-01',
      expected_session: '2026-10-02',
      is_current: false,
    })).toEqual({
      severity: 'warning',
      text: 'Last published data as of 2026-10-01. When this scan was created, the latest completed session was 2026-10-02.',
    });
  });

  it('labels a current snapshot answer as informational', () => {
    expect(describePublishedSource({
      data_mode: 'current',
      as_of_date: '2026-10-02',
      is_current: true,
    })).toEqual({
      severity: 'info',
      text: 'Served from the published snapshot as of 2026-10-02, the latest completed session when this scan was created.',
    });
  });
});
