import { describe, expect, it } from 'vitest';

import { formatScanDropdownLabel } from './scanLabel';

describe('formatScanDropdownLabel', () => {
  it('formats manual scan labels with NYSE datetime and man prefix', () => {
    expect(formatScanDropdownLabel({
      trigger_source: 'manual',
      started_at: '2026-03-29T21:45:00Z',
      universe_type: 'all',
      passed_stocks: 42,
      total_stocks: 100,
    })).toBe('man Mar 29, 2026, 5:45 PM | All (42/100)');
  });

  it('formats auto scan labels with NYSE date and auto prefix', () => {
    expect(formatScanDropdownLabel({
      trigger_source: 'auto',
      started_at: '2026-03-29T21:45:00Z',
      universe_type: 'all',
      passed_stocks: 42,
      total_stocks: 100,
    })).toBe('auto Mar 29, 2026 | All (42/100)');
  });

  it('treats naive timestamps as UTC before converting to NYSE time', () => {
    expect(formatScanDropdownLabel({
      trigger_source: 'manual',
      started_at: '2026-03-29T21:45:00',
      universe_type: 'all',
      passed_stocks: 42,
      total_stocks: 100,
    })).toBe('man Mar 29, 2026, 5:45 PM | All (42/100)');
  });

  it('renders market universe labels from universe_market metadata', () => {
    expect(formatScanDropdownLabel({
      trigger_source: 'manual',
      started_at: '2026-03-29T21:45:00Z',
      universe_type: 'market',
      universe_market: 'HK',
      passed_stocks: 42,
      total_stocks: 100,
    })).toBe('man Mar 29, 2026, 5:45 PM | HK (42/100)');
  });
});
