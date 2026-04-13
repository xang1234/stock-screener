import { describe, expect, it } from 'vitest';

import { formatScanDropdownLabel } from './scanLabel';

describe('formatScanDropdownLabel', () => {
  it('formats manual scan labels with NYSE datetime and man prefix', () => {
    expect(formatScanDropdownLabel({
      trigger_source: 'manual',
      started_at: '2026-03-29T21:45:00Z',
      universe_def: { type: 'all' },
      passed_stocks: 42,
      total_stocks: 100,
    })).toBe('man Mar 29, 2026, 5:45 PM | All (42/100)');
  });

  it('formats auto scan labels with NYSE date and auto prefix', () => {
    expect(formatScanDropdownLabel({
      trigger_source: 'auto',
      started_at: '2026-03-29T21:45:00Z',
      universe_def: { type: 'all' },
      passed_stocks: 42,
      total_stocks: 100,
    })).toBe('auto Mar 29, 2026 | All (42/100)');
  });

  it('treats naive timestamps as UTC before converting to NYSE time', () => {
    expect(formatScanDropdownLabel({
      trigger_source: 'manual',
      started_at: '2026-03-29T21:45:00',
      universe_def: { type: 'all' },
      passed_stocks: 42,
      total_stocks: 100,
    })).toBe('man Mar 29, 2026, 5:45 PM | All (42/100)');
  });

  it('renders market universe labels from typed universe_def', () => {
    expect(formatScanDropdownLabel({
      trigger_source: 'manual',
      started_at: '2026-03-29T21:45:00Z',
      universe_def: { type: 'market', market: 'HK' },
      passed_stocks: 42,
      total_stocks: 100,
    })).toBe('man Mar 29, 2026, 5:45 PM | HK (42/100)');
  });

  it('renders exchange and index universes from typed universe_def', () => {
    expect(formatScanDropdownLabel({
      trigger_source: 'manual',
      started_at: '2026-03-29T21:45:00Z',
      universe_def: { type: 'exchange', exchange: 'NASDAQ' },
      passed_stocks: 10,
      total_stocks: 20,
    })).toBe('man Mar 29, 2026, 5:45 PM | NASDAQ (10/20)');

    expect(formatScanDropdownLabel({
      trigger_source: 'manual',
      started_at: '2026-03-29T21:45:00Z',
      universe_def: { type: 'index', index: 'SP500' },
      passed_stocks: 10,
      total_stocks: 20,
    })).toBe('man Mar 29, 2026, 5:45 PM | S&P500 (10/20)');
  });

  it('renders test and custom universes with symbol counts', () => {
    expect(formatScanDropdownLabel({
      trigger_source: 'manual',
      started_at: '2026-03-29T21:45:00Z',
      universe_def: { type: 'test', symbols: ['AAPL', 'MSFT', 'NVDA'] },
      passed_stocks: 2,
      total_stocks: 3,
    })).toBe('man Mar 29, 2026, 5:45 PM | Test (3) (2/3)');
  });

  it('falls back to Unknown when universe_def is missing', () => {
    expect(formatScanDropdownLabel({
      trigger_source: 'manual',
      started_at: '2026-03-29T21:45:00Z',
      passed_stocks: 0,
      total_stocks: 0,
    })).toBe('man Mar 29, 2026, 5:45 PM | Unknown (0/0)');
  });
});
