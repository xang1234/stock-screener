import { describe, expect, it } from 'vitest';

import {
  formatSocialScore, noActionableReason, socialFreshness, socialStateLabel, stateReasonText,
  visibleSocialRows,
} from './socialSignalPresentation';

describe('Social Signal presentation', () => {
  it('keeps missing scores distinct from zero', () => {
    expect(formatSocialScore(null)).toBe('—');
    expect(formatSocialScore(0)).toBe('0');
    expect(formatSocialScore(88.6)).toBe('89');
  });

  it('uses deterministic state and freshness labels', () => {
    const now = Date.parse('2026-09-07T15:00:00Z');
    expect(socialStateLabel('risk_off')).toBe('Risk-off');
    expect(socialFreshness({ stale: false, published_at: '2026-09-07T12:00:00Z' }, now))
      .toEqual({ label: 'Updated 3h ago', color: 'success' });
    expect(socialFreshness({ stale: false, published_at: '2026-09-07T14:35:00Z' }, now).label)
      .toBe('Updated 25m ago');
    expect(socialFreshness({ stale: true, published_at: '2026-09-07T12:00:00Z' }, now))
      .toEqual({
        label: `Delayed, last update ${new Date('2026-09-07T12:00:00Z').toLocaleString()}`,
        color: 'warning',
      });
    expect(socialFreshness({ stale: false }, now).label).toBe('Update time unknown');
  });

  it('turns state reason codes into plain text', () => {
    expect(stateReasonText('stale_features')).toBe('Price features out of date');
    expect(stateReasonText('market_exposure_below_50')).toBe('Market exposure below 50');
    expect(stateReasonText('some_new_reason')).toBe('Some new reason');
  });

  it('names the most common Watch reason when exposure is not the cause', () => {
    const watch = (reasons) => ({ explanation: { market_exposure: '65', state_reasons: reasons } });
    expect(noActionableReason([watch(['setup_not_ready']), watch(['setup_not_ready', 'missing_liquidity'])]))
      .toBe('No actionable signals: setup not ready for most signals.');
    expect(noActionableReason([])).toBe('No actionable signals right now.');
  });

  it('filters sources using the published source-name projection', () => {
    const rows = [{
      state: 'watch', canonical_symbol: 'AAA',
      explanation: { source_names: ['Minervini Research List'] },
    }];

    expect(visibleSocialRows(rows, { source: 'minervini' })).toEqual(rows);
    expect(visibleSocialRows(rows, { source: 'asia' })).toEqual([]);
  });
});
