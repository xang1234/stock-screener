import { describe, expect, it } from 'vitest';

import {
  fallbackText, formatSocialScore, handleText, socialFreshness, socialStateLabel, stateReasonText,
  visibleSocialRows,
} from './socialSignalPresentation';

describe('Social Signal presentation', () => {
  it('shows one @ whether or not the stored handle has one', () => {
    expect(handleText('@JonahLupton')).toBe('@JonahLupton');
    expect(handleText('analyst')).toBe('@analyst');
  });

  it('treats a null stale flag as not fresh', () => {
    const publishedAt = '2026-11-02T10:00:00Z';
    expect(socialFreshness({ stale: null, published_at: publishedAt }).color).toBe('warning');
    expect(socialFreshness({ stale: false, published_at: publishedAt }, Date.parse('2026-11-02T13:00:00Z')))
      .toEqual({ label: 'Updated 3h ago', color: 'success' });
  });

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
    expect(socialFreshness({ published_at: '2026-09-07T12:00:00Z' }, now))
      .toEqual({
        label: `Delayed, last update ${new Date('2026-09-07T12:00:00Z').toLocaleString()}`,
        color: 'warning',
      });
  });

  it('turns state reason codes into plain text', () => {
    expect(stateReasonText('stale_features')).toBe('Price features out of date');
    expect(stateReasonText('market_exposure_below_50')).toBe('Market exposure below 50');
    expect(stateReasonText('some_new_reason')).toBe('Some new reason');
  });

  it('says why Actionable is empty from the server fallback reason', () => {
    const fallback = (reasonCode, reasonParams, watchCount, riskOffCount = 0) => fallbackText({
      from: 'actionable', to: 'watch', reason_code: reasonCode, reason_params: reasonParams,
      watch_count: watchCount, risk_off_count: riskOffCount,
    });
    expect(fallback('market_exposure_below_50', { market_exposure: '42.4', floor: 50 }, 3, 2))
      .toBe('No actionable signals: market exposure is 42, below 50, so 2 ready setups are under Risk-off. Showing Watch.');
    expect(fallback('market_exposure_below_50', { market_exposure: '42', floor: 50 }, 0, 1))
      .toBe('No actionable signals: market exposure is 42, below 50, so 1 ready setup is under Risk-off.');
    expect(fallback('stale_features', { count: 2, required_session: '2026-10-30' }, 3))
      .toBe('No actionable signals: price features for 2 of 3 Watch signals are not yet updated for 2026-10-30. Showing Watch.');
    expect(fallback('setup_not_ready', { count: 2 }, 3))
      .toBe('No actionable signals: setup not ready for 2 of 3 Watch signals. Showing Watch.');
    expect(fallback(null, {}, 1)).toBe('No actionable signals. Showing Watch.');
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
