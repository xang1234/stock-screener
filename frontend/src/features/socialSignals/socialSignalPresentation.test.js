import { describe, expect, it } from 'vitest';

import {
  formatSocialScore, socialCoverage, socialFreshnessLabel, socialStateLabel,
} from './socialSignalPresentation';

describe('Social Signal presentation', () => {
  it('keeps missing scores distinct from zero', () => {
    expect(formatSocialScore(null)).toBe('—');
    expect(formatSocialScore(0)).toBe('0');
    expect(formatSocialScore(88.6)).toBe('89');
  });

  it('uses deterministic state, freshness, and warming-up labels', () => {
    expect(socialStateLabel('risk_off')).toBe('Risk-off');
    expect(socialFreshnessLabel({ stale: false })).toBe('Fresh');
    expect(socialFreshnessLabel({ stale: true })).toBe('Stale');
    expect(socialCoverage({ observed_list_count: 1, enabled_list_count: 3,
      coverage: ['limited_history'] }, '14d')).toEqual({
      label: '1/3 lists · 14D', tone: 'warning', detail: 'Warming up · limited history',
    });
  });
});
