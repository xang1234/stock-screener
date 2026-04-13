import { describe, it, expect } from 'vitest';
import { buildFilterParams } from './filterUtils';

describe('buildFilterParams — cross-market USD filters (3axp)', () => {
  it('omits USD params when range values are null', () => {
    const params = buildFilterParams({
      marketCapUsd: { min: null, max: null },
      advUsd: { min: null, max: null },
      markets: [],
    });
    expect(params.min_market_cap_usd).toBeUndefined();
    expect(params.max_market_cap_usd).toBeUndefined();
    expect(params.min_adv_usd).toBeUndefined();
    expect(params.max_adv_usd).toBeUndefined();
    expect(params.markets).toBeUndefined();
  });

  it('maps range values to snake_case API params', () => {
    const params = buildFilterParams({
      marketCapUsd: { min: 1_000_000_000, max: 50_000_000_000 },
      advUsd: { min: 10_000_000, max: null },
    });
    expect(params.min_market_cap_usd).toBe(1_000_000_000);
    expect(params.max_market_cap_usd).toBe(50_000_000_000);
    expect(params.min_adv_usd).toBe(10_000_000);
    expect(params.max_adv_usd).toBeUndefined();
  });

  it('joins markets array as comma-separated when non-empty', () => {
    const params = buildFilterParams({ markets: ['US', 'HK', 'JP'] });
    expect(params.markets).toBe('US,HK,JP');
  });

  it('omits markets when the array is empty', () => {
    const params = buildFilterParams({ markets: [] });
    expect(params.markets).toBeUndefined();
  });
});
