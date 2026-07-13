import { describe, expect, it } from 'vitest';

import { groupRequestParams } from './groups';

describe('groupRequestParams', () => {
  it('adds as_of_date only when an anchor is present', () => {
    expect(groupRequestParams({ market: 'HK', limit: 197 }, '2026-03-16')).toEqual({
      market: 'HK',
      limit: 197,
      as_of_date: '2026-03-16',
    });
    expect(groupRequestParams({ market: 'HK', limit: 197 }, null)).toEqual({
      market: 'HK',
      limit: 197,
    });
  });
});
