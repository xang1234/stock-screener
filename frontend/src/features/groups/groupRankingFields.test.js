import { describe, expect, it } from 'vitest';
import { GROUP_RS_FIELDS, formatGroupRs } from './groupRankingFields';

describe('groupRankingFields', () => {
  it('keeps live and static overall/1M/3M fields identical', () => {
    expect(GROUP_RS_FIELDS).toEqual([
      { field: 'avg_rs_rating', label: 'RS', staticLabel: 'Avg RS' },
      { field: 'avg_rs_rating_1m', label: '1M RS', staticLabel: '1M RS' },
      { field: 'avg_rs_rating_3m', label: '3M RS', staticLabel: '3M RS' },
    ]);
  });

  it('formats finite ratings and renders missing values safely', () => {
    expect(formatGroupRs(87.25)).toBe('87.3');
    expect(formatGroupRs(null)).toBe('-');
    expect(formatGroupRs(Number.NaN)).toBe('-');
  });
});
