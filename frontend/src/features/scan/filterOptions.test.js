import { describe, expect, it } from 'vitest';

import { normalizeScanFilterOptions } from './filterOptions';

describe('normalizeScanFilterOptions', () => {
  it('maps exported snake_case keys to the FilterPanel camelCase shape', () => {
    expect(
      normalizeScanFilterOptions({
        ibd_industries: ['Semiconductors'],
        gics_sectors: ['Technology'],
        ratings: ['Strong Buy'],
      })
    ).toEqual({
      ibdIndustries: ['Semiconductors'],
      gicsSectors: ['Technology'],
      ratings: ['Strong Buy'],
    });
  });

  it('preserves the live camelCase shape and defaults missing values to empty arrays', () => {
    expect(
      normalizeScanFilterOptions({
        ibdIndustries: ['Software'],
      })
    ).toEqual({
      ibdIndustries: ['Software'],
      gicsSectors: [],
      ratings: [],
    });
  });
});
