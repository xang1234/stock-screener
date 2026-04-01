import { describe, expect, it } from 'vitest';

import { buildDefaultScanFilters } from '../features/scan/defaultFilters';
import {
  filterStaticScanRows,
  paginateStaticScanRows,
  sortStaticScanRows,
} from './scanClient';

const rows = [
  {
    symbol: 'NVDA',
    company_name: 'NVIDIA Corporation',
    stage: 2,
    rating: 'Strong Buy',
    ibd_industry_group: 'Semiconductors',
    gics_sector: 'Technology',
    volume: 26_000_000,
    market_cap: 3_000_000_000_000,
    ipo_date: '1999-01-22',
    composite_score: 97.5,
    rs_rating: 95,
    current_price: 145.4,
    passes_template: true,
    ma_alignment: true,
  },
  {
    symbol: 'MSFT',
    company_name: 'Microsoft Corporation',
    stage: 2,
    rating: 'Buy',
    ibd_industry_group: 'Software',
    gics_sector: 'Technology',
    volume: 14_000_000,
    market_cap: 3_200_000_000_000,
    ipo_date: '1986-03-13',
    composite_score: 89.2,
    rs_rating: 90,
    current_price: 430.2,
    passes_template: true,
    ma_alignment: true,
  },
  {
    symbol: 'SNOW',
    company_name: 'Snowflake',
    stage: 1,
    rating: 'Watch',
    ibd_industry_group: 'Cloud Software',
    gics_sector: 'Technology',
    volume: 5_500_000,
    market_cap: 60_000_000_000,
    ipo_date: '2020-09-16',
    composite_score: 55.0,
    rs_rating: 67,
    current_price: 176.0,
    passes_template: false,
    ma_alignment: false,
  },
];

describe('static scan client', () => {
  it('filters rows with the exported read-only criteria set', () => {
    const filters = buildDefaultScanFilters();
    filters.symbolSearch = 'nv';
    filters.stage = 2;
    filters.ratings = ['Strong Buy'];
    filters.ibdIndustries = { values: ['Semiconductors'], mode: 'include' };
    filters.minVolume = 20_000_000;
    filters.price = { min: 100, max: 200 };
    filters.rsRating = { min: 90, max: null };
    filters.maAlignment = true;
    filters.passesTemplate = true;

    const filtered = filterStaticScanRows(rows, filters);

    expect(filtered).toEqual([rows[0]]);
  });

  it('sorts and paginates rows in-browser without any backend assistance', () => {
    const sortedByRating = sortStaticScanRows(rows, 'rating', 'desc');
    const sortedByScore = sortStaticScanRows(rows, 'composite_score', 'asc');
    const pageTwo = paginateStaticScanRows(sortedByRating, 2, 1);

    expect(sortedByRating.map((row) => row.symbol)).toEqual(['NVDA', 'MSFT', 'SNOW']);
    expect(sortedByScore.map((row) => row.symbol)).toEqual(['SNOW', 'MSFT', 'NVDA']);
    expect(pageTwo.map((row) => row.symbol)).toEqual(['MSFT']);
  });
});
