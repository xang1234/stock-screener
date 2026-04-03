import { describe, expect, it } from 'vitest';

import {
  applyScanFilterDefaults,
  buildDefaultScanFilters,
} from '../features/scan/defaultFilters';
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
    volume: 126_000_000,
    market_cap: 3_000_000_000_000,
    ipo_date: '1999-01-22',
    composite_score: 97.5,
    rs_rating: 95,
    current_price: 145.4,
    passes_template: true,
    ma_alignment: true,
    eps_growth_qq: 45,
    price_change_1d: 4.2,
  },
  {
    symbol: 'MSFT',
    company_name: 'Microsoft Corporation',
    stage: 2,
    rating: 'Buy',
    ibd_industry_group: 'Software',
    gics_sector: 'Technology',
    volume: 95_000_000,
    market_cap: 3_200_000_000_000,
    ipo_date: '1986-03-13',
    composite_score: 89.2,
    rs_rating: 90,
    current_price: 430.2,
    passes_template: true,
    ma_alignment: true,
    eps_growth_qq: 12,
    price_change_1d: 1.8,
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
    eps_growth_qq: -8,
    price_change_1d: -3.5,
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

  it('applies the static default dollar-volume filter contract', () => {
    const filters = applyScanFilterDefaults({ minVolume: 100_000_000 });

    const filtered = filterStaticScanRows(rows, filters);

    expect(filtered.map((row) => row.symbol)).toEqual(['NVDA']);
  });

  it('supports market-cap, categorical, date, range, and boolean filters together', () => {
    const filters = buildDefaultScanFilters();
    filters.minMarketCap = 100_000_000_000;
    filters.ibdIndustries = { values: ['Semiconductors', 'Software'], mode: 'include' };
    filters.gicsSectors = { values: ['Technology'], mode: 'include' };
    filters.ipoAfter = '1990-01-01';
    filters.epsGrowth = { min: 20, max: null };
    filters.perfDay = { min: 0, max: null };
    filters.passesTemplate = true;
    filters.maAlignment = true;

    const filtered = filterStaticScanRows(rows, filters);

    expect(filtered.map((row) => row.symbol)).toEqual(['NVDA']);
  });

  it('supports exclude-mode categorical filters', () => {
    const filters = buildDefaultScanFilters();
    filters.ibdIndustries = { values: ['Semiconductors'], mode: 'exclude' };

    const filtered = filterStaticScanRows(rows, filters);

    expect(filtered.map((row) => row.symbol)).toEqual(['MSFT', 'SNOW']);
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
