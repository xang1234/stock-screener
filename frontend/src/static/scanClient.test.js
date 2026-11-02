import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import {
  applyScanFilterDefaults,
  buildDefaultScanFilters,
} from '../features/scan/defaultFilters';
import { fieldValueOptions } from '../features/scan/scanFilterFields';
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
  beforeEach(() => {
    vi.useFakeTimers();
    // Fix "today" to 2024-01-15 UTC so IPO boundary tests are deterministic.
    vi.setSystemTime(new Date(Date.UTC(2024, 0, 15)));
  });

  afterEach(() => {
    vi.useRealTimers();
  });

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

  it('filters by recent RS line blue-dot leadership', () => {
    const testRows = [
      { symbol: 'BDOT', rs_line_blue_dot_recent: true },
      { symbol: 'NOPE', rs_line_blue_dot_recent: false },
      { symbol: 'MISS', rs_line_blue_dot_recent: false },
    ];
    const filters = buildDefaultScanFilters();
    filters.rsLineBlueDotRecent = true;

    const filtered = filterStaticScanRows(testRows, filters);

    expect(filtered.map((row) => row.symbol)).toEqual(['BDOT']);
  });

  it('filters correction survivors without treating missing legacy values as false', () => {
    const testRows = [
      { symbol: 'READY', correction_survivor: true },
      { symbol: 'NOPE', correction_survivor: false },
      { symbol: 'LEGACY' },
    ];
    const filters = buildDefaultScanFilters();
    expect(filters.correctionSurvivor).toBeNull();
    filters.correctionSurvivor = true;

    const filtered = filterStaticScanRows(testRows, filters);

    expect(filtered.map((row) => row.symbol)).toEqual(['READY']);
  });

  it('publishes the fixed action-state filter options', () => {
    expect(fieldValueOptions('action_state')).toEqual([
      'exit_risk',
      'deteriorating',
      'event_risk',
      'extended',
      'data_limited',
      'setup_ready',
      'watch',
    ]);
  });

  it('does not coerce missing static RS blue-dot values to false', () => {
    const testRows = [
      { symbol: 'BDOT', rs_line_blue_dot_recent: true },
      { symbol: 'NOPE', rs_line_blue_dot_recent: false },
      { symbol: 'LEGACY' },
    ];
    const filters = buildDefaultScanFilters();
    filters.rsLineBlueDotRecent = false;

    const filtered = filterStaticScanRows(testRows, filters);

    expect(filtered.map((row) => row.symbol)).toEqual(['NOPE']);
  });

  it('keeps the legacy setup-engine blue-dot filter wired in static mode', () => {
    const testRows = [
      { symbol: 'SETUP', se_rs_line_blue_dot: true },
      { symbol: 'RESTING', se_rs_line_blue_dot: false },
    ];
    const filters = buildDefaultScanFilters();
    filters.seRsLineBlueDot = true;

    const filtered = filterStaticScanRows(testRows, filters);

    expect(filtered.map((row) => row.symbol)).toEqual(['SETUP']);
  });

  it('supports exclude-mode categorical filters', () => {
    const filters = buildDefaultScanFilters();
    filters.ibdIndustries = { values: ['Semiconductors'], mode: 'exclude' };

    const filtered = filterStaticScanRows(rows, filters);

    expect(filtered.map((row) => row.symbol)).toEqual(['MSFT', 'SNOW']);
  });

  it('filters by IBD group rank range', () => {
    const testRows = [
      { ...rows[0], symbol: 'LEADER', ibd_group_rank: 40 },
      { ...rows[1], symbol: 'LAGGING_GROUP', ibd_group_rank: 41 },
      { ...rows[2], symbol: 'UNKNOWN_GROUP', ibd_group_rank: null },
    ];
    const filters = buildDefaultScanFilters();
    filters.ibdGroupRank = { min: null, max: 40 };

    const filtered = filterStaticScanRows(testRows, filters);

    expect(filtered.map((row) => row.symbol)).toEqual(['LEADER']);
  });

  it('resolves IPO date presets to a cutoff (not raw string comparison)', () => {
    // Frozen clock: 2024-01-15 UTC. Derived cutoffs:
    //   1y  → 2023-01-15   5y → 2019-01-15   6m → 2023-07-15
    const testRows = [
      { ...rows[0], symbol: 'OLD', ipo_date: '1999-01-22' },
      { ...rows[0], symbol: 'NEW', ipo_date: '2023-07-15' },
    ];

    const filtersOneY = buildDefaultScanFilters();
    filtersOneY.ipoAfter = '1y';
    expect(filterStaticScanRows(testRows, filtersOneY).map((r) => r.symbol)).toEqual(['NEW']);

    const filtersFiveY = buildDefaultScanFilters();
    filtersFiveY.ipoAfter = '5y';
    expect(filterStaticScanRows(testRows, filtersFiveY).map((r) => r.symbol)).toEqual(['NEW']);

    const filtersSixM = buildDefaultScanFilters();
    filtersSixM.ipoAfter = '6m';
    expect(filterStaticScanRows(testRows, filtersSixM).map((r) => r.symbol)).not.toContain('OLD');
  });

  it('filters by EPS Rating, Market Cap, and RS 12M ranges', () => {
    const testRows = [
      { symbol: 'A', eps_rating: 85, market_cap: 5_000_000_000, market_cap_usd: 500_000_000, rs_rating_12m: 90 },
      { symbol: 'B', eps_rating: 45, market_cap: 500_000_000, market_cap_usd: 2_000_000_000, rs_rating_12m: 55 },
      { symbol: 'C', eps_rating: null, market_cap: null, rs_rating_12m: null },
    ];

    const f1 = buildDefaultScanFilters();
    f1.epsRating = { min: 70, max: null };
    expect(filterStaticScanRows(testRows, f1).map((r) => r.symbol)).toEqual(['A']);

    const f2 = buildDefaultScanFilters();
    f2.minMarketCap = 1_000_000_000;
    expect(filterStaticScanRows(testRows, f2).map((r) => r.symbol)).toEqual(['A']);

    const fUsd = buildDefaultScanFilters();
    fUsd.marketCapUsd = { min: 1_000_000_000, max: null };
    expect(filterStaticScanRows(testRows, fUsd).map((r) => r.symbol)).toEqual(['B']);

    const f3 = buildDefaultScanFilters();
    f3.rs12m = { min: 80, max: null };
    expect(filterStaticScanRows(testRows, f3).map((r) => r.symbol)).toEqual(['A']);
  });

  it('sorts and paginates rows in-browser without any backend assistance', () => {
    const sortedByRating = sortStaticScanRows(rows, 'rating', 'desc');
    const sortedByScore = sortStaticScanRows(rows, 'composite_score', 'asc');
    const pageTwo = paginateStaticScanRows(sortedByRating, 2, 1);

    expect(sortedByRating.map((row) => row.symbol)).toEqual(['NVDA', 'MSFT', 'SNOW']);
    expect(sortedByScore.map((row) => row.symbol)).toEqual(['SNOW', 'MSFT', 'NVDA']);
    expect(pageTwo.map((row) => row.symbol)).toEqual(['MSFT']);
  });

  it('keeps searched listing-only IPO rows visible despite the default min-volume filter', () => {
    const filters = applyScanFilterDefaults({ minVolume: 100_000_000 });
    filters.symbolSearch = '0100';

    const filtered = filterStaticScanRows([
      {
        symbol: '0100.HK',
        company_name: 'MINIMAX-W',
        scan_mode: 'listing_only',
        data_status: 'insufficient_history',
        is_scannable: false,
        volume: null,
      },
    ], filters);

    expect(filtered.map((row) => row.symbol)).toEqual(['0100.HK']);
  });

  it('ranks rows a core screener scored ahead of IPO-only rows when sorting composite desc', () => {
    const sorted = sortStaticScanRows([
      { symbol: 'IPO95', scan_mode: 'ipo_weighted', composite_score: 95 },
      { symbol: 'FULL80', scan_mode: 'full', composite_score: 80, minervini_score: 80 },
      { symbol: 'NEW1', scan_mode: 'listing_only', composite_score: null },
      { symbol: 'FULL70', scan_mode: 'full', composite_score: 70, canslim_score: 70 },
    ], 'composite_score', 'desc');

    expect(sorted.map((row) => row.symbol)).toEqual(['FULL80', 'FULL70', 'IPO95', 'NEW1']);
  });

  it('keeps null composite scores last while sorting desc across scan modes', () => {
    const sorted = sortStaticScanRows([
      { symbol: 'FULLNULL', scan_mode: 'full', composite_score: null },
      { symbol: 'FULL80', scan_mode: 'full', composite_score: 80 },
      { symbol: 'FULL70', scan_mode: 'full', composite_score: 70 },
      { symbol: 'IPO95', scan_mode: 'ipo_weighted', composite_score: 95 },
    ], 'composite_score', 'desc');

    expect(sorted.map((row) => row.symbol)).toEqual(['IPO95', 'FULL80', 'FULL70', 'FULLNULL']);
  });

  it('keeps ascending composite sorts numeric instead of forcing scan-mode grouping', () => {
    const sorted = sortStaticScanRows([
      { symbol: 'IPO95', scan_mode: 'ipo_weighted', composite_score: 95 },
      { symbol: 'FULL80', scan_mode: 'full', composite_score: 80, minervini_score: 80 },
      { symbol: 'NEW1', scan_mode: 'listing_only', composite_score: null },
      { symbol: 'FULL70', scan_mode: 'full', composite_score: 70, canslim_score: 70 },
    ], 'composite_score', 'asc');

    expect(sorted.map((row) => row.symbol)).toEqual(['FULL70', 'FULL80', 'IPO95', 'NEW1']);
  });

  it('sorts resilience descending numerically with nulls last and symbol ties', () => {
    const sorted = sortStaticScanRows([
      { symbol: 'LEGACY', resilience_score: null },
      { symbol: 'BETA', resilience_score: '84' },
      { symbol: 'LOW', resilience_score: '9' },
      { symbol: 'ALPHA', resilience_score: '84' },
    ], 'resilience_score', 'desc');

    expect(sorted.map((row) => row.symbol)).toEqual(['ALPHA', 'BETA', 'LOW', 'LEGACY']);
  });

  it('sorts resilience ascending numerically with nulls last and symbol ties', () => {
    const sorted = sortStaticScanRows([
      { symbol: 'LEGACY', resilience_score: null },
      { symbol: 'BETA', resilience_score: 84 },
      { symbol: 'LOW', resilience_score: 9 },
      { symbol: 'ALPHA', resilience_score: 84 },
    ], 'resilience_score', 'asc');

    expect(sorted.map((row) => row.symbol)).toEqual(['LOW', 'ALPHA', 'BETA', 'LEGACY']);
  });

  it('uses symbol ties when both resilience scores are omitted', () => {
    const sorted = sortStaticScanRows([
      { symbol: 'LEGACY-Z' },
      { symbol: 'LEGACY-A' },
    ], 'resilience_score', 'desc');

    expect(sorted.map((row) => row.symbol)).toEqual(['LEGACY-A', 'LEGACY-Z']);
  });

  it('keeps ascending composite score nulls last on the shared score path', () => {
    const sorted = sortStaticScanRows([
      { symbol: 'LEGACY', composite_score: null },
      { symbol: 'HIGH', composite_score: 84 },
      { symbol: 'LOW', composite_score: 9 },
    ], 'composite_score', 'asc');

    expect(sorted.map((row) => row.symbol)).toEqual(['LOW', 'HIGH', 'LEGACY']);
  });

  it('uses symbol tiebreaks for equal ascending composite scores', () => {
    const sorted = sortStaticScanRows([
      { symbol: 'ZFULL', scan_mode: 'full', composite_score: 80 },
      { symbol: 'AIPO', scan_mode: 'ipo_weighted', composite_score: 80 },
    ], 'composite_score', 'asc');

    expect(sorted.map((row) => row.symbol)).toEqual(['AIPO', 'ZFULL']);
  });
});
