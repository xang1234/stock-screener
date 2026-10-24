import { describe, expect, it } from 'vitest';

import { TEST_SYMBOLS, UNIVERSE_SCOPES_BY_MARKET } from './constants';
import {
  buildUniverseDef,
  getSelectionCount,
  parseLegacyUniverseDefault,
  withWatchlistUniverse,
} from './universeSelection';

describe('buildUniverseDef', () => {
  it('returns null when market is missing', () => {
    expect(buildUniverseDef(null, 'market')).toBeNull();
  });

  it('returns null when market is set but scope is missing (non-TEST)', () => {
    expect(buildUniverseDef('US', null)).toBeNull();
  });

  it('maps TEST market to a typed test universe with the preset symbol list', () => {
    expect(buildUniverseDef('TEST', null)).toEqual({
      type: 'test',
      symbols: TEST_SYMBOLS,
    });
  });

  it('maps scope "market" to a typed market universe', () => {
    expect(buildUniverseDef('HK', 'market')).toEqual({ type: 'market', market: 'HK' });
  });

  it('maps exchange and index scopes to their typed forms', () => {
    expect(buildUniverseDef('US', 'exchange:NYSE')).toEqual({
      type: 'exchange',
      market: 'US',
      exchange: 'NYSE',
    });
    expect(buildUniverseDef('CN', 'exchange:BJSE')).toEqual({
      type: 'exchange',
      market: 'CN',
      exchange: 'BJSE',
    });
    expect(buildUniverseDef('US', 'index:SP500')).toEqual({ type: 'index', index: 'SP500' });
  });

  it('returns the backend-provided universe definition for runtime option scopes', () => {
    const universeSelections = {
      scopesByMarket: {
        HK: [
          {
            value: 'market:HK:mic:XHKG:tier:main_board',
            universe_def: {
              type: 'market',
              market: 'HK',
              mic: 'XHKG',
              listing_tier: 'main_board',
            },
          },
        ],
      },
    };

    expect(
      buildUniverseDef('HK', 'market:HK:mic:XHKG:tier:main_board', universeSelections)
    ).toEqual({
      type: 'market',
      market: 'HK',
      mic: 'XHKG',
      listing_tier: 'main_board',
    });
  });
});

describe('getSelectionCount', () => {
  const stats = {
    active: 7500,
    sp500: 500,
    by_exchange: { NYSE: 2500, NASDAQ: 3000, AMEX: 400 },
    by_market: {
      US: { counts: { active: 5900 } },
      HK: { counts: { active: 2400 } },
      CN: { counts: { active: 5492 } },
    },
  };

  it('returns null when stats are not loaded yet', () => {
    expect(getSelectionCount('US', 'market', null)).toBeNull();
  });

  it('returns the per-market active count for market scope', () => {
    expect(getSelectionCount('US', 'market', stats)).toBe(5900);
    expect(getSelectionCount('HK', 'market', stats)).toBe(2400);
    expect(getSelectionCount('CN', 'market', stats)).toBe(5492);
  });

  it('returns the by_exchange count for exchange scopes', () => {
    expect(getSelectionCount('US', 'exchange:NYSE', stats)).toBe(2500);
  });

  it('returns the sp500 count for the SP500 index', () => {
    expect(getSelectionCount('US', 'index:SP500', stats)).toBe(500);
  });

  it('returns the TEST_SYMBOLS length for TEST market', () => {
    expect(getSelectionCount('TEST', null, stats)).toBe(TEST_SYMBOLS.length);
  });

  it('returns null for a market that has no stats entry', () => {
    expect(getSelectionCount('JP', 'market', stats)).toBeNull();
  });
});

describe('parseLegacyUniverseDefault', () => {
  it('maps exchange legacy strings to US + exchange scope', () => {
    expect(parseLegacyUniverseDefault('nyse')).toEqual({
      market: 'US',
      scope: 'exchange:NYSE',
    });
    expect(parseLegacyUniverseDefault('nasdaq')).toEqual({
      market: 'US',
      scope: 'exchange:NASDAQ',
    });
  });

  it('maps sp500 to US + index scope', () => {
    expect(parseLegacyUniverseDefault('sp500')).toEqual({
      market: 'US',
      scope: 'index:SP500',
    });
  });

  it('maps market:hk and friends to market scope', () => {
    expect(parseLegacyUniverseDefault('market:hk')).toEqual({ market: 'HK', scope: 'market' });
    expect(parseLegacyUniverseDefault('market:jp')).toEqual({ market: 'JP', scope: 'market' });
    expect(parseLegacyUniverseDefault('market:kr')).toEqual({ market: 'KR', scope: 'market' });
    expect(parseLegacyUniverseDefault('market:cn')).toEqual({ market: 'CN', scope: 'market' });
    expect(parseLegacyUniverseDefault('market:au')).toEqual({ market: 'AU', scope: 'market' });
  });

  // The 'all' default is deliberately ambiguous (it used to mean "all US"), so
  // we force the user to pick explicitly rather than silently defaulting.
  it('maps legacy "all" to an unselected state', () => {
    expect(parseLegacyUniverseDefault('all')).toEqual({ market: null, scope: null });
  });

  it('yields unselected state for null / unknown values', () => {
    expect(parseLegacyUniverseDefault(null)).toEqual({ market: null, scope: null });
    expect(parseLegacyUniverseDefault('bogus')).toEqual({ market: null, scope: null });
  });
});

describe('UNIVERSE_SCOPES_BY_MARKET', () => {
  it('exposes KOSPI and KOSDAQ scopes for Korea', () => {
    expect(UNIVERSE_SCOPES_BY_MARKET.KR).toEqual([
      { value: 'market', label: 'All South Korea' },
      { value: 'exchange:KOSPI', label: 'KOSPI' },
      { value: 'exchange:KOSDAQ', label: 'KOSDAQ' },
      { value: 'index:KOSPI', label: 'KOSPI Composite' },
    ]);
  });

  it('exposes SSE, SZSE, and BJSE scopes for China', () => {
    expect(UNIVERSE_SCOPES_BY_MARKET.CN).toEqual([
      { value: 'market', label: 'All China A-shares' },
      { value: 'exchange:SSE', label: 'Shanghai Stock Exchange' },
      { value: 'exchange:SZSE', label: 'Shenzhen Stock Exchange' },
      { value: 'exchange:BJSE', label: 'Beijing Stock Exchange' },
      { value: 'index:CSI300', label: 'CSI 300' },
    ]);
  });

  it('exposes ASX and ASX 200 scopes for Australia', () => {
    expect(UNIVERSE_SCOPES_BY_MARKET.AU).toEqual([
      { value: 'market', label: 'All Australia' },
      { value: 'exchange:XASX', label: 'ASX' },
      { value: 'index:ASX200', label: 'S&P/ASX 200' },
    ]);
  });
});

describe('withWatchlistUniverse', () => {
  const base = {
    markets: [{ value: 'US', label: 'United States' }, { value: 'TEST', label: 'Test Mode' }],
    scopesByMarket: { US: [], TEST: [] },
  };
  const watchlists = [{ id: 7, name: 'Leaders' }, { id: 8, name: 'Laggards' }];

  it('lists watchlists under a Watchlist market placed before Test Mode', () => {
    const selections = withWatchlistUniverse(base, { watchlists });

    expect(selections.markets.map((option) => option.value)).toEqual(['US', 'WATCHLIST', 'TEST']);
    expect(selections.scopesByMarket.WATCHLIST.map((option) => [option.value, option.label])).toEqual([
      ['7', 'Leaders'],
      ['8', 'Laggards'],
    ]);
  });

  it('turns the loaded watchlist into a counted custom universe', () => {
    const selections = withWatchlistUniverse(base, { watchlists, selected: { id: '7', symbols: ['NVDA', 'ANET'] } });
    const [leaders] = selections.scopesByMarket.WATCHLIST;

    expect(buildUniverseDef('WATCHLIST', '7', selections)).toEqual({
      type: 'custom',
      symbols: ['NVDA', 'ANET'],
    });
    expect(getSelectionCount('WATCHLIST', '7', null, leaders.universe_def)).toBe(2);
  });

  it('blocks an empty watchlist and one over the 2000-symbol limit', () => {
    const empty = withWatchlistUniverse(base, { watchlists, selected: { id: '7', symbols: [] } });
    const oversized = withWatchlistUniverse(base, {
      watchlists,
      selected: { id: '7', symbols: Array.from({ length: 2001 }, (_, index) => `S${index}`) },
    });

    expect(empty.scopesByMarket.WATCHLIST[0].disabledReason).toBe('Empty');
    expect(oversized.scopesByMarket.WATCHLIST[0].disabledReason).toBe('Over 2000 symbols');
    expect(buildUniverseDef('WATCHLIST', '7', oversized)).toBeNull();
  });
});
