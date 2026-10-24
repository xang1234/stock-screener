import {
  MAX_CUSTOM_UNIVERSE_SYMBOLS,
  TEST_SYMBOLS,
  UNIVERSE_GEOGRAPHIC_MARKETS,
  WATCHLIST_MARKET,
} from './constants';

function findRuntimeScopeOption(market, scope, universeSelections) {
  return universeSelections?.scopesByMarket?.[market]?.find((option) => option.value === scope) ?? null;
}

// Build a typed universe_def payload from the two-step picker state.
// Returns null when the selection is incomplete (caller should disable submit).
export function buildUniverseDef(market, scope, universeSelections = null) {
  if (market === 'TEST') {
    return { type: 'test', symbols: TEST_SYMBOLS };
  }
  if (!market || !scope) {
    return null;
  }
  const runtimeOption = findRuntimeScopeOption(market, scope, universeSelections);
  if (runtimeOption?.universe_def) {
    return runtimeOption.universe_def;
  }
  if (scope === 'market') {
    return { type: 'market', market };
  }
  if (scope.startsWith('exchange:')) {
    return { type: 'exchange', market, exchange: scope.slice('exchange:'.length) };
  }
  if (scope.startsWith('index:')) {
    return { type: 'index', index: scope.slice('index:'.length) };
  }
  return null;
}

function watchlistDisabledReason({ symbols, failed }) {
  if (failed) {
    return 'Unavailable';
  }
  if (!symbols) {
    return 'Loading';
  }
  if (symbols.length === 0) {
    return 'Empty';
  }
  if (symbols.length > MAX_CUSTOM_UNIVERSE_SYMBOLS) {
    return `Over ${MAX_CUSTOM_UNIVERSE_SYMBOLS} symbols`;
  }
  return null;
}

function watchlistListHint(watchlists, listFailed) {
  if (!watchlists) {
    return listFailed ? 'Could not load watchlists' : 'Loading watchlists';
  }
  return watchlists.length === 0 ? 'No watchlists yet' : null;
}

function watchlistScope(watchlist, selected) {
  const value = String(watchlist.id);
  if (value !== selected?.id) {
    return { value, label: watchlist.name, disabled: false, disabledReason: null };
  }
  const disabledReason = watchlistDisabledReason(selected);
  return {
    value,
    label: watchlist.name,
    disabled: Boolean(disabledReason),
    disabledReason,
    universe_def: disabledReason ? undefined : { type: 'custom', symbols: selected.symbols },
  };
}

// The selected watchlist's scope option carries its custom universe_def, so
// buildUniverseDef and getSelectionCount read it like any runtime scope.
export function withWatchlistUniverse(
  universeSelections,
  { watchlists, listFailed = false, selected = null } = {},
) {
  const hint = watchlistListHint(watchlists, listFailed);
  const scopes = hint
    ? [{ value: 'none', label: hint, disabled: true, disabledReason: null }]
    : watchlists.map((watchlist) => watchlistScope(watchlist, selected));
  const markets = universeSelections?.markets ?? [];
  const testIndex = markets.findIndex((option) => option.value === 'TEST');
  const insertAt = testIndex === -1 ? markets.length : testIndex;
  return {
    markets: [
      ...markets.slice(0, insertAt),
      { value: WATCHLIST_MARKET, label: 'Watchlist', disabled: false, disabledReason: null },
      ...markets.slice(insertAt),
    ],
    scopesByMarket: { ...universeSelections?.scopesByMarket, [WATCHLIST_MARKET]: scopes },
  };
}

// Count of stocks matching a (market, scope) selection, derived from the
// universeStats bootstrap payload. Returns null when data isn't available yet.
export function getSelectionCount(market, scope, universeStats, universeDef = null) {
  if (universeDef?.type === 'custom') {
    return universeDef.symbols.length;
  }
  if (!universeStats) {
    return null;
  }
  if (market === 'TEST') {
    return TEST_SYMBOLS.length;
  }
  if (!market || !scope) {
    return null;
  }
  if (universeDef?.type === 'market') {
    if (universeDef.listing_tier) {
      return null;
    }
    if (universeDef.mic) {
      return universeStats.by_exchange?.[universeDef.mic] ?? null;
    }
    return universeStats.by_market?.[universeDef.market]?.counts?.active ?? null;
  }
  if (universeDef?.type === 'index') {
    return universeDef.index === 'SP500' ? universeStats.sp500 ?? null : null;
  }
  if (scope === 'market') {
    return universeStats.by_market?.[market]?.counts?.active ?? null;
  }
  if (scope.startsWith('exchange:')) {
    const exchange = scope.slice('exchange:'.length);
    return universeStats.by_exchange?.[exchange] ?? null;
  }
  if (scope === 'index:SP500') {
    return universeStats.sp500 ?? null;
  }
  return null;
}

// Map a legacy saved-default string (e.g. 'nyse', 'sp500', 'market:hk') to the
// new (market, scope) picker state. Ambiguous 'all' yields (null, null) so
// users must explicitly pick a market, matching the bead's design intent.
export function parseLegacyUniverseDefault(legacy) {
  if (typeof legacy !== 'string') {
    return { market: null, scope: null };
  }
  const value = legacy.trim().toLowerCase();
  if (value === 'test') {
    return { market: 'TEST', scope: null };
  }
  if (value === 'nyse' || value === 'nasdaq' || value === 'amex') {
    return { market: 'US', scope: `exchange:${value.toUpperCase()}` };
  }
  if (value === 'sp500') {
    return { market: 'US', scope: 'index:SP500' };
  }
  if (value.startsWith('market:')) {
    const market = value.slice('market:'.length).toUpperCase();
    if (UNIVERSE_GEOGRAPHIC_MARKETS.includes(market)) {
      return { market, scope: 'market' };
    }
  }
  return { market: null, scope: null };
}
