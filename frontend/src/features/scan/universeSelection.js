import { TEST_SYMBOLS } from './constants';

// Build a typed universe_def payload from the two-step picker state.
// Returns null when the selection is incomplete (caller should disable submit).
export function buildUniverseDef(market, scope) {
  if (market === 'TEST') {
    return { type: 'test', symbols: TEST_SYMBOLS };
  }
  if (!market || !scope) {
    return null;
  }
  if (scope === 'market') {
    return { type: 'market', market };
  }
  if (scope.startsWith('exchange:')) {
    return { type: 'exchange', exchange: scope.slice('exchange:'.length) };
  }
  if (scope.startsWith('index:')) {
    return { type: 'index', index: scope.slice('index:'.length) };
  }
  return null;
}

// Count of stocks matching a (market, scope) selection, derived from the
// universeStats bootstrap payload. Returns null when data isn't available yet.
export function getSelectionCount(market, scope, universeStats) {
  if (!universeStats) {
    return null;
  }
  if (market === 'TEST') {
    return TEST_SYMBOLS.length;
  }
  if (!market || !scope) {
    return null;
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
    if (market === 'US' || market === 'HK' || market === 'JP' || market === 'TW') {
      return { market, scope: 'market' };
    }
  }
  return { market: null, scope: null };
}
