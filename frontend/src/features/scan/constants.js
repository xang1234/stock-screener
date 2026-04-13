import { buildDefaultScanFilters } from './defaultFilters';
import { getStableFilterKey } from '../../utils/filterUtils';

export const TEST_SYMBOLS = [
  'AAPL',
  'MSFT',
  'GOOGL',
  'AMZN',
  'NVDA',
  'META',
  'TSLA',
  'BRK.B',
  'V',
  'JPM',
  'WMT',
  'MA',
  'JNJ',
  'PG',
  'XOM',
  'UNH',
  'HD',
  'CVX',
  'ABBV',
  'KO',
];

export const DEFAULT_FILTER_KEY = getStableFilterKey(buildDefaultScanFilters());

export const SCREENER_OPTIONS = [
  { id: 'minervini', label: 'Min' },
  { id: 'canslim', label: 'CAN' },
  { id: 'ipo', label: 'IPO' },
  { id: 'custom', label: 'Cust' },
  { id: 'volume_breakthrough', label: 'VolB' },
  { id: 'setup_engine', label: 'Setup' },
];

// Two-step universe picker: pick a market first, then a scope within that market.
// TEST is a developer utility that bypasses market selection.
export const UNIVERSE_MARKETS = [
  { value: 'US', label: 'United States' },
  { value: 'HK', label: 'Hong Kong' },
  { value: 'JP', label: 'Japan' },
  { value: 'TW', label: 'Taiwan' },
  { value: 'TEST', label: 'Test Mode' },
];

// Scope values are encoded as "kind:value" so a single Select can handle exchanges,
// indices, and whole-market. "market" alone means every symbol in the parent market.
// Asia indices (HSI/NIKKEI225/TAIEX) are deferred to StockScreenClaude-7hwc — they
// require index-membership data that doesn't exist yet.
export const UNIVERSE_SCOPES_BY_MARKET = {
  US: [
    { value: 'market', label: 'All US' },
    { value: 'exchange:NYSE', label: 'NYSE' },
    { value: 'exchange:NASDAQ', label: 'NASDAQ' },
    { value: 'exchange:AMEX', label: 'AMEX' },
    { value: 'index:SP500', label: 'S&P 500' },
  ],
  HK: [{ value: 'market', label: 'All Hong Kong' }],
  JP: [{ value: 'market', label: 'All Japan' }],
  TW: [{ value: 'market', label: 'All Taiwan' }],
  TEST: [],
};
