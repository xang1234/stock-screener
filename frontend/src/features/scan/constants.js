import { buildDefaultScanFilters } from './defaultFilters';
import { stableExpressionKey } from './filterExpressionModel';
import { legacyFiltersToExpression } from './legacyFilterExpression';

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

export const DEFAULT_FILTER_KEY = stableExpressionKey(
  legacyFiltersToExpression(buildDefaultScanFilters()),
);

export const SCREENER_OPTIONS = [
  { id: 'minervini', label: 'Min' },
  { id: 'canslim', label: 'CAN' },
  { id: 'ipo', label: 'IPO' },
  { id: 'custom', label: 'Cust' },
  { id: 'volume_breakthrough', label: 'VolB' },
  { id: 'setup_engine', label: 'Setup' },
];

// Geographic markets the backend supports. TEST is a developer utility
// and deliberately excluded — it's a pseudo-market for bypassing real
// market selection.
export const UNIVERSE_GEOGRAPHIC_MARKETS = ['US', 'HK', 'IN', 'JP', 'KR', 'TW', 'CN', 'CA', 'DE', 'SG', 'AU', 'MY'];

export const UNIVERSE_MARKETS = [
  { value: 'US', label: 'United States' },
  { value: 'HK', label: 'Hong Kong' },
  { value: 'IN', label: 'India' },
  { value: 'JP', label: 'Japan' },
  { value: 'KR', label: 'South Korea' },
  { value: 'TW', label: 'Taiwan' },
  { value: 'CN', label: 'China A-shares' },
  { value: 'CA', label: 'Canada' },
  { value: 'DE', label: 'Germany' },
  { value: 'SG', label: 'Singapore' },
  { value: 'AU', label: 'Australia' },
  { value: 'MY', label: 'Malaysia' },
  { value: 'TEST', label: 'Test Mode' },
];

// Scope values are encoded as "kind:value" so a single Select can handle
// exchanges, indices, and whole-market. "market" alone means every symbol
// in the parent market. Index scopes resolve via stock_universe_index_membership;
// an unseeded index will still show here but return zero symbols when scanned.
export const UNIVERSE_SCOPES_BY_MARKET = {
  US: [
    { value: 'market', label: 'All United States' },
    { value: 'exchange:NYSE', label: 'NYSE' },
    { value: 'exchange:NASDAQ', label: 'NASDAQ' },
    { value: 'exchange:AMEX', label: 'AMEX' },
    { value: 'index:SP500', label: 'S&P 500' },
  ],
  HK: [
    { value: 'market', label: 'All Hong Kong' },
    { value: 'exchange:XHKG', label: 'Hong Kong Exchange' },
    { value: 'index:HSI', label: 'Hang Seng Index' },
  ],
  IN: [
    { value: 'market', label: 'All India' },
    { value: 'exchange:XNSE', label: 'National Stock Exchange' },
    { value: 'exchange:XBOM', label: 'Bombay Stock Exchange' },
    { value: 'index:NIFTY50', label: 'NIFTY 50' },
  ],
  JP: [
    { value: 'market', label: 'All Japan' },
    { value: 'exchange:XTKS', label: 'Tokyo Stock Exchange' },
    { value: 'index:NIKKEI225', label: 'Nikkei 225' },
  ],
  KR: [
    { value: 'market', label: 'All South Korea' },
    { value: 'exchange:KOSPI', label: 'KOSPI' },
    { value: 'exchange:KOSDAQ', label: 'KOSDAQ' },
    { value: 'index:KOSPI', label: 'KOSPI Composite' },
  ],
  TW: [
    { value: 'market', label: 'All Taiwan' },
    { value: 'exchange:XTAI', label: 'Taiwan Stock Exchange' },
    { value: 'index:TAIEX', label: 'TAIEX' },
  ],
  CN: [
    { value: 'market', label: 'All China A-shares' },
    { value: 'exchange:SSE', label: 'Shanghai Stock Exchange' },
    { value: 'exchange:SZSE', label: 'Shenzhen Stock Exchange' },
    { value: 'exchange:BJSE', label: 'Beijing Stock Exchange' },
    { value: 'index:CSI300', label: 'CSI 300' },
  ],
  CA: [
    { value: 'market', label: 'All Canada' },
    { value: 'exchange:TSX', label: 'Toronto Stock Exchange' },
    { value: 'exchange:TSXV', label: 'TSX Venture Exchange' },
    { value: 'index:TSX_COMPOSITE', label: 'S&P/TSX Composite' },
  ],
  DE: [
    { value: 'market', label: 'All Germany' },
    { value: 'exchange:XETR', label: 'Xetra' },
    { value: 'exchange:XFRA', label: 'Frankfurt' },
    { value: 'index:DAX', label: 'DAX' },
    { value: 'index:MDAX', label: 'MDAX' },
    { value: 'index:SDAX', label: 'SDAX' },
  ],
  SG: [
    { value: 'market', label: 'All Singapore' },
    { value: 'exchange:XSES', label: 'Singapore Exchange' },
    { value: 'index:STI', label: 'Straits Times Index' },
  ],
  AU: [
    { value: 'market', label: 'All Australia' },
    { value: 'exchange:XASX', label: 'ASX' },
    { value: 'index:ASX200', label: 'S&P/ASX 200' },
  ],
  MY: [
    { value: 'market', label: 'All Malaysia' },
    { value: 'exchange:XKLS', label: 'Bursa Malaysia' },
    { value: 'index:FBMKLCI', label: 'FTSE Bursa Malaysia KLCI' },
  ],
  TEST: [],
};
