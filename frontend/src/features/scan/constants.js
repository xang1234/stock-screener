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

// Geographic markets the backend supports. TEST is a developer utility
// and deliberately excluded — it's a pseudo-market for bypassing real
// market selection.
export const UNIVERSE_GEOGRAPHIC_MARKETS = ['US', 'HK', 'IN', 'JP', 'KR', 'TW', 'CN', 'CA', 'DE'];

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
  { value: 'TEST', label: 'Test Mode' },
];

// Scope values are encoded as "kind:value" so a single Select can handle
// exchanges, indices, and whole-market. "market" alone means every symbol
// in the parent market. Asia index scopes (HSI/NIKKEI225/TAIEX) resolve
// via stock_universe_index_membership — an unseeded index will still show
// here but return zero symbols when scanned.
export const UNIVERSE_SCOPES_BY_MARKET = {
  US: [
    { value: 'market', label: 'All US' },
    { value: 'exchange:NYSE', label: 'NYSE' },
    { value: 'exchange:NASDAQ', label: 'NASDAQ' },
    { value: 'exchange:AMEX', label: 'AMEX' },
    { value: 'index:SP500', label: 'S&P 500' },
  ],
  HK: [
    { value: 'market', label: 'All Hong Kong' },
    { value: 'index:HSI', label: 'Hang Seng Index' },
  ],
  IN: [
    { value: 'market', label: 'All India' },
  ],
  JP: [
    { value: 'market', label: 'All Japan' },
    { value: 'index:NIKKEI225', label: 'Nikkei 225' },
  ],
  KR: [
    { value: 'market', label: 'All Korea' },
    { value: 'exchange:KOSPI', label: 'KOSPI' },
    { value: 'exchange:KOSDAQ', label: 'KOSDAQ' },
  ],
  TW: [
    { value: 'market', label: 'All Taiwan' },
    { value: 'index:TAIEX', label: 'TAIEX 50' },
  ],
  CN: [
    { value: 'market', label: 'All China A-shares' },
    { value: 'exchange:SSE', label: 'Shanghai Stock Exchange' },
    { value: 'exchange:SZSE', label: 'Shenzhen Stock Exchange' },
    { value: 'exchange:BJSE', label: 'Beijing Stock Exchange' },
  ],
  CA: [
    { value: 'market', label: 'All Canada' },
    { value: 'exchange:TSX', label: 'Toronto Stock Exchange' },
    { value: 'exchange:TSXV', label: 'TSX Venture Exchange' },
  ],
  DE: [
    { value: 'market', label: 'All Germany' },
    { value: 'exchange:XETR', label: 'Xetra' },
    { value: 'exchange:XFRA', label: 'Frankfurt' },
    { value: 'index:DAX', label: 'DAX 40' },
    { value: 'index:MDAX', label: 'MDAX' },
    { value: 'index:SDAX', label: 'SDAX' },
  ],
  TEST: [],
};
