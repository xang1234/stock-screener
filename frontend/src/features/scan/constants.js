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
