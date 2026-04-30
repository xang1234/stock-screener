/**
 * Centralized formatting utility functions for consistent display across components
 *
 * These functions handle formatting of large numbers, dates, and percentages
 * for stock data display.
 */

/**
 * Format a large number with K/M/B/T suffix
 * @param {number|null} value - Number to format
 * @param {string} prefix - Optional prefix (e.g., '$')
 * @returns {string} Formatted string with suffix
 */
export const formatLargeNumber = (value, prefix = '') => {
  if (value == null) return '-';
  if (value >= 1e12) return `${prefix}${(value / 1e12).toFixed(1)}T`;
  if (value >= 1e9) return `${prefix}${(value / 1e9).toFixed(1)}B`;
  if (value >= 1e6) return `${prefix}${(value / 1e6).toFixed(1)}M`;
  if (value >= 1e3) return `${prefix}${(value / 1e3).toFixed(0)}K`;
  return `${prefix}${value}`;
};

/**
 * Format market cap with higher precision (2 decimal places)
 * @param {number|null} value - Market cap value
 * @returns {string} Formatted string with $ prefix
 */
export const formatMarketCap = (value) => {
  if (value == null) return '-';
  if (value >= 1e12) return `$${(value / 1e12).toFixed(2)}T`;
  if (value >= 1e9) return `$${(value / 1e9).toFixed(2)}B`;
  if (value >= 1e6) return `$${(value / 1e6).toFixed(2)}M`;
  if (value >= 1e3) return `$${(value / 1e3).toFixed(2)}K`;
  return `$${value.toFixed(2)}`;
};

/**
 * Format IPO date as age in years or months
 * @param {string|null} ipoDate - ISO date string
 * @returns {string} Formatted age string (e.g., "2.5y" or "8mo")
 */
export const formatIpoAge = (ipoDate) => {
  if (!ipoDate) return '-';
  const ipo = new Date(ipoDate);
  const now = new Date();
  const diffMs = now - ipo;
  const diffDays = diffMs / (1000 * 60 * 60 * 24);
  const diffMonths = diffDays / 30.44;
  const diffYears = diffDays / 365.25;

  if (diffYears >= 1) {
    return `${diffYears.toFixed(1)}y`;
  } else {
    return `${Math.round(diffMonths)}mo`;
  }
};

/**
 * Get color for IPO age (newer IPOs get highlight)
 * @param {string|null} ipoDate - ISO date string
 * @returns {string} MUI color path
 */
export const getIpoAgeColor = (ipoDate) => {
  if (!ipoDate) return 'text.secondary';
  const ipo = new Date(ipoDate);
  const now = new Date();
  const diffDays = (now - ipo) / (1000 * 60 * 60 * 24);
  const diffYears = diffDays / 365.25;

  if (diffYears <= 1) return 'success.main';
  if (diffYears <= 3) return 'warning.main';
  return 'text.secondary';
};

/**
 * Format a percentage with optional sign
 * @param {number|null} value - Percentage value
 * @param {number} decimals - Decimal places (default: 1)
 * @returns {string} Formatted percentage string
 */
export const formatPercent = (value, decimals = 1) => {
  if (value == null) return '-';
  return `${value >= 0 ? '+' : ''}${value.toFixed(decimals)}%`;
};

/**
 * Format a 0..1 confidence value as a percentage string ("92%").
 * Returns null (not '-') so callers can omit the badge entirely when
 * confidence is unavailable.
 * @param {number|null|undefined} confidence
 * @returns {string|null}
 */
export const formatConfidence = (confidence) => {
  if (confidence === null || confidence === undefined) return null;
  return `${Math.round(confidence * 100)}%`;
};

/**
 * Format a ratio value
 * @param {number|null} value - Ratio value
 * @param {number} decimals - Decimal places (default: 2)
 * @returns {string} Formatted ratio string
 */
export const formatRatio = (value, decimals = 2) => {
  if (value == null) return '-';
  return value.toFixed(decimals);
};

/**
 * Convert snake_case pattern name to Title Case
 * @param {string|null} name - Snake case name (e.g., "three_weeks_tight")
 * @returns {string} Title case name (e.g., "Three Weeks Tight")
 */
export const formatPatternName = (name) => {
  if (!name) return '-';
  return name
    .split('_')
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(' ');
};

/**
 * Get color for a 0-100 score: green >= 70, amber >= 40, red < 40
 * @param {number|null} score - Score value 0-100
 * @returns {string|undefined} Hex color or undefined for null
 */
export const getScoreColor = (score) => {
  if (score == null) return undefined;
  if (score >= 70) return '#4caf50';
  if (score >= 40) return '#ff9800';
  return '#f44336';
};

// Per-market currency-symbol map. Matches the conventions used in the rest
// of the product (ADR ASIA-E2 + E5). Unknown / null currency falls back to
// "$" so single-market (US) output stays unchanged.
const CURRENCY_PREFIX_BY_CODE = Object.freeze({
  USD: '$',
  HKD: 'HK$',
  INR: '₹',
  JPY: '¥',
  KRW: '₩',
  TWD: 'NT$',
  CNY: '¥',
});

/**
 * Return the short currency-symbol prefix for a row's local currency code.
 * @param {string|null|undefined} currencyCode - ISO 4217 code from stock_universe
 * @returns {string} Currency symbol (e.g. '$', 'HK$', '₹', '¥', '₩', 'NT$'). '$' fallback.
 */
export const getCurrencyPrefix = (currencyCode) => {
  if (!currencyCode) return '$';
  return CURRENCY_PREFIX_BY_CODE[String(currencyCode).toUpperCase()] ?? '$';
};

/**
 * Format a numeric value with the appropriate per-market currency prefix.
 * Returns '-' for null/undefined so empty cells stay visually consistent
 * with other "missing" cells in the results table.
 * @param {number|null|undefined} value - Raw numeric value (local currency units)
 * @param {string|null|undefined} currencyCode - ISO 4217 code (USD/HKD/INR/JPY/KRW/TWD/CNY)
 * @param {number} decimals - Decimal places (default: 2)
 * @returns {string}
 */
export const formatLocalCurrency = (value, currencyCode, decimals = 2) => {
  if (value == null) return '-';
  return `${getCurrencyPrefix(currencyCode)}${value.toFixed(decimals)}`;
};
