/**
 * Price history API functions and query utilities
 * Shared across components that need to fetch/prefetch price data
 */
import apiClient from './client';

/**
 * Fetch price history data for a symbol
 * @param {string} symbol - Stock symbol
 * @param {string} period - Time period (default: '6mo')
 * @returns {Promise<Array>} Price history data with OHLCV and moving averages
 */
export const fetchPriceHistory = async (symbol, period = '6mo') => {
  const response = await apiClient.get(`/v1/stocks/${symbol}/history`, {
    params: { period },
  });
  return response.data;
};

/**
 * Fetch OHLCV history for many symbols in a single request.
 * @param {string[]} symbols - Ticker symbols (case-insensitive, deduped server-side)
 * @param {string} period - Time period (default: '6mo')
 * @returns {Promise<{data: Record<string, Array>, missing: string[]}>}
 */
export const fetchPriceHistoryBatch = async (symbols, period = '6mo') => {
  const response = await apiClient.post('/v1/stocks/history/batch', { symbols, period });
  return response.data;
};

/**
 * Query key factory for price history queries
 * Ensures consistent cache key usage across the app
 */
export const priceHistoryKeys = {
  all: ['priceHistory'],
  symbol: (symbol, period = '6mo') => ['priceHistory', symbol, period],
  batch: (symbols, period = '6mo') => [
    'priceHistory',
    'batch',
    period,
    [...symbols].map((s) => String(s).toUpperCase()).sort().join(','),
  ],
};

/**
 * Default stale time for price history queries (5 minutes)
 * Price data doesn't change frequently during market hours
 */
export const PRICE_HISTORY_STALE_TIME = 300000;
