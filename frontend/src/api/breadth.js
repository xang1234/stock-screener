/**
 * API client for market breadth endpoints.
 */
import apiClient from './client';

/**
 * Get the most recent market breadth data.
 */
export const getCurrentBreadth = async (market = 'US') => {
  const response = await apiClient.get('/v1/breadth/current', {
    params: { market }
  });
  return response.data;
};

/**
 * Get the published breadth bootstrap snapshot for the default dashboard view.
 */
export const getBreadthBootstrap = async (market = 'US') => {
  const response = await apiClient.get('/v1/breadth/bootstrap', {
    params: { market }
  });
  return response.data;
};

/**
 * Get historical market breadth data for a date range.
 *
 * @param {string} startDate - Start date in YYYY-MM-DD format
 * @param {string} endDate - End date in YYYY-MM-DD format
 * @param {number} limit - Maximum number of records to return (default: 365)
 */
export const getHistoricalBreadth = async (startDate, endDate, limit = 365, market = 'US') => {
  const response = await apiClient.get('/v1/breadth/historical', {
    params: { start_date: startDate, end_date: endDate, limit, market }
  });
  return response.data;
};

/**
 * Get time series data for a specific breadth indicator.
 *
 * @param {string} indicator - The indicator name (e.g., 'stocks_up_4pct')
 * @param {number} days - Number of days to retrieve (default: 30)
 */
export const getIndicatorTrend = async (indicator, days = 30, market = 'US') => {
  const response = await apiClient.get(`/v1/breadth/trend/${indicator}`, {
    params: { days, market }
  });
  return response.data;
};

/**
 * Get summary statistics for market breadth.
 */
export const getBreadthSummary = async (market = 'US') => {
  const response = await apiClient.get('/v1/breadth/summary', {
    params: { market }
  });
  return response.data;
};

/**
 * Manually trigger a breadth calculation for a specific date.
 *
 * @param {string|null} date - Date in YYYY-MM-DD format (null for today)
 */
export const triggerCalculation = async (date = null, market = 'US') => {
  const response = await apiClient.post('/v1/breadth/calculate', {
    calculation_date: date,
    market
  });
  return response.data;
};

/**
 * Trigger a historical backfill for a date range.
 *
 * @param {string} startDate - Start date in YYYY-MM-DD format
 * @param {string} endDate - End date in YYYY-MM-DD format
 */
export const triggerBackfill = async (startDate, endDate, market = 'US') => {
  const response = await apiClient.post('/v1/breadth/backfill', {
    start_date: startDate,
    end_date: endDate,
    market
  });
  return response.data;
};
