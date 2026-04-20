/**
 * API client for IBD Industry Group Rankings endpoints.
 */
import apiClient from './client';

/**
 * Get current IBD group rankings.
 *
 * @param {number} limit - Maximum number of groups to return (default: 50)
 * @returns {Promise<Object>} Rankings response with date, total_groups, and rankings array
 */
export const getCurrentRankings = async (limit = 50, market = 'US') => {
  const response = await apiClient.get('/v1/groups/rankings/current', {
    params: { limit, market }
  });
  return response.data;
};

/**
 * Get the published groups bootstrap snapshot for the default dashboard view.
 */
export const getGroupsBootstrap = async (market = 'US') => {
  const response = await apiClient.get('/v1/groups/bootstrap', {
    params: { market },
  });
  return response.data;
};

/**
 * Get rank movers (gainers and losers) for a period.
 *
 * @param {string} period - Time period: '1w', '1m', '3m', or '6m'
 * @param {number} limit - Number of movers per direction (default: 20)
 * @returns {Promise<Object>} Object with period, gainers, and losers arrays
 */
export const getRankMovers = async (period = '1w', limit = 20, market = 'US') => {
  const response = await apiClient.get('/v1/groups/rankings/movers', {
    params: { period, limit, market }
  });
  return response.data;
};

/**
 * Get detailed ranking history for a specific industry group.
 *
 * @param {string} industryGroup - IBD industry group name
 * @param {number} days - Days of history to retrieve (default: 180)
 * @returns {Promise<Object>} Group detail with history and rank changes
 */
export const getGroupDetail = async (industryGroup, days = 180, market = 'US') => {
  const response = await apiClient.get('/v1/groups/rankings/detail', {
    params: { group: industryGroup, days, market }
  });
  return response.data;
};

/**
 * Get summary statistics for ranking data.
 *
 * @returns {Promise<Object>} Summary with total_records, date range, etc.
 */
export const getRankingsSummary = async () => {
  const response = await apiClient.get('/v1/groups/summary');
  return response.data;
};

/**
 * Manually trigger a ranking calculation for a specific date.
 * Returns immediately with a task_id for status polling.
 *
 * @param {string|null} date - Date in YYYY-MM-DD format (null for today)
 * @returns {Promise<Object>} Task response with task_id, status, message
 */
export const triggerCalculation = async (date = null) => {
  const response = await apiClient.post('/v1/groups/rankings/calculate', {
    calculation_date: date
  });
  return response.data;
};

/**
 * Get the status of a ranking calculation task.
 *
 * @param {string} taskId - Celery task ID from triggerCalculation
 * @returns {Promise<Object>} Status response with task_id, status, result/error
 */
export const getCalculationStatus = async (taskId) => {
  const response = await apiClient.get(`/v1/groups/rankings/calculate/status/${taskId}`);
  return response.data;
};

/**
 * Trigger a historical backfill for a date range.
 *
 * @param {string} startDate - Start date in YYYY-MM-DD format
 * @param {string} endDate - End date in YYYY-MM-DD format
 * @returns {Promise<Object>} Backfill result with statistics
 */
export const triggerBackfill = async (startDate, endDate) => {
  const response = await apiClient.post('/v1/groups/rankings/backfill', {
    start_date: startDate,
    end_date: endDate
  });
  return response.data;
};
