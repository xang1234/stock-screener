/**
 * Cache management API client functions.
 *
 * Provides functions for:
 * - Fetching cache statistics
 * - Triggering manual cache refreshes (fundamentals, prices, SPY)
 * - Monitoring cache health and freshness
 */
import apiClient from './client';

/**
 * Get comprehensive cache statistics for dashboard display.
 *
 * Returns aggregated stats for:
 * - Fundamentals cache (total stocks, freshness, hit rates)
 * - Price cache (SPY status, total symbols cached)
 * - Market status (open/closed)
 *
 * @returns {Promise<Object>} Dashboard cache statistics
 */
export const getCacheStats = async () => {
  const response = await apiClient.get('/v1/cache/dashboard-stats');
  return response.data;
};

/**
 * Trigger price cache warmup for all symbols.
 *
 * This queues a background task to warm the price cache for
 * top symbols. Expected duration: ~30 minutes.
 *
 * @returns {Promise<Object>} Task response with task_id
 */
export const triggerPriceRefresh = async () => {
  const response = await apiClient.post('/v1/cache/warm/all');
  return response.data;
};

/**
 * Trigger fundamental data refresh for all stocks.
 *
 * This queues a background task to refresh fundamentals for
 * all ~7,000 active stocks. Expected duration: ~4 hours.
 *
 * **Warning**: This is an expensive operation.
 *
 * @returns {Promise<Object>} Task response with task_id
 */
export const triggerFundamentalsRefresh = async () => {
  const response = await apiClient.post('/v1/fundamentals/refresh/all');
  return response.data;
};

/**
 * Trigger SPY benchmark cache warmup.
 *
 * This refreshes the SPY benchmark data (1-year and 2-year periods)
 * used for relative strength calculations.
 *
 * @returns {Promise<Object>} Task response with task_id
 */
export const triggerSpyRefresh = async () => {
  const response = await apiClient.post('/v1/cache/warm/spy');
  return response.data;
};

/**
 * Get detailed fundamentals cache statistics.
 *
 * Returns stats for a sample of 100 stocks including:
 * - Total checked
 * - Redis/DB cache hit counts
 * - Fresh vs stale counts
 * - Hit rates
 *
 * @returns {Promise<Object>} Fundamentals cache statistics
 */
export const getFundamentalsStats = async () => {
  const response = await apiClient.get('/v1/fundamentals/stats');
  return response.data;
};

/**
 * Get intraday data staleness status.
 *
 * Returns information about symbols with stale intraday data
 * (data fetched during market hours that is now outdated after close).
 *
 * @returns {Promise<Object>} Staleness status including:
 *   - stale_intraday_count: Number of affected symbols
 *   - stale_symbols: Array of first 10 stale symbol names
 *   - market_is_open: Boolean if market is currently open
 *   - current_time_et: Current time in ET
 *   - has_stale_data: Boolean if any stale data exists
 */
export const getStalenessStatus = async () => {
  const response = await apiClient.get('/v1/cache/staleness-status');
  return response.data;
};

/**
 * Force refresh stale intraday price data.
 *
 * Triggers a background task to refresh symbols that have stale
 * intraday data (data fetched during market hours that is now
 * outdated after market close).
 *
 * @param {Object} options - Refresh options
 * @param {Array<string>|null} options.symbols - Optional list of symbols to refresh.
 * @param {boolean} options.refreshAll - If true, refresh ALL cached symbols (not just stale ones).
 * @returns {Promise<Object>} Task response with task_id for tracking
 */
export const forceRefreshPriceCache = async ({ symbols = null, refreshAll = false } = {}) => {
  const response = await apiClient.post('/v1/cache/force-refresh', {
    symbols: symbols,
    refresh_all: refreshAll
  });
  return response.data;
};

/**
 * Get cache health status (NEW unified endpoint).
 *
 * Uses SPY as a proxy for overall cache health (O(1) check).
 * Includes warmup metadata for detecting partial failures.
 *
 * Returns one of 6 states:
 * - fresh: Cache is up to date
 * - updating: Refresh task is currently running
 * - stuck: Task running but no progress for >30 minutes
 * - partial: Last warmup incomplete (some symbols failed)
 * - stale: SPY missing expected trading date
 * - error: Redis unavailable or other error
 *
 * @returns {Promise<Object>} Health status including:
 *   - status: 'fresh'|'updating'|'stuck'|'partial'|'stale'|'error'
 *   - spy_last_date: Last date in SPY data
 *   - expected_date: Date cache should have
 *   - message: Human-readable explanation
 *   - can_refresh: Whether refresh is allowed
 *   - task_running: Task info if updating (includes progress)
 *   - last_warmup: Warmup metadata (status, count, total)
 */
export const getCacheHealth = async () => {
  const response = await apiClient.get('/v1/cache/health');
  return response.data;
};

/**
 * Trigger smart cache refresh (NEW unified endpoint).
 *
 * Replaces the confusing split between triggerPriceRefresh and forceRefreshPriceCache.
 *
 * Features:
 * - Always warms SPY first (required for RS calculations)
 * - Fetches symbols in market cap order (high cap first)
 * - Prevents double-refresh (returns existing task info if running)
 *
 * @param {string} mode - Refresh mode:
 *   - 'auto' (default): Refresh all currently cached symbols
 *   - 'full': Refresh entire universe (~5000 symbols, ~2 hours)
 * @returns {Promise<Object>} Response with:
 *   - status: 'queued' or 'already_running'
 *   - task_id: Celery task ID for tracking
 *   - message: Human-readable status
 */
export const refreshCache = async (mode = 'auto', market = null) => {
  const response = await apiClient.post('/v1/cache/refresh', { mode, market });
  return response.data;
};

/**
 * Force-cancel a stuck refresh task.
 *
 * Use when a task appears stuck (no progress for >30 minutes).
 * Releases the lock so a new refresh can be started.
 *
 * Safety: Won't cancel actively progressing tasks (requires >30 min no heartbeat).
 *
 * @returns {Promise<Object>} Response with:
 *   - status: 'cancelled', 'no_task', or 'active'
 *   - message: Human-readable status
 */
export const forceCancelRefresh = async () => {
  const response = await apiClient.post('/v1/cache/force-cancel');
  return response.data;
};
