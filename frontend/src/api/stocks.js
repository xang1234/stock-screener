/**
 * Stock API endpoints
 */
import apiClient from './client';

/**
 * Get basic stock information
 * @param {string} symbol - Stock ticker symbol
 * @returns {Promise} Stock info data
 */
export const getStockInfo = async (symbol) => {
  const response = await apiClient.get(`/v1/stocks/${symbol}/info`);
  return response.data;
};

/**
 * Get stock fundamentals
 * @param {string} symbol - Stock ticker symbol
 * @param {boolean} forceRefresh - Force cache refresh
 * @param {boolean} useAlphaVantage - Use Alpha Vantage API
 * @returns {Promise} Fundamental data
 */
export const getStockFundamentals = async (
  symbol,
  forceRefresh = false,
  useAlphaVantage = false
) => {
  const response = await apiClient.get(`/v1/stocks/${symbol}/fundamentals`, {
    params: { force_refresh: forceRefresh, use_alpha_vantage: useAlphaVantage },
  });
  return response.data;
};

/**
 * Get stock technical indicators
 * @param {string} symbol - Stock ticker symbol
 * @param {boolean} forceRefresh - Force cache refresh
 * @returns {Promise} Technical data
 */
export const getStockTechnicals = async (symbol, forceRefresh = false) => {
  const response = await apiClient.get(`/v1/stocks/${symbol}/technicals`, {
    params: { force_refresh: forceRefresh },
  });
  return response.data;
};

/**
 * Get complete stock data (info + fundamentals + technicals)
 * @param {string} symbol - Stock ticker symbol
 * @param {object} options - Options for data fetching
 * @returns {Promise} Complete stock data
 */
export const getStockData = async (
  symbol,
  options = {
    includeFundamentals: true,
    includeTechnicals: true,
    forceRefresh: false,
  }
) => {
  const response = await apiClient.get(`/v1/stocks/${symbol}`, {
    params: {
      include_fundamentals: options.includeFundamentals,
      include_technicals: options.includeTechnicals,
      force_refresh: options.forceRefresh,
    },
  });
  return response.data;
};

/**
 * Get stock industry classification
 * @param {string} symbol - Stock ticker symbol
 * @returns {Promise} Industry classification data
 */
export const getStockIndustry = async (symbol) => {
  const response = await apiClient.get(`/v1/stocks/${symbol}/industry`);
  return response.data;
};

/**
 * Get Minervini template scan for a stock
 * @param {string} symbol - Stock ticker symbol
 * @param {boolean} includeVCP - Include VCP pattern detection
 * @returns {Promise} Minervini scan results
 */
export const getMinerviniScan = async (symbol, includeVCP = true) => {
  const response = await apiClient.get(`/v1/technical/${symbol}/minervini`, {
    params: { include_vcp: includeVCP },
  });
  return response.data;
};

/**
 * Get Relative Strength (RS) rating for a stock
 * @param {string} symbol - Stock ticker symbol
 * @returns {Promise} RS rating data
 */
export const getRSRating = async (symbol) => {
  const response = await apiClient.get(`/v1/technical/${symbol}/rs-rating`);
  return response.data;
};

/**
 * Get Weinstein Stage Analysis for a stock
 * @param {string} symbol - Stock ticker symbol
 * @returns {Promise} Stage analysis data
 */
export const getStageAnalysis = async (symbol) => {
  const response = await apiClient.get(`/v1/technical/${symbol}/stage`);
  return response.data;
};

/**
 * Get Moving Average analysis for a stock
 * @param {string} symbol - Stock ticker symbol
 * @returns {Promise} MA analysis data
 */
export const getMAAnalysis = async (symbol) => {
  const response = await apiClient.get(`/v1/technical/${symbol}/ma-analysis`);
  return response.data;
};

/**
 * Get VCP pattern detection for a stock
 * @param {string} symbol - Stock ticker symbol
 * @returns {Promise} VCP detection data
 */
export const getVCPDetection = async (symbol) => {
  const response = await apiClient.get(`/v1/technical/${symbol}/vcp`);
  return response.data;
};

/**
 * Get 52-week position analysis for a stock
 * @param {string} symbol - Stock ticker symbol
 * @returns {Promise} 52-week position data
 */
export const get52WeekPosition = async (symbol) => {
  const response = await apiClient.get(`/v1/technical/${symbol}/52-week-position`);
  return response.data;
};

/**
 * Get historical price data for a stock
 * @param {string} symbol - Stock ticker symbol
 * @param {string} period - Time period (1mo, 3mo, 6mo, 1y, 2y, 5y)
 * @returns {Promise} Historical OHLCV data with moving averages
 */
export const getPriceHistory = async (symbol, period = '6mo') => {
  const response = await apiClient.get(`/v1/stocks/${symbol}/history`, {
    params: { period },
  });
  return response.data;
};

/**
 * Get consolidated chart modal data for a stock
 * Prioritizes data from recent scans (fast), falls back to computation (slow).
 * @param {string} symbol - Stock ticker symbol
 * @returns {Promise} All chart modal data (RS, industry, VCP, growth metrics, etc.)
 */
export const getChartData = async (symbol) => {
  const response = await apiClient.get(`/v1/stocks/${symbol}/chart-data`);
  return response.data;
};

/**
 * Search symbols across the active universe.
 * @param {string} query - Symbol or company-name query
 * @param {number} limit - Maximum number of results
 * @returns {Promise<Object[]>} Search results
 */
export const searchStocks = async (query, limit = 8) => {
  const response = await apiClient.get('/v1/stocks/search', {
    params: { q: query, limit },
  });
  return response.data;
};

/**
 * Get the full stock decision workspace payload.
 * @param {string} symbol - Stock ticker symbol
 * @returns {Promise<Object>} Decision dashboard payload
 */
export const getStockDecisionDashboard = async (symbol) => {
  const response = await apiClient.get(`/v1/stocks/${symbol}/decision-dashboard`);
  return response.data;
};

/**
 * Get historical validation metrics for one symbol.
 * @param {string} symbol - Stock ticker symbol
 * @param {number} lookbackDays - Lookback window in days
 * @returns {Promise<Object>} Validation payload
 */
export const getStockValidation = async (symbol, lookbackDays = 365) => {
  const response = await apiClient.get(`/v1/stocks/${symbol}/validation`, {
    params: { lookback_days: lookbackDays },
  });
  return response.data;
};

/**
 * Get industry/sector peers from the latest published feature run.
 * @param {string} symbol - Stock ticker symbol
 * @param {string} [peerType='industry'] - 'industry' or 'sector'
 * @returns {Promise<Array>} Peer stock items
 */
export const getStockPeers = async (symbol, peerType = 'industry') => {
  const response = await apiClient.get(`/v1/stocks/${symbol}/peers`, {
    params: { peer_type: peerType },
  });
  return response.data;
};
