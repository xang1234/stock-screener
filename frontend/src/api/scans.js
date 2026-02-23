/**
 * API functions for bulk stock scanning
 */
import apiClient from './client';

/**
 * Create a new bulk scan
 * @param {Object} params - Scan parameters
 * @param {string} params.universe - Universe filter: 'all', 'nyse', 'nasdaq', 'amex', 'sp500', 'custom', or 'test'
 * @param {Array<string>} params.symbols - Custom symbol list (required if universe='custom' or 'test')
 * @param {Object} [params.universe_def] - Structured universe definition (takes precedence over legacy universe field)
 * @param {string} params.universe_def.type - Universe type: 'all', 'exchange', 'index', 'custom', 'test'
 * @param {string} [params.universe_def.exchange] - Exchange name: 'NYSE', 'NASDAQ', 'AMEX' (if type='exchange')
 * @param {string} [params.universe_def.index] - Index name: 'SP500' (if type='index')
 * @param {Array<string>} [params.universe_def.symbols] - Symbol list (if type='custom' or 'test')
 * @param {Object} params.criteria - Scan criteria
 * @param {Array<string>} params.screeners - Screeners to run: ['minervini', 'canslim', 'ipo', 'custom', 'volume_breakthrough', 'setup_engine']
 * @param {string} params.composite_method - How to combine scores: 'weighted_average', 'maximum', 'minimum'
 * @returns {Promise<Object>} Scan creation response with scan_id
 */
export const createScan = async ({
  universe = 'all',
  symbols = [],
  criteria = {},
  screeners = ['minervini'],
  composite_method = 'weighted_average'
}) => {
  const response = await apiClient.post('/v1/scans', {
    universe,
    symbols,
    criteria,
    screeners,
    composite_method,
  });
  return response.data;
};

/**
 * Get scan status and progress
 * @param {string} scanId - Scan ID
 * @returns {Promise<Object>} Scan status with progress information
 */
export const getScanStatus = async (scanId) => {
  const response = await apiClient.get(`/v1/scans/${scanId}/status`);
  return response.data;
};

/**
 * Cancel a running scan
 * @param {string} scanId - Scan ID
 * @returns {Promise<Object>} Cancellation confirmation
 */
export const cancelScan = async (scanId) => {
  const response = await apiClient.post(`/v1/scans/${scanId}/cancel`);
  return response.data;
};

/**
 * Get scan results with pagination, sorting, and filtering
 * @param {string} scanId - Scan ID
 * @param {Object} params - Query parameters (supports all filter types)
 * @returns {Promise<Object>} Paginated scan results
 */
export const getScanResults = async (scanId, params = {}) => {
  const response = await apiClient.get(`/v1/scans/${scanId}/results`, {
    params: {
      detail_level: 'table',
      ...params,
    },
  });
  return response.data;
};

/**
 * Get lightweight filtered symbols for chart navigation
 * @param {string} scanId - Scan ID
 * @param {Object} params - Query parameters (filters/sort/passes_only)
 * @returns {Promise<Object>} Symbol list response with total + symbols
 */
export const getScanSymbols = async (scanId, params = {}) => {
  const response = await apiClient.get(`/v1/scans/${scanId}/symbols`, {
    params,
  });
  return response.data;
};

/**
 * Get filter options (unique industries, sectors, ratings) for a scan
 * @param {string} scanId - Scan ID
 * @returns {Promise<Object>} Filter options with ibd_industries, gics_sectors, ratings arrays
 */
export const getFilterOptions = async (scanId) => {
  const response = await apiClient.get(`/v1/scans/${scanId}/filter-options`);
  return response.data;
};

/**
 * Get list of all scans
 * @param {Object} params - Query parameters
 * @param {number} params.page - Page number
 * @param {number} params.per_page - Results per page
 * @returns {Promise<Array>} List of scans
 */
export const getScans = async (params = {}) => {
  const response = await apiClient.get('/v1/scans', { params });
  return response.data;
};

/**
 * Delete a scan and its results
 * @param {string} scanId - Scan ID
 * @returns {Promise<Object>} Deletion confirmation
 */
export const deleteScan = async (scanId) => {
  const response = await apiClient.delete(`/v1/scans/${scanId}`);
  return response.data;
};

/**
 * Get universe statistics
 * @returns {Promise<Object>} Universe stats (total, active, by_exchange)
 */
export const getUniverseStats = async () => {
  const response = await apiClient.get('/v1/universe/stats');
  return response.data;
};

/**
 * Export scan results to CSV
 * @param {string} scanId - Scan ID
 * @param {Object} params - Query parameters
 * @param {string} params.format - Export format (csv or excel)
 * @param {number} params.min_score - Minimum Minervini score filter
 * @param {number} params.stage - Stage filter (1-4)
 * @param {boolean} params.passes_only - Filter for passing stocks only
 * @param {number} params.min_eps_growth - Minimum EPS growth Q/Q filter (%)
 * @param {number} params.min_sales_growth - Minimum sales growth Q/Q filter (%)
 * @returns {Promise<Blob>} CSV file blob
 */
export const exportScanResults = async (scanId, params = {}) => {
  const response = await apiClient.get(`/v1/scans/${scanId}/export`, {
    params: { format: 'csv', ...params },
    responseType: 'blob',
  });
  return response.data;
};

/**
 * Get a single stock result from a scan by symbol
 * Optimized endpoint for fetching individual stock data
 * @param {string} scanId - Scan ID
 * @param {string} symbol - Stock symbol
 * @returns {Promise<Object>} Single scan result item
 */
export const getSingleResult = async (scanId, symbol, params = {}) => {
  const response = await apiClient.get(
    `/v1/scans/${scanId}/result/${symbol}`,
    {
      params: {
        detail_level: 'core',
        ...params,
      },
    }
  );
  return response.data;
};

/**
 * Get setup-engine explain payload for a single symbol
 * @param {string} scanId - Scan ID
 * @param {string} symbol - Stock symbol
 * @returns {Promise<Object>} Setup details payload
 */
export const getSetupDetails = async (scanId, symbol) => {
  const response = await apiClient.get(`/v1/scans/${scanId}/setup/${symbol}`);
  return response.data;
};

/**
 * Get all filtered symbols from scan results (across all pages)
 * Used for chart viewer navigation
 * @param {string} scanId - Scan ID
 * @param {Object} params - API filter parameters (from buildFilterParams utility)
 * @returns {Promise<Array<string>>} Complete list of symbols matching filters
 */
export const getAllFilteredSymbols = async (scanId, params = {}) => {
  try {
    const response = await getScanSymbols(scanId, params);
    if (Array.isArray(response?.symbols)) {
      return response.symbols;
    }
  } catch {
    // Fallback to paginated /results crawl for backward compatibility.
  }

  // Fetch first page to get total count
  const firstPage = await getScanResults(scanId, {
    ...params,
    page: 1,
    per_page: 100,
    include_sparklines: false,
  });

  const total = firstPage.total;
  const symbols = firstPage.results.map((r) => r.symbol);

  // If more results exist, fetch remaining pages in batches (rate-limited)
  if (total > 100) {
    const pageCount = Math.ceil(total / 100);
    const pageTasks = [];

    for (let page = 2; page <= pageCount; page++) {
      pageTasks.push(() => getScanResults(scanId, {
          ...params,
          page,
          per_page: 100,
          include_sparklines: false,
        }));
    }

    // Fetch in batches of 3 to avoid overwhelming the server
    const BATCH_SIZE = 3;
    const allResults = [];

    for (let i = 0; i < pageTasks.length; i += BATCH_SIZE) {
      const batch = pageTasks.slice(i, i + BATCH_SIZE);
      const batchResults = await Promise.all(batch.map((task) => task()));
      allResults.push(...batchResults);
    }

    allResults.forEach((pageData) => {
      symbols.push(...pageData.results.map((r) => r.symbol));
    });
  }

  return symbols;
};
