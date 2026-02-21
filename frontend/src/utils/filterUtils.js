/**
 * Filter utilities for converting frontend filter state to API parameters
 */

/**
 * Convert frontend filter state object to API query parameters
 * @param {Object} filters - The filter state from ScanPage
 * @param {Object} options - Additional options like page, perPage, sortBy, sortOrder
 * @returns {Object} API-compatible query parameters
 */
export const buildFilterParams = (filters, options = {}) => {
  const params = {};

  // Add pagination/sorting if provided
  if (options.page) params.page = options.page;
  if (options.perPage) params.per_page = options.perPage;
  if (options.sortBy) params.sort_by = options.sortBy;
  if (options.sortOrder) params.sort_order = options.sortOrder;

  // Text search
  if (filters.symbolSearch) params.symbol_search = filters.symbolSearch;

  // Categorical
  if (filters.stage != null) params.stage = filters.stage;
  if (filters.ratings?.length) params.ratings = filters.ratings.join(',');

  // IBD Industries - supports both old array format and new object format with mode
  if (filters.ibdIndustries?.values?.length) {
    params.ibd_industries = filters.ibdIndustries.values.join(',');
    if (filters.ibdIndustries.mode === 'exclude') {
      params.ibd_industries_mode = 'exclude';
    }
  } else if (Array.isArray(filters.ibdIndustries) && filters.ibdIndustries.length) {
    // Backward compatibility with old array format
    params.ibd_industries = filters.ibdIndustries.join(',');
  }

  // GICS Sectors - supports both old array format and new object format with mode
  if (filters.gicsSectors?.values?.length) {
    params.gics_sectors = filters.gicsSectors.values.join(',');
    if (filters.gicsSectors.mode === 'exclude') {
      params.gics_sectors_mode = 'exclude';
    }
  } else if (Array.isArray(filters.gicsSectors) && filters.gicsSectors.length) {
    // Backward compatibility with old array format
    params.gics_sectors = filters.gicsSectors.join(',');
  }

  // Score ranges
  if (filters.compositeScore?.min != null) params.min_composite = filters.compositeScore.min;
  if (filters.compositeScore?.max != null) params.max_composite = filters.compositeScore.max;
  if (filters.minerviniScore?.min != null) params.min_score = filters.minerviniScore.min;
  if (filters.minerviniScore?.max != null) params.max_score = filters.minerviniScore.max;
  if (filters.canslimScore?.min != null) params.min_canslim = filters.canslimScore.min;
  if (filters.canslimScore?.max != null) params.max_canslim = filters.canslimScore.max;
  if (filters.ipoScore?.min != null) params.min_ipo = filters.ipoScore.min;
  if (filters.ipoScore?.max != null) params.max_ipo = filters.ipoScore.max;
  if (filters.customScore?.min != null) params.min_custom = filters.customScore.min;
  if (filters.customScore?.max != null) params.max_custom = filters.customScore.max;
  if (filters.volBreakthroughScore?.min != null) params.min_vol_breakthrough = filters.volBreakthroughScore.min;
  if (filters.volBreakthroughScore?.max != null) params.max_vol_breakthrough = filters.volBreakthroughScore.max;

  // Setup Engine
  if (filters.seSetupScore?.min != null) params.min_se_setup_score = filters.seSetupScore.min;
  if (filters.seSetupScore?.max != null) params.max_se_setup_score = filters.seSetupScore.max;
  if (filters.seDistanceToPivot?.min != null) params.min_se_distance_to_pivot_pct = filters.seDistanceToPivot.min;
  if (filters.seDistanceToPivot?.max != null) params.max_se_distance_to_pivot_pct = filters.seDistanceToPivot.max;
  if (filters.seBbSqueeze?.min != null) params.min_se_bb_width_pctile_252 = filters.seBbSqueeze.min;
  if (filters.seBbSqueeze?.max != null) params.max_se_bb_width_pctile_252 = filters.seBbSqueeze.max;
  if (filters.seVolumeVs50d?.min != null) params.min_se_volume_vs_50d = filters.seVolumeVs50d.min;
  if (filters.seVolumeVs50d?.max != null) params.max_se_volume_vs_50d = filters.seVolumeVs50d.max;
  if (filters.seSetupReady != null) params.se_setup_ready = filters.seSetupReady;
  if (filters.seRsLineNewHigh != null) params.se_rs_line_new_high = filters.seRsLineNewHigh;

  // RS ranges
  if (filters.rsRating?.min != null) params.min_rs = filters.rsRating.min;
  if (filters.rsRating?.max != null) params.max_rs = filters.rsRating.max;
  if (filters.rs1m?.min != null) params.min_rs_1m = filters.rs1m.min;
  if (filters.rs1m?.max != null) params.max_rs_1m = filters.rs1m.max;
  if (filters.rs3m?.min != null) params.min_rs_3m = filters.rs3m.min;
  if (filters.rs3m?.max != null) params.max_rs_3m = filters.rs3m.max;
  if (filters.rs12m?.min != null) params.min_rs_12m = filters.rs12m.min;
  if (filters.rs12m?.max != null) params.max_rs_12m = filters.rs12m.max;

  // EPS Rating
  if (filters.epsRating?.min != null) params.min_eps_rating = filters.epsRating.min;
  if (filters.epsRating?.max != null) params.max_eps_rating = filters.epsRating.max;

  // Price & Growth
  if (filters.price?.min != null) params.min_price = filters.price.min;
  if (filters.price?.max != null) params.max_price = filters.price.max;
  if (filters.adrPercent?.min != null) params.min_adr = filters.adrPercent.min;
  if (filters.adrPercent?.max != null) params.max_adr = filters.adrPercent.max;
  if (filters.epsGrowth?.min != null) params.min_eps_growth = filters.epsGrowth.min;
  if (filters.epsGrowth?.max != null) params.max_eps_growth = filters.epsGrowth.max;
  if (filters.salesGrowth?.min != null) params.min_sales_growth = filters.salesGrowth.min;
  if (filters.salesGrowth?.max != null) params.max_sales_growth = filters.salesGrowth.max;

  // VCP
  if (filters.vcpScore?.min != null) params.min_vcp_score = filters.vcpScore.min;
  if (filters.vcpScore?.max != null) params.max_vcp_score = filters.vcpScore.max;
  if (filters.vcpPivot?.min != null) params.min_vcp_pivot = filters.vcpPivot.min;
  if (filters.vcpPivot?.max != null) params.max_vcp_pivot = filters.vcpPivot.max;
  if (filters.vcpDetected != null) params.vcp_detected = filters.vcpDetected;
  if (filters.vcpReady != null) params.vcp_ready = filters.vcpReady;

  // Booleans
  if (filters.maAlignment != null) params.ma_alignment = filters.maAlignment;
  if (filters.passesTemplate != null) params.passes_only = filters.passesTemplate;

  // Volume & Market Cap
  if (filters.minVolume != null) params.min_volume = filters.minVolume;
  if (filters.minMarketCap != null) params.min_market_cap = filters.minMarketCap;

  // Performance filters (price change %)
  if (filters.perfDay?.min != null) params.min_perf_day = filters.perfDay.min;
  if (filters.perfDay?.max != null) params.max_perf_day = filters.perfDay.max;
  if (filters.perfWeek?.min != null) params.min_perf_week = filters.perfWeek.min;
  if (filters.perfWeek?.max != null) params.max_perf_week = filters.perfWeek.max;
  if (filters.perfMonth?.min != null) params.min_perf_month = filters.perfMonth.min;
  if (filters.perfMonth?.max != null) params.max_perf_month = filters.perfMonth.max;

  // Qullamaggie extended performance filters
  if (filters.perf3m?.min != null) params.min_perf_3m = filters.perf3m.min;
  if (filters.perf3m?.max != null) params.max_perf_3m = filters.perf3m.max;
  if (filters.perf6m?.min != null) params.min_perf_6m = filters.perf6m.min;
  if (filters.perf6m?.max != null) params.max_perf_6m = filters.perf6m.max;

  // Episodic Pivot filters
  if (filters.gapPercent?.min != null) params.min_gap_percent = filters.gapPercent.min;
  if (filters.gapPercent?.max != null) params.max_gap_percent = filters.gapPercent.max;
  if (filters.volumeSurge?.min != null) params.min_volume_surge = filters.volumeSurge.min;
  if (filters.volumeSurge?.max != null) params.max_volume_surge = filters.volumeSurge.max;

  // EMA distance filters
  if (filters.ema10Distance?.min != null) params.min_ema_10 = filters.ema10Distance.min;
  if (filters.ema10Distance?.max != null) params.max_ema_10 = filters.ema10Distance.max;
  if (filters.ema20Distance?.min != null) params.min_ema_20 = filters.ema20Distance.min;
  if (filters.ema20Distance?.max != null) params.max_ema_20 = filters.ema20Distance.max;
  if (filters.ema50Distance?.min != null) params.min_ema_50 = filters.ema50Distance.min;
  if (filters.ema50Distance?.max != null) params.max_ema_50 = filters.ema50Distance.max;

  // 52-week distance filters
  if (filters.week52HighDistance?.min != null) params.min_52w_high = filters.week52HighDistance.min;
  if (filters.week52HighDistance?.max != null) params.max_52w_high = filters.week52HighDistance.max;
  if (filters.week52LowDistance?.min != null) params.min_52w_low = filters.week52LowDistance.min;
  if (filters.week52LowDistance?.max != null) params.max_52w_low = filters.week52LowDistance.max;

  // IPO date filter
  if (filters.ipoAfter) params.ipo_after = filters.ipoAfter;

  // Beta and Beta-Adjusted RS filters
  if (filters.beta?.min != null) params.min_beta = filters.beta.min;
  if (filters.beta?.max != null) params.max_beta = filters.beta.max;
  if (filters.betaAdjRs?.min != null) params.min_beta_adj_rs = filters.betaAdjRs.min;
  if (filters.betaAdjRs?.max != null) params.max_beta_adj_rs = filters.betaAdjRs.max;

  return params;
};

/**
 * Generate a stable cache key for react-query based on filters
 * Only includes non-null/non-empty active filters, sorted deterministically
 * @param {Object} filters - The filter state
 * @returns {string} JSON string for use in query keys
 */
export const getStableFilterKey = (filters) => {
  const activeFilters = {};

  // Helper to check if a value is "active" (not null/undefined/empty)
  const isActive = (value) => {
    if (value === null || value === undefined) return false;
    if (typeof value === 'string') return value.length > 0;
    if (Array.isArray(value)) return value.length > 0;
    if (typeof value === 'object') {
      // Range filters { min, max }
      if ('min' in value && 'max' in value) {
        return value.min != null || value.max != null;
      }
      // Multi-select with mode { values, mode }
      if ('values' in value) {
        return value.values?.length > 0;
      }
    }
    return true;
  };

  // Only include filters that have active values
  Object.keys(filters).sort().forEach(key => {
    const value = filters[key];
    if (isActive(value)) {
      // Normalize range filters to only include non-null values
      if (typeof value === 'object' && !Array.isArray(value)) {
        if ('min' in value && 'max' in value) {
          const normalized = {};
          if (value.min != null) normalized.min = value.min;
          if (value.max != null) normalized.max = value.max;
          if (Object.keys(normalized).length > 0) {
            activeFilters[key] = normalized;
          }
        } else if ('values' in value && value.values?.length > 0) {
          activeFilters[key] = {
            values: [...value.values].sort(),
            mode: value.mode || 'include'
          };
        } else {
          activeFilters[key] = value;
        }
      } else if (Array.isArray(value)) {
        activeFilters[key] = [...value].sort();
      } else {
        activeFilters[key] = value;
      }
    }
  });

  return JSON.stringify(activeFilters);
};

// Keep the old function for backward compatibility but mark as deprecated
/** @deprecated Use getStableFilterKey instead */
export const getFilterCacheKey = (filters) => {
  return JSON.stringify(filters);
};
