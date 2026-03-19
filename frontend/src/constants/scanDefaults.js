export const DEFAULT_SCAN_DEFAULTS = {
  universe: 'all',
  screeners: ['minervini', 'canslim', 'ipo', 'custom', 'volume_breakthrough', 'setup_engine'],
  composite_method: 'weighted_average',
  criteria: {
    include_vcp: true,
    custom_filters: {
      price_min: 20,
      price_max: 500,
      rs_rating_min: 75,
      volume_min: 1000000,
      market_cap_min: 1000000000,
      eps_growth_min: 20,
      sales_growth_min: 15,
      ma_alignment: true,
      min_score: 70,
    },
  },
};
