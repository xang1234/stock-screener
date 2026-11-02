const STATE_LABELS = {
  actionable: 'Actionable', watch: 'Watch', risk_off: 'Risk-off',
  context: 'Context', unresolved: 'Needs resolution',
};

const STATE_REASON_TEXT = {
  market_context: 'Broad market or macro instrument',
  unresolved_security: 'Ticker not matched to a listing',
  inactive_security: 'Listing is inactive',
  unsupported_market: 'Market not supported',
  unknown_feature_freshness: 'Price feature date unknown',
  stale_features: 'Price features out of date',
  unknown_market_freshness: 'Market exposure date unknown',
  stale_market: 'Market exposure out of date',
  missing_liquidity: 'Trading volume unknown',
  liquidity_rejected: 'Trading volume below the minimum',
  missing_setup_readiness: 'Setup readiness unknown',
  setup_not_ready: 'Setup not ready',
  missing_setup_score: 'Setup score unavailable',
  invalid_setup_score: 'Setup score unavailable',
  missing_market_exposure: 'Market exposure unavailable',
  invalid_market_exposure: 'Market exposure unavailable',
  market_exposure_below_50: 'Market exposure below 50',
};

export const ATTEMPT_TITLES = {
  collection_failed: 'Latest collection did not publish',
  failed: 'Latest analysis failed',
  processing: 'Posts collected; analysis is in progress',
  collecting: 'Collection is in progress',
  published: 'Latest run published',
};

export const formatSocialScore = (value) => (
  value == null || Number.isNaN(Number(value)) ? '—' : String(Math.round(Number(value)))
);

export const socialStateLabel = (state) => STATE_LABELS[state] || String(state || 'Unknown');

export const stateReasonText = (code) => {
  const text = STATE_REASON_TEXT[code] || String(code || '').replaceAll('_', ' ');
  return text.charAt(0).toUpperCase() + text.slice(1);
};

const age = (minutes) => (
  minutes < 60 ? `${minutes}m` : minutes < 1440 ? `${Math.floor(minutes / 60)}h` : `${Math.floor(minutes / 1440)}d`
);

export const socialFreshness = ({ stale = false, published_at: publishedAt } = {}, now = Date.now()) => {
  const at = Date.parse(publishedAt);
  if (Number.isNaN(at)) return { label: 'Update time unknown', color: 'default' };
  if (stale) return { label: `Delayed, last update ${new Date(at).toLocaleString()}`, color: 'warning' };
  return { label: `Updated ${age(Math.max(0, Math.floor((now - at) / 60000)))} ago`, color: 'success' };
};

export const explanationMap = (item) => item?.explanation || {};

export const visibleSocialRows = (rows = [], filters = {}) => rows.filter((row) => {
  if (['context', 'unresolved'].includes(row.state)) return false;
  const explanation = explanationMap(row);
  const haystacks = {
    source: JSON.stringify(explanation.source_names || '').toLowerCase(),
    theme: JSON.stringify(explanation.theme || '').toLowerCase(),
    instrument: String(explanation.security_kind || 'stock').toLowerCase(),
    state: String(row.state || '').toLowerCase(),
    ticker: String(row.canonical_symbol || '').toLowerCase(),
  };
  return Object.entries(filters).every(([key, value]) => (
    !value || (haystacks[key] || '').includes(String(value).trim().toLowerCase())
  ));
});
