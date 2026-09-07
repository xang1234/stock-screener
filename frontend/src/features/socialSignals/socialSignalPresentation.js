const STATE_LABELS = {
  actionable: 'Actionable', watch: 'Watch', risk_off: 'Risk-off',
  context: 'Context', unresolved: 'Needs resolution',
};

export const formatSocialScore = (value) => (
  value == null || Number.isNaN(Number(value)) ? '—' : String(Math.round(Number(value)))
);

export const socialStateLabel = (state) => STATE_LABELS[state] || String(state || 'Unknown');

export const socialFreshnessLabel = ({ available = true, stale = false } = {}) => {
  if (!available) return 'Unavailable';
  return stale ? 'Stale' : 'Fresh';
};

export const socialCoverage = (item = {}, window = '7d') => {
  const reasons = Array.isArray(item.coverage) ? item.coverage : [];
  const warming = reasons.some((reason) => /limited|warming|history/i.test(reason))
    || Number(item.observed_list_count || 0) < Number(item.enabled_list_count || 0);
  return {
    label: `${item.observed_list_count ?? 0}/${item.enabled_list_count ?? 0} lists · ${window.toUpperCase()}`,
    tone: warming ? 'warning' : 'success',
    detail: warming
      ? `Warming up${reasons.length ? ` · ${reasons.join(', ').replaceAll('_', ' ')}` : ''}`
      : 'Complete source coverage',
  };
};

export const explanationMap = (item) => item?.explanation || {};

export const visibleSocialRows = (rows = [], filters = {}) => rows.filter((row) => {
  if (['context', 'unresolved'].includes(row.state)) return false;
  const explanation = explanationMap(row);
  const haystacks = {
    source: JSON.stringify(explanation.sources || explanation.source_names || '').toLowerCase(),
    theme: JSON.stringify(explanation.themes || explanation.theme || '').toLowerCase(),
    instrument: String(explanation.security_kind || 'stock').toLowerCase(),
    state: String(row.state || '').toLowerCase(),
    ticker: String(row.canonical_symbol || '').toLowerCase(),
  };
  return Object.entries(filters).every(([key, value]) => (
    !value || (haystacks[key] || '').includes(String(value).trim().toLowerCase())
  ));
});
