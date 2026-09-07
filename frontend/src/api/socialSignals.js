import apiClient from './client';

const cleanFilters = (filters = {}) => Object.fromEntries(
  Object.entries(filters).filter(([, value]) => value != null && String(value).trim() !== ''),
);

const filterKey = (filters = {}) => {
  const params = new URLSearchParams();
  Object.entries(cleanFilters(filters)).sort(([left], [right]) => left.localeCompare(right))
    .forEach(([key, value]) => params.set(key, value));
  return params.toString();
};

export const socialQueueKey = ({
  market, window = '7d', view = 'actionable', rankMode = 'blended', page = 1,
  pageSize = 50, filters = {},
}) => [
  'socialSignals', 'queue', market, window, view, rankMode, page, pageSize,
  filterKey(filters),
];

export const getSocialQueue = async ({
  market, window = '7d', view = 'actionable', rankMode = 'blended', page = 1,
  pageSize = 50, filters = {},
}) => {
  const response = await apiClient.get('/v1/social-signals/queue', { params: {
    market, window, view, rank_mode: rankMode, page, page_size: pageSize,
    ...cleanFilters(filters),
  } });
  return response.data;
};

export const getSocialContext = async ({ market, window = '7d', page = 1, pageSize = 50 }) => {
  const response = await apiClient.get('/v1/social-signals/context', {
    params: { market, window, page, page_size: pageSize },
  });
  return response.data;
};

export const getSocialUnresolved = async ({ market, window = '7d', page = 1, pageSize = 50 }) => {
  const response = await apiClient.get('/v1/social-signals/unresolved', {
    params: { scope: 'unknown', market, window, page, page_size: pageSize },
  });
  return response.data;
};

export const getSocialEvidence = async (candidateKey, window = '7d') => {
  const response = await apiClient.get(
    `/v1/social-signals/candidates/${encodeURIComponent(candidateKey)}/evidence`,
    { params: { window } },
  );
  return response.data;
};

export const getSocialSummary = async (market) => {
  const response = await apiClient.get('/v1/social-signals/summary', { params: { market } });
  return response.data;
};

export const getSocialThemePulse = async (market) => {
  const response = await apiClient.get('/v1/social-signals/theme-pulse', { params: { market } });
  return response.data;
};
