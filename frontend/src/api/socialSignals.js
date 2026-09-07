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

const adminConfig = (adminKey, config = {}) => ({
  ...config,
  headers: { ...(config.headers || {}), 'X-Admin-Key': adminKey },
});

export const getSocialAdminRuntime = async (adminKey) => (
  (await apiClient.get('/v1/social-signals/admin/runtime', adminConfig(adminKey))).data
);
export const updateSocialAdminRuntime = async (adminKey, body) => (
  (await apiClient.patch('/v1/social-signals/admin/runtime', body, adminConfig(adminKey))).data
);
export const getSocialAdminHealth = async (adminKey) => (
  (await apiClient.get('/v1/social-signals/admin/health', adminConfig(adminKey))).data
);
export const getSocialAdminSources = async (adminKey, includeArchived = false) => (
  (await apiClient.get('/v1/social-signals/admin/sources', adminConfig(adminKey, {
    params: { include_archived: includeArchived },
  }))).data
);
export const createSocialSource = async (adminKey, body) => (
  (await apiClient.post('/v1/social-signals/admin/sources', body, adminConfig(adminKey))).data
);
export const renameSocialSource = async (adminKey, sourceId, name, expectedVersion) => (
  (await apiClient.patch(`/v1/social-signals/admin/sources/${sourceId}`,
    { name, expected_version: expectedVersion }, adminConfig(adminKey))).data
);
export const testSocialSource = async (adminKey, sourceId, expectedVersion) => (
  (await apiClient.post(`/v1/social-signals/admin/sources/${sourceId}/test`,
    { expected_version: expectedVersion }, adminConfig(adminKey))).data
);
export const transitionSocialSource = async (
  adminKey, sourceId, target, expectedVersion,
) => ((await apiClient.post(`/v1/social-signals/admin/sources/${sourceId}/transition`,
  { target, expected_version: expectedVersion }, adminConfig(adminKey))).data);
export const refreshSocialSignals = async (adminKey) => (
  (await apiClient.post('/v1/social-signals/admin/refresh', null, adminConfig(adminKey))).data
);
export const getSocialAnalysis = async (adminKey, state = null) => (
  (await apiClient.get('/v1/social-signals/admin/analysis', adminConfig(adminKey, {
    params: state ? { state } : {},
  }))).data
);
export const retrySocialAnalysis = async (adminKey, workId) => (
  (await apiClient.post(`/v1/social-signals/admin/analysis/${workId}/retry`, null,
    adminConfig(adminKey))).data
);
export const getSocialAssociations = async (adminKey) => (
  (await apiClient.get('/v1/social-signals/admin/associations', adminConfig(adminKey))).data
);
export const decideSocialAssociation = async (adminKey, associationId, body) => (
  (await apiClient.post(`/v1/social-signals/admin/associations/${associationId}/decision`,
    body, adminConfig(adminKey))).data
);
export const getSocialCompanyIdentities = async (adminKey) => (
  (await apiClient.get('/v1/social-signals/admin/company-identities', adminConfig(adminKey))).data
);
export const updateSocialCompanyIdentities = async (adminKey, body) => (
  (await apiClient.patch('/v1/social-signals/admin/company-identities', body,
    adminConfig(adminKey))).data
);
export const getSocialRuns = async (adminKey) => (
  (await apiClient.get('/v1/social-signals/admin/runs', adminConfig(adminKey))).data
);
export const getSocialValidation = async (adminKey, runId) => (
  (await apiClient.get(`/v1/social-signals/admin/validation/${encodeURIComponent(runId)}`,
    adminConfig(adminKey))).data
);
