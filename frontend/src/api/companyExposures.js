import apiClient from './client';

// Company-exposure research operations (shadow slice). Paths omit the /api
// prefix; the client's baseURL supplies it. Every call is an administrator
// call and sends the admin key only in its own request headers.
const BASE_PATH = '/v1/company-exposures';

const adminConfig = (adminKey) => ({ headers: { 'X-Admin-Key': adminKey } });

// Operational job status is mutable; the preview is pinned to the exact
// dossier revision the job produced. They never share a query key with
// generation-bound product data.
export const researchJobKey = (jobId) => ['companyExposure', 'researchJob', jobId];
export const researchPreviewKey = (jobId, revisionId) => [
  'companyExposure', 'researchPreview', jobId, revisionId,
];

// The server decides when a job stops making progress on its own.
export const isResearchJobSettled = (job) => Boolean(job?.settled);

export const requestExposureResearch = async (adminKey, {
  kind = 'verify', symbol, securityId, economicThemeId, idempotencyKey, suppliedCik,
}) => {
  const body = {
    kind,
    economic_theme_id: economicThemeId,
    idempotency_key: idempotencyKey,
  };
  if (securityId != null) body.security_id = securityId;
  else body.symbol = symbol;
  if (suppliedCik) body.supplied_cik = suppliedCik;
  const response = await apiClient.post(
    `${BASE_PATH}/research-requests`, body, adminConfig(adminKey),
  );
  return response.data;
};

export const getResearchJob = async (adminKey, jobId) => {
  const response = await apiClient.get(
    `${BASE_PATH}/research-jobs/${encodeURIComponent(jobId)}`, adminConfig(adminKey),
  );
  return response.data;
};

export const getResearchJobPreview = async (adminKey, jobId) => {
  const response = await apiClient.get(
    `${BASE_PATH}/research-jobs/${encodeURIComponent(jobId)}/preview`, adminConfig(adminKey),
  );
  return response.data;
};
