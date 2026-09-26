import { beforeEach, describe, expect, it, vi } from 'vitest';

import apiClient from './client';
import {
  getResearchJob, getResearchJobPreview, isResearchJobSettled, requestExposureResearch,
  researchJobKey, researchPreviewKey,
} from './companyExposures';

vi.mock('./client', () => ({ default: { get: vi.fn(), post: vi.fn() } }));

describe('company exposure research client', () => {
  beforeEach(() => vi.clearAllMocks());

  it('requests research with the admin key and no actor field', async () => {
    apiClient.post.mockResolvedValue({ data: { job_id: 'j1' } });
    await requestExposureResearch('secret', {
      symbol: 'EXMP', economicThemeId: 't1', idempotencyKey: 'k1',
    });
    expect(apiClient.post).toHaveBeenCalledWith(
      '/v1/company-exposures/research-requests',
      { kind: 'verify', economic_theme_id: 't1', idempotency_key: 'k1', symbol: 'EXMP' },
      { headers: { 'X-Admin-Key': 'secret' } },
    );
    expect(apiClient.post.mock.calls[0][1]).not.toHaveProperty('actor');
  });

  it('prefers an explicit security id and forwards a supplied CIK', async () => {
    apiClient.post.mockResolvedValue({ data: {} });
    await requestExposureResearch('secret', {
      kind: 'refresh', securityId: 42, economicThemeId: 't1', idempotencyKey: 'k2',
      suppliedCik: '1234567',
    });
    expect(apiClient.post.mock.calls[0][1]).toEqual({
      kind: 'refresh', economic_theme_id: 't1', idempotency_key: 'k2',
      security_id: 42, supplied_cik: '1234567',
    });
  });

  it('reads job status and the job-scoped preview without the /api prefix', async () => {
    apiClient.get.mockResolvedValue({ data: {} });
    await getResearchJob('secret', 'job/1');
    await getResearchJobPreview('secret', 'job/1');
    expect(apiClient.get.mock.calls).toEqual([
      ['/v1/company-exposures/research-jobs/job%2F1', { headers: { 'X-Admin-Key': 'secret' } }],
      ['/v1/company-exposures/research-jobs/job%2F1/preview', { headers: { 'X-Admin-Key': 'secret' } }],
    ]);
  });

  it('keeps job and preview query keys distinct', () => {
    expect(researchJobKey('j1')).toEqual(['companyExposure', 'researchJob', 'j1']);
    expect(researchPreviewKey('j1', 'r1')).toEqual(['companyExposure', 'researchPreview', 'j1', 'r1']);
  });

  it('stops polling once a job is settled or paused', () => {
    expect(isResearchJobSettled({ state: 'researching' })).toBe(false);
    expect(isResearchJobSettled({ state: 'partial' })).toBe(true);
    expect(isResearchJobSettled({ state: 'review_required' })).toBe(true);
    expect(isResearchJobSettled(undefined)).toBe(false);
  });
});
