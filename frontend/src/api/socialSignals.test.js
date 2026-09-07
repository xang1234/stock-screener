import { beforeEach, describe, expect, it, vi } from 'vitest';

import apiClient from './client';
import {
  createSocialSource, getSocialAdminHealth, getSocialAdminSources,
  getSocialContext, getSocialEvidence, getSocialQueue, getSocialUnresolved,
  testSocialSource,
  socialQueueKey,
} from './socialSignals';

vi.mock('./client', () => ({ default: { get: vi.fn() } }));

describe('Social Signal read client', () => {
  beforeEach(() => vi.clearAllMocks());

  it('sends exact public controls without an admin credential', async () => {
    apiClient.get.mockResolvedValue({ data: { items: [] } });
    const controls = {
      market: 'HK', window: '14d', view: 'all', rankMode: 'pure_social',
      page: 2, pageSize: 25,
      filters: { source: 'Asia', theme: 'chips', instrument: 'stock',
        state: 'watch', ticker: '0700' },
    };
    await getSocialQueue(controls);
    expect(apiClient.get).toHaveBeenCalledWith('/v1/social-signals/queue', {
      params: {
        market: 'HK', window: '14d', view: 'all', rank_mode: 'pure_social',
        page: 2, page_size: 25, source: 'Asia', theme: 'chips',
        instrument: 'stock', state: 'watch', ticker: '0700',
      },
    });
    expect(apiClient.get.mock.calls[0][1].headers).toBeUndefined();
    expect(socialQueueKey(controls)).toEqual([
      'socialSignals', 'queue', 'HK', '14d', 'all', 'pure_social', 2, 25,
      'instrument=stock&source=Asia&state=watch&theme=chips&ticker=0700',
    ]);
  });

  it('uses separately cached context, global unresolved, and encoded evidence routes', async () => {
    apiClient.get.mockResolvedValue({ data: { items: [] } });
    await getSocialContext({ market: 'JP', window: '7d', page: 1, pageSize: 50 });
    await getSocialUnresolved({ market: 'JP', window: '7d', page: 3, pageSize: 10 });
    await getSocialEvidence('JP:6758/primary', '1d');
    expect(apiClient.get.mock.calls).toEqual([
      ['/v1/social-signals/context', { params: {
        market: 'JP', window: '7d', page: 1, page_size: 50,
      } }],
      ['/v1/social-signals/unresolved', { params: {
        scope: 'unknown', market: 'JP', window: '7d', page: 3, page_size: 10,
      } }],
      ['/v1/social-signals/candidates/JP%3A6758%2Fprimary/evidence', {
        params: { window: '1d' },
      }],
    ]);
  });

  it('sends the admin key only on administrator calls', async () => {
    apiClient.get.mockResolvedValue({ data: {} });
    apiClient.post = vi.fn().mockResolvedValue({ data: {} });
    await getSocialAdminHealth('secret');
    await getSocialAdminSources('secret', true);
    await createSocialSource('secret', { name: 'Readable', list_ref: '123' });
    await testSocialSource('secret', 9, 3);
    expect(apiClient.get.mock.calls).toEqual([
      ['/v1/social-signals/admin/health', { headers: { 'X-Admin-Key': 'secret' } }],
      ['/v1/social-signals/admin/sources', {
        params: { include_archived: true }, headers: { 'X-Admin-Key': 'secret' },
      }],
    ]);
    expect(apiClient.post.mock.calls).toEqual([
      ['/v1/social-signals/admin/sources', { name: 'Readable', list_ref: '123' },
        { headers: { 'X-Admin-Key': 'secret' } }],
      ['/v1/social-signals/admin/sources/9/test', { expected_version: 3 },
        { headers: { 'X-Admin-Key': 'secret' } }],
    ]);
  });
});
