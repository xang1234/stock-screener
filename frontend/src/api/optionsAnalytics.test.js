import { beforeEach, describe, expect, it, vi } from 'vitest';

import apiClient from './client';
import {
  getOptionsCommandCenter,
  getOptionsSymbolDetail,
  refreshOptionsAnalytics,
} from './optionsAnalytics';
import {
  commandCenterFixture,
  symbolDetailFixture,
} from '../features/options/__fixtures__/optionsResponses';

vi.mock('./client', () => ({
  default: { get: vi.fn(), post: vi.fn() },
}));

describe('live options analytics client', () => {
  beforeEach(() => vi.clearAllMocks());

  it('uses only protected options-analytics routes and normalizes responses', async () => {
    apiClient.get
      .mockResolvedValueOnce({ data: commandCenterFixture })
      .mockResolvedValueOnce({ data: symbolDetailFixture });

    expect((await getOptionsCommandCenter()).run_id).toBe(7);
    expect((await getOptionsSymbolDetail('aapl')).item.symbol).toBe('AAPL');
    expect(apiClient.get.mock.calls).toEqual([
      ['/v1/options-analytics/command-center'],
      ['/v1/options-analytics/symbols/AAPL'],
    ]);
  });

  it('posts one explicit refresh request', async () => {
    apiClient.post.mockResolvedValue({ data: { status: 'accepted', task_id: 'task-1' } });

    const result = await refreshOptionsAnalytics({ sourceRunId: 33, force: true });

    expect(result.task_id).toBe('task-1');
    expect(apiClient.post).toHaveBeenCalledWith('/v1/options-analytics/refresh', {
      source_run_id: 33,
      force: true,
    });
  });
});
