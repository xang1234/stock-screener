import { act, render } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { PipelineProvider } from './PipelineContext';
import { usePipeline } from './usePipeline';

const api = vi.hoisted(() => ({ getPipelineStatus: vi.fn() }));
vi.mock('../api/themes', () => api);

let pipeline;
function Capture() {
  pipeline = usePipeline();
  return null;
}

describe('PipelineProvider', () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  it('stops polling once a run reports skipped', async () => {
    // #472: a run queued before cutover to economic authority is skipped.
    vi.useFakeTimers();
    api.getPipelineStatus.mockResolvedValue({ status: 'skipped', error_message: 'economic_authority' });
    render(
      <QueryClientProvider client={new QueryClient()}>
        <PipelineProvider>
          <Capture />
        </PipelineProvider>
      </QueryClientProvider>,
    );

    act(() => pipeline.startPipeline('run-1'));
    await act(async () => {
      await vi.advanceTimersByTimeAsync(10000);
    });

    expect(api.getPipelineStatus).toHaveBeenCalledTimes(1);
    expect(pipeline.isPipelineRunning).toBe(false);
    expect(pipeline.isCardVisible).toBe(true);
  });

  it('keeps the latest status when an earlier response arrives late', async () => {
    vi.useFakeTimers();
    let resolveSlow;
    api.getPipelineStatus
      .mockImplementationOnce(() => new Promise((resolve) => { resolveSlow = resolve; }))
      .mockResolvedValue({ status: 'completed', percent: 100 });
    const queryClient = new QueryClient();
    const invalidate = vi.spyOn(queryClient, 'invalidateQueries');
    render(
      <QueryClientProvider client={queryClient}>
        <PipelineProvider>
          <Capture />
        </PipelineProvider>
      </QueryClientProvider>,
    );

    act(() => pipeline.startPipeline('run-2'));
    await act(async () => {
      await vi.advanceTimersByTimeAsync(5000);
    });
    await act(async () => {
      resolveSlow({ status: 'running', percent: 20 });
    });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(10000);
    });

    expect(pipeline.pipelineStatus.status).toBe('completed');
    expect(pipeline.isPipelineRunning).toBe(false);
    expect(invalidate).toHaveBeenCalledWith({ queryKey: ['themeRankings'] });
    const calls = api.getPipelineStatus.mock.calls.length;
    await act(async () => {
      await vi.advanceTimersByTimeAsync(10000);
    });
    expect(api.getPipelineStatus).toHaveBeenCalledTimes(calls);
  });
});
