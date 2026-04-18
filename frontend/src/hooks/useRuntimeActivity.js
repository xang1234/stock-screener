import { useQuery } from '@tanstack/react-query';
import { getRuntimeActivity } from '../api/appRuntime';

export const DEFAULT_RUNTIME_ACTIVITY = {
  bootstrap: {
    state: 'idle',
    app_ready: true,
    primary_market: 'US',
    enabled_markets: ['US'],
    current_stage: null,
    percent: 0,
    message: null,
    background_warning: null,
  },
  summary: {
    active_market_count: 0,
    active_markets: [],
    status: 'idle',
  },
  markets: [],
};

export function useRuntimeActivity({ enabled = true } = {}) {
  return useQuery({
    queryKey: ['runtimeActivity'],
    queryFn: getRuntimeActivity,
    enabled,
    placeholderData: DEFAULT_RUNTIME_ACTIVITY,
    retry: 1,
    staleTime: 3_000,
    refetchInterval: enabled
      ? (query) => {
        const data = query.state.data ?? DEFAULT_RUNTIME_ACTIVITY;
        const isBootstrapRunning = data.bootstrap?.state === 'running';
        const activeCount = data.summary?.active_market_count ?? 0;
        const hasWarning = data.summary?.status === 'warning';
        if (isBootstrapRunning || activeCount > 0 || hasWarning) {
          return 5_000;
        }
        return 30_000;
      }
      : false,
  });
}
