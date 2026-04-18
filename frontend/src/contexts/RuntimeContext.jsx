/* eslint-disable react-refresh/only-export-components */

import { createContext, useContext, useEffect, useMemo } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { setUnauthorizedResponseHandler } from '../api/client';
import {
  getAppCapabilities,
  startRuntimeBootstrap,
  updateRuntimeMarkets,
} from '../api/appRuntime';
import { loginServer, logoutServer } from '../api/auth';
import { DEFAULT_SCAN_DEFAULTS } from '../constants/scanDefaults';
import { DEFAULT_RUNTIME_ACTIVITY } from '../hooks/useRuntimeActivity';

export const DEFAULT_CAPABILITIES = {
  features: {
    themes: true,
    chatbot: true,
    tasks: true,
  },
  auth: {
    required: false,
    configured: true,
    authenticated: true,
    mode: 'session_cookie',
    message: null,
  },
  ui_snapshots: {
    enabled: false,
    scan: false,
    breadth: false,
    groups: false,
    themes: false,
  },
  scan_defaults: DEFAULT_SCAN_DEFAULTS,
  bootstrap_required: false,
  primary_market: 'US',
  enabled_markets: ['US'],
  bootstrap_state: 'not_started',
  supported_markets: ['US', 'HK', 'JP', 'TW'],
  api_base_path: '/api',
};

const RuntimeContext = createContext(null);
const BOOTSTRAP_BACKGROUND_WARNING = (
  'Data loading continues in the background after bootstrap. '
  + 'Additional enabled markets keep syncing after the app becomes usable.'
);

function buildBootstrapSeed(primaryMarket, enabledMarkets, taskId) {
  return enabledMarkets.map((market) => ({
    market,
    lifecycle: 'bootstrap',
    stage_key: 'universe',
    stage_label: 'Universe Refresh',
    status: 'queued',
    progress_mode: 'indeterminate',
    percent: null,
    current: null,
    total: null,
    message: market === primaryMarket
      ? 'Bootstrap queued.'
      : `Queued until ${primaryMarket} is ready.`,
    task_name: 'runtime_bootstrap',
    task_id: market === primaryMarket ? (taskId ?? null) : null,
    updated_at: null,
  }));
}

export function RuntimeProvider({ children }) {
  const queryClient = useQueryClient();

  useEffect(() => {
    const handleUnauthorized = () => {
      queryClient.invalidateQueries({ queryKey: ['appCapabilities'] });
    };

    setUnauthorizedResponseHandler(handleUnauthorized);
    return () => setUnauthorizedResponseHandler(null);
  }, [queryClient]);

  const capabilitiesQuery = useQuery({
    queryKey: ['appCapabilities'],
    queryFn: getAppCapabilities,
    placeholderData: DEFAULT_CAPABILITIES,
    retry: 1,
    staleTime: 60_000,
    refetchInterval: (query) => {
      const bootstrapRequired = Boolean(query.state.data?.bootstrap_required);
      const bootstrapState = query.state.data?.bootstrap_state;
      if (bootstrapRequired || bootstrapState === 'running') {
        return 15_000;
      }
      return false;
    },
  });

  const loginMutation = useMutation({
    mutationFn: ({ password }) => loginServer({ password }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['appCapabilities'] });
    },
  });

  const logoutMutation = useMutation({
    mutationFn: logoutServer,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['appCapabilities'] });
    },
  });

  const bootstrapMutation = useMutation({
    mutationFn: ({ primaryMarket, enabledMarkets }) => (
      startRuntimeBootstrap({ primaryMarket, enabledMarkets })
    ),
    onSuccess: (data) => {
      queryClient.setQueryData(['appCapabilities'], (previous) => ({
        ...(previous ?? DEFAULT_CAPABILITIES),
        bootstrap_required: Boolean(data.bootstrap_required),
        primary_market: data.primary_market ?? previous?.primary_market ?? 'US',
        enabled_markets: data.enabled_markets ?? previous?.enabled_markets ?? ['US'],
        bootstrap_state: data.bootstrap_state ?? 'running',
        supported_markets: data.supported_markets
          ?? previous?.supported_markets
          ?? ['US', 'HK', 'JP', 'TW'],
      }));
      queryClient.setQueryData(['runtimeActivity'], (previous) => {
        const primaryMarket = data.primary_market ?? 'US';
        const enabledMarkets = data.enabled_markets ?? [primaryMarket];
        return {
          ...(previous ?? DEFAULT_RUNTIME_ACTIVITY),
          bootstrap: {
            ...(previous?.bootstrap ?? DEFAULT_RUNTIME_ACTIVITY.bootstrap),
            state: data.bootstrap_state ?? 'running',
            app_ready: !data.bootstrap_required,
            primary_market: primaryMarket,
            enabled_markets: enabledMarkets,
            current_stage: 'Universe Refresh',
            progress_mode: 'indeterminate',
            percent: null,
            message: 'Bootstrap queued.',
            background_warning: enabledMarkets.length > 1
              ? BOOTSTRAP_BACKGROUND_WARNING
              : null,
          },
          summary: {
            active_market_count: enabledMarkets.length,
            active_markets: enabledMarkets,
            status: 'active',
          },
          markets: buildBootstrapSeed(primaryMarket, enabledMarkets, data.task_id),
        };
      });
      queryClient.invalidateQueries({ queryKey: ['appCapabilities'] });
      queryClient.invalidateQueries({ queryKey: ['runtimeActivity'] });
    },
  });

  const updateMarketsMutation = useMutation({
    mutationFn: ({ primaryMarket, enabledMarkets }) => (
      updateRuntimeMarkets({ primaryMarket, enabledMarkets })
    ),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['appCapabilities'] });
      queryClient.invalidateQueries({ queryKey: ['runtimeActivity'] });
    },
  });

  const capabilities = capabilitiesQuery.data ?? DEFAULT_CAPABILITIES;
  const runtimeReady = (
    !capabilitiesQuery.isPlaceholderData
    || capabilitiesQuery.isFetchedAfterMount
    || capabilitiesQuery.isError
  );

  const value = useMemo(() => {
    const features = capabilities.features ?? DEFAULT_CAPABILITIES.features;
    const auth = capabilities.auth ?? DEFAULT_CAPABILITIES.auth;

    return {
      capabilities,
      auth,
      features,
      runtimeReady,
      uiSnapshots: capabilities.ui_snapshots ?? DEFAULT_CAPABILITIES.ui_snapshots,
      scanDefaults: capabilities.scan_defaults ?? DEFAULT_SCAN_DEFAULTS,
      bootstrapRequired: Boolean(capabilities.bootstrap_required),
      primaryMarket: capabilities.primary_market ?? 'US',
      enabledMarkets: capabilities.enabled_markets ?? ['US'],
      bootstrapState: capabilities.bootstrap_state ?? 'not_started',
      supportedMarkets: capabilities.supported_markets ?? ['US', 'HK', 'JP', 'TW'],
      login: (password) => loginMutation.mutateAsync({ password }),
      logout: () => logoutMutation.mutateAsync(),
      startBootstrap: ({ primaryMarket, enabledMarkets }) => (
        bootstrapMutation.mutateAsync({ primaryMarket, enabledMarkets })
      ),
      updateMarkets: ({ primaryMarket, enabledMarkets }) => (
        updateMarketsMutation.mutateAsync({ primaryMarket, enabledMarkets })
      ),
      isLoggingIn: loginMutation.isPending,
      loginError: loginMutation.error?.response?.data?.detail || loginMutation.error?.message || null,
      isLoggingOut: logoutMutation.isPending,
      isStartingBootstrap: bootstrapMutation.isPending,
      bootstrapError: bootstrapMutation.error?.response?.data?.detail || bootstrapMutation.error?.message || null,
      isUpdatingMarkets: updateMarketsMutation.isPending,
    };
  }, [
    bootstrapMutation,
    capabilities,
    loginMutation,
    logoutMutation,
    runtimeReady,
    updateMarketsMutation,
  ]);

  return <RuntimeContext.Provider value={value}>{children}</RuntimeContext.Provider>;
}

export function useRuntime() {
  const context = useContext(RuntimeContext);
  if (!context) {
    throw new Error('useRuntime must be used within a RuntimeProvider');
  }
  return context;
}
