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

export const DEFAULT_MARKET_CATALOG = {
  version: 'fallback.v1',
  markets: [
    {
      code: 'US',
      label: 'United States',
      currency: 'USD',
      timezone: 'America/New_York',
      calendar_id: 'XNYS',
      exchanges: ['NYSE', 'NASDAQ', 'AMEX'],
      indexes: ['SP500'],
      capabilities: {},
    },
    {
      code: 'HK',
      label: 'Hong Kong',
      currency: 'HKD',
      timezone: 'Asia/Hong_Kong',
      calendar_id: 'XHKG',
      exchanges: ['HKEX', 'SEHK', 'XHKG'],
      indexes: ['HSI'],
      capabilities: {},
    },
    {
      code: 'IN',
      label: 'India',
      currency: 'INR',
      timezone: 'Asia/Kolkata',
      calendar_id: 'XNSE',
      exchanges: ['NSE', 'XNSE', 'BSE', 'XBOM'],
      indexes: [],
      capabilities: {},
    },
    {
      code: 'JP',
      label: 'Japan',
      currency: 'JPY',
      timezone: 'Asia/Tokyo',
      calendar_id: 'XTKS',
      exchanges: ['TSE', 'JPX', 'XTKS'],
      indexes: ['NIKKEI225'],
      capabilities: {},
    },
    {
      code: 'KR',
      label: 'South Korea',
      currency: 'KRW',
      timezone: 'Asia/Seoul',
      calendar_id: 'XKRX',
      exchanges: ['KOSPI', 'KOSDAQ', 'KRX', 'XKRX'],
      indexes: [],
      capabilities: {},
    },
    {
      code: 'TW',
      label: 'Taiwan',
      currency: 'TWD',
      timezone: 'Asia/Taipei',
      calendar_id: 'XTAI',
      exchanges: ['TWSE', 'TPEX', 'XTAI'],
      indexes: ['TAIEX'],
      capabilities: {},
    },
    {
      code: 'CN',
      label: 'China A-shares',
      currency: 'CNY',
      timezone: 'Asia/Shanghai',
      calendar_id: 'XSHG',
      exchanges: ['SSE', 'SZSE', 'BJSE', 'XSHG', 'XSHE', 'XBSE'],
      indexes: [],
      capabilities: {},
    },
    {
      code: 'CA',
      label: 'Canada',
      currency: 'CAD',
      timezone: 'America/Toronto',
      calendar_id: 'XTSE',
      exchanges: ['TSX', 'TSXV', 'XTSE', 'XTNX'],
      indexes: ['TSX_COMPOSITE'],
      capabilities: {},
    },
    {
      code: 'DE',
      label: 'Germany',
      currency: 'EUR',
      timezone: 'Europe/Berlin',
      calendar_id: 'XETR',
      exchanges: ['XETR', 'XETRA', 'XFRA', 'FRA', 'FWB'],
      indexes: ['DAX', 'MDAX', 'SDAX'],
      capabilities: {},
    },
    {
      code: 'SG',
      label: 'Singapore',
      currency: 'SGD',
      timezone: 'Asia/Singapore',
      calendar_id: 'XSES',
      exchanges: ['SGX', 'SES', 'XSES'],
      indexes: ['STI'],
      capabilities: {},
    },
  ],
};

const DEFAULT_SUPPORTED_MARKETS = DEFAULT_MARKET_CATALOG.markets
  .map((market) => market.code)
  .filter(Boolean);

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
  market_catalog: DEFAULT_MARKET_CATALOG,
  supported_markets: DEFAULT_SUPPORTED_MARKETS,
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

function supportedMarketsFromCatalog(marketCatalog) {
  const markets = marketCatalog?.markets;
  if (!Array.isArray(markets) || markets.length === 0) {
    return null;
  }
  return markets.map((market) => market.code).filter(Boolean);
}

export function mergeBootstrapCapabilities(previous, data) {
  return {
    ...(previous ?? DEFAULT_CAPABILITIES),
    bootstrap_required: Boolean(data.bootstrap_required),
    primary_market: data.primary_market ?? previous?.primary_market ?? 'US',
    enabled_markets: data.enabled_markets ?? previous?.enabled_markets ?? ['US'],
    bootstrap_state: data.bootstrap_state ?? 'running',
    market_catalog: data.market_catalog
      ?? previous?.market_catalog
      ?? DEFAULT_CAPABILITIES.market_catalog,
    supported_markets: data.supported_markets
      ?? previous?.supported_markets
      ?? supportedMarketsFromCatalog(data.market_catalog)
      ?? supportedMarketsFromCatalog(previous?.market_catalog)
      ?? DEFAULT_CAPABILITIES.supported_markets,
  };
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
      queryClient.setQueryData(['appCapabilities'], (previous) => (
        mergeBootstrapCapabilities(previous, data)
      ));
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
    const marketCatalog = capabilities.market_catalog ?? DEFAULT_CAPABILITIES.market_catalog;
    const supportedMarkets = capabilities.supported_markets
      ?? supportedMarketsFromCatalog(marketCatalog)
      ?? supportedMarketsFromCatalog(DEFAULT_MARKET_CATALOG);

    return {
      capabilities,
      auth,
      features,
      marketCatalog,
      runtimeReady,
      uiSnapshots: capabilities.ui_snapshots ?? DEFAULT_CAPABILITIES.ui_snapshots,
      scanDefaults: capabilities.scan_defaults ?? DEFAULT_SCAN_DEFAULTS,
      bootstrapRequired: Boolean(capabilities.bootstrap_required),
      primaryMarket: capabilities.primary_market ?? 'US',
      enabledMarkets: capabilities.enabled_markets ?? ['US'],
      bootstrapState: capabilities.bootstrap_state ?? 'not_started',
      supportedMarkets,
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
