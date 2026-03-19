/* eslint-disable react-refresh/only-export-components */

import { createContext, useContext, useEffect, useMemo, useRef } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import {
  getAppCapabilities,
  getDesktopBootstrapStatus,
  startDesktopBootstrap,
} from '../api/appRuntime';
import { DEFAULT_SCAN_DEFAULTS } from '../constants/scanDefaults';

export const DEFAULT_BOOTSTRAP = {
  status: 'completed',
  job_id: null,
  message: null,
  current_step: null,
  started_at: null,
  completed_at: null,
  current: 0,
  total: 0,
  percent: 100,
  steps: [],
  warnings: [],
  error: null,
};

export const DEFAULT_CAPABILITIES = {
  desktop_mode: false,
  features: {
    themes: true,
    chatbot: true,
    tasks: true,
  },
  ui_snapshots: {
    enabled: false,
    scan: false,
    breadth: false,
    groups: false,
    themes: false,
  },
  scan_defaults: DEFAULT_SCAN_DEFAULTS,
  api_base_path: '/api',
  bootstrap_required: false,
  bootstrap: DEFAULT_BOOTSTRAP,
};

const RuntimeContext = createContext(null);

export function RuntimeProvider({ children }) {
  const queryClient = useQueryClient();
  const autoBootstrapRequested = useRef(false);

  const capabilitiesQuery = useQuery({
    queryKey: ['appCapabilities'],
    queryFn: getAppCapabilities,
    placeholderData: DEFAULT_CAPABILITIES,
    retry: 1,
    staleTime: 60_000,
  });

  const desktopMode = Boolean(capabilitiesQuery.data?.desktop_mode);

  const bootstrapQuery = useQuery({
    queryKey: ['desktopBootstrapStatus'],
    queryFn: getDesktopBootstrapStatus,
    enabled: desktopMode,
    initialData: capabilitiesQuery.data?.bootstrap ?? DEFAULT_BOOTSTRAP,
    retry: 1,
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      return status === 'queued' || status === 'running' ? 2000 : false;
    },
  });

  const startBootstrapMutation = useMutation({
    mutationFn: startDesktopBootstrap,
    onSuccess: (data) => {
      queryClient.setQueryData(['desktopBootstrapStatus'], data);
      queryClient.invalidateQueries({ queryKey: ['appCapabilities'] });
    },
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: ['desktopBootstrapStatus'] });
    },
  });

  useEffect(() => {
    const bootstrap = bootstrapQuery.data ?? capabilitiesQuery.data?.bootstrap ?? DEFAULT_BOOTSTRAP;
    if (!desktopMode || autoBootstrapRequested.current) {
      return;
    }
    if (bootstrap.status === 'idle') {
      autoBootstrapRequested.current = true;
      startBootstrapMutation.mutate(false);
    }
  }, [desktopMode, bootstrapQuery.data, capabilitiesQuery.data, startBootstrapMutation]);

  const capabilities = capabilitiesQuery.data ?? DEFAULT_CAPABILITIES;
  const bootstrap = desktopMode
    ? (bootstrapQuery.data ?? capabilities.bootstrap ?? DEFAULT_BOOTSTRAP)
    : DEFAULT_BOOTSTRAP;

  const value = useMemo(() => {
    const features = capabilities.features ?? DEFAULT_CAPABILITIES.features;
    const bootstrapIncomplete = desktopMode && bootstrap.status !== 'completed';

    return {
      capabilities,
      bootstrap,
      desktopMode,
      features,
      runtimeReady: !capabilitiesQuery.isPlaceholderData,
      uiSnapshots: capabilities.ui_snapshots ?? DEFAULT_CAPABILITIES.ui_snapshots,
      scanDefaults: capabilities.scan_defaults ?? DEFAULT_SCAN_DEFAULTS,
      bootstrapIncomplete,
      bootstrapRunning: bootstrap.status === 'queued' || bootstrap.status === 'running',
      bootstrapFailed: bootstrap.status === 'failed',
      bootstrapWarnings: bootstrap.warnings ?? [],
      startBootstrap: (force = false) => startBootstrapMutation.mutate(force),
      isStartingBootstrap: startBootstrapMutation.isPending,
    };
  }, [bootstrap, capabilities, capabilitiesQuery.isPlaceholderData, desktopMode, startBootstrapMutation]);

  return <RuntimeContext.Provider value={value}>{children}</RuntimeContext.Provider>;
}

export function useRuntime() {
  const context = useContext(RuntimeContext);
  if (!context) {
    throw new Error('useRuntime must be used within a RuntimeProvider');
  }
  return context;
}
