/* eslint-disable react-refresh/only-export-components */

import { createContext, useContext, useMemo } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import {
  getAppCapabilities,
  getDesktopSetupStatus,
  getDesktopUpdateStatus,
  runDesktopUpdateNow,
  startDesktopSetup,
} from '../api/appRuntime';
import { loginServer, logoutServer } from '../api/auth';
import { DEFAULT_SCAN_DEFAULTS } from '../constants/scanDefaults';

export const DEFAULT_DATA_STATUS = {
  local_data_present: false,
  starter_baseline_active: false,
  setup_completed_at: null,
  prices: { ready: false, last_success_at: null, message: null },
  breadth: { ready: false, last_success_at: null, message: null },
  groups: { ready: false, last_success_at: null, message: null },
  fundamentals: { ready: false, last_success_at: null, message: null },
  universe: { ready: false, last_success_at: null, message: null },
};

export const DEFAULT_SETUP = {
  status: 'completed',
  mode: null,
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
  starter_baseline_active: false,
  app_ready: true,
  data_status: DEFAULT_DATA_STATUS,
};

export const DEFAULT_UPDATE = {
  status: 'idle',
  scope: null,
  triggered_by: null,
  job_id: null,
  message: null,
  current_step: null,
  started_at: null,
  completed_at: null,
  last_success_at: null,
  current: 0,
  total: 0,
  percent: 0,
  steps: [],
  warnings: [],
  error: null,
  data_status: DEFAULT_DATA_STATUS,
};

export const DEFAULT_CAPABILITIES = {
  desktop_mode: false,
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
  api_base_path: '/api',
  bootstrap_required: false,
  bootstrap: DEFAULT_SETUP,
  setup_required: false,
  setup: DEFAULT_SETUP,
  setup_options: [],
  update: DEFAULT_UPDATE,
  data_status: DEFAULT_DATA_STATUS,
};

const RuntimeContext = createContext(null);

function getPollingInterval(status) {
  return status === 'queued' || status === 'running' ? 2000 : false;
}

export function RuntimeProvider({ children }) {
  const queryClient = useQueryClient();

  const capabilitiesQuery = useQuery({
    queryKey: ['appCapabilities'],
    queryFn: getAppCapabilities,
    placeholderData: DEFAULT_CAPABILITIES,
    retry: 1,
    staleTime: 60_000,
  });

  const desktopMode = Boolean(capabilitiesQuery.data?.desktop_mode);

  const setupQuery = useQuery({
    queryKey: ['desktopSetupStatus'],
    queryFn: getDesktopSetupStatus,
    enabled: desktopMode,
    initialData: capabilitiesQuery.data?.setup ?? DEFAULT_SETUP,
    retry: 1,
    refetchInterval: (query) => getPollingInterval(query.state.data?.status),
  });

  const updateQuery = useQuery({
    queryKey: ['desktopUpdateStatus'],
    queryFn: getDesktopUpdateStatus,
    enabled: desktopMode,
    initialData: capabilitiesQuery.data?.update ?? DEFAULT_UPDATE,
    retry: 1,
    refetchInterval: (query) => getPollingInterval(query.state.data?.status),
  });

  const startSetupMutation = useMutation({
    mutationFn: startDesktopSetup,
    onSuccess: (data) => {
      queryClient.setQueryData(['desktopSetupStatus'], data);
      queryClient.invalidateQueries({ queryKey: ['appCapabilities'] });
    },
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: ['desktopSetupStatus'] });
      queryClient.invalidateQueries({ queryKey: ['desktopUpdateStatus'] });
      queryClient.invalidateQueries({ queryKey: ['appCapabilities'] });
    },
  });

  const refreshNowMutation = useMutation({
    mutationFn: runDesktopUpdateNow,
    onSuccess: (data) => {
      queryClient.setQueryData(['desktopUpdateStatus'], data);
    },
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: ['desktopUpdateStatus'] });
      queryClient.invalidateQueries({ queryKey: ['desktopSetupStatus'] });
      queryClient.invalidateQueries({ queryKey: ['appCapabilities'] });
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

  const capabilities = capabilitiesQuery.data ?? DEFAULT_CAPABILITIES;
  const setup = desktopMode
    ? (setupQuery.data ?? capabilities.setup ?? DEFAULT_SETUP)
    : DEFAULT_SETUP;
  const update = desktopMode
    ? (updateQuery.data ?? capabilities.update ?? DEFAULT_UPDATE)
    : DEFAULT_UPDATE;
  const dataStatus = setup.data_status ?? update.data_status ?? capabilities.data_status ?? DEFAULT_DATA_STATUS;

  const value = useMemo(() => {
    const features = capabilities.features ?? DEFAULT_CAPABILITIES.features;
    const auth = capabilities.auth ?? DEFAULT_CAPABILITIES.auth;
    const setupRequired = desktopMode && !setup.app_ready;
    const setupRunning = setup.status === 'queued' || setup.status === 'running';
    const setupFailed = setup.status === 'failed';
    const updateRunning = update.status === 'queued' || update.status === 'running';
    const updateFailed = update.status === 'failed';

    return {
      capabilities,
      auth,
      bootstrap: setup,
      bootstrapIncomplete: setupRequired,
      bootstrapRunning: setupRunning,
      bootstrapFailed: setupFailed,
      bootstrapWarnings: setup.warnings ?? [],
      desktopMode,
      features,
      runtimeReady: !capabilitiesQuery.isPlaceholderData,
      uiSnapshots: capabilities.ui_snapshots ?? DEFAULT_CAPABILITIES.ui_snapshots,
      scanDefaults: capabilities.scan_defaults ?? DEFAULT_SCAN_DEFAULTS,
      setup,
      setupOptions: capabilities.setup_options ?? [],
      setupRequired,
      setupRunning,
      setupFailed,
      setupWarnings: setup.warnings ?? [],
      update,
      updateRunning,
      updateFailed,
      dataStatus,
      login: (password) => loginMutation.mutateAsync({ password }),
      logout: () => logoutMutation.mutateAsync(),
      isLoggingIn: loginMutation.isPending,
      loginError: loginMutation.error?.response?.data?.detail || loginMutation.error?.message || null,
      isLoggingOut: logoutMutation.isPending,
      startSetup: (mode = 'quick_start', force = false) => startSetupMutation.mutate({ mode, force }),
      refreshNow: (scope = 'manual', force = false) => refreshNowMutation.mutate({ scope, force }),
      isStartingSetup: startSetupMutation.isPending,
      isRefreshingNow: refreshNowMutation.isPending,
      startBootstrap: (force = false) => startSetupMutation.mutate({ mode: 'quick_start', force }),
      isStartingBootstrap: startSetupMutation.isPending,
    };
  }, [
    capabilities,
    capabilitiesQuery.isPlaceholderData,
    dataStatus,
    desktopMode,
    loginMutation,
    logoutMutation,
    refreshNowMutation,
    setup,
    startSetupMutation,
    update,
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
