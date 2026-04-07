/* eslint-disable react-refresh/only-export-components */

import { createContext, useContext, useEffect, useMemo } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { setUnauthorizedResponseHandler } from '../api/client';
import { getAppCapabilities } from '../api/appRuntime';
import { loginServer, logoutServer } from '../api/auth';
import { DEFAULT_SCAN_DEFAULTS } from '../constants/scanDefaults';

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
  api_base_path: '/api',
};

const RuntimeContext = createContext(null);

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
      login: (password) => loginMutation.mutateAsync({ password }),
      logout: () => logoutMutation.mutateAsync(),
      isLoggingIn: loginMutation.isPending,
      loginError: loginMutation.error?.response?.data?.detail || loginMutation.error?.message || null,
      isLoggingOut: logoutMutation.isPending,
    };
  }, [
    capabilities,
    loginMutation,
    logoutMutation,
    runtimeReady,
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
