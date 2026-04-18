import apiClient from './client';

export const getAppCapabilities = async () => {
  const response = await apiClient.get('/v1/app-capabilities');
  return response.data;
};

export const getBootstrapStatus = async () => {
  const response = await apiClient.get('/v1/runtime/bootstrap-status');
  return response.data;
};

export const getRuntimeActivity = async () => {
  const response = await apiClient.get('/v1/runtime/activity');
  return response.data;
};

export const startRuntimeBootstrap = async ({ primaryMarket, enabledMarkets }) => {
  const response = await apiClient.post('/v1/runtime/bootstrap', {
    primary_market: primaryMarket,
    enabled_markets: enabledMarkets,
  });
  return response.data;
};

export const updateRuntimeMarkets = async ({ primaryMarket, enabledMarkets }) => {
  const response = await apiClient.patch('/v1/runtime/markets', {
    primary_market: primaryMarket,
    enabled_markets: enabledMarkets,
  });
  return response.data;
};
