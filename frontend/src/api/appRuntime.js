import apiClient from './client';

export const getAppCapabilities = async () => {
  const response = await apiClient.get('/v1/app-capabilities');
  return response.data;
};

export const startDesktopSetup = async ({ mode = 'quick_start', force = false } = {}) => {
  const response = await apiClient.post('/v1/app/setup', null, {
    params: { mode, force },
  });
  return response.data;
};

export const getDesktopSetupStatus = async () => {
  const response = await apiClient.get('/v1/app/setup/status');
  return response.data;
};

export const runDesktopUpdateNow = async ({ scope = 'manual', force = false } = {}) => {
  const response = await apiClient.post('/v1/app/update/run-now', null, {
    params: { scope, force },
  });
  return response.data;
};

export const getDesktopUpdateStatus = async () => {
  const response = await apiClient.get('/v1/app/update/status');
  return response.data;
};
