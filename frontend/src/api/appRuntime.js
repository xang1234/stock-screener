import apiClient from './client';

export const getAppCapabilities = async () => {
  const response = await apiClient.get('/v1/app-capabilities');
  return response.data;
};

export const startDesktopBootstrap = async (force = false) => {
  const response = await apiClient.post('/v1/app/bootstrap', null, {
    params: { force },
  });
  return response.data;
};

export const getDesktopBootstrapStatus = async () => {
  const response = await apiClient.get('/v1/app/bootstrap/status');
  return response.data;
};
