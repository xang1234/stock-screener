import apiClient from './client';

export const getAppCapabilities = async () => {
  const response = await apiClient.get('/v1/app-capabilities');
  return response.data;
};
