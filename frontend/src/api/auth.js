import apiClient from './client';

export const loginServer = async ({ password }) => {
  const response = await apiClient.post('/v1/auth/login', { password });
  return response.data;
};

export const logoutServer = async () => {
  const response = await apiClient.post('/v1/auth/logout');
  return response.data;
};

export const getServerAuthStatus = async () => {
  const response = await apiClient.get('/v1/auth/status');
  return response.data;
};
