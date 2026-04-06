import apiClient from './client';

export const getStrategyProfiles = async () => {
  const response = await apiClient.get('/v1/strategy-profiles');
  return response.data;
};

export const getStrategyProfile = async (profile) => {
  const response = await apiClient.get(`/v1/strategy-profiles/${encodeURIComponent(profile)}`);
  return response.data;
};
