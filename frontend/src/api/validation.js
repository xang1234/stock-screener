import apiClient from './client';

export const getValidationOverview = async (sourceKind = 'scan_pick', lookbackDays = 90) => {
  const response = await apiClient.get('/v1/validation/overview', {
    params: {
      source_kind: sourceKind,
      lookback_days: lookbackDays,
    },
  });
  return response.data;
};
