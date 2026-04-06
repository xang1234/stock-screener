import apiClient from './client';

export const getDailyDigest = async (asOfDate, profile) => {
  const response = await apiClient.get('/v1/digest/daily', {
    params: {
      ...(asOfDate ? { as_of_date: asOfDate } : {}),
      ...(profile ? { profile } : {}),
    },
  });
  return response.data;
};

export const getDailyDigestMarkdown = async (asOfDate, profile) => {
  const response = await apiClient.get('/v1/digest/daily/markdown', {
    params: {
      ...(asOfDate ? { as_of_date: asOfDate } : {}),
      ...(profile ? { profile } : {}),
    },
    responseType: 'text',
  });
  return response.data;
};
