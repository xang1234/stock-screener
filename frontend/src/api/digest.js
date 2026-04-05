import apiClient from './client';

export const getDailyDigest = async (asOfDate) => {
  const response = await apiClient.get('/v1/digest/daily', {
    params: asOfDate ? { as_of_date: asOfDate } : undefined,
  });
  return response.data;
};

export const getDailyDigestMarkdown = async (asOfDate) => {
  const response = await apiClient.get('/v1/digest/daily/markdown', {
    params: asOfDate ? { as_of_date: asOfDate } : undefined,
    responseType: 'text',
  });
  return response.data;
};
