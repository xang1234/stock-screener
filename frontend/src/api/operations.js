import apiClient from './client';

export const fetchOperationsJobs = async () => {
  const response = await apiClient.get('/v1/operations/jobs');
  return response.data;
};

export const cancelOperationsJob = async (taskId) => {
  const response = await apiClient.post(`/v1/operations/jobs/${taskId}/cancel`);
  return response.data;
};

