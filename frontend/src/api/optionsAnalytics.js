import apiClient from './client';
import {
  normalizeOptionsCommandCenter,
  normalizeOptionsSymbolDetail,
} from '../features/options/optionsContract';

const normalizeSymbol = (symbol) => {
  const normalized = String(symbol || '').trim().toUpperCase();
  if (!normalized) throw new Error('A symbol is required');
  return normalized;
};

export const getOptionsCommandCenter = async () => {
  const response = await apiClient.get('/v1/options-analytics/command-center');
  return normalizeOptionsCommandCenter(response.data);
};

export const getOptionsSymbolDetail = async (symbol) => {
  const normalized = normalizeSymbol(symbol);
  const response = await apiClient.get(`/v1/options-analytics/symbols/${encodeURIComponent(normalized)}`);
  return normalizeOptionsSymbolDetail(response.data, { expectedSymbol: normalized });
};

export const refreshOptionsAnalytics = async ({ sourceRunId = null, force = false } = {}) => {
  const response = await apiClient.post('/v1/options-analytics/refresh', {
    source_run_id: sourceRunId,
    force,
  });
  return response.data;
};
