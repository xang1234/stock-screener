/**
 * API client for per-market telemetry endpoints (beads asia.10.1 + 10.2).
 */
import apiClient from './client';

/** Latest gauges + today's counters across every supported market. */
export const fetchMarketSummaries = async () => {
  const response = await apiClient.get('/v1/telemetry/markets');
  return response.data;
};

/** Detailed summary for one market. */
export const fetchMarketDetail = async (market) => {
  const response = await apiClient.get(`/v1/telemetry/markets/${encodeURIComponent(market)}`);
  return response.data;
};

/** Recent raw events for one (market, metric_key). Capped at 15 days, 1000 rows. */
export const fetchMetricHistory = async (market, metricKey, { days = 15, limit = 200 } = {}) => {
  const response = await apiClient.get(
    `/v1/telemetry/markets/${encodeURIComponent(market)}/${encodeURIComponent(metricKey)}`,
    { params: { days, limit } },
  );
  return response.data;
};

/**
 * Active alerts (open + acknowledged). When ``evaluate`` is true (default),
 * the backend re-evaluates thresholds before returning so the result reflects
 * current gauge state.
 */
export const fetchAlerts = async ({ evaluate = true } = {}) => {
  const response = await apiClient.get('/v1/telemetry/alerts', { params: { evaluate } });
  return response.data;
};

/** Acknowledge an open alert. ``acknowledgedBy`` is recorded on the row. */
export const acknowledgeAlert = async (alertId, acknowledgedBy) => {
  const response = await apiClient.post(
    `/v1/telemetry/alerts/${alertId}/acknowledge`,
    { acknowledged_by: acknowledgedBy },
  );
  return response.data;
};
