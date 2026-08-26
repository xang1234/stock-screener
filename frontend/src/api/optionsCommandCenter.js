import apiClient from './client';

/** GET /v1/options-command-center/ -- macro SPY/QQQ levels, the six
 * universe-wide ranking tables, and generated alerts, all derived from
 * persisted options snapshots. See the backend endpoint's docstring
 * (app/api/v1/options_command_center.py) for why individual ranking lists
 * can legitimately come back short or empty. */
export async function getOptionsCommandCenterSnapshot() {
  const response = await apiClient.get('/v1/options-command-center/');
  return response.data;
}
