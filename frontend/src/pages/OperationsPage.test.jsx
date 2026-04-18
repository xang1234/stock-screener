import { fireEvent, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { renderWithProviders } from '../test/renderWithProviders';
import OperationsPage from './OperationsPage';

const fetchAlerts = vi.fn();
const acknowledgeAlert = vi.fn();
const fetchOperationsJobs = vi.fn();
const cancelOperationsJob = vi.fn();
const useRuntimeActivity = vi.fn();

vi.mock('../api/telemetry', () => ({
  fetchAlerts: (...args) => fetchAlerts(...args),
  acknowledgeAlert: (...args) => acknowledgeAlert(...args),
}));

vi.mock('../api/operations', () => ({
  fetchOperationsJobs: (...args) => fetchOperationsJobs(...args),
  cancelOperationsJob: (...args) => cancelOperationsJob(...args),
}));

vi.mock('../hooks/useRuntimeActivity', () => ({
  useRuntimeActivity: (...args) => useRuntimeActivity(...args),
}));

const SUMMARY_US = {
  market: 'US',
  freshness_lag: { lag_seconds: 1800, last_refresh_at_epoch: 1700000000 },
  benchmark_age: { age_seconds: 3600, benchmark_symbol: 'SPY' },
  universe_drift: { current_size: 5000, prior_size: 4990, delta: 10 },
  completeness_distribution: {
    bucket_counts: { '0-25': 50, '25-50': 100, '50-75': 200, '75-90': 1000, '90-100': 3650 },
    symbols_total: 5000,
  },
  extraction_today: { by_language: {} },
};

const ALERT_US = {
  id: 1,
  market: 'US',
  metric_key: 'freshness_lag',
  severity: 'warning',
  state: 'open',
  owner: 'us-ops',
  title: '[WARNING] freshness_lag on US',
  description: 'freshness_lag = 9000.00 crossed the warning threshold (7200) on market US.',
  metrics: { value: 9000, thresholds: { warning: 7200, critical: 21600 } },
  opened_at: '2026-04-15T12:34:56+00:00',
  acknowledged_at: null,
  acknowledged_by: null,
  closed_at: null,
};

const OPERATIONS_PAYLOAD = {
  jobs: [
    {
      task_id: 'task-fetch-hk',
      task_name: 'app.tasks.cache_tasks.smart_refresh_cache',
      queue: 'data_fetch_hk',
      market: 'HK',
      state: 'waiting',
      worker: null,
      age_seconds: 45,
      wait_reason: 'waiting_for_external_fetch_global',
      heartbeat_lag_seconds: null,
      cancel_strategy: 'revoke_and_remove_from_queue',
      progress_mode: 'determinate',
      percent: 60,
      current: 600,
      total: 1000,
      message: 'Batch 3/5 · refreshing prices',
    },
    {
      task_id: 'task-scan-us',
      task_name: 'app.tasks.scan_tasks.run_bulk_scan',
      queue: 'user_scans_us',
      market: 'US',
      state: 'running',
      worker: 'userscans-us@host',
      age_seconds: 120,
      wait_reason: null,
      heartbeat_lag_seconds: null,
      cancel_strategy: 'scan_cancel',
      progress_mode: 'indeterminate',
      percent: null,
      current: null,
      total: null,
      message: 'Running scan',
    },
  ],
  queues: [
    { queue: 'data_fetch_hk', depth: 1, oldest_age_seconds: 45 },
    { queue: 'user_scans_us', depth: 0, oldest_age_seconds: null },
  ],
  workers: [
    { worker: 'general@host', status: 'online', queues: ['celery'], active: 0, reserved: 0, scheduled: 0 },
    { worker: 'userscans-us@host', status: 'online', queues: ['user_scans_us'], active: 1, reserved: 0, scheduled: 0 },
  ],
  leases: {
    external_fetch_global: {
      task_id: 'task-fetch-us',
      task_name: 'app.tasks.cache_tasks.smart_refresh_cache',
      started_at: '2026-04-18T11:50:00Z',
    },
    market_workload: {
      US: { task_id: 'task-scan-us', task_name: 'app.tasks.scan_tasks.run_bulk_scan' },
      HK: null,
      JP: null,
      TW: null,
    },
  },
  generated_at: '2026-04-18T12:00:00Z',
};

describe('OperationsPage', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    fetchAlerts.mockResolvedValue({ summaries: [SUMMARY_US], alerts: [ALERT_US] });
    fetchOperationsJobs.mockResolvedValue(OPERATIONS_PAYLOAD);
    cancelOperationsJob.mockResolvedValue({
      status: 'accepted',
      cancel_strategy: 'scan_cancel',
      message: 'Cancelled task-scan-us',
    });
    useRuntimeActivity.mockReturnValue({
      data: {
        bootstrap: {
          background_warning: 'Additional data loading continues in the background.',
        },
        markets: [
          {
            market: 'US',
            lifecycle: 'daily_refresh',
            stage_label: 'Price Refresh',
            status: 'running',
            percent: 42,
            current: 42,
            total: 100,
            message: 'Refreshing prices',
            task_name: 'smart_refresh_cache',
            updated_at: '2026-04-18T12:00:00Z',
          },
        ],
      },
      isLoading: false,
      isError: false,
    });
  });

  it('renders market activity, job console, and telemetry sections', async () => {
    renderWithProviders(<OperationsPage />);

    expect(screen.getByText('Market activity')).toBeInTheDocument();
    expect(screen.getByText('Job console')).toBeInTheDocument();
    expect(screen.getByText('Lease status')).toBeInTheDocument();
    expect(screen.getByText('Price Refresh')).toBeInTheDocument();
    expect(screen.getByText(/Refreshing prices/)).toBeInTheDocument();
    expect(screen.getByText(/Additional data loading continues in the background/)).toBeInTheDocument();

    await waitFor(() => {
      expect(screen.getAllByText('app.tasks.cache_tasks.smart_refresh_cache').length).toBeGreaterThan(0);
    });
    expect(screen.getByText('waiting_for_external_fetch_global')).toBeInTheDocument();
    expect(screen.getByText(/600\/1000/)).toBeInTheDocument();
    expect(screen.getByText(/Batch 3\/5 · refreshing prices/)).toBeInTheDocument();
    expect(screen.getByText('External Fetch Lease')).toBeInTheDocument();
    expect(screen.getByText('Workers')).toBeInTheDocument();
    expect(screen.getAllByText('Queues').length).toBeGreaterThan(0);

    await waitFor(() => {
      expect(screen.getByText(/Freshness:/)).toBeInTheDocument();
    });
    expect(screen.queryByText(/No active alerts/)).not.toBeInTheDocument();
    expect(screen.getByText('warning')).toBeInTheDocument();
    expect(screen.getByText('open')).toBeInTheDocument();
  });

  it('clicking Ack calls acknowledgeAlert', async () => {
    acknowledgeAlert.mockResolvedValue({ ...ALERT_US, state: 'acknowledged' });
    renderWithProviders(<OperationsPage />);

    const ackButton = await screen.findByRole('button', { name: /ack/i });
    fireEvent.click(ackButton);

    await waitFor(() => {
      expect(acknowledgeAlert).toHaveBeenCalledWith(1, 'operator');
    });
  });

  it('clicking a job action calls cancelOperationsJob', async () => {
    renderWithProviders(<OperationsPage />);

    const cancelButtons = await screen.findAllByRole('button', { name: /cancel/i });
    fireEvent.click(cancelButtons[0]);

    await waitFor(() => {
      expect(cancelOperationsJob).toHaveBeenCalled();
    });
    expect(cancelOperationsJob.mock.calls[0][0]).toBe('task-fetch-hk');
  });

  it('clears the Working state after a cancel mutation settles', async () => {
    let resolveCancel;
    cancelOperationsJob.mockImplementation(
      () =>
        new Promise((resolve) => {
          resolveCancel = resolve;
        })
    );

    renderWithProviders(<OperationsPage />);

    const cancelButton = (await screen.findAllByRole('button', { name: /cancel/i }))[0];
    fireEvent.click(cancelButton);

    expect(await screen.findByRole('button', { name: /working/i })).toBeDisabled();

    resolveCancel({
      status: 'accepted',
      cancel_strategy: 'revoke_and_remove_from_queue',
      message: 'Removed task-fetch-hk',
    });

    await waitFor(() => {
      expect(screen.queryByRole('button', { name: /working/i })).not.toBeInTheDocument();
      expect(screen.getAllByRole('button', { name: /cancel/i })[0]).toBeEnabled();
    });
  });

  it('shows empty-state message when no alerts active', async () => {
    fetchAlerts.mockResolvedValue({ summaries: [SUMMARY_US], alerts: [] });
    renderWithProviders(<OperationsPage />);

    await waitFor(() => {
      expect(screen.getByText(/No active alerts/)).toBeInTheDocument();
    });
  });

  it('keeps runtime activity cards visible when the activity poll errors', async () => {
    useRuntimeActivity.mockReturnValue({
      data: {
        bootstrap: {
          background_warning: 'Additional data loading continues in the background.',
        },
        markets: [
          {
            market: 'US',
            lifecycle: 'daily_refresh',
            stage_label: 'Price Refresh',
            status: 'running',
            percent: 42,
            current: 42,
            total: 100,
            message: 'Refreshing prices',
            task_name: 'smart_refresh_cache',
            updated_at: '2026-04-18T12:00:00Z',
          },
        ],
      },
      isLoading: false,
      isError: true,
    });

    renderWithProviders(<OperationsPage />);

    expect(screen.getByText('Price Refresh')).toBeInTheDocument();
    expect(screen.getByText(/Refreshing prices/)).toBeInTheDocument();
    expect(screen.getByText(/Failed to refresh runtime activity/)).toBeInTheDocument();
  });

  it('derives percent for determinate jobs that only expose counts', async () => {
    fetchOperationsJobs.mockResolvedValue({
      ...OPERATIONS_PAYLOAD,
      jobs: [
        {
          task_id: 'task-fetch-hk',
          task_name: 'app.tasks.cache_tasks.smart_refresh_cache',
          queue: 'data_fetch_hk',
          market: 'HK',
          state: 'running',
          worker: 'datafetch-global@host',
          age_seconds: 45,
          wait_reason: null,
          heartbeat_lag_seconds: 5,
          cancel_strategy: 'unsupported',
          progress_mode: 'determinate',
          percent: null,
          current: 600,
          total: 1000,
          message: 'Batch 3/5 · refreshing prices',
        },
      ],
    });

    renderWithProviders(<OperationsPage />);

    expect(await screen.findByText(/60%/)).toBeInTheDocument();
    expect(screen.getByText(/600\/1000/)).toBeInTheDocument();
    expect(screen.getByText(/Batch 3\/5 · refreshing prices/)).toBeInTheDocument();
  });
});
