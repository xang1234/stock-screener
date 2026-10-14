import { act, fireEvent, screen, waitFor, within } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { renderWithProviders } from '../test/renderWithProviders';
import OperationsPage from './OperationsPage';

const fetchAlerts = vi.fn();
const acknowledgeAlert = vi.fn();
const fetchOperationsJobs = vi.fn();
const cancelOperationsJob = vi.fn();
const releasePipelineChain = vi.fn();
const useRuntimeActivity = vi.fn();
const getBootstrapStatus = vi.fn();
const runtimeState = vi.hoisted(() => ({ features: { themes: true } }));

vi.mock('../contexts/RuntimeContext', () => ({
  useRuntime: () => runtimeState,
}));

vi.mock('../api/telemetry', () => ({
  fetchAlerts: (...args) => fetchAlerts(...args),
  acknowledgeAlert: (...args) => acknowledgeAlert(...args),
}));

vi.mock('../api/operations', () => ({
  fetchOperationsJobs: (...args) => fetchOperationsJobs(...args),
  cancelOperationsJob: (...args) => cancelOperationsJob(...args),
  releasePipelineChain: (...args) => releasePipelineChain(...args),
}));

vi.mock('../api/appRuntime', () => ({
  getBootstrapStatus: (...args) => getBootstrapStatus(...args),
}));

vi.mock('../hooks/useRuntimeActivity', () => ({
  useRuntimeActivity: (...args) => useRuntimeActivity(...args),
}));

vi.mock('../api/tasks', () => ({
  getScheduledTasks: () => Promise.resolve({ tasks: [], total_tasks: 0 }),
  triggerTask: vi.fn(),
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
    runtimeState.features = { themes: true };
    delete runtimeState.enabledMarkets;
    delete runtimeState.supportedMarkets;
    vi.clearAllMocks();
    fetchAlerts.mockResolvedValue({ summaries: [SUMMARY_US], alerts: [ALERT_US] });
    fetchOperationsJobs.mockResolvedValue(OPERATIONS_PAYLOAD);
    getBootstrapStatus.mockResolvedValue({
      market_stages: { US: { breadth: 'ready', exposure: 'ready', groups: 'ready' } },
      features: { cot: 'ready', options: 'disabled', social: 'disabled' },
    });
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
            stage_label: 'Scan',
            status: 'running',
            percent: 42,
            current: 42,
            total: 100,
            message: 'Running market scan',
            task_name: 'build_daily_snapshot',
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
    expect(screen.getByText('Scan')).toBeInTheDocument();
    expect(screen.getByText(/Running market scan/)).toBeInTheDocument();
    expect(screen.getByText(/Additional data loading continues in the background/)).toBeInTheDocument();

    await waitFor(() => {
      expect(screen.getAllByText('app.tasks.cache_tasks.smart_refresh_cache').length).toBeGreaterThan(0);
    });
    expect(screen.getByText('waiting_for_external_fetch_global')).toBeInTheDocument();
    expect(screen.getByText(/600\/1000/)).toBeInTheDocument();
    expect(screen.getByText(/Batch 3\/5 · refreshing prices/)).toBeInTheDocument();
    expect(screen.getByText('External Fetch Lease')).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: 'Workers' })).toBeInTheDocument();
    expect(screen.getAllByText('Queues').length).toBeGreaterThan(0);

    await waitFor(() => {
      expect(screen.getByText(/Freshness:/)).toBeInTheDocument();
    });
    expect(screen.queryByText(/No active alerts/)).not.toBeInTheDocument();
    expect(screen.getByText('warning')).toBeInTheDocument();
    expect(screen.getByText('open')).toBeInTheDocument();
  });

  it('shows scheduled jobs only when the tasks feature is enabled', async () => {
    const { unmount } = renderWithProviders(<OperationsPage />);
    expect(screen.queryByRole('heading', { name: 'Scheduled jobs' })).not.toBeInTheDocument();
    unmount();

    runtimeState.features = { themes: true, tasks: true };
    renderWithProviders(<OperationsPage />);

    expect(screen.getByRole('heading', { name: 'Scheduled jobs' })).toBeInTheDocument();
    expect(await screen.findByText(/No scheduled tasks/)).toBeInTheDocument();
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

  const confirmCancel = async (index = 0) => {
    const cancelButtons = await screen.findAllByRole('button', { name: /cancel/i });
    fireEvent.click(cancelButtons[index]);
    const dialog = await screen.findByRole('dialog');
    fireEvent.click(within(dialog).getAllByRole('button').at(-1));
    return dialog;
  };

  it('confirms with the plain-language effect before cancelling, and sends the strategy it showed', async () => {
    renderWithProviders(<OperationsPage />);

    const cancelButtons = await screen.findAllByRole('button', { name: /cancel/i });
    fireEvent.click(cancelButtons[0]);

    const dialog = await screen.findByRole('dialog');
    expect(within(dialog).getByText('Remove from queue — it never runs.')).toBeInTheDocument();
    expect(cancelOperationsJob).not.toHaveBeenCalled();

    fireEvent.click(within(dialog).getByRole('button', { name: 'Cancel' }));

    await waitFor(() => {
      expect(cancelOperationsJob).toHaveBeenCalledWith('task-fetch-hk', 'revoke_and_remove_from_queue');
    });
  });

  describe('when the jobs request fails (#567)', () => {
    it('says so in every jobs section instead of rendering empty data as healthy', async () => {
      fetchOperationsJobs.mockRejectedValue(new Error('backend down'));
      renderWithProviders(<OperationsPage />);

      // The jobs table, Lease status, Queues and Workers.
      await waitFor(
        () => expect(screen.getAllByText('Failed to load Operations job inventory.')).toHaveLength(4),
        { timeout: 4000 },
      );
      // An empty fallback would claim nothing holds the lease.
      expect(screen.queryByText('No task currently holds the external fetch lease.')).not.toBeInTheDocument();
      expect(screen.queryByText('No pipeline chain claims are held.')).not.toBeInTheDocument();
      expect(screen.queryByText('No queue data available.')).not.toBeInTheDocument();
      expect(screen.queryByText('No worker inspect data available.')).not.toBeInTheDocument();
    });

    it('keeps the last known data, marked as such, when a refetch fails', async () => {
      const { queryClient } = renderWithProviders(<OperationsPage />);
      expect(await screen.findByText('External Fetch Lease')).toBeInTheDocument();

      fetchOperationsJobs.mockRejectedValue(new Error('backend down'));
      await act(() => queryClient.refetchQueries({ queryKey: ['operations', 'jobs'] }));

      await waitFor(
        () => expect(screen.getAllByText(/Couldn't refresh\. Showing the last known status from/)).toHaveLength(4),
        { timeout: 4000 },
      );
      // With the date: data from yesterday must not read as today's.
      expect(screen.getAllByText(new RegExp(new Date().toLocaleDateString().replace(/[.*+?^${}()|[\]\\/]/g, '\\$&')))
        .length).toBeGreaterThanOrEqual(4);
      // The lease holder from the last success is still shown, not hidden.
      expect(screen.getAllByText(/task-fetch-us/).length).toBeGreaterThan(0);
      expect(screen.queryByText('Failed to load Operations job inventory.')).not.toBeInTheDocument();
    });
  });

  describe('pipeline chain claims', () => {
    const withClaim = {
      ...OPERATIONS_PAYLOAD,
      leases: {
        ...OPERATIONS_PAYLOAD.leases,
        pipeline_chain: {
          HK: { task_id: 'tok-hk', task_name: 'daily_market_pipeline', started_at: '2026-10-13T08:30:00Z', ttl_seconds: 7200 },
        },
      },
    };

    it('says when no pipeline holds a claim', async () => {
      renderWithProviders(<OperationsPage />);

      expect(await screen.findByText('No pipeline chain claims are held.')).toBeInTheDocument();
    });

    it('releases a claim only after confirming, with the token it showed', async () => {
      fetchOperationsJobs.mockResolvedValue(withClaim);
      releasePipelineChain.mockResolvedValue({ status: 'accepted', cancel_strategy: 'release_pipeline_chain', message: 'ok' });
      renderWithProviders(<OperationsPage />);

      expect(await screen.findByText('Claim: tok-hk')).toBeInTheDocument();
      fireEvent.click(screen.getByRole('button', { name: 'Release HK pipeline chain claim' }));
      const dialog = await screen.findByRole('dialog');
      expect(within(dialog).getByText(/otherwise the next Run starts a second pipeline/)).toBeInTheDocument();
      expect(releasePipelineChain).not.toHaveBeenCalled();

      fireEvent.click(within(dialog).getByRole('button', { name: 'Release' }));

      await waitFor(() => expect(releasePipelineChain).toHaveBeenCalledWith('HK', 'tok-hk'));
    });

    it('says why when the claim was already gone', async () => {
      fetchOperationsJobs.mockResolvedValue(withClaim);
      releasePipelineChain.mockResolvedValue({
        status: 'blocked', cancel_strategy: 'release_pipeline_chain', message: 'The HK pipeline chain claim tok-hk is no longer held.',
      });
      renderWithProviders(<OperationsPage />);

      fireEvent.click(await screen.findByRole('button', { name: 'Release HK pipeline chain claim' }));
      fireEvent.click(within(await screen.findByRole('dialog')).getByRole('button', { name: 'Release' }));

      expect(await screen.findByText(/is no longer held/)).toBeInTheDocument();
    });
  });

  it('backing out of the dialog does not cancel', async () => {
    renderWithProviders(<OperationsPage />);

    fireEvent.click((await screen.findAllByRole('button', { name: /cancel/i }))[0]);
    fireEvent.click(within(await screen.findByRole('dialog')).getByRole('button', { name: 'Back' }));

    await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument());
    expect(cancelOperationsJob).not.toHaveBeenCalled();
  });

  it('says why when the job moved before the cancel landed (409)', async () => {
    cancelOperationsJob.mockRejectedValue({
      response: { status: 409, data: { detail: 'Job moved to running, so this action no longer applies. Refresh and retry.' } },
    });
    renderWithProviders(<OperationsPage />);

    await confirmCancel();

    expect(await screen.findByText(/Job moved to running/)).toBeInTheDocument();
  });

  it('shows Cancel requested instead of the button until the job leaves', async () => {
    fetchOperationsJobs.mockResolvedValue({
      ...OPERATIONS_PAYLOAD,
      jobs: [
        {
          ...OPERATIONS_PAYLOAD.jobs[1],
          cancel_requested_at: '2026-10-13T10:00:00+00:00',
          cancel_requested_strategy: 'scan_cancel',
        },
      ],
    });
    renderWithProviders(<OperationsPage />);

    expect(await screen.findByText('Cancel requested')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Cancel scan' })).not.toBeInTheDocument();
  });

  it('labels the live force-release action instead of Unavailable', async () => {
    fetchOperationsJobs.mockResolvedValue({
      ...OPERATIONS_PAYLOAD,
      jobs: [
        {
          ...OPERATIONS_PAYLOAD.jobs[1],
          state: 'stale',
          cancel_strategy: 'force_release_market_lease',
        },
      ],
    });
    renderWithProviders(<OperationsPage />);

    fireEvent.click(await screen.findByRole('button', { name: 'Force release' }));
    expect(
      within(await screen.findByRole('dialog')).getByText(/the worker may still be running; use only if it is dead/),
    ).toBeInTheDocument();
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

    await confirmCancel();

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

  it('opens with market coverage that names a market enabled without workers', async () => {
    runtimeState.enabledMarkets = ['US', 'JP'];
    runtimeState.supportedMarkets = ['US', 'HK', 'JP'];
    fetchOperationsJobs.mockResolvedValue({
      ...OPERATIONS_PAYLOAD,
      queues: [
        { queue: 'user_scans_us', depth: 0, oldest_age_seconds: null, consumers: 1 },
        { queue: 'data_fetch_jp', depth: 0, oldest_age_seconds: null, consumers: 0 },
      ],
    });

    renderWithProviders(<OperationsPage />);

    expect(await screen.findByText('1 market needs attention: JP.')).toBeInTheDocument();
    expect(screen.getByText('No worker fetches its prices (data_fetch_jp).')).toBeInTheDocument();
    // HK has a queued job, so it is active and gets a coverage row too.
    expect(screen.getAllByRole('rowheader').map((cell) => cell.textContent)).toEqual(['US', 'JP', 'HK']);
  });

  it('puts queues, idle leases and telemetry of inactive markets behind disclosures', async () => {
    runtimeState.enabledMarkets = ['US'];
    runtimeState.supportedMarkets = ['US', 'HK', 'DE'];
    fetchAlerts.mockResolvedValue({ summaries: [SUMMARY_US, { ...SUMMARY_US, market: 'DE' }], alerts: [] });
    fetchOperationsJobs.mockResolvedValue({
      ...OPERATIONS_PAYLOAD,
      queues: [
        ...OPERATIONS_PAYLOAD.queues,
        { queue: 'data_fetch_de', depth: 0, oldest_age_seconds: null, consumers: 0 },
      ],
      leases: {
        ...OPERATIONS_PAYLOAD.leases,
        market_workload: { US: OPERATIONS_PAYLOAD.leases.market_workload.US, HK: null, DE: null },
      },
    });

    renderWithProviders(<OperationsPage />);

    // HK has a queued job, so it stays; DE has nothing, so it waits behind
    // one disclosure per section.
    const toggles = await screen.findAllByRole('button', { name: 'Inactive markets (1): DE' });
    expect(toggles).toHaveLength(3);
    expect(screen.getAllByText('data_fetch_hk').length).toBeGreaterThan(0);
    expect(screen.queryByText('data_fetch_de')).not.toBeInTheDocument();
    expect(screen.getByText('HK:')).toBeInTheDocument();
    expect(screen.queryByText('DE:')).not.toBeInTheDocument();

    toggles.forEach((toggle) => fireEvent.click(toggle));

    expect(screen.getByText('data_fetch_de')).toBeVisible();
    expect(screen.getByText('DE:')).toBeVisible();
    expect(screen.getByText('DE')).toBeVisible();
  });

  it('says when a queued job sits on a queue no worker reads', async () => {
    fetchOperationsJobs.mockResolvedValue({
      ...OPERATIONS_PAYLOAD,
      jobs: [{ ...OPERATIONS_PAYLOAD.jobs[0], queue: 'data_fetch_jp', market: 'JP', wait_reason: 'no_worker_for_queue' }],
      queues: [
        { queue: 'data_fetch_jp', depth: 1, oldest_age_seconds: null, consumers: 0 },
        { queue: 'data_fetch_kr', depth: 0, oldest_age_seconds: null, consumers: 0 },
        { queue: 'user_scans_us', depth: 0, oldest_age_seconds: null, consumers: 1 },
      ],
    });

    renderWithProviders(<OperationsPage />);

    expect(await screen.findByText(/No responding worker reads data_fetch_jp\. Start one, or cancel this job\./)).toBeInTheDocument();
    expect(screen.queryByText('no_worker_for_queue')).not.toBeInTheDocument();
    // Only a queue holding work is flagged; an idle unserved queue is just 0.
    const jpRow = screen.getAllByRole('row').find((row) => within(row).queryByText('data_fetch_jp', { exact: true }) && within(row).queryByText('No worker'));
    expect(jpRow).toBeTruthy();
    expect(screen.getAllByText('No worker')).toHaveLength(1);
  });

  it('shows an unknown worker count as a dash, not as unserved', async () => {
    fetchOperationsJobs.mockResolvedValue({
      ...OPERATIONS_PAYLOAD,
      queues: [{ queue: 'data_fetch_hk', depth: 1, oldest_age_seconds: 45, consumers: null }],
    });

    renderWithProviders(<OperationsPage />);

    expect(await screen.findAllByText('data_fetch_hk', { selector: 'td' })).toHaveLength(2);
    expect(screen.queryByText('No worker')).not.toBeInTheDocument();
    const queueRow = screen.getByRole('row', { name: /^data_fetch_hk 1 45s/ });
    expect(within(queueRow).getAllByRole('cell').at(-1)).toHaveTextContent(/^—$/);
  });

  it('falls back to the pending state when determinate jobs have unusable totals', async () => {
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
          current: 0,
          total: 0,
          message: 'Batch sizing pending',
        },
      ],
    });

    renderWithProviders(<OperationsPage />);

    expect(await screen.findByText('Batch sizing pending')).toBeInTheDocument();
    expect(screen.queryByText(/0%/)).not.toBeInTheDocument();
  });

  it('shows exposure research only when the themes feature is enabled', async () => {
    renderWithProviders(<OperationsPage />);
    expect(await screen.findByText('Company exposure research (shadow)')).toBeInTheDocument();
  });

  it('hides exposure research when its routes are not registered', async () => {
    runtimeState.features = { themes: false };
    renderWithProviders(<OperationsPage />);
    expect(await screen.findByText('Batch 3/5 · refreshing prices')).toBeInTheDocument();
    expect(screen.queryByText('Company exposure research (shadow)')).not.toBeInTheDocument();
  });

  it('lists derived stages and features that have not published (#393)', async () => {
    getBootstrapStatus.mockResolvedValue({
      market_stages: {
        US: { breadth: 'missing', exposure: 'ready', groups: 'missing' },
        HK: { breadth: 'ready', exposure: 'ready', groups: 'ready' },
      },
      features: { cot: 'missing', options: 'disabled', social: 'missing' },
    });
    renderWithProviders(<OperationsPage />);

    expect(await screen.findByText('Not yet published')).toBeInTheDocument();
    expect(screen.getByText('US: breadth, groups')).toBeInTheDocument();
    expect(screen.getByText('Features: cot, social')).toBeInTheDocument();
    const alert = screen.getByText('Not yet published').closest('[role="status"]');
    expect(alert).not.toHaveTextContent('HK');
    expect(alert).not.toHaveTextContent('options');
  });

  it('shows no publication warning when everything enabled has published', async () => {
    renderWithProviders(<OperationsPage />);
    expect(await screen.findByText('Batch 3/5 · refreshing prices')).toBeInTheDocument();
    await waitFor(() => expect(getBootstrapStatus).toHaveBeenCalled());
    expect(screen.queryByText('Not yet published')).not.toBeInTheDocument();
  });

  it('says when publication status cannot be loaded instead of looking complete', async () => {
    getBootstrapStatus.mockRejectedValue(new Error('boom'));
    renderWithProviders(<OperationsPage />);
    expect(await screen.findByText('Failed to load publication status.')).toBeInTheDocument();
  });
});
