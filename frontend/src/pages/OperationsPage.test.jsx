import { screen, waitFor, fireEvent } from '@testing-library/react';
import { describe, expect, it, vi, beforeEach } from 'vitest';

import { renderWithProviders } from '../test/renderWithProviders';
import OperationsPage from './OperationsPage';

const fetchAlerts = vi.fn();
const acknowledgeAlert = vi.fn();

vi.mock('../api/telemetry', () => ({
  fetchAlerts: (...args) => fetchAlerts(...args),
  acknowledgeAlert: (...args) => acknowledgeAlert(...args),
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

describe('OperationsPage', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    fetchAlerts.mockResolvedValue({ summaries: [SUMMARY_US], alerts: [ALERT_US] });
  });

  it('renders per-market summary card and active alerts table', async () => {
    renderWithProviders(<OperationsPage />);

    // Market card content (the card heading + a few labels)
    await waitFor(() => {
      expect(screen.getByText(/Freshness:/)).toBeInTheDocument();
    });
    expect(screen.getByText(/Universe:/)).toBeInTheDocument();
    // "US" appears in both the card label and the alert row, so use getAllByText.
    expect(screen.getAllByText('US').length).toBeGreaterThanOrEqual(1);

    // Alert row content
    await waitFor(() => {
      expect(screen.getByText('warning')).toBeInTheDocument();
    });
    expect(screen.getByText('open')).toBeInTheDocument();
    expect(screen.getByText('freshness_lag')).toBeInTheDocument();
    expect(screen.getByText('us-ops')).toBeInTheDocument();
  });

  it('clicking Ack calls acknowledgeAlert and refetches', async () => {
    acknowledgeAlert.mockResolvedValue({ ...ALERT_US, state: 'acknowledged' });
    renderWithProviders(<OperationsPage />);

    const ackButton = await screen.findByRole('button', { name: /ack/i });
    fireEvent.click(ackButton);

    await waitFor(() => {
      expect(acknowledgeAlert).toHaveBeenCalledWith(1, 'operator');
    });
  });

  it('shows empty-state message when no alerts active', async () => {
    fetchAlerts.mockResolvedValue({ summaries: [SUMMARY_US], alerts: [] });
    renderWithProviders(<OperationsPage />);

    await waitFor(() => {
      expect(screen.getByText(/No active alerts/)).toBeInTheDocument();
    });
  });
});
