import { MemoryRouter } from 'react-router-dom';
import { fireEvent, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import { renderWithProviders } from '../test/renderWithProviders';
import ValidationPage from './ValidationPage';

const getValidationOverview = vi.fn();
const formatDate = (value) => new Intl.DateTimeFormat('en-US', { timeZone: 'UTC' }).format(new Date(`${value}T00:00:00Z`));

vi.mock('../api/validation', () => ({
  getValidationOverview: (...args) => getValidationOverview(...args),
}));

describe('ValidationPage', () => {
  it('renders overview metrics, events, and failure clusters', async () => {
    getValidationOverview.mockResolvedValue({
      source_kind: 'scan_pick',
      lookback_days: 90,
      horizons: [
        { horizon_sessions: 1, sample_size: 12, positive_rate: 0.58, avg_return_pct: 1.2, median_return_pct: 0.9, avg_mfe_pct: null, avg_mae_pct: null, skipped_missing_history: 1 },
        { horizon_sessions: 5, sample_size: 11, positive_rate: 0.55, avg_return_pct: 2.4, median_return_pct: 1.8, avg_mfe_pct: 4.7, avg_mae_pct: -2.1, skipped_missing_history: 2 },
      ],
      recent_events: [
        {
          source_kind: 'scan_pick',
          source_ref: 'run:12',
          event_at: '2026-04-01',
          entry_at: '2026-04-02',
          return_1s_pct: 1.4,
          return_5s_pct: 3.1,
          mfe_5s_pct: 4.2,
          mae_5s_pct: -1.2,
          attributes: { symbol: 'NVDA' },
        },
      ],
      failure_clusters: [
        {
          cluster_key: 'rating:Buy',
          label: 'Rating: Buy',
          sample_size: 4,
          avg_return_pct: -3.2,
          median_return_pct: -2.5,
        },
      ],
      freshness: {
        latest_feature_as_of_date: '2026-04-01',
        latest_theme_alert_at: '2026-04-02T03:00:00Z',
        price_cache_period: '2y',
      },
      degraded_reasons: [],
    });

    renderWithProviders(
      <MemoryRouter>
        <ValidationPage />
      </MemoryRouter>
    );

    expect(await screen.findByRole('heading', { name: 'Validation' })).toBeInTheDocument();
    expect(screen.getByText('Recent Events')).toBeInTheDocument();
    expect(screen.getByText('Failure Clusters')).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'NVDA' })).toHaveAttribute('href', '/stocks/NVDA');
    expect(screen.getByText('Rating: Buy')).toBeInTheDocument();
    expect(screen.getByText(formatDate('2026-04-01'))).toBeInTheDocument();
    expect(screen.getByText(formatDate('2026-04-02'))).toBeInTheDocument();
  });

  it('updates the query when source and lookback controls change', async () => {
    getValidationOverview
      .mockResolvedValueOnce({
        source_kind: 'scan_pick',
        lookback_days: 90,
        horizons: [],
        recent_events: [],
        failure_clusters: [],
        freshness: {},
        degraded_reasons: [],
      })
      .mockResolvedValueOnce({
        source_kind: 'theme_alert',
        lookback_days: 90,
        horizons: [],
        recent_events: [],
        failure_clusters: [],
        freshness: {},
        degraded_reasons: [],
      })
      .mockResolvedValueOnce({
        source_kind: 'theme_alert',
        lookback_days: 180,
        horizons: [],
        recent_events: [],
        failure_clusters: [],
        freshness: {},
        degraded_reasons: ['missing_price_cache'],
      });

    renderWithProviders(
      <MemoryRouter>
        <ValidationPage />
      </MemoryRouter>
    );

    await screen.findByRole('heading', { name: 'Validation' });
    fireEvent.click(screen.getByRole('button', { name: 'Theme Alerts' }));
    fireEvent.click(screen.getByRole('button', { name: '180D' }));

    expect(await screen.findByText(/validation is partially degraded/i)).toBeInTheDocument();
    expect(getValidationOverview).toHaveBeenCalledWith('theme_alert', 180);
  });
});
