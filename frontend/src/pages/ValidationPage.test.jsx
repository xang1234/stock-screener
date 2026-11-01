import { MemoryRouter } from 'react-router-dom';
import { fireEvent, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import MarketSelector from '../components/Layout/MarketSelector';
import { MarketProvider } from '../contexts/MarketContext';
import { renderWithProviders } from '../test/renderWithProviders';
import ValidationPage from './ValidationPage';

const getValidationOverview = vi.fn();
const formatDate = (value) => new Intl.DateTimeFormat('en-US', { timeZone: 'UTC' }).format(new Date(`${value}T00:00:00Z`));

vi.mock('../api/validation', () => ({
  getValidationOverview: (...args) => getValidationOverview(...args),
}));

vi.mock('../contexts/RuntimeContext', () => ({
  useRuntime: () => ({
    primaryMarket: 'US',
    enabledMarkets: ['US', 'HK'],
    supportedMarkets: ['US', 'HK'],
    marketCatalog: {
      markets: [
        { code: 'US', label: 'United States' },
        { code: 'HK', label: 'Hong Kong' },
      ],
    },
  }),
}));

const emptyOverview = (market) => ({
  source_kind: 'scan_pick',
  lookback_days: 90,
  market,
  benchmark_symbol: null,
  horizons: [],
  recent_events: [],
  failure_clusters: [],
  freshness: {},
  degraded_reasons: [],
});

describe('ValidationPage', () => {
  beforeEach(() => {
    getValidationOverview.mockReset();
    window.localStorage.clear();
  });

  it('renders overview metrics, events, and failure clusters', async () => {
    getValidationOverview.mockResolvedValue({
      source_kind: 'scan_pick',
      lookback_days: 90,
      market: 'US',
      benchmark_symbol: 'SPY',
      stop_loss_pct: 7,
      horizons: [
        { horizon_sessions: 1, sample_size: 12, positive_rate: 0.58, avg_return_pct: 1.2, median_return_pct: 0.9, avg_excess_return_pct: 0.4, median_excess_return_pct: 0.2, excess_sample_size: 10, avg_stopped_return_pct: 1.0, median_stopped_return_pct: 0.9, stopped_out_rate: 0.08, avg_mfe_pct: 2.0, avg_mae_pct: -1.0, skipped_missing_history: 1 },
        { horizon_sessions: 5, sample_size: 11, positive_rate: 0.55, avg_return_pct: 2.4, median_return_pct: 1.8, avg_excess_return_pct: 1.1, median_excess_return_pct: 0.6, excess_sample_size: 9, avg_stopped_return_pct: 1.9, median_stopped_return_pct: 1.5, stopped_out_rate: 0.18, avg_mfe_pct: 4.7, avg_mae_pct: -2.1, skipped_missing_history: 2 },
        { horizon_sessions: 20, sample_size: 8, positive_rate: 0.5, avg_return_pct: 3.1, median_return_pct: 1.2, avg_excess_return_pct: 1.4, median_excess_return_pct: 0.3, excess_sample_size: 8, avg_stopped_return_pct: -1.5, median_stopped_return_pct: -2.0, stopped_out_rate: 0.35, avg_mfe_pct: 9.4, avg_mae_pct: -5.2, skipped_missing_history: 3 },
        { horizon_sessions: 60, sample_size: 0, positive_rate: null, avg_return_pct: null, median_return_pct: null, avg_excess_return_pct: null, median_excess_return_pct: null, excess_sample_size: 0, avg_stopped_return_pct: null, median_stopped_return_pct: null, stopped_out_rate: null, avg_mfe_pct: null, avg_mae_pct: null, skipped_missing_history: 1, pending_windows: 10 },
      ],
      recent_events: [
        {
          source_kind: 'scan_pick',
          source_ref: 'run:12',
          event_at: '2026-04-01',
          entry_at: '2026-04-02',
          return_1s_pct: 1.4,
          return_5s_pct: 3.1,
          return_20s_pct: 6.5,
          return_60s_pct: 12.0,
          excess_return_5s_pct: 2.2,
          mfe_5s_pct: 4.2,
          mae_5s_pct: -1.2,
          counted: true,
          counted_horizons: [1, 5, 20, 60],
          attributes: { symbol: 'NVDA' },
        },
        {
          source_kind: 'scan_pick',
          source_ref: 'run:11',
          event_at: '2026-03-20',
          entry_at: '2026-03-23',
          return_1s_pct: 0.5,
          return_5s_pct: 1.0,
          return_20s_pct: 2.0,
          return_60s_pct: null,
          counted: false,
          counted_horizons: [1, 20],
          attributes: { symbol: 'AMD' },
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
    const cardTitles = screen.getAllByText(/^\d+ Sessions?$/).map((title) => title.textContent);
    expect(cardTitles).toEqual(['1 Session', '5 Sessions', '20 Sessions', '60 Sessions']);
    const fiveSessions = within(screen.getByText('5 Sessions').closest('.MuiCardContent-root'));
    expect(fiveSessions.getByText('Avg vs SPY (no stop, n=9)').nextSibling).toHaveTextContent('1.1%');
    expect(fiveSessions.getByText('Median vs SPY (no stop, n=9)').nextSibling).toHaveTextContent('0.6%');
    expect(fiveSessions.getByText('Avg MFE (no stop)').nextSibling).toHaveTextContent('4.7%');
    expect(fiveSessions.getByText('Avg MAE (no stop)').nextSibling).toHaveTextContent('-2.1%');
    const twentySessions = within(screen.getByText('20 Sessions').closest('.MuiCardContent-root'));
    expect(twentySessions.getByText('Avg with 7% stop').nextSibling).toHaveTextContent('-1.5%');
    expect(twentySessions.getByText('Median with 7% stop').nextSibling).toHaveTextContent('-2.0%');
    expect(twentySessions.getByText('Stopped out').nextSibling).toHaveTextContent('35%');
    const sixtySessions = within(screen.getByText('60 Sessions').closest('.MuiCardContent-root'));
    expect(sixtySessions.getByText('Pending').nextSibling).toHaveTextContent('10');
    expect(sixtySessions.getByText('Skipped').nextSibling).toHaveTextContent('1');
    const eventRow = screen.getByRole('link', { name: 'NVDA' }).closest('tr');
    const headers = screen.getAllByRole('columnheader').map((cell) => cell.textContent);
    expect(headers.slice(headers.indexOf('1S'), headers.indexOf('60S') + 1)).toEqual(['1S', '5S', '20S', '60S']);
    expect(eventRow.cells[headers.indexOf('5S vs SPY')]).toHaveTextContent('2.2%');
    expect(eventRow.cells[headers.indexOf('20S')]).toHaveTextContent('6.5%');
    expect(eventRow.cells[headers.indexOf('60S')]).toHaveTextContent('12.0%');
    expect(within(eventRow).queryByText('repeat')).not.toBeInTheDocument();
    const repeatRow = screen.getByRole('link', { name: 'AMD' }).closest('tr');
    expect(within(repeatRow).getByText('repeat')).toBeInTheDocument();
    expect(repeatRow).not.toHaveStyle({ opacity: '0.55' });
    for (const header of ['1S', '20S']) {
      expect(repeatRow.cells[headers.indexOf(header)].className).toBe(eventRow.cells[headers.indexOf(header)].className);
    }
    for (const header of ['5S', '60S', '5S vs SPY', 'MFE', 'MAE']) {
      const cell = repeatRow.cells[headers.indexOf(header)];
      expect(cell.className).not.toBe(eventRow.cells[headers.indexOf(header)].className);
      expect(cell).toHaveStyle({ color: 'rgb(139, 147, 167)' });
      expect(cell).not.toHaveStyle({ opacity: '0.55' });
    }
  });

  it('updates the query when source and lookback controls change', async () => {
    getValidationOverview
      .mockResolvedValueOnce({
        source_kind: 'scan_pick',
        lookback_days: 90,
        market: 'US',
        horizons: [],
        recent_events: [],
        failure_clusters: [],
        freshness: {},
        degraded_reasons: [],
      })
      .mockResolvedValueOnce({
        source_kind: 'theme_alert',
        lookback_days: 90,
        market: 'US',
        horizons: [],
        recent_events: [],
        failure_clusters: [],
        freshness: {},
        degraded_reasons: [],
      })
      .mockResolvedValueOnce({
        source_kind: 'theme_alert',
        lookback_days: 180,
        market: 'US',
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
    expect(getValidationOverview).toHaveBeenCalledWith('theme_alert', 180, 'US');
  });

  it('scopes the scorecard to the market chosen in the header selector', async () => {
    getValidationOverview.mockImplementation((sourceKind, lookbackDays, market) => (
      Promise.resolve(emptyOverview(market))
    ));
    const user = userEvent.setup();

    renderWithProviders(
      <MemoryRouter>
        <MarketProvider>
          <MarketSelector />
          <ValidationPage />
        </MarketProvider>
      </MemoryRouter>
    );

    await screen.findByRole('heading', { name: 'Validation' });
    expect(getValidationOverview).toHaveBeenLastCalledWith('scan_pick', 90, 'US');
    fireEvent.mouseDown(screen.getByRole('combobox', { name: /market/i }));
    await user.click(await screen.findByRole('option', { name: /Hong Kong/i }));

    await waitFor(() => {
      expect(getValidationOverview).toHaveBeenLastCalledWith('scan_pick', 90, 'HK');
    });
  });

  it("shows the loading state, not the previous market's scorecard, while a new market loads", async () => {
    let resolveHk;
    getValidationOverview.mockImplementation((sourceKind, lookbackDays, market) => (
      market === 'HK'
        ? new Promise((resolve) => { resolveHk = resolve; })
        : Promise.resolve({ ...emptyOverview('US'), benchmark_symbol: 'SPY' })
    ));
    const user = userEvent.setup();

    renderWithProviders(
      <MemoryRouter>
        <MarketProvider>
          <MarketSelector />
          <ValidationPage />
        </MarketProvider>
      </MemoryRouter>
    );

    expect(await screen.findByRole('columnheader', { name: '5S vs SPY' })).toBeInTheDocument();
    fireEvent.mouseDown(screen.getByRole('combobox', { name: /market/i }));
    await user.click(await screen.findByRole('option', { name: /Hong Kong/i }));
    await waitFor(() => {
      expect(getValidationOverview).toHaveBeenLastCalledWith('scan_pick', 90, 'HK');
    });

    expect(screen.queryByRole('columnheader', { name: '5S vs SPY' })).not.toBeInTheDocument();
    expect(screen.getByRole('progressbar')).toBeInTheDocument();
    resolveHk({ ...emptyOverview('HK'), benchmark_symbol: '2800.HK' });
    expect(await screen.findByRole('columnheader', { name: '5S vs 2800.HK' })).toBeInTheDocument();
  });
});
