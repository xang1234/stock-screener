import { createTheme, ThemeProvider } from '@mui/material';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter } from 'react-router-dom';
import { fireEvent, screen, waitFor } from '@testing-library/react';
import { describe, expect, it, vi, beforeEach, afterEach } from 'vitest';

import { renderWithProviders } from '../test/renderWithProviders';
import DigestPage from './DigestPage';

const getDailyDigest = vi.fn();
const getDailyDigestMarkdown = vi.fn();
const originalClipboard = navigator.clipboard;
const originalCreateObjectURL = window.URL.createObjectURL;
const originalRevokeObjectURL = window.URL.revokeObjectURL;

vi.mock('../api/digest', () => ({
  getDailyDigest: (...args) => getDailyDigest(...args),
  getDailyDigestMarkdown: (...args) => getDailyDigestMarkdown(...args),
}));

describe('DigestPage', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    Object.defineProperty(navigator, 'clipboard', {
      configurable: true,
      value: {
        writeText: vi.fn().mockResolvedValue(undefined),
      },
    });
    window.URL.createObjectURL = vi.fn(() => 'blob:digest');
    window.URL.revokeObjectURL = vi.fn();
    vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {});
  });

  afterEach(() => {
    Object.defineProperty(navigator, 'clipboard', {
      configurable: true,
      value: originalClipboard,
    });
    window.URL.createObjectURL = originalCreateObjectURL;
    window.URL.revokeObjectURL = originalRevokeObjectURL;
    vi.restoreAllMocks();
  });

  it('renders digest sections and exports markdown', async () => {
    getDailyDigest.mockResolvedValue({
      as_of_date: '2026-04-04',
      freshness: {
        latest_feature_as_of_date: '2026-04-04',
        latest_breadth_date: '2026-04-04',
        latest_theme_metrics_date: '2026-04-04',
        latest_theme_alert_at: '2026-04-04T15:30:00Z',
        validation_lookback_days: 90,
      },
      market: {
        stance: 'offense',
        summary: 'Current stance is offense.',
        breadth_metrics: {
          up_4pct: 120,
          down_4pct: 35,
          ratio_5day: 1.7,
          ratio_10day: 1.4,
          total_stocks_scanned: 4200,
        },
      },
      leaders: [
        {
          symbol: 'NVDA',
          name: 'NVIDIA',
          composite_score: 96,
          rating: 'Strong Buy',
          industry_group: 'Semiconductors',
          reason_summary: 'Strengths led by stage, rs rating.',
        },
      ],
      themes: {
        leaders: [
          { theme_id: 1, display_name: 'AI Infrastructure', momentum_score: 84, mention_velocity: 1.8, basket_return_1m: 12.5, status: 'trending' },
        ],
        laggards: [
          { theme_id: 2, display_name: 'Solar', momentum_score: 18, mention_velocity: 0.6, basket_return_1m: -11.2, status: 'fading' },
        ],
        recent_alerts: [
          { alert_id: 7, alert_type: 'breakout', severity: 'warning', triggered_at: '2026-04-04T15:30:00Z', theme: 'AI Infrastructure', title: 'AI breakout', related_tickers: ['NVDA', 'AVGO'] },
        ],
      },
      validation: {
        lookback_days: 90,
        scan_pick: {
          source_kind: 'scan_pick',
          horizons: [
            { horizon_sessions: 1, sample_size: 12, positive_rate: 0.67, avg_return_pct: 1.8, median_return_pct: 1.4, avg_mfe_pct: null, avg_mae_pct: null, skipped_missing_history: 0 },
            { horizon_sessions: 5, sample_size: 11, positive_rate: 0.64, avg_return_pct: 3.6, median_return_pct: 2.9, avg_mfe_pct: 5.4, avg_mae_pct: -2.0, skipped_missing_history: 1 },
          ],
          degraded_reasons: [],
        },
        theme_alert: {
          source_kind: 'theme_alert',
          horizons: [
            { horizon_sessions: 1, sample_size: 6, positive_rate: 0.33, avg_return_pct: -0.4, median_return_pct: -0.2, avg_mfe_pct: null, avg_mae_pct: null, skipped_missing_history: 0 },
            { horizon_sessions: 5, sample_size: 5, positive_rate: 0.4, avg_return_pct: -1.1, median_return_pct: -0.7, avg_mfe_pct: 1.8, avg_mae_pct: -3.2, skipped_missing_history: 0 },
          ],
          degraded_reasons: ['missing_price_cache'],
        },
      },
      watchlists: [
        {
          watchlist_id: 1,
          watchlist_name: 'Core Leaders',
          matched_symbols: ['NVDA'],
          alert_symbols: ['NVDA'],
          notes: '1 leader overlap and 1 alert overlap out of 2 tracked symbols.',
        },
      ],
      risks: [
        {
          kind: 'validation',
          message: 'Recent theme-alert follow-through is weak.',
          severity: 'warning',
        },
      ],
      degraded_reasons: ['missing_recent_theme_alerts'],
    });
    getDailyDigestMarkdown.mockResolvedValue('# Daily Digest (2026-04-04)\n');

    renderWithProviders(
      <MemoryRouter>
        <DigestPage />
      </MemoryRouter>
    );

    expect(await screen.findByRole('heading', { name: 'Daily Digest' })).toBeInTheDocument();
    expect(screen.getByText('Current stance is offense.')).toBeInTheDocument();
    expect(screen.getAllByRole('link', { name: 'NVDA' })[0]).toHaveAttribute('href', '/stocks/NVDA');
    expect(screen.getAllByText('AI Infrastructure').length).toBeGreaterThan(0);
    expect(screen.getByText('Core Leaders')).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: 'Copy Markdown' }));
    await waitFor(() => expect(navigator.clipboard.writeText).toHaveBeenCalledWith('# Daily Digest (2026-04-04)\n'));
    expect(await screen.findByText('Markdown copied to clipboard.')).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: 'Download Markdown' }));
    await waitFor(() => expect(window.URL.createObjectURL).toHaveBeenCalled());
    expect(await screen.findByText('Markdown download started.')).toBeInTheDocument();
  });

  it('renders degraded empty states without crashing', async () => {
    getDailyDigest.mockResolvedValue({
      as_of_date: '2026-04-04',
      freshness: {},
      market: {
        stance: 'unavailable',
        summary: 'Market breadth snapshot is unavailable for the digest date.',
        breadth_metrics: {},
      },
      leaders: [],
      themes: {
        leaders: [],
        laggards: [],
        recent_alerts: [],
      },
      validation: {
        lookback_days: 90,
        scan_pick: {
          source_kind: 'scan_pick',
          horizons: [],
          degraded_reasons: ['missing_feature_run'],
        },
        theme_alert: {
          source_kind: 'theme_alert',
          horizons: [],
          degraded_reasons: ['missing_price_cache'],
        },
      },
      watchlists: [],
      risks: [],
      degraded_reasons: ['missing_published_feature_run', 'missing_breadth_snapshot'],
    });

    renderWithProviders(
      <MemoryRouter>
        <DigestPage />
      </MemoryRouter>
    );

    expect(await screen.findByRole('heading', { name: 'Daily Digest' })).toBeInTheDocument();
    expect(screen.getByText(/digest sections are partially degraded/i)).toBeInTheDocument();
    expect(screen.getAllByText(/no ranked themes are available/i)).toHaveLength(2);
    expect(screen.getByText(/no leader candidates are available/i)).toBeInTheDocument();
    expect(screen.getByText(/no watchlist highlights are available/i)).toBeInTheDocument();
  });

  it('keeps rendering cached digest data when a refetch fails', async () => {
    const cachedDigest = {
      as_of_date: '2026-04-04',
      freshness: {},
      market: {
        stance: 'balanced',
        summary: 'Cached digest remains available.',
        breadth_metrics: {},
      },
      leaders: [],
      themes: {
        leaders: [],
        laggards: [],
        recent_alerts: [],
      },
      validation: {
        lookback_days: 90,
        scan_pick: {
          source_kind: 'scan_pick',
          horizons: [],
          degraded_reasons: [],
        },
        theme_alert: {
          source_kind: 'theme_alert',
          horizons: [],
          degraded_reasons: [],
        },
      },
      watchlists: [],
      risks: [],
      degraded_reasons: [],
    };
    const queryClient = new QueryClient({
      defaultOptions: {
        queries: {
          retry: false,
        },
      },
    });
    queryClient.setQueryData(['dailyDigest'], cachedDigest);
    queryClient.invalidateQueries({ queryKey: ['dailyDigest'] });
    getDailyDigest.mockRejectedValue(new Error('Network down'));

    renderWithProviders(
      <QueryClientProvider client={queryClient}>
        <ThemeProvider theme={createTheme()}>
          <MemoryRouter>
            <DigestPage />
          </MemoryRouter>
        </ThemeProvider>
      </QueryClientProvider>,
      { wrapper: ({ children }) => children }
    );

    expect(await screen.findByRole('heading', { name: 'Daily Digest' })).toBeInTheDocument();
    expect(screen.getByText('Cached digest remains available.')).toBeInTheDocument();
    await waitFor(() => expect(getDailyDigest).toHaveBeenCalled());
    expect(screen.queryByText(/failed to load daily digest/i)).not.toBeInTheDocument();
  });
});
