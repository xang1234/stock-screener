import { render, screen, waitFor, cleanup, act } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('./pages/ScanPage', () => ({ default: () => <div>Live Scan Page</div> }));
vi.mock('./pages/MarketScanPage', () => ({ default: () => <div>Live Market Scan Page</div> }));
vi.mock('./pages/BreadthPage', () => ({ default: () => <div>Live Breadth Page</div> }));
vi.mock('./pages/GroupRankingsPage', () => ({ default: () => <div>Live Group Rankings Page</div> }));
vi.mock('./pages/ThemesPage', () => ({ default: () => <div>Live Themes Page</div> }));
vi.mock('./pages/ChatbotPage', () => ({ default: () => <div>Live Chatbot Page</div> }));
vi.mock('./components/Stock/StockDetails', () => ({ default: () => <div>Live Stock Details</div> }));
vi.mock('./components/Layout/Layout', () => ({ default: ({ children }) => <div>{children}</div> }));
vi.mock('./components/App/DesktopSetupScreen', () => ({ default: () => <div>Desktop Setup</div> }));
vi.mock('./components/App/ServerLoginScreen', () => ({ default: () => <div>Server Login</div> }));
vi.mock('./contexts/PipelineContext', () => ({ PipelineProvider: ({ children }) => <>{children}</> }));
vi.mock('./contexts/RuntimeContext', () => ({
  RuntimeProvider: ({ children }) => <>{children}</>,
  useRuntime: () => ({
    auth: null,
    desktopMode: false,
    features: {
      themes: true,
      chatbot: true,
    },
    isLoggingIn: false,
    login: vi.fn(),
    loginError: null,
    runtimeReady: true,
    setupRequired: false,
  }),
}));
vi.mock('./components/Scan/FilterPanel', () => ({
  default: ({ presetsEnabled }) => (
    <div data-testid="static-filter-panel">{presetsEnabled ? 'presets-enabled' : 'presets-disabled'}</div>
  ),
}));
vi.mock('./static/StaticChartViewerModal', () => ({ default: () => null }));
vi.mock('./components/Scan/ResultsTable', () => ({
  default: ({ showActions, total }) => (
    <div data-testid="static-results-table">
      {showActions ? 'actions-visible' : 'actions-hidden'}:{total}
    </div>
  ),
}));
vi.mock('./components/Charts/BreadthChart', () => ({
  default: ({ availableRanges = [] }) => (
    <div data-testid="breadth-chart-ranges">{availableRanges.join(',')}</div>
  ),
}));

const staticPayloads = {
  'manifest.json': {
    schema_version: 'static-site-v1',
    generated_at: '2026-03-31T22:00:00Z',
    as_of_date: '2026-03-31',
    features: {
      scan: true,
      breadth: true,
      groups: true,
    },
    pages: {
      home: { path: 'home.json' },
      scan: { path: 'scan/manifest.json' },
      breadth: { path: 'breadth.json' },
      groups: { path: 'groups.json' },
    },
    warnings: [],
  },
  'home.json': {
    generated_at: '2026-03-31T22:00:00Z',
    as_of_date: '2026-03-31',
    freshness: {
      scan_as_of_date: '2026-03-31',
      scan_run_id: 7,
      breadth_latest_date: '2026-03-31',
      groups_latest_date: '2026-03-31',
    },
    key_markets: [],
    scan_summary: {
      top_results: [{ symbol: 'NVDA', composite_score: 97.5, current_price: 145.4, rating: 'Strong Buy' }],
    },
    top_groups: [{ industry_group: 'Semiconductors', rank: 1 }],
  },
  'scan/manifest.json': {
    generated_at: '2026-03-31T22:00:00Z',
    as_of_date: '2026-03-31',
    run_id: 7,
    sort: { field: 'composite_score', order: 'desc' },
    default_page_size: 50,
    rows_total: 2,
    filter_options: {
      ibd_industries: ['Semiconductors', 'Software'],
      gics_sectors: ['Technology'],
      ratings: ['Strong Buy', 'Buy'],
    },
    initial_rows: [
      { symbol: 'NVDA', company_name: 'NVIDIA Corporation', composite_score: 97.5, rating: 'Strong Buy' },
      { symbol: 'MSFT', company_name: 'Microsoft Corporation', composite_score: 89.2, rating: 'Buy' },
    ],
    chunks: [{ path: 'scan/chunks/chunk-0001.json', count: 2 }],
    charts: {
      path: 'charts/index.json',
      limit: 200,
      symbols_total: 2,
      available: true,
    },
  },
  'scan/chunks/chunk-0001.json': {
    rows: [
      { symbol: 'NVDA', company_name: 'NVIDIA Corporation', composite_score: 97.5, rating: 'Strong Buy' },
      { symbol: 'MSFT', company_name: 'Microsoft Corporation', composite_score: 89.2, rating: 'Buy' },
    ],
  },
  'charts/index.json': {
    schema_version: 'static-charts-v1',
    generated_at: '2026-03-31T22:00:00Z',
    as_of_date: '2026-03-31',
    limit: 200,
    symbols_total: 2,
    skipped_symbols: [],
    symbols: [
      { symbol: 'NVDA', rank: 1, path: 'charts/NVDA.json' },
      { symbol: 'MSFT', rank: 2, path: 'charts/MSFT.json' },
    ],
  },
  'breadth.json': {
    generated_at: '2026-03-31T22:00:00Z',
    published_at: '2026-03-31T22:00:00Z',
    payload: {
      current: {
        date: '2026-03-31',
        stocks_up_4pct: 123,
        stocks_down_4pct: 42,
        ratio_10day: 1.8,
      },
      chart_data: [],
      spy_overlay: [],
      history_90d: [{ date: '2026-03-31', stocks_up_4pct: 123, stocks_down_4pct: 42, ratio_5day: 1.5, ratio_10day: 1.8 }],
    },
  },
  'groups.json': {
    available: true,
    payload: {
      movers_period: '3m',
      rankings: {
        date: '2026-03-31',
        rankings: [{ industry_group: 'Semiconductors', rank: 1, avg_rs_rating: 92.5, num_stocks: 14, rank_change_1w: 2, rank_change_1m: 4, rank_change_3m: 7 }],
      },
      movers: {
        gainers: [{ industry_group: 'Semiconductors', rank: 1 }],
        losers: [{ industry_group: 'Retail', rank: 197 }],
      },
    },
  },
};

const installFetchMock = () => {
  globalThis.fetch = vi.fn(async (url) => {
    const path = String(url).split('/static-data/')[1];
    const payload = staticPayloads[path];
    if (!payload) {
      return {
        ok: false,
        status: 404,
        json: async () => ({}),
      };
    }
    return {
      ok: true,
      status: 200,
      json: async () => JSON.parse(JSON.stringify(payload)),
    };
    });
};

const renderStaticAppAtHash = async (hash) => {
  vi.stubEnv('VITE_STATIC_SITE', 'true');
  vi.resetModules();
  installFetchMock();
  const { default: App } = await import('./App');
  await act(async () => {
    window.location.hash = hash;
    render(<App />);
  });
};

describe('App static mode', () => {
  beforeEach(() => {
    window.location.hash = '#/';
  });

  afterEach(() => {
    cleanup();
    vi.unstubAllEnvs();
    vi.restoreAllMocks();
    window.location.hash = '#/';
  });

  it.each([
    ['#/', 'Daily Market Snapshot'],
    ['#/scan', 'Daily Scan'],
    ['#/breadth', 'Market Breadth'],
    ['#/groups', 'Industry Group Rankings'],
    ['#/themes', 'Daily Market Snapshot'],
  ])('renders the static hash route %s without any /api requests', async (hash, heading) => {
    await renderStaticAppAtHash(hash);

    expect(await screen.findByText(heading, {}, { timeout: 10000 })).toBeInTheDocument();
    expect(screen.getAllByText('Read-only').length).toBeGreaterThan(0);
    expect(screen.queryByText('Sign out')).not.toBeInTheDocument();

    await waitFor(() => {
      expect(globalThis.fetch).toHaveBeenCalled();
    });

    const requestedUrls = globalThis.fetch.mock.calls.map(([url]) => String(url));
    expect(requestedUrls.length).toBeGreaterThan(0);
    expect(requestedUrls.every((url) => url.includes('/static-data/') && !url.includes('/api'))).toBe(true);
  }, 15000);

  it('keeps scan controls read-only in the static route', async () => {
    await renderStaticAppAtHash('#/scan');

    expect(await screen.findByRole('heading', { name: 'Daily Scan' })).toBeInTheDocument();
    await waitFor(() => {
      expect(screen.getByTestId('static-filter-panel')).toHaveTextContent('presets-disabled');
      expect(screen.getByTestId('static-results-table')).toHaveTextContent('actions-visible:2');
    });
  }, 10000);

  it('locks the breadth view to the exported range in the static route', async () => {
    await renderStaticAppAtHash('#/breadth');

    expect(await screen.findByRole('heading', { name: 'Market Breadth' })).toBeInTheDocument();
    expect(screen.getByTestId('breadth-chart-ranges')).toHaveTextContent('1M');
    expect(screen.getByTestId('breadth-chart-ranges')).not.toHaveTextContent('3M');
  }, 10000);
});
