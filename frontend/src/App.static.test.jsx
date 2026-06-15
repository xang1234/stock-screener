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
vi.mock('./components/App/ServerLoginScreen', () => ({ default: () => <div>Server Login</div> }));
vi.mock('./contexts/PipelineContext', () => ({ PipelineProvider: ({ children }) => <>{children}</> }));
vi.mock('./contexts/RuntimeContext', () => ({
  RuntimeProvider: ({ children }) => <>{children}</>,
  useRuntime: () => ({
    auth: null,
    features: {
      themes: true,
      chatbot: true,
    },
    isLoggingIn: false,
    login: vi.fn(),
    loginError: null,
    runtimeReady: true,
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
    schema_version: 'static-site-v2',
    generated_at: '2026-03-31T22:00:00Z',
    as_of_date: '2026-03-31',
    default_market: 'US',
    supported_markets: ['US', 'HK'],
    features: {
      scan: true,
      breadth: true,
      groups: true,
    },
    pages: {
      home: { path: 'markets/us/home.json' },
      scan: { path: 'markets/us/scan/manifest.json' },
      breadth: { path: 'markets/us/breadth.json' },
      groups: { path: 'markets/us/groups.json' },
    },
    markets: {
      US: {
        display_name: 'United States',
        pages: {
          home: { path: 'markets/us/home.json' },
          scan: { path: 'markets/us/scan/manifest.json' },
          breadth: { path: 'markets/us/breadth.json' },
          groups: { path: 'markets/us/groups.json' },
        },
        assets: {
          charts: { path: 'markets/us/charts/index.json' },
        },
      },
      HK: {
        display_name: 'Hong Kong',
        pages: {
          home: { path: 'markets/hk/home.json' },
          scan: { path: 'markets/hk/scan/manifest.json' },
          breadth: { path: 'markets/hk/breadth.json' },
          groups: { path: 'markets/hk/groups.json' },
        },
        assets: {
          charts: { path: 'markets/hk/charts/index.json' },
        },
      },
    },
    warnings: [],
  },
  'markets/us/home.json': {
    generated_at: '2026-03-31T22:00:00Z',
    as_of_date: '2026-03-31',
    market: 'US',
    market_display_name: 'United States',
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
    top_groups: [{ industry_group: 'Semiconductors', rank: 1, top_symbol: 'NVDA' }],
  },
  'markets/hk/home.json': {
    generated_at: '2026-03-31T22:00:00Z',
    as_of_date: '2026-03-31',
    market: 'HK',
    market_display_name: 'Hong Kong',
    freshness: {
      scan_as_of_date: '2026-03-31',
      scan_run_id: 17,
      breadth_latest_date: '2026-03-31',
      groups_latest_date: '2026-03-31',
    },
    key_markets: [],
    scan_summary: {
      top_results: [{ symbol: '0700.HK', composite_score: 91.2, current_price: 398.4, rating: 'Buy' }],
    },
    top_groups: [{ industry_group: 'Internet', rank: 1, top_symbol: '0700.HK' }],
  },
  'markets/us/scan/manifest.json': {
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
    preset_screens: [
      {
        id: 'minervini',
        name: 'Minervini Trend Template',
        short_name: 'Minervini',
        description: 'Stage 2 uptrend stocks',
        tier: 1,
        filters: { minerviniScore: { min: 70, max: null }, stage: 2 },
        sort_by: 'minervini_score',
        sort_order: 'desc',
      },
    ],
    chunks: [{ path: 'markets/us/scan/chunks/chunk-0001.json', count: 2 }],
    charts: {
      path: 'markets/us/charts/index.json',
      limit: 200,
      symbols_total: 2,
      available: true,
    },
  },
  'markets/hk/scan/manifest.json': {
    generated_at: '2026-03-31T22:00:00Z',
    as_of_date: '2026-03-31',
    run_id: 17,
    sort: { field: 'composite_score', order: 'desc' },
    default_page_size: 50,
    rows_total: 1,
    filter_options: {
      ibd_industries: ['Internet'],
      gics_sectors: ['Communication Services'],
      ratings: ['Buy'],
    },
    initial_rows: [
      { symbol: '0700.HK', company_name: 'Tencent Holdings', composite_score: 91.2, rating: 'Buy' },
    ],
    chunks: [{ path: 'markets/hk/scan/chunks/chunk-0001.json', count: 1 }],
    charts: {
      path: 'markets/hk/charts/index.json',
      limit: 200,
      symbols_total: 1,
      available: true,
    },
  },
  'markets/us/scan/chunks/chunk-0001.json': {
    rows: [
      { symbol: 'NVDA', company_name: 'NVIDIA Corporation', composite_score: 97.5, rating: 'Strong Buy' },
      { symbol: 'MSFT', company_name: 'Microsoft Corporation', composite_score: 89.2, rating: 'Buy' },
    ],
  },
  'markets/hk/scan/chunks/chunk-0001.json': {
    rows: [
      { symbol: '0700.HK', company_name: 'Tencent Holdings', composite_score: 91.2, rating: 'Buy' },
    ],
  },
  'markets/us/charts/index.json': {
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
  'markets/hk/charts/index.json': {
    schema_version: 'static-charts-v1',
    generated_at: '2026-03-31T22:00:00Z',
    as_of_date: '2026-03-31',
    limit: 200,
    symbols_total: 1,
    skipped_symbols: [],
    symbols: [
      { symbol: '0700.HK', rank: 1, path: 'charts/0700.HK.json' },
    ],
  },
  'markets/us/breadth.json': {
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
  'markets/hk/breadth.json': {
    generated_at: '2026-03-31T22:00:00Z',
    published_at: '2026-03-31T22:00:00Z',
    payload: {
      current: {
        date: '2026-03-31',
        stocks_up_4pct: 54,
        stocks_down_4pct: 12,
        ratio_10day: 2.4,
      },
      chart_data: [],
      spy_overlay: [],
      history_90d: [{ date: '2026-03-31', stocks_up_4pct: 54, stocks_down_4pct: 12, ratio_5day: 2.1, ratio_10day: 2.4 }],
    },
  },
  'markets/us/groups.json': {
    available: true,
    payload: {
      movers_period: '1w',
      rankings: {
        date: '2026-03-31',
        rankings: [{ industry_group: 'Semiconductors', rank: 1, avg_rs_rating: 92.5, num_stocks: 14, rank_change_1w: 2, rank_change_1m: 4, rank_change_3m: 7 }],
      },
      movers: {
        gainers: [{ industry_group: 'Semiconductors', rank: 1, rank_change_1w: 3 }],
        losers: [{ industry_group: 'Retail', rank: 197, rank_change_1w: -5 }],
      },
    },
  },
  'markets/hk/groups.json': {
    available: true,
    payload: {
      movers_period: '1w',
      rankings: {
        date: '2026-03-31',
        rankings: [{ industry_group: 'Internet', rank: 1, avg_rs_rating: 88.1, num_stocks: 9, rank_change_1w: 1, rank_change_1m: 3, rank_change_3m: 5 }],
      },
      movers: {
        gainers: [{ industry_group: 'Internet', rank: 1, rank_change_1w: 2 }],
        losers: [{ industry_group: 'Property', rank: 44, rank_change_1w: -4 }],
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
    window.localStorage.clear();
  });

  afterEach(() => {
    cleanup();
    vi.unstubAllEnvs();
    vi.restoreAllMocks();
    window.location.hash = '#/';
    window.localStorage.clear();
  });

  it.each([
    ['#/', 'United States Snapshot'],
    ['#/scan', 'Daily Scan'],
    ['#/breadth', 'United States Breadth'],
    ['#/groups', 'United States Group Rankings'],
    ['#/themes', 'United States Snapshot'],
  ])('renders the static hash route %s without any /api requests', async (hash, heading) => {
    await renderStaticAppAtHash(hash);

    const headingMatcher = (_content, element) => {
      if (!element) return false;
      const hasText = (el) => el.textContent?.includes(heading);
      return hasText(element) && Array.from(element.children).every((child) => !hasText(child));
    };
    expect(await screen.findByText(headingMatcher, {}, { timeout: 10000 })).toBeInTheDocument();
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

  it('offers 1M and 3M ranges on the breadth page in the static route', async () => {
    await renderStaticAppAtHash('#/breadth');

    expect(await screen.findByRole('heading', { name: 'United States Breadth' })).toBeInTheDocument();
    expect(screen.getByTestId('breadth-chart-ranges')).toHaveTextContent('1M');
    expect(screen.getByTestId('breadth-chart-ranges')).toHaveTextContent('3M');
  }, 10000);

  it('honors the market query parameter and loads market-scoped breadth assets', async () => {
    await renderStaticAppAtHash('#/breadth?market=HK');

    expect(await screen.findByRole('heading', { name: 'Hong Kong Breadth' })).toBeInTheDocument();
    expect(window.location.hash).toContain('#/breadth');
    expect(window.location.hash).toContain('market=HK');
    expect(window.localStorage.getItem('static-site:selected-market')).toBe('HK');

    const requestedUrls = globalThis.fetch.mock.calls.map(([url]) => String(url));
    expect(requestedUrls.some((url) => url.includes('/static-data/markets/hk/breadth.json'))).toBe(true);
  }, 10000);

  it('uses the persisted market selection when no query parameter is present', async () => {
    window.localStorage.setItem('static-site:selected-market', 'HK');

    await renderStaticAppAtHash('#/groups');

    expect(await screen.findByRole('heading', { name: 'Hong Kong Group Rankings' })).toBeInTheDocument();
    expect(screen.getByRole('combobox', { name: 'Static market selector' })).toHaveTextContent('Hong Kong');
  }, 10000);
});
