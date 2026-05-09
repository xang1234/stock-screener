import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import {
  DEFAULT_CAPABILITIES,
  RuntimeProvider,
  mergeBootstrapCapabilities,
  useRuntime,
} from './RuntimeContext';

const {
  getAppCapabilities,
  loginServer,
  logoutServer,
  startRuntimeBootstrap,
  setUnauthorizedResponseHandler,
  updateRuntimeMarkets,
} = vi.hoisted(() => ({
  getAppCapabilities: vi.fn(),
  loginServer: vi.fn(),
  logoutServer: vi.fn(),
  startRuntimeBootstrap: vi.fn(),
  setUnauthorizedResponseHandler: vi.fn(),
  updateRuntimeMarkets: vi.fn(),
}));

vi.mock('../api/client', () => ({
  setUnauthorizedResponseHandler,
}));

vi.mock('../api/appRuntime', () => ({
  getAppCapabilities: (...args) => getAppCapabilities(...args),
  startRuntimeBootstrap: (...args) => startRuntimeBootstrap(...args),
  updateRuntimeMarkets: (...args) => updateRuntimeMarkets(...args),
}));

vi.mock('../api/auth', () => ({
  loginServer: (...args) => loginServer(...args),
  logoutServer: (...args) => logoutServer(...args),
}));

function RuntimeProbe() {
  const {
    auth,
    runtimeReady,
    bootstrapRequired,
    bootstrapState,
    primaryMarket,
    enabledMarkets,
    marketCatalog,
    supportedMarkets,
    startBootstrap,
  } = useRuntime();

  return (
    <>
      <div data-testid="runtime-ready">{String(runtimeReady)}</div>
      <div data-testid="auth-required">{String(auth.required)}</div>
      <div data-testid="bootstrap-required">{String(bootstrapRequired)}</div>
      <div data-testid="bootstrap-state">{bootstrapState}</div>
      <div data-testid="primary-market">{primaryMarket}</div>
      <div data-testid="enabled-markets">{enabledMarkets.join(',')}</div>
      <div data-testid="market-catalog-labels">
        {marketCatalog.markets.map((market) => market.label).join(',')}
      </div>
      <div data-testid="supported-markets">{supportedMarkets.join(',')}</div>
      <button
        type="button"
        onClick={() => startBootstrap({ primaryMarket: 'HK', enabledMarkets: ['HK', 'US'] })}
      >
        Start bootstrap
      </button>
    </>
  );
}

function renderRuntime() {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retryDelay: 1,
        gcTime: Infinity,
      },
    },
  });

  const rendered = render(
    <QueryClientProvider client={queryClient}>
      <RuntimeProvider>
        <RuntimeProbe />
      </RuntimeProvider>
    </QueryClientProvider>
  );

  return {
    queryClient,
    ...rendered,
  };
}

describe('RuntimeProvider', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    getAppCapabilities.mockReset();
    loginServer.mockReset();
    logoutServer.mockReset();
    startRuntimeBootstrap.mockReset();
    setUnauthorizedResponseHandler.mockReset();
    updateRuntimeMarkets.mockReset();
  });

  it('marks runtime ready after app capabilities errors out', async () => {
    getAppCapabilities.mockRejectedValue(new Error('backend unavailable'));

    renderRuntime();

    expect(screen.getByTestId('runtime-ready')).toHaveTextContent('false');

    await waitFor(() => {
      expect(screen.getByTestId('runtime-ready')).toHaveTextContent('true');
    });

    expect(screen.getByTestId('auth-required')).toHaveTextContent('false');
    expect(getAppCapabilities).toHaveBeenCalledTimes(2);
  });

  it('exposes bootstrap metadata from app capabilities', async () => {
    getAppCapabilities.mockResolvedValue({
      features: { themes: true, chatbot: true, tasks: true },
      auth: { required: false, configured: true, authenticated: true, mode: 'session_cookie', message: null },
      ui_snapshots: { enabled: false, scan: false, breadth: false, groups: false, themes: false },
      scan_defaults: { universe: 'all', screeners: ['minervini'], composite_method: 'weighted_average', criteria: {} },
      bootstrap_required: true,
      primary_market: 'HK',
      enabled_markets: ['HK', 'US'],
      market_catalog: {
        version: 'test.v1',
        markets: [
          {
            code: 'US',
            label: 'United States',
            currency: 'USD',
            timezone: 'America/New_York',
            calendar_id: 'XNYS',
            exchanges: ['NYSE', 'NASDAQ'],
            indexes: ['SP500'],
            capabilities: {},
          },
          {
            code: 'HK',
            label: 'Hong Kong',
            currency: 'HKD',
            timezone: 'Asia/Hong_Kong',
            calendar_id: 'XHKG',
            exchanges: ['HKEX'],
            indexes: ['HSI'],
            capabilities: {},
          },
        ],
      },
      bootstrap_state: 'running',
      supported_markets: ['US', 'HK', 'IN', 'JP', 'KR', 'TW', 'CN', 'CA', 'DE'],
      api_base_path: '/api',
    });

    renderRuntime();

    await waitFor(() => {
      expect(screen.getByTestId('runtime-ready')).toHaveTextContent('true');
    });

    expect(screen.getByTestId('bootstrap-required')).toHaveTextContent('true');
    expect(screen.getByTestId('primary-market')).toHaveTextContent('HK');
    expect(screen.getByTestId('enabled-markets')).toHaveTextContent('HK,US');
    expect(screen.getByTestId('market-catalog-labels')).toHaveTextContent('United States,Hong Kong');
  });

  it('updates cached bootstrap state immediately after bootstrap starts', async () => {
    getAppCapabilities
      .mockResolvedValueOnce({
        ...DEFAULT_CAPABILITIES,
        bootstrap_required: true,
        bootstrap_state: 'not_started',
      })
      .mockImplementationOnce(() => new Promise(() => {}));
    startRuntimeBootstrap.mockResolvedValue({
      bootstrap_required: true,
      empty_system: true,
      primary_market: 'HK',
      enabled_markets: ['HK', 'US'],
      bootstrap_state: 'running',
      supported_markets: ['US', 'HK', 'IN', 'JP', 'KR', 'TW', 'CN', 'CA', 'DE'],
      task_id: 'task-bootstrap-123',
    });

    const { queryClient } = renderRuntime();

    await waitFor(() => {
      expect(screen.getByTestId('runtime-ready')).toHaveTextContent('true');
    });
    expect(screen.getByTestId('bootstrap-state')).toHaveTextContent('not_started');

    fireEvent.click(screen.getByRole('button', { name: 'Start bootstrap' }));

    await waitFor(() => {
      expect(startRuntimeBootstrap).toHaveBeenCalledWith({
        primaryMarket: 'HK',
        enabledMarkets: ['HK', 'US'],
      });
    });
    await waitFor(() => {
      expect(screen.getByTestId('bootstrap-required')).toHaveTextContent('true');
      expect(screen.getByTestId('bootstrap-state')).toHaveTextContent('running');
      expect(screen.getByTestId('primary-market')).toHaveTextContent('HK');
      expect(screen.getByTestId('enabled-markets')).toHaveTextContent('HK,US');
    });

    await waitFor(() => {
      expect(queryClient.getQueryData(['runtimeActivity'])?.markets).toEqual([
        expect.objectContaining({ market: 'HK', task_id: 'task-bootstrap-123' }),
        expect.objectContaining({ market: 'US', task_id: null }),
      ]);
      expect(queryClient.getQueryData(['appCapabilities'])?.market_catalog).toEqual(
        DEFAULT_CAPABILITIES.market_catalog
      );
    });
  });

  it('preserves the non-US markets in bootstrap fallback supported markets', async () => {
    getAppCapabilities
      .mockResolvedValueOnce({
        ...DEFAULT_CAPABILITIES,
        bootstrap_required: true,
        bootstrap_state: 'not_started',
      })
      .mockImplementationOnce(() => new Promise(() => {}));
    startRuntimeBootstrap.mockResolvedValue({
      bootstrap_required: true,
      empty_system: true,
      primary_market: 'HK',
      enabled_markets: ['HK', 'US'],
      bootstrap_state: 'running',
      supported_markets: null,
      task_id: 'task-bootstrap-123',
    });

    const { queryClient } = renderRuntime();

    await waitFor(() => {
      expect(screen.getByTestId('runtime-ready')).toHaveTextContent('true');
    });

    queryClient.setQueryData(['appCapabilities'], (previous) => ({
      ...(previous ?? DEFAULT_CAPABILITIES),
      supported_markets: null,
    }));

    fireEvent.click(screen.getByRole('button', { name: 'Start bootstrap' }));

    await waitFor(() => {
      expect(startRuntimeBootstrap).toHaveBeenCalledTimes(1);
      expect(queryClient.getQueryData(['appCapabilities'])?.supported_markets).toEqual(
        DEFAULT_CAPABILITIES.supported_markets
      );
      expect(screen.getByTestId('supported-markets')).toHaveTextContent('US,HK,IN,JP,KR,TW,CN,CA');
    });
  });
});

describe('mergeBootstrapCapabilities', () => {
  it('falls back to the default supported markets when bootstrap omits them', () => {
    const merged = mergeBootstrapCapabilities(null, {
      bootstrap_required: true,
      primary_market: 'HK',
      enabled_markets: ['HK', 'US'],
      bootstrap_state: 'running',
      supported_markets: null,
    });

    expect(merged.supported_markets).toEqual(DEFAULT_CAPABILITIES.supported_markets);
  });
});
