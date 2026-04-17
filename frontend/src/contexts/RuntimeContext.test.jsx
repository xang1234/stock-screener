import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { RuntimeProvider, useRuntime } from './RuntimeContext';

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
  const { auth, runtimeReady, bootstrapRequired, primaryMarket, enabledMarkets } = useRuntime();

  return (
    <>
      <div data-testid="runtime-ready">{String(runtimeReady)}</div>
      <div data-testid="auth-required">{String(auth.required)}</div>
      <div data-testid="bootstrap-required">{String(bootstrapRequired)}</div>
      <div data-testid="primary-market">{primaryMarket}</div>
      <div data-testid="enabled-markets">{enabledMarkets.join(',')}</div>
    </>
  );
}

function renderRuntime() {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retryDelay: 1,
        gcTime: 0,
      },
    },
  });

  return render(
    <QueryClientProvider client={queryClient}>
      <RuntimeProvider>
        <RuntimeProbe />
      </RuntimeProvider>
    </QueryClientProvider>
  );
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
      bootstrap_state: 'running',
      supported_markets: ['US', 'HK', 'JP', 'TW'],
      api_base_path: '/api',
    });

    renderRuntime();

    await waitFor(() => {
      expect(screen.getByTestId('runtime-ready')).toHaveTextContent('true');
    });

    expect(screen.getByTestId('bootstrap-required')).toHaveTextContent('true');
    expect(screen.getByTestId('primary-market')).toHaveTextContent('HK');
    expect(screen.getByTestId('enabled-markets')).toHaveTextContent('HK,US');
  });
});
