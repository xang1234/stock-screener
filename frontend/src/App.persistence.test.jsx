import { QueryClient, dehydrate } from '@tanstack/react-query';
import { cleanup, render, screen, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import {
  PERSISTED_QUERY_CACHE_BUSTER,
  shouldDehydratePersistedQuery,
} from './appQueryPersistence';
import { DEFAULT_CAPABILITIES } from './contexts/RuntimeContext';

const { getAppCapabilities } = vi.hoisted(() => ({ getAppCapabilities: vi.fn() }));

vi.mock('./config/runtimeMode', () => ({ STATIC_SITE_MODE: false }));
vi.mock('./api/appRuntime', () => ({
  getAppCapabilities: (...args) => getAppCapabilities(...args),
  startRuntimeBootstrap: vi.fn(),
  updateRuntimeMarkets: vi.fn(),
}));
vi.mock('./components/App/ServerLoginScreen', () => ({ default: () => <div>Server Login</div> }));
vi.mock('./components/Layout/Layout', () => ({ default: ({ children }) => <div>{children}</div> }));
vi.mock('./contexts/PipelineContext', () => ({
  PipelineProvider: ({ children }) => <>{children}</>,
}));
vi.mock('./contexts/StrategyProfileContext', () => ({
  StrategyProfileProvider: ({ children }) => <>{children}</>,
}));
vi.mock('./pageLoaders', async (importOriginal) => {
  const loaders = {
    '/': async () => ({ default: () => <div>Market Scan Page</div> }),
    '/themes': async () => ({ default: () => <div>Themes Page</div> }),
  };
  return {
    ...(await importOriginal()),
    loadMarketScanPage: loaders['/'],
    loadThemesPage: loaders['/themes'],
    // App's import-time preload goes through loaderForPath. The real one
    // returns the real loaders, and a cold MarketScanPage or ThemesPage import
    // (~1 s) ran beside the test and outlasted findByText on busy CI runners.
    loaderForPath: (path) => loaders[path] ?? null,
  };
});

const successfulQuery = (queryKey) => ({
  queryKey,
  state: { status: 'success' },
});

const failedQuery = (queryKey) => ({
  queryKey,
  state: { status: 'error' },
});

describe('app query persistence', () => {
  it('persists app capabilities but not volatile runtime activity', () => {
    expect(shouldDehydratePersistedQuery(successfulQuery(['appCapabilities']))).toBe(true);
    expect(shouldDehydratePersistedQuery(successfulQuery(['runtimeActivity']))).toBe(false);
  });

  it('never restores a scannable watchlist symbol list from storage', () => {
    expect(shouldDehydratePersistedQuery(successfulQuery(['userWatchlistSymbols', 7]))).toBe(false);
    expect(shouldDehydratePersistedQuery(successfulQuery(['userWatchlists']))).toBe(true);
  });

  it('persists ordinary successful data queries only', () => {
    expect(shouldDehydratePersistedQuery(successfulQuery(['scanHistory', 'US']))).toBe(true);
    expect(shouldDehydratePersistedQuery(failedQuery(['scanHistory', 'US']))).toBe(false);
  });

  it('persists only the most recently loaded scan results page', () => {
    const queryClient = new QueryClient();
    const queryCache = queryClient.getQueryCache();
    queryClient.setQueryData(['scanResultsQuery', 'scan-1', 'page-1'], { rows: 1 }, { updatedAt: 1000 });
    queryClient.setQueryData(['scanResultsQuery', 'scan-1', 'page-2'], { rows: 2 }, { updatedAt: 2000 });
    queryClient.setQueryData(['scanHistory', 'US'], { scans: [] }, { updatedAt: 500 });

    const find = (queryKey) => queryCache.find({ queryKey, exact: true });

    expect(shouldDehydratePersistedQuery(find(['scanResultsQuery', 'scan-1', 'page-1']), queryCache)).toBe(false);
    expect(shouldDehydratePersistedQuery(find(['scanResultsQuery', 'scan-1', 'page-2']), queryCache)).toBe(true);
    expect(shouldDehydratePersistedQuery(find(['scanHistory', 'US']), queryCache)).toBe(true);
  });
});

// App.jsx holds its QueryClient at module scope, so each refresh imports a
// fresh copy, as a real page load starts with an empty in-memory cache.
async function importAppAt(path) {
  window.history.replaceState(null, '', path);
  vi.resetModules();
  return (await import('./App')).default;
}

// Call after importAppAt, right before render: an earlier test's persister can
// still write on its 1 s throttle, and a cold import can outlast that.
function persistCapabilities(capabilities) {
  const previousPageClient = new QueryClient();
  previousPageClient.setQueryData(['appCapabilities'], capabilities, { updatedAt: Date.now() - 60_000 });
  window.localStorage.setItem('stockscanner-query-cache', JSON.stringify({
    timestamp: Date.now(),
    buster: PERSISTED_QUERY_CACHE_BUSTER,
    clientState: dehydrate(previousPageClient, {
      shouldDehydrateQuery: shouldDehydratePersistedQuery,
    }),
  }));
}

function deferLiveCapabilities() {
  let resolveLive;
  getAppCapabilities.mockReturnValue(new Promise((resolve) => { resolveLive = resolve; }));
  return (capabilities) => resolveLive(capabilities);
}

const withThemes = (themes) => ({
  ...DEFAULT_CAPABILITIES,
  features: { ...DEFAULT_CAPABILITIES.features, themes },
});

describe('App refresh with a persisted cache', () => {
  afterEach(() => {
    cleanup();
    window.localStorage.clear();
    window.history.replaceState(null, '', '/');
    getAppCapabilities.mockReset();
  });

  it('paints the app from an authenticated cache, then shows login once the live check says the session expired', async () => {
    const App = await importAppAt('/');
    persistCapabilities({
      ...DEFAULT_CAPABILITIES,
      auth: { ...DEFAULT_CAPABILITIES.auth, required: true, authenticated: true },
    });
    const resolveLive = deferLiveCapabilities();

    render(<App />);

    expect(await screen.findByText('Market Scan Page')).toBeInTheDocument();
    expect(getAppCapabilities).toHaveBeenCalledTimes(1);

    resolveLive({
      ...DEFAULT_CAPABILITIES,
      auth: { ...DEFAULT_CAPABILITIES.auth, required: true, authenticated: false },
    });

    expect(await screen.findByText('Server Login')).toBeInTheDocument();
    expect(screen.queryByText('Market Scan Page')).not.toBeInTheDocument();
  });

  it('keeps a deep link the cached features hide until the live check enables it', async () => {
    const App = await importAppAt('/themes');
    persistCapabilities(withThemes(false));
    const resolveLive = deferLiveCapabilities();

    render(<App />);

    expect(await screen.findByText('Loading page')).toBeInTheDocument();
    expect(getAppCapabilities).toHaveBeenCalledTimes(1);
    expect(window.location.pathname).toBe('/themes');
    expect(screen.queryByText('Market Scan Page')).not.toBeInTheDocument();

    resolveLive(withThemes(true));

    expect(await screen.findByText('Themes Page')).toBeInTheDocument();
    expect(window.location.pathname).toBe('/themes');
  });

  it('redirects a deep link home once the live check confirms the feature is off', async () => {
    const App = await importAppAt('/themes');
    persistCapabilities(withThemes(false));
    const resolveLive = deferLiveCapabilities();

    render(<App />);

    expect(await screen.findByText('Loading page')).toBeInTheDocument();
    resolveLive(withThemes(false));

    expect(await screen.findByText('Market Scan Page')).toBeInTheDocument();
    await waitFor(() => expect(window.location.pathname).toBe('/'));
  });
});
