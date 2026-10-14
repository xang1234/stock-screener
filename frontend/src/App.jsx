import { useEffect, useMemo, lazy, Suspense } from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { PersistQueryClientProvider, removeOldestQuery } from '@tanstack/react-query-persist-client';
import { createSyncStoragePersister } from '@tanstack/query-sync-storage-persister';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import {
  AppBar,
  Box,
  Container,
  CssBaseline,
  ThemeProvider,
  Toolbar,
  createTheme,
} from '@mui/material';

import { STATIC_SITE_MODE } from './config/runtimeMode';

import Layout from './components/Layout/Layout';
import { BrandMark } from './components/Layout/AppNav';
import { PageSkeleton } from './components/common/LoadingSkeletons';
import { getDesignTokens, useColorMode } from './theme';
import { PipelineProvider } from './contexts/PipelineContext';
import { MarketProvider } from './contexts/MarketContext';
import { useMarket } from './contexts/MarketContext';
import { RuntimeProvider, useRuntime } from './contexts/RuntimeContext';
import { StrategyProfileProvider } from './contexts/StrategyProfileContext';
import { ColorModeContext } from './contexts/ColorModeContext';
import {
  PERSISTED_QUERY_CACHE_BUSTER,
  shouldDehydratePersistedQuery,
} from './appQueryPersistence';
import { isLiveOptionsAvailable } from './features/options/optionsAvailability';
import {
  loadBreadthPage,
  loadChatbotPage,
  loadGroupRankingsPage,
  loadMarketScanPage,
  loadOperationsPage,
  loadOptionsPage,
  loadOptionsSymbolPage,
  loadScanPage,
  loadStockDetails,
  loadThemesPage,
  loadValidationPage,
  loaderForPath,
} from './pageLoaders';

// All pages are lazy-loaded so the initial bundle stays free of heavy
// page-specific chunks (MarketScanPage alone pulls in recharts, ~400KB).
const BootstrapSetupScreen = lazy(() => import('./components/App/BootstrapSetupScreen'));
const ServerLoginScreen = lazy(() => import('./components/App/ServerLoginScreen'));
const MarketScanPage = lazy(loadMarketScanPage);
const ScanPage = lazy(loadScanPage);
const StockDetails = lazy(loadStockDetails);
const BreadthPage = lazy(loadBreadthPage);
const GroupRankingsPage = lazy(loadGroupRankingsPage);
const ValidationPage = lazy(loadValidationPage);
const ThemesPage = lazy(loadThemesPage);
const ChatbotPage = lazy(loadChatbotPage);
const OperationsPage = lazy(loadOperationsPage);
const OptionsPage = lazy(loadOptionsPage);
const OptionsSymbolPage = lazy(loadOptionsSymbolPage);
const StaticAppShell = lazy(() => import('./static/StaticAppShell'));

// Fetch the landing page's code alongside /app-capabilities rather than after
// it: the router only asks for it once capabilities confirm the app renders.
// A failed fetch surfaces through lazy() when the route renders, so it is
// dropped here.
if (!STATIC_SITE_MODE && typeof window !== 'undefined') {
  loaderForPath(window.location.pathname)?.().catch(() => {});
}

function LiveOptionsRoute({ detail = false }) {
  const { features, marketCatalog, primaryMarket } = useRuntime();
  const { selectedMarket } = useMarket();
  const available = isLiveOptionsAvailable({
    features,
    marketCatalog,
    selectedMarket,
    primaryMarket,
  });
  if (!available) return <Navigate to="/" replace />;
  return detail ? <OptionsSymbolPage /> : <OptionsPage />;
}

// In-app fallback for lazy page transitions (Layout chrome is already mounted,
// so a page-shaped skeleton in the content area is the right scope).
const PageLoadingFallback = () => <PageSkeleton />;

// Cold-start fallback shown while appCapabilities is first loading and we don't
// yet know whether to render the app, the login screen, or the bootstrap setup.
// Renders only static header chrome — no nav links, no providers, no data
// queries — so it's safe to show before auth state is known, while still giving
// the user immediate visual structure instead of a bare spinner. On refresh,
// persisted page data can still paint quickly once live capabilities confirm
// whether the app shell, login screen, or bootstrap setup should render.
const AppLoadingScreen = ({ label = 'Stock Scanner' }) => (
  <Box sx={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
    <AppBar position="static">
      <Toolbar variant="dense" sx={{ minHeight: 52 }}>
        <BrandMark label={label} />
      </Toolbar>
    </AppBar>
    <Container maxWidth="xl" sx={{ mt: 1 }}>
      <PageSkeleton label="Loading Stock Scanner" />
    </Container>
  </Box>
);

// Create React Query client with optimized settings
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      // Retry transient (network / 5xx) failures once; 4xx responses are
      // deterministic, so retrying them only doubles the failed requests.
      retry: (failureCount, error) => {
        const status = error?.response?.status;
        if (status != null && status >= 400 && status < 500) {
          return false;
        }
        return failureCount < 1;
      },
      refetchIntervalInBackground: false,
      staleTime: 5 * 60 * 1000, // 5 minutes - data considered fresh
      gcTime: 30 * 60 * 1000, // 30 minutes - keep in cache (was cacheTime in v4)
      // No global keep-previous-data placeholder: most queries here are keyed
      // by market/entity, where carrying the previous key's data across a
      // switch displays wrong data under the new label (and once poisoned a
      // bootstrap-seeded cache). Queries that benefit from previous data
      // (e.g. paginated tables) opt in explicitly.
    },
  },
});

// Persist the query cache to localStorage so a page refresh paints last-known
// data immediately instead of an empty shell. Restored entries are real data
// (not placeholderData), which also satisfies the runtimeReady gate on
// refresh — first-time visitors still wait for live capabilities before the
// app renders. Stale restored data refetches in the background per staleTime.
const queryCachePersister = createSyncStoragePersister({
  storage: typeof window !== 'undefined' ? window.localStorage : undefined,
  key: 'stockscanner-query-cache',
  throttleTime: 1000,
  // On a quota error, drop the oldest query and retry instead of silently
  // skipping the write (which would leave an older snapshot to restore).
  retry: removeOldestQuery,
});

// Keep persistence policy outside this component file so Fast Refresh remains
// limited to React component exports.
const persistOptions = {
  persister: queryCachePersister,
  maxAge: 24 * 60 * 60 * 1000,
  buster: PERSISTED_QUERY_CACHE_BUSTER,
  dehydrateOptions: {
    shouldDehydrateQuery: (query) => (
      shouldDehydratePersistedQuery(query, queryClient.getQueryCache())
    ),
  },
};

function App() {
  const { mode, toggleColorMode } = useColorMode();
  const colorMode = useMemo(() => ({ mode, toggleColorMode }), [mode, toggleColorMode]);

  const theme = useMemo(() => createTheme(getDesignTokens(mode)), [mode]);

  // Keep the pre-React background (index.css) and the mobile browser bar in step.
  useEffect(() => {
    document.documentElement.dataset.colorMode = mode;
    document.querySelector('meta[name="theme-color"]')
      ?.setAttribute('content', theme.palette.background.paper);
  }, [mode, theme]);

  const appShell = STATIC_SITE_MODE ? (
    // The static shell mounts its own chrome, so its cold start shows the same
    // app bar + skeleton frame instead of a bare skeleton flush to the edges.
    <Suspense fallback={<AppLoadingScreen label="Stock Scanner Daily" />}>
      <StaticAppShell />
    </Suspense>
  ) : (
    <RuntimeProvider>
      <AppShell />
    </RuntimeProvider>
  );

  const themedApp = (
    <ColorModeContext.Provider value={colorMode}>
      <ThemeProvider theme={theme}>
        <CssBaseline />
        {appShell}
      </ThemeProvider>
    </ColorModeContext.Provider>
  );

  // Static mode serves pre-baked JSON bundles that resolve synchronously —
  // persistence would only pause those queries during cache restore. Only
  // the live app persists its cache across reloads.
  if (STATIC_SITE_MODE) {
    return <QueryClientProvider client={queryClient}>{themedApp}</QueryClientProvider>;
  }

  return (
    <PersistQueryClientProvider client={queryClient} persistOptions={persistOptions}>
      {themedApp}
    </PersistQueryClientProvider>
  );
}

function AppShell() {
  const {
    auth,
    bootstrapRequired,
    bootstrapState,
    enabledMarkets,
    features,
    isLoggingIn,
    isStartingBootstrap,
    login,
    marketCatalog,
    primaryMarket,
    loginError,
    runtimeReady,
    startBootstrap,
    deploymentMarkets,
    bootstrapError,
  } = useRuntime();

  if (!runtimeReady) {
    return <AppLoadingScreen />;
  }

  if (auth?.required && !auth?.authenticated) {
    return (
      <Suspense fallback={<AppLoadingScreen />}>
        <ServerLoginScreen
          auth={auth}
          isLoggingIn={isLoggingIn}
          loginError={loginError}
          onLogin={login}
        />
      </Suspense>
    );
  }

  if (bootstrapRequired) {
    return (
      <Suspense fallback={<AppLoadingScreen />}>
        <BootstrapSetupScreen
          primaryMarket={primaryMarket}
          enabledMarkets={enabledMarkets}
          deploymentMarkets={deploymentMarkets}
          marketCatalog={marketCatalog}
          bootstrapState={bootstrapState}
          isStartingBootstrap={isStartingBootstrap}
          bootstrapError={bootstrapError}
          onStartBootstrap={startBootstrap}
        />
      </Suspense>
    );
  }

  const appRoutes = (
    <Router>
      <MarketProvider>
        <Layout>
          <Suspense fallback={<PageLoadingFallback />}>
            <Routes>
              <Route path="/" element={<MarketScanPage />} />
              <Route path="/scan" element={<ScanPage />} />
              <Route path="/breadth" element={<BreadthPage />} />
              <Route path="/groups" element={<GroupRankingsPage />} />
              <Route path="/validation" element={<ValidationPage />} />
              {features.themes && <Route path="/themes" element={<ThemesPage />} />}
              {features.chatbot && <Route path="/chatbot" element={<ChatbotPage />} />}
              <Route path="/options" element={<LiveOptionsRoute />} />
              <Route path="/options/:symbol" element={<LiveOptionsRoute detail />} />
              <Route path="/stocks/:ticker" element={<StockDetails />} />
              <Route path="/operations" element={<OperationsPage />} />
              <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
          </Suspense>
        </Layout>
      </MarketProvider>
    </Router>
  );

  const routedApp = (
    <StrategyProfileProvider>
      {appRoutes}
    </StrategyProfileProvider>
  );

  if (features.themes) {
    return <PipelineProvider>{routedApp}</PipelineProvider>;
  }

  return routedApp;
}

export default App;
