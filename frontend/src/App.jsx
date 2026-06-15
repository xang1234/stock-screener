import { useState, useMemo, lazy, Suspense } from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { PersistQueryClientProvider } from '@tanstack/react-query-persist-client';
import { createSyncStoragePersister } from '@tanstack/query-sync-storage-persister';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import {
  AppBar,
  Box,
  CircularProgress,
  CssBaseline,
  ThemeProvider,
  Toolbar,
  Typography,
  createTheme,
} from '@mui/material';
import ShowChartIcon from '@mui/icons-material/ShowChart';

import { STATIC_SITE_MODE } from './config/runtimeMode';

import Layout from './components/Layout/Layout';
import { PipelineProvider } from './contexts/PipelineContext';
import { MarketProvider } from './contexts/MarketContext';
import { RuntimeProvider, useRuntime } from './contexts/RuntimeContext';
import { StrategyProfileProvider } from './contexts/StrategyProfileContext';
import { ColorModeContext } from './contexts/ColorModeContext';
import {
  PERSISTED_QUERY_CACHE_BUSTER,
  shouldDehydratePersistedQuery,
} from './appQueryPersistence';

// All pages are lazy-loaded so the initial bundle stays free of heavy
// page-specific chunks (MarketScanPage alone pulls in recharts, ~400KB).
const BootstrapSetupScreen = lazy(() => import('./components/App/BootstrapSetupScreen'));
const ServerLoginScreen = lazy(() => import('./components/App/ServerLoginScreen'));
const MarketScanPage = lazy(() => import('./pages/MarketScanPage'));
const ScanPage = lazy(() => import('./pages/ScanPage'));
const StockDetails = lazy(() => import('./components/Stock/StockDetails'));
const BreadthPage = lazy(() => import('./pages/BreadthPage'));
const GroupRankingsPage = lazy(() => import('./pages/GroupRankingsPage'));
const ValidationPage = lazy(() => import('./pages/ValidationPage'));
const ThemesPage = lazy(() => import('./pages/ThemesPage'));
const ChatbotPage = lazy(() => import('./pages/ChatbotPage'));
const OperationsPage = lazy(() => import('./pages/OperationsPage'));
const StaticAppShell = lazy(() => import('./static/StaticAppShell'));

// In-app fallback for lazy page transitions (Layout chrome is already mounted,
// so a spinner in the content area is the right scope).
const PageLoadingFallback = () => (
  <Box
    sx={{
      display: 'flex',
      justifyContent: 'center',
      alignItems: 'center',
      minHeight: '50vh',
    }}
  >
    <CircularProgress />
  </Box>
);

// Cold-start fallback shown while appCapabilities is first loading and we don't
// yet know whether to render the app, the login screen, or the bootstrap setup.
// Renders only static header chrome — no nav links, no providers, no data
// queries — so it's safe to show before auth state is known, while still giving
// the user immediate visual structure instead of a bare spinner. On refresh,
// persisted page data can still paint quickly once live capabilities confirm
// whether the app shell, login screen, or bootstrap setup should render.
const AppLoadingScreen = () => (
  <Box sx={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
    <AppBar position="static" sx={{ minHeight: 48 }}>
      <Toolbar variant="dense" sx={{ minHeight: 48 }}>
        <ShowChartIcon sx={{ mr: 1, fontSize: 20 }} />
        <Typography variant="subtitle1" component="div" sx={{ fontWeight: 600 }}>
          STOCK SCANNER
        </Typography>
      </Toolbar>
    </AppBar>
    <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', flex: 1 }}>
      <CircularProgress />
    </Box>
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
});

// Keep persistence policy outside this component file so Fast Refresh remains
// limited to React component exports.
const persistOptions = {
  persister: queryCachePersister,
  maxAge: 24 * 60 * 60 * 1000,
  buster: PERSISTED_QUERY_CACHE_BUSTER,
  dehydrateOptions: {
    shouldDehydrateQuery: shouldDehydratePersistedQuery,
  },
};

// Function to create theme based on mode
const getDesignTokens = (mode) => ({
  palette: {
    mode,
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
    success: {
      main: '#2e7d32',
      light: '#4caf50',
    },
    error: {
      main: '#d32f2f',
      light: '#f44336',
    },
    background: {
      default: mode === 'light' ? '#f5f5f5' : '#0c0c11',
      paper: mode === 'light' ? '#ffffff' : '#141419',
    },
  },
  shape: {
    borderRadius: 10,
  },
  typography: {
    fontFamily: '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
    fontSize: 13,
    body1: {
      fontSize: '14px',
    },
    body2: {
      fontSize: '13px',
      lineHeight: 1.5,
    },
    caption: {
      fontSize: '11px',
    },
    h6: {
      fontSize: '15px',
      fontWeight: 600,
    },
  },
  components: {
    MuiTableCell: {
      styleOverrides: {
        root: {
          padding: '4px 6px',
          fontSize: '11px',
          lineHeight: 1.3,
          borderBottom: mode === 'light' ? '1px solid #e0e0e0' : '1px solid #333',
        },
        head: {
          backgroundColor: '#1a1a2e',
          color: '#ffffff',
          fontWeight: 600,
          fontSize: '10px',
          textTransform: 'uppercase',
          letterSpacing: '0.5px',
          padding: '6px 6px',
          whiteSpace: 'nowrap',
          borderBottom: '2px solid #333',
        },
        sizeSmall: {
          padding: '3px 5px',
        },
      },
    },
    MuiTableRow: {
      styleOverrides: {
        root: {
          height: 24,
          '&:nth-of-type(odd)': {
            backgroundColor: mode === 'light' ? '#fafafa' : '#1e1e1e',
          },
          '&:nth-of-type(even)': {
            backgroundColor: mode === 'light' ? '#ffffff' : '#252525',
          },
          '&:hover': {
            backgroundColor: mode === 'light' ? '#e3f2fd !important' : '#333 !important',
          },
          '&.MuiTableRow-head': {
            height: 28,
            '&:nth-of-type(odd)': {
              backgroundColor: '#1a1a2e',
            },
          },
        },
      },
    },
    MuiTableSortLabel: {
      styleOverrides: {
        root: {
          color: '#ffffff !important',
          '&:hover': {
            color: '#90caf9 !important',
          },
          '&.Mui-active': {
            color: '#90caf9 !important',
          },
        },
        icon: {
          color: '#90caf9 !important',
        },
      },
    },
    MuiChip: {
      styleOverrides: {
        sizeSmall: {
          height: 18,
          fontSize: '10px',
        },
        labelSmall: {
          padding: '0 6px',
        },
      },
    },
    MuiIconButton: {
      styleOverrides: {
        sizeSmall: {
          padding: 2,
        },
      },
    },
    MuiCardContent: {
      styleOverrides: {
        root: {
          padding: 12,
          '&:last-child': {
            paddingBottom: 12,
          },
        },
      },
    },
    MuiTab: {
      styleOverrides: {
        root: {
          minHeight: 40,
          fontSize: '12px',
        },
      },
    },
    MuiTabs: {
      styleOverrides: {
        root: {
          minHeight: 40,
        },
      },
    },
  },
});

function App() {
  const [mode, setMode] = useState('dark');

  const colorMode = useMemo(
    () => ({
      toggleColorMode: () => {
        setMode((prevMode) => (prevMode === 'light' ? 'dark' : 'light'));
      },
      mode,
    }),
    [mode]
  );

  const theme = useMemo(() => createTheme(getDesignTokens(mode)), [mode]);

  const appShell = STATIC_SITE_MODE ? (
    <Suspense fallback={<PageLoadingFallback />}>
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
    supportedMarkets,
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
          supportedMarkets={supportedMarkets}
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
