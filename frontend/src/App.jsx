import { useState, useMemo, lazy, Suspense } from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { CssBaseline, ThemeProvider, createTheme, CircularProgress, Box } from '@mui/material';

import { STATIC_SITE_MODE } from './config/runtimeMode';
import StaticAppShell from './static/StaticAppShell';

// Eagerly loaded pages (most frequently used)
import ScanPage from './pages/ScanPage';
import MarketScanPage from './pages/MarketScanPage';
import StockDetails from './components/Stock/StockDetails';
import Layout from './components/Layout/Layout';
import DesktopSetupScreen from './components/App/DesktopSetupScreen';
import ServerLoginScreen from './components/App/ServerLoginScreen';
import { PipelineProvider } from './contexts/PipelineContext';
import { RuntimeProvider, useRuntime } from './contexts/RuntimeContext';
import { ColorModeContext } from './contexts/ColorModeContext';

// Lazy loaded pages (secondary pages)
const BreadthPage = lazy(() => import('./pages/BreadthPage'));
const GroupRankingsPage = lazy(() => import('./pages/GroupRankingsPage'));
const ValidationPage = lazy(() => import('./pages/ValidationPage'));
const ThemesPage = lazy(() => import('./pages/ThemesPage'));
const ChatbotPage = lazy(() => import('./pages/ChatbotPage'));

// Loading fallback component
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

// Create React Query client with optimized settings
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
      staleTime: 5 * 60 * 1000, // 5 minutes - data considered fresh
      gcTime: 30 * 60 * 1000, // 30 minutes - keep in cache (was cacheTime in v4)
      placeholderData: (previousData) => previousData, // Use previous data while loading
    },
  },
});

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
      default: mode === 'light' ? '#f5f5f5' : '#121212',
      paper: mode === 'light' ? '#ffffff' : '#1e1e1e',
    },
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
          borderRadius: 3,
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

  const appShell = STATIC_SITE_MODE ? <StaticAppShell /> : (
    <RuntimeProvider>
      <AppShell />
    </RuntimeProvider>
  );

  return (
    <QueryClientProvider client={queryClient}>
      <ColorModeContext.Provider value={colorMode}>
        <ThemeProvider theme={theme}>
          <CssBaseline />
          {appShell}
        </ThemeProvider>
      </ColorModeContext.Provider>
    </QueryClientProvider>
  );
}

function AppShell() {
  const {
    auth,
    desktopMode,
    features,
    isLoggingIn,
    login,
    loginError,
    runtimeReady,
    setupRequired,
  } = useRuntime();

  if (!runtimeReady) {
    return <PageLoadingFallback />;
  }

  if (desktopMode && setupRequired) {
    return <DesktopSetupScreen />;
  }

  if (auth?.required && !auth?.authenticated) {
    return (
      <ServerLoginScreen
        auth={auth}
        isLoggingIn={isLoggingIn}
        loginError={loginError}
        onLogin={login}
      />
    );
  }

  const appRoutes = (
    <Router>
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
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </Suspense>
      </Layout>
    </Router>
  );

  if (features.themes) {
    return <PipelineProvider>{appRoutes}</PipelineProvider>;
  }

  return appRoutes;
}

export default App;
