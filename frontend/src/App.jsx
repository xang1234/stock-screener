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
import ServerLoginScreen from './components/App/ServerLoginScreen';
import { AssistantChatProvider } from './contexts/AssistantChatContext';
import { PipelineProvider } from './contexts/PipelineContext';
import { RuntimeProvider, useRuntime } from './contexts/RuntimeContext';
import { StrategyProfileProvider } from './contexts/StrategyProfileContext';
import { ColorModeContext } from './contexts/ColorModeContext';

// Lazy loaded pages (secondary pages)
const BreadthPage = lazy(() => import('./pages/BreadthPage'));
const GroupRankingsPage = lazy(() => import('./pages/GroupRankingsPage'));
const DigestPage = lazy(() => import('./pages/DigestPage'));
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
      main: '#5C6BC0',
    },
    secondary: {
      main: '#7C4DFF',
    },
    success: {
      main: '#00C853',
      light: '#69F0AE',
    },
    error: {
      main: '#FF1744',
      light: '#FF5252',
    },
    background: {
      default: mode === 'light' ? '#f5f5f5' : '#0A0A0C',
      paper: mode === 'light' ? '#ffffff' : '#111114',
    },
    ...(mode === 'dark' && {
      text: {
        primary: '#E0E0E0',
        secondary: '#787880',
      },
      divider: '#1E1E22',
    }),
  },
  typography: {
    fontFamily: '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
    fontSize: 12,
    body1: {
      fontSize: '13px',
    },
    body2: {
      fontSize: '12px',
      lineHeight: 1.4,
    },
    caption: {
      fontSize: '10px',
    },
    h4: {
      fontSize: '18px',
      fontWeight: 600,
      letterSpacing: '-0.02em',
    },
    h6: {
      fontSize: '13px',
      fontWeight: 600,
      letterSpacing: '0.02em',
      textTransform: 'uppercase',
    },
  },
  components: {
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundImage: 'none',
          ...(mode === 'dark' && {
            border: '1px solid #1E1E22',
            boxShadow: 'none',
          }),
        },
      },
    },
    MuiAppBar: {
      styleOverrides: {
        root: {
          ...(mode === 'dark' && {
            backgroundColor: '#0A0A0C',
            borderBottom: '1px solid #1E1E22',
            boxShadow: 'none',
          }),
        },
      },
    },
    MuiTableCell: {
      styleOverrides: {
        root: {
          padding: '3px 6px',
          fontSize: '11px',
          lineHeight: 1.2,
          borderBottom: mode === 'light' ? '1px solid #e0e0e0' : '1px solid #1E1E22',
        },
        head: {
          backgroundColor: '#0F0F14',
          color: '#787880',
          fontWeight: 600,
          fontSize: '9px',
          textTransform: 'uppercase',
          letterSpacing: '0.8px',
          padding: '5px 6px',
          whiteSpace: 'nowrap',
          borderBottom: '1px solid #1E1E22',
        },
        sizeSmall: {
          padding: '2px 5px',
        },
      },
    },
    MuiTableRow: {
      styleOverrides: {
        root: {
          height: 22,
          '&:nth-of-type(odd)': {
            backgroundColor: mode === 'light' ? '#fafafa' : '#111114',
          },
          '&:nth-of-type(even)': {
            backgroundColor: mode === 'light' ? '#ffffff' : '#141418',
          },
          '&:hover': {
            backgroundColor: mode === 'light' ? '#e3f2fd !important' : '#1A1A22 !important',
          },
          '&.MuiTableRow-head': {
            height: 26,
            '&:nth-of-type(odd)': {
              backgroundColor: '#0F0F14',
            },
          },
        },
      },
    },
    MuiTableSortLabel: {
      styleOverrides: {
        root: {
          color: '#787880 !important',
          '&:hover': {
            color: '#9FA8DA !important',
          },
          '&.Mui-active': {
            color: '#9FA8DA !important',
          },
        },
        icon: {
          color: '#9FA8DA !important',
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
    features,
    isLoggingIn,
    login,
    loginError,
    runtimeReady,
  } = useRuntime();

  if (!runtimeReady) {
    return <PageLoadingFallback />;
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
            <Route path="/digest" element={<DigestPage />} />
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

  const routedApp = (
    <StrategyProfileProvider>
      <AssistantChatProvider>
        {appRoutes}
      </AssistantChatProvider>
    </StrategyProfileProvider>
  );

  if (features.themes) {
    return <PipelineProvider>{routedApp}</PipelineProvider>;
  }

  return routedApp;
}

export default App;
