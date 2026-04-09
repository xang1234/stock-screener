import { render } from '@testing-library/react';
import { ThemeProvider, createTheme } from '@mui/material';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

// Match the dark theme from App.jsx getDesignTokens('dark')
const darkTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: { main: '#1976d2' },
    secondary: { main: '#dc004e' },
    success: { main: '#2e7d32', light: '#4caf50' },
    error: { main: '#d32f2f', light: '#f44336' },
    background: { default: '#121212', paper: '#1e1e1e' },
  },
});

export function renderWithProviders(ui, options = {}) {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
      },
    },
  });

  function Wrapper({ children }) {
    return (
      <QueryClientProvider client={queryClient}>
        <ThemeProvider theme={darkTheme}>{children}</ThemeProvider>
      </QueryClientProvider>
    );
  }
  return {
    queryClient,
    ...render(ui, { wrapper: Wrapper, ...options }),
  };
}
