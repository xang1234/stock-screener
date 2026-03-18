import { createTheme, ThemeProvider } from '@mui/material';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import ScanPage from './ScanPage';

vi.mock('../contexts/RuntimeContext', () => ({
  useRuntime: () => ({
    runtimeReady: false,
    uiSnapshots: {
      scan: false,
    },
  }),
}));

vi.mock('../hooks/useFilterPresets', () => ({
  useFilterPresets: () => ({
    presets: [],
    isLoading: false,
    createPresetAsync: vi.fn(),
    updatePresetAsync: vi.fn(),
    deletePreset: vi.fn(),
    isCreating: false,
    isUpdating: false,
  }),
}));

describe('ScanPage', () => {
  it('renders without a temporal-dead-zone crash before runtime bootstrap completes', () => {
    const queryClient = new QueryClient({
      defaultOptions: {
        queries: {
          retry: false,
        },
      },
    });

    render(
      <QueryClientProvider client={queryClient}>
        <ThemeProvider theme={createTheme()}>
          <ScanPage />
        </ThemeProvider>
      </QueryClientProvider>
    );

    expect(screen.getByRole('progressbar')).toBeInTheDocument();
  });
});
