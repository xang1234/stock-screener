import { createTheme, ThemeProvider } from '@mui/material';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen, waitFor } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import BreadthPage from './BreadthPage';

const runtimeState = {
  bootstrap: {
    message: 'Desktop bootstrap is still preparing breadth data.',
  },
  bootstrapIncomplete: true,
};

vi.mock('../contexts/RuntimeContext', () => ({
  useRuntime: () => runtimeState,
}));

vi.mock('../api/breadth', () => ({
  getCurrentBreadth: vi.fn().mockRejectedValue({
    message: 'Not found',
    response: { status: 404 },
  }),
  getHistoricalBreadth: vi.fn().mockResolvedValue([]),
  getBreadthSummary: vi.fn().mockResolvedValue({}),
}));

vi.mock('../api/stocks', () => ({
  getPriceHistory: vi.fn().mockResolvedValue([]),
}));

vi.mock('../components/Charts/BreadthChart', () => ({
  default: () => <div>breadth-chart</div>,
}));

describe('BreadthPage desktop initialization state', () => {
  it('shows a bootstrap message instead of an error while data is still initializing', async () => {
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
          <BreadthPage />
        </ThemeProvider>
      </QueryClientProvider>
    );

    await waitFor(() => {
      expect(screen.getByText(/preparing breadth data/i)).toBeInTheDocument();
    });
  });
});
