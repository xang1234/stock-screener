import { createTheme, ThemeProvider } from '@mui/material';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen, waitFor } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import WatchlistsTab from './WatchlistsTab';

const runtimeState = {
  bootstrap: {
    message: 'Desktop setup is still warming market data.',
  },
  bootstrapIncomplete: true,
};

vi.mock('../../contexts/RuntimeContext', () => ({
  useRuntime: () => runtimeState,
}));

vi.mock('../../api/userWatchlists', () => ({
  getWatchlists: vi.fn().mockResolvedValue({ watchlists: [] }),
  getWatchlistData: vi.fn(),
}));

vi.mock('./WatchlistTable', () => ({
  default: () => <div>watchlist-table</div>,
}));

vi.mock('./UserWatchlistManager', () => ({
  default: () => null,
}));

vi.mock('./WatchlistChartModal', () => ({
  default: () => null,
}));

describe('WatchlistsTab desktop empty state', () => {
  it('shows an initialization message while desktop bootstrap is incomplete', async () => {
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
          <WatchlistsTab />
        </ThemeProvider>
      </QueryClientProvider>
    );

    await waitFor(() => {
      expect(screen.getByText(/warming market data/i)).toBeInTheDocument();
    });
  });
});
