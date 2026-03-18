import { createTheme, ThemeProvider } from '@mui/material';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen, waitFor } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import GroupRankingsPage from './GroupRankingsPage';

const runtimeState = {
  runtimeReady: true,
  bootstrap: {
    message: 'Desktop bootstrap is still preparing group rankings.',
  },
  bootstrapIncomplete: true,
  features: {
    tasks: false,
  },
  uiSnapshots: {
    groups: true,
  },
};

vi.mock('../contexts/RuntimeContext', () => ({
  useRuntime: () => runtimeState,
}));

vi.mock('../api/groups', () => ({
  getGroupsBootstrap: vi.fn().mockRejectedValue({
    message: 'Not found',
    response: { status: 404 },
  }),
  getCurrentRankings: vi.fn().mockRejectedValue({
    message: 'Not found',
    response: { status: 404 },
  }),
  getRankMovers: vi.fn().mockResolvedValue({ gainers: [], losers: [] }),
  getGroupDetail: vi.fn(),
  triggerCalculation: vi.fn(),
  getCalculationStatus: vi.fn(),
}));

describe('GroupRankingsPage desktop initialization state', () => {
  it('shows bootstrap progress messaging and hides manual refresh when task controls are disabled', async () => {
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
          <GroupRankingsPage />
        </ThemeProvider>
      </QueryClientProvider>
    );

    await waitFor(() => {
      expect(screen.getByText(/preparing group rankings/i)).toBeInTheDocument();
    });

    expect(screen.queryByRole('button', { name: /refresh/i })).not.toBeInTheDocument();
  });
});
