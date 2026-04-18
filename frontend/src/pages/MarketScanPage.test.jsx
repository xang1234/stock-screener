import { createTheme, ThemeProvider } from '@mui/material';
import { render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import MarketScanPage from './MarketScanPage';

const runtimeState = {
  features: {
    themes: false,
  },
};

vi.mock('../contexts/RuntimeContext', () => ({
  useRuntime: () => runtimeState,
}));

vi.mock('../components/MarketScan/KeyMarketsTab', () => ({
  default: () => <div>key-markets</div>,
}));

vi.mock('../components/MarketScan/DailyMarketSnapshotTab', () => ({
  default: () => <div>daily-snapshot-tab</div>,
}));

vi.mock('../components/MarketScan/ThemesTab', () => ({
  default: () => <div>themes-tab</div>,
}));

vi.mock('../components/MarketScan/WatchlistsTab', () => ({
  default: () => <div>watchlists-tab</div>,
}));

vi.mock('../components/MarketScan/StockbeeMmTab', () => ({
  default: () => <div>stockbee-tab</div>,
}));

describe('MarketScanPage capability gating', () => {
  it('removes the Themes tab when themes are disabled', () => {
    render(
      <ThemeProvider theme={createTheme()}>
        <MarketScanPage />
      </ThemeProvider>
    );

    expect(screen.queryByRole('tab', { name: 'Themes' })).not.toBeInTheDocument();
    expect(screen.getByRole('tab', { name: 'Key Markets' })).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: 'Daily Snapshot' })).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: 'Watchlists' })).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: 'Stockbee MM' })).toBeInTheDocument();
  });
});
