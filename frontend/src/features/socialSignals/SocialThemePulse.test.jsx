import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { createTheme, ThemeProvider } from '@mui/material';
import { render, screen } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { getSocialThemePulse } from '../../api/socialSignals';
import SocialThemePulse from './SocialThemePulse';

vi.mock('../../api/socialSignals', () => ({ getSocialThemePulse: vi.fn() }));

function renderPulse(enabled = true, market = 'HK') {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <ThemeProvider theme={createTheme()}>
        <SocialThemePulse enabled={enabled} market={market} />
      </ThemeProvider>
    </QueryClientProvider>,
  );
}

describe('SocialThemePulse', () => {
  beforeEach(() => vi.clearAllMocks());

  it('keeps Social and Market strength distinct and explains company coverage', async () => {
    getSocialThemePulse.mockResolvedValue({ supported: true, available: true, items: [
      { theme_key: 'ai_datacentres', name: 'AI Datacentres', status: 'confirmed',
        social_strength: 84, market_strength: 71, accepted_company_count: 5,
        measured_company_count: 4, benchmark_symbol: 'HSI', reasons: [] },
      { theme_key: 'robotics', name: 'Robotics', status: 'discovering',
        social_strength: 66, market_strength: null, accepted_company_count: 1,
        measured_company_count: 0, benchmark_symbol: null, reasons: ['candidate_theme'] },
      { theme_key: 'chips', name: 'Chips', status: 'insufficient_market_data',
        social_strength: 75, market_strength: null, accepted_company_count: 3,
        measured_company_count: 1, benchmark_symbol: 'HSI', reasons: ['insufficient_coverage'] },
    ] });
    renderPulse();

    expect(await screen.findByText('AI Datacentres')).toBeInTheDocument();
    expect(screen.getByText('Social strength 84')).toBeInTheDocument();
    expect(screen.getByText('Market strength 71')).toBeInTheDocument();
    expect(screen.getByText(/4\/5 companies · HSI/)).toBeInTheDocument();
    expect(screen.getByText('Discovering')).toBeInTheDocument();
    expect(screen.getByText('Insufficient market data')).toBeInTheDocument();
  });

  it('does not fetch or render when the capability is off', () => {
    renderPulse(false);
    expect(getSocialThemePulse).not.toHaveBeenCalled();
    expect(screen.queryByText('Social Pulse')).not.toBeInTheDocument();
  });

  it('renders an unsupported-market state without fetching', async () => {
    renderPulse(true, 'IN');

    expect(await screen.findByText(/available for US, HK, CN, JP, and TW markets/)).toBeInTheDocument();
    expect(getSocialThemePulse).not.toHaveBeenCalled();
    expect(screen.queryByText(/could not be loaded/)).not.toBeInTheDocument();
  });
});
