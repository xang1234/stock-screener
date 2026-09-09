import { fireEvent, render, screen } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { createTheme, ThemeProvider } from '@mui/material';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import DailySocialSignalsCard from './DailySocialSignalsCard';
import { getSocialSummary } from '../../api/socialSignals';

vi.mock('../../api/socialSignals', () => ({ getSocialSummary: vi.fn() }));

function renderCard(onOpen = vi.fn(), market = 'US') {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={client}>
      <ThemeProvider theme={createTheme()}>
        <DailySocialSignalsCard market={market} onOpen={onOpen} />
      </ThemeProvider>
    </QueryClientProvider>,
  );
  return onOpen;
}

describe('DailySocialSignalsCard', () => {
  beforeEach(() => vi.clearAllMocks());

  it('shows the published top five, themes, posture, coverage, freshness and navigation', async () => {
    getSocialSummary.mockResolvedValue({
      supported: true, available: true, stale: true,
      published_at: '2026-09-07T06:00:00Z',
      enabled_source_count: 3, participating_source_count: 3,
      top_signals: Array.from({ length: 6 }, (_, index) => ({
        candidate_key: `US:S${index}`, canonical_symbol: `S${index}`,
        queue_score: 90 - index,
        explanation: { market_exposure: 65 },
      })),
      dominant_themes: [{ theme_key: 'ai_infrastructure', accepted_company_count: 4 }],
    });
    const onOpen = renderCard();

    expect(screen.getByText('Social Signals')).toBeInTheDocument();
    expect(await screen.findByText('S0')).toBeInTheDocument();
    expect(screen.getByText('S4')).toBeInTheDocument();
    expect(screen.queryByText('S5')).not.toBeInTheDocument();
    expect(screen.getByText(/AI infrastructure · 4 companies/i)).toBeInTheDocument();
    expect(screen.getByText(/Exposure 65/)).toBeInTheDocument();
    expect(screen.getByText(/3\/3 sources/)).toBeInTheDocument();
    expect(screen.getByText('Stale')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Open Social Signals' }));
    expect(onOpen).toHaveBeenCalledTimes(1);
  });

  it('uses typed unavailable copy without inventing zeros', async () => {
    getSocialSummary.mockResolvedValue({
      supported: true, available: false, reason_code: 'no_published_run',
    });
    renderCard();
    expect(await screen.findByText(/first Social Signal run/i)).toBeInTheDocument();
  });

  it('shows unsupported markets without requesting a summary', async () => {
    renderCard(vi.fn(), 'KR');

    expect(await screen.findByText(/available for US, HK, CN, JP, and TW markets/)).toBeInTheDocument();
    expect(getSocialSummary).not.toHaveBeenCalled();
    expect(screen.getByRole('button', { name: 'Open Social Signals' })).toBeDisabled();
  });
});
