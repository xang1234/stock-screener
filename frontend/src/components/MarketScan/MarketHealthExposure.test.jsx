import { describe, it, expect } from 'vitest';
import { screen } from '@testing-library/react';
import { renderWithProviders } from '../../test/renderWithProviders';
import MarketHealthExposure from './MarketHealthExposure';

const SAMPLE = {
  market: 'US',
  date: '2026-06-16',
  exposure_score: 42,
  stance: 'Uptrend Under Pressure',
  distribution_day_count: 4,
  follow_through_day: false,
  trend: 'neutral',
  vix: 18.5,
  benchmark_symbol: 'SPY',
  components: { base: 50, dist_from_200dma: 23, ma_alignment: 8, distribution_drag: -9, heavy_distribution_cap: 45 },
  history: [
    { date: '2026-06-12', exposure_score: 60, stance: 'Uptrend Under Pressure', follow_through: true },
    { date: '2026-06-16', exposure_score: 42, stance: 'Uptrend Under Pressure', follow_through: false },
  ],
};

describe('MarketHealthExposure', () => {
  it('renders the score, stance, and a why-breakdown', () => {
    renderWithProviders(<MarketHealthExposure exposure={SAMPLE} />);
    expect(screen.getByText('42')).toBeInTheDocument();
    expect(screen.getAllByText('Uptrend Under Pressure').length).toBeGreaterThan(0);
    expect(screen.getByText('Distribution days')).toBeInTheDocument();
    // a "_cap" contribution renders as "cap N"
    expect(screen.getByText('cap 45')).toBeInTheDocument();
    // the timeline shows a follow-through-day legend when any history point is flagged
    expect(screen.getByText('● follow-through day')).toBeInTheDocument();
  });

  it('omits the follow-through legend when no day is flagged', () => {
    const noFtd = { ...SAMPLE, history: SAMPLE.history.map((p) => ({ ...p, follow_through: false })) };
    renderWithProviders(<MarketHealthExposure exposure={noFtd} />);
    expect(screen.queryByText('● follow-through day')).not.toBeInTheDocument();
  });

  it('renders a placeholder when there is no exposure data', () => {
    renderWithProviders(<MarketHealthExposure exposure={null} />);
    expect(screen.getByText(/No exposure data yet/i)).toBeInTheDocument();
  });
});
