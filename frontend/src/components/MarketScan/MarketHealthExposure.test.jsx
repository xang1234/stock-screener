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
  follow_through_date: null,
  cycle_state: 'rally_attempt',
  rally_day: 3,
  trend: 'neutral',
  vix: 18.5,
  benchmark_symbol: 'SPY',
  components: { base: 50, dist_from_200dma: 23, ma_alignment: 8, distribution_drag: -9, heavy_distribution_cap: 45 },
  history: [
    { date: '2026-06-12', exposure_score: 60, stance: 'Uptrend Under Pressure', cycle_state: 'uptrend', follow_through: true, failed_ftd: false },
    { date: '2026-06-15', exposure_score: 45, stance: 'Uptrend Under Pressure', cycle_state: 'correction', follow_through: false, failed_ftd: true },
    { date: '2026-06-16', exposure_score: 42, stance: 'Uptrend Under Pressure', cycle_state: 'rally_attempt', rally_day: 3, follow_through: false, failed_ftd: false },
  ],
};

const unflagged = (point) => ({ ...point, follow_through: false, failed_ftd: false });

describe('MarketHealthExposure', () => {
  it('renders the score, stance, and a why-breakdown', () => {
    renderWithProviders(<MarketHealthExposure exposure={SAMPLE} />);
    expect(screen.getByText('42')).toBeInTheDocument();
    expect(screen.getAllByText('Uptrend Under Pressure').length).toBeGreaterThan(0);
    expect(screen.getByText('Distribution days')).toBeInTheDocument();
    // a "_cap" contribution renders as "cap N"
    expect(screen.getByText('cap 45')).toBeInTheDocument();
  });

  it.each([
    [{ cycle_state: 'rally_attempt', rally_day: 3 }, 'Rally attempt, day 3'],
    [{ cycle_state: 'correction', rally_day: null }, 'Correction'],
    [{ cycle_state: 'uptrend', rally_day: null, follow_through_date: '2026-04-08' }, 'Uptrend (FTD 2026-04-08)'],
    [{ cycle_state: 'uptrend', rally_day: null }, 'Uptrend'],
  ])('shows the market cycle %o as "%s"', (cycle, label) => {
    renderWithProviders(<MarketHealthExposure exposure={{ ...SAMPLE, ...cycle }} />);
    expect(screen.getByText('Market cycle')).toBeInTheDocument();
    expect(screen.getByText(label)).toBeInTheDocument();
  });

  it('marks follow-through and failed follow-through days apart on the timeline', () => {
    renderWithProviders(<MarketHealthExposure exposure={SAMPLE} />);
    expect(screen.getByText('● follow-through day')).toBeInTheDocument();
    expect(screen.getByText('○ failed FTD')).toBeInTheDocument();
  });

  it('omits both timeline legends when no day is flagged', () => {
    renderWithProviders(<MarketHealthExposure exposure={{ ...SAMPLE, history: SAMPLE.history.map(unflagged) }} />);
    expect(screen.queryByText('● follow-through day')).not.toBeInTheDocument();
    expect(screen.queryByText('○ failed FTD')).not.toBeInTheDocument();
  });

  it('renders a placeholder when there is no exposure data', () => {
    renderWithProviders(<MarketHealthExposure exposure={null} />);
    expect(screen.getByText(/No exposure data yet/i)).toBeInTheDocument();
  });
});
