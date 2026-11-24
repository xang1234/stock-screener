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

  const DIVERGENCE = {
    index_off_high_pct: -1.2,
    above_sma200_pct: 34,
    new_high_52week_count: 40,
    new_low_52week_count: 12,
    flagged: true,
  };

  it('states a flagged breadth divergence in one line', () => {
    renderWithProviders(<MarketHealthExposure exposure={{ ...SAMPLE, breadth_divergence: DIVERGENCE }} />);
    expect(screen.getByText('Divergence: index 1.2% off high, 34% of stocks above 200-day')).toBeInTheDocument();
  });

  it('names new lows outnumbering new highs when that drives the divergence', () => {
    const divergence = { ...DIVERGENCE, above_sma200_pct: 62, new_high_52week_count: 12, new_low_52week_count: 40 };
    renderWithProviders(<MarketHealthExposure exposure={{ ...SAMPLE, breadth_divergence: divergence }} />);
    expect(screen.getByText(
      'Divergence: index 1.2% off high, 62% of stocks above 200-day, 40 new lows vs 12 new highs',
    )).toBeInTheDocument();
  });

  it.each([false, null])('says nothing about divergence when the flag is %s', (flagged) => {
    renderWithProviders(<MarketHealthExposure exposure={{ ...SAMPLE, breadth_divergence: { ...DIVERGENCE, flagged } }} />);
    expect(screen.queryByText(/Divergence/)).not.toBeInTheDocument();
  });

  it('reads FTD quality as the share of 4% gainers near their highs', () => {
    const ftdQuality = { ftd_date: '2026-11-11', stocks_up_4pct: 137, up_4pct_near_high_count: 52, near_high_share: 52 / 137 };
    renderWithProviders(<MarketHealthExposure exposure={{ ...SAMPLE, ftd_quality: ftdQuality }} />);
    expect(screen.getByText('FTD 11/11: 38% of 4% gainers near highs (52 of 137)')).toBeInTheDocument();
  });

  it.each([
    [5, 'FTD 11/11: 38% of 4% gainers near highs (52 of 137) · 42 new template entries in 5 sessions'],
    [3, 'FTD 11/11: 38% of 4% gainers near highs (52 of 137) · 42 new template entries (3 of 5 sessions so far)'],
  ])('adds new template entries after the FTD with %s sessions counted', (sessions, text) => {
    const ftdQuality = {
      ftd_date: '2026-11-11', stocks_up_4pct: 137, up_4pct_near_high_count: 52, near_high_share: 52 / 137,
      new_template_entries: 42, template_entry_sessions: sessions,
    };
    renderWithProviders(<MarketHealthExposure exposure={{ ...SAMPLE, ftd_quality: ftdQuality }} />);
    expect(screen.getByText(text)).toBeInTheDocument();
  });

  it('shows template entries alone when the near-high share is unknown', () => {
    const ftdQuality = {
      ftd_date: '2026-11-11', stocks_up_4pct: 0, up_4pct_near_high_count: 0, near_high_share: null,
      new_template_entries: 1, template_entry_sessions: 1,
    };
    renderWithProviders(<MarketHealthExposure exposure={{ ...SAMPLE, ftd_quality: ftdQuality }} />);
    expect(screen.getByText('FTD 11/11: 1 new template entry (1 of 5 sessions so far)')).toBeInTheDocument();
  });

  it('hides template entries while none of the sessions after the FTD has a count', () => {
    const ftdQuality = {
      ftd_date: '2026-11-11', stocks_up_4pct: 137, up_4pct_near_high_count: 52, near_high_share: 52 / 137,
      new_template_entries: null, template_entry_sessions: 0,
    };
    renderWithProviders(<MarketHealthExposure exposure={{ ...SAMPLE, ftd_quality: ftdQuality }} />);
    expect(screen.getByText('FTD 11/11: 38% of 4% gainers near highs (52 of 137)')).toBeInTheDocument();
  });

  it('omits the FTD line when the share is unknown', () => {
    const ftdQuality = { ftd_date: '2026-11-11', stocks_up_4pct: null, up_4pct_near_high_count: null, near_high_share: null };
    renderWithProviders(<MarketHealthExposure exposure={{ ...SAMPLE, ftd_quality: ftdQuality }} />);
    expect(screen.queryByText(/^FTD /)).not.toBeInTheDocument();
  });

  it('renders a placeholder when there is no exposure data', () => {
    renderWithProviders(<MarketHealthExposure exposure={null} />);
    expect(screen.getByText(/No exposure data yet/i)).toBeInTheDocument();
  });
});
