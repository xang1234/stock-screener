import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import { renderWithProviders } from '../../test/renderWithProviders';
import StockDetails from './StockDetails';

const getStockDecisionDashboard = vi.fn();
const getStockValidation = vi.fn();
const candlestickChartPropsSpy = vi.fn();

vi.mock('../../api/stocks', () => ({
  getStockDecisionDashboard: (...args) => getStockDecisionDashboard(...args),
  getStockValidation: (...args) => getStockValidation(...args),
}));

vi.mock('../Charts/CandlestickChart', () => ({
  default: (props) => {
    candlestickChartPropsSpy(props);
    return <div data-testid="candlestick-chart">{props.symbol}</div>;
  },
}));

vi.mock('../common/AddToWatchlistMenu', () => ({
  default: () => <div data-testid="add-to-watchlist" />,
}));

vi.mock('../Scan/StockMetricsSidebar', () => ({
  default: () => <div data-testid="stock-metrics-sidebar" />,
}));

describe('StockDetails', () => {
  it('renders the decision workspace from dashboard payload', async () => {
    candlestickChartPropsSpy.mockClear();
    getStockDecisionDashboard.mockResolvedValue({
      symbol: 'NVDA',
      info: {
        symbol: 'NVDA',
        name: 'NVIDIA Corp',
        sector: 'Technology',
        industry: 'Semiconductors',
        current_price: 921.45,
        market_cap: 2300000000000,
      },
      fundamentals: {
        symbol: 'NVDA',
        market_cap: 2300000000000,
        eps_growth_quarterly: 54.2,
        revenue_growth: 42.1,
      },
      technicals: {
        symbol: 'NVDA',
        current_price: 921.45,
        high_52w: 974,
        low_52w: 410,
      },
      chart: {
        price_history: [{ date: '2026-04-01', open: 900, high: 930, low: 890, close: 921.45, volume: 1000 }],
        chart_data: { stage: 2, rs_rating: 97, eps_rating: 96, vcp_detected: true, adr_percent: 4.2, current_price: 921.45, composite_score: 88.5, minervini_score: 86, company_name: 'NVIDIA Corp', rating: 'Strong Buy', screeners_run: ['minervini'] },
      },
      decision_summary: {
        composite_score: 88.5,
        rating: 'Strong Buy',
        screeners_passed: 2,
        screeners_total: 2,
        composite_method: 'weighted_average',
        top_strengths: [{ screener_name: 'minervini', criterion_name: 'rs_rating', score: 18, max_score: 20, passed: true }],
        top_weaknesses: [{ screener_name: 'minervini', criterion_name: 'vcp', score: 0, max_score: 20, passed: false }],
        freshness: {
          feature_run_id: 77,
          feature_as_of_date: '2026-04-02',
          feature_completed_at: '2026-04-02T20:05:00Z',
          breadth_date: '2026-04-02',
          has_price_history: true,
        },
      },
      screener_explanations: [
        {
          screener_name: 'minervini',
          score: 86,
          passes: true,
          rating: 'Strong Buy',
          criteria: [{ name: 'rs_rating', score: 18, max_score: 20, passed: true }],
        },
      ],
      peers: [{ symbol: 'AVGO', company_name: 'Broadcom', composite_score: 82.1, rs_rating: 93, stage: 2 }],
      themes: [{ theme_id: 1, display_name: 'AI Infrastructure', momentum_score: 81.2, mention_velocity: 1.8, basket_return_1m: 12.4, status: 'trending', lifecycle_state: 'active' }],
      regime: { label: 'offense', summary: 'Current stance: offense.', ratio_5day: 1.6, ratio_10day: 1.3 },
      degraded_reasons: [],
    });
    getStockValidation.mockResolvedValue({
      symbol: 'NVDA',
      lookback_days: 365,
      source_breakdown: [
        {
          source_kind: 'scan_pick',
          horizons: [
            { horizon_sessions: 1, sample_size: 5, positive_rate: 0.6, avg_return_pct: 1.2, median_return_pct: 1.0, avg_mfe_pct: null, avg_mae_pct: null, skipped_missing_history: 0 },
            { horizon_sessions: 5, sample_size: 4, positive_rate: 0.5, avg_return_pct: 2.1, median_return_pct: 1.7, avg_mfe_pct: 4.2, avg_mae_pct: -1.8, skipped_missing_history: 1 },
          ],
          recent_events: [],
          failure_clusters: [],
          degraded_reasons: [],
        },
        {
          source_kind: 'theme_alert',
          horizons: [
            { horizon_sessions: 1, sample_size: 2, positive_rate: 0.5, avg_return_pct: 0.8, median_return_pct: 0.8, avg_mfe_pct: null, avg_mae_pct: null, skipped_missing_history: 0 },
            { horizon_sessions: 5, sample_size: 2, positive_rate: 0.5, avg_return_pct: 1.1, median_return_pct: 1.1, avg_mfe_pct: 2.5, avg_mae_pct: -1.1, skipped_missing_history: 0 },
          ],
          recent_events: [],
          failure_clusters: [],
          degraded_reasons: [],
        },
      ],
      recent_events: [
        {
          source_kind: 'scan_pick',
          source_ref: 'run:12',
          event_at: '2026-04-01',
          entry_at: '2026-04-02',
          return_1s_pct: 1.4,
          return_5s_pct: 3.1,
          mfe_5s_pct: 4.2,
          mae_5s_pct: -1.2,
          attributes: { symbol: 'NVDA' },
        },
      ],
      failure_clusters: [
        {
          cluster_key: 'rating:Buy',
          label: 'Rating: Buy',
          sample_size: 4,
          avg_return_pct: -3.2,
          median_return_pct: -2.8,
        },
      ],
      freshness: {
        latest_feature_as_of_date: '2026-04-02',
      },
      degraded_reasons: [],
    });

    renderWithProviders(
      <MemoryRouter initialEntries={['/stocks/NVDA']}>
        <Routes>
          <Route path="/stocks/:ticker" element={<StockDetails />} />
        </Routes>
      </MemoryRouter>
    );

    expect(await screen.findByRole('heading', { name: 'NVDA' })).toBeInTheDocument();
    expect(screen.getByText('NVIDIA Corp')).toBeInTheDocument();
    expect(screen.getByText('Decision Summary')).toBeInTheDocument();
    expect(screen.getByText('Historical Validation')).toBeInTheDocument();
    expect(screen.getByText('Industry Peers')).toBeInTheDocument();
    expect(screen.getByText('AI Infrastructure')).toBeInTheDocument();
    expect(screen.getByText('Rating: Buy')).toBeInTheDocument();
    expect(screen.getByTestId('candlestick-chart')).toHaveTextContent('NVDA');
    expect(candlestickChartPropsSpy.mock.calls.at(-1)?.[0]?.dataUpdatedAtOverride).toBeUndefined();
  }, 10000);

  it('renders degraded workspace payloads without decision or regime cards throwing', async () => {
    getStockDecisionDashboard.mockResolvedValue({
      symbol: 'NVDA',
      freshness: {
        feature_run_id: null,
        feature_as_of_date: null,
        feature_completed_at: null,
        breadth_date: null,
        has_price_history: false,
      },
      info: {
        symbol: 'NVDA',
        name: 'NVIDIA Corp',
      },
      fundamentals: null,
      technicals: null,
      chart: {
        price_history: [],
        chart_data: { current_price: 921.45 },
      },
      screener_explanations: [],
      peers: [],
      themes: [],
      degraded_reasons: ['missing_explanation', 'missing_breadth'],
    });
    getStockValidation.mockResolvedValue({
      symbol: 'NVDA',
      lookback_days: 365,
      source_breakdown: [
        {
          source_kind: 'scan_pick',
          horizons: [],
          recent_events: [],
          failure_clusters: [],
          degraded_reasons: ['no_matching_scan_picks'],
        },
        {
          source_kind: 'theme_alert',
          horizons: [],
          recent_events: [],
          failure_clusters: [],
          degraded_reasons: ['no_matching_theme_alerts'],
        },
      ],
      recent_events: [],
      failure_clusters: [],
      freshness: {},
      degraded_reasons: ['missing_price_cache'],
    });

    renderWithProviders(
      <MemoryRouter initialEntries={['/stocks/NVDA']}>
        <Routes>
          <Route path="/stocks/:ticker" element={<StockDetails />} />
        </Routes>
      </MemoryRouter>
    );

    expect(await screen.findByRole('heading', { name: 'NVDA' })).toBeInTheDocument();
    expect(screen.getByText(/workspace is partially degraded/i)).toBeInTheDocument();
    expect(screen.getByTestId('stock-metrics-sidebar')).toBeInTheDocument();
    expect(screen.getByTestId('candlestick-chart')).toHaveTextContent('NVDA');
    expect(screen.getByText('Decision Summary')).toBeInTheDocument();
    expect(screen.getByText('Historical Validation')).toBeInTheDocument();
    expect(screen.getAllByText(/validation is partially degraded/i)).toHaveLength(3);
    expect(screen.getByText('Market regime context is unavailable for this symbol.')).toBeInTheDocument();
  }, 10000);
});
