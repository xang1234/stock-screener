import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import { renderWithProviders } from '../../test/renderWithProviders';
import StockDetails from './StockDetails';

const getStockDecisionDashboard = vi.fn();
const candlestickChartPropsSpy = vi.fn();

vi.mock('../../api/stocks', () => ({
  getStockDecisionDashboard: (...args) => getStockDecisionDashboard(...args),
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
        chart_data: { stage: 2, rs_rating: 97, eps_rating: 96, vcp_detected: true, adr_percent: 4.2, current_price: 921.45 },
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

    renderWithProviders(
      <MemoryRouter initialEntries={['/stock/NVDA']}>
        <Routes>
          <Route path="/stock/:symbol" element={<StockDetails />} />
        </Routes>
      </MemoryRouter>
    );

    expect(await screen.findByRole('heading', { name: 'NVDA' })).toBeInTheDocument();
    expect(screen.getByText('NVIDIA Corp')).toBeInTheDocument();
    expect(screen.getByText('Decision Summary')).toBeInTheDocument();
    expect(screen.getByText('Industry Peers')).toBeInTheDocument();
    expect(screen.getByText('AI Infrastructure')).toBeInTheDocument();
    expect(screen.getByTestId('candlestick-chart')).toHaveTextContent('NVDA');
    expect(candlestickChartPropsSpy.mock.calls.at(-1)?.[0]?.dataUpdatedAtOverride).toBeUndefined();
  });
});
