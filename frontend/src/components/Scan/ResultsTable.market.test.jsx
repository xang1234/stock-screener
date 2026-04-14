/**
 * ResultsTable — 3axp multi-market rendering.
 *
 * Verifies market badge + currency prefixes + MCap USD/Local toggle behave
 * end-to-end against an HK row. Uses the same jsdom mocks as the parent
 * ResultsTable.test.jsx since virtualization, recharts, and React Query
 * don't play nicely without a real browser.
 */

import { describe, it, expect, vi } from 'vitest';
import { fireEvent, screen, within } from '@testing-library/react';
import { renderWithProviders } from '../../test/renderWithProviders';
import ResultsTable from './ResultsTable';

vi.mock('@tanstack/react-virtual', () => ({
  useVirtualizer: ({ count }) => ({
    getVirtualItems: () =>
      Array.from({ length: count }, (_, i) => ({
        index: i,
        start: i * 32,
        end: (i + 1) * 32,
        size: 32,
        key: i,
      })),
    getTotalSize: () => count * 32,
  }),
}));
vi.mock('./RSSparkline', () => ({ default: () => <span data-testid="rs-sparkline" /> }));
vi.mock('./PriceSparkline', () => ({ default: () => <span data-testid="price-sparkline" /> }));
vi.mock('../common/AddToWatchlistMenu', () => ({ default: () => null }));

const tencentHk = {
  symbol: '0700.HK',
  company_name: 'Tencent',
  composite_score: 82.5,
  rating: 'Buy',
  current_price: 410.5,
  volume: 15_000_000,
  market_cap: 3_900_000_000_000, // in HKD
  market_cap_usd: 500_000_000_000,
  adv_usd: 800_000_000,
  market: 'HK',
  exchange: 'HKEX',
  currency: 'HKD',
  field_availability: null,
  growth_metric_basis: null,
};

const defaultProps = {
  total: 1,
  page: 1,
  perPage: 25,
  sortBy: 'composite_score',
  sortOrder: 'desc',
  onPageChange: vi.fn(),
  onPerPageChange: vi.fn(),
  onSortChange: vi.fn(),
  onOpenChart: vi.fn(),
  loading: false,
};

describe('ResultsTable — 3axp market / currency / USD toggle', () => {
  it('renders the HK market badge next to a suffixed symbol', () => {
    renderWithProviders(<ResultsTable {...defaultProps} results={[tencentHk]} />);
    expect(screen.getByText('0700.HK')).toBeInTheDocument();
    expect(screen.getByTestId('market-badge-HK')).toBeInTheDocument();
  });

  it('prefixes the Price cell with HK$ based on the row currency', () => {
    renderWithProviders(<ResultsTable {...defaultProps} results={[tencentHk]} />);
    // Price column shows 410.5 with HK$ prefix at 2 decimals.
    expect(screen.getByText('HK$410.50')).toBeInTheDocument();
  });

  it('defaults Market Cap column to USD and toggles to local HK$ when clicked', () => {
    renderWithProviders(<ResultsTable {...defaultProps} results={[tencentHk]} />);

    // Default display: USD
    expect(screen.getByTestId('mcap-display-toggle').textContent).toContain('USD');
    // Header reflects USD
    expect(screen.getByText('MCap ($)')).toBeInTheDocument();
    // Cell shows the USD value formatted via formatLargeNumber ($500B)
    expect(screen.getByText('$500.0B')).toBeInTheDocument();

    // Flip to Local
    fireEvent.click(screen.getByTestId('mcap-display-toggle'));
    expect(screen.getByTestId('mcap-display-toggle').textContent).toContain('Local');
    expect(screen.getByText('MCap (local)')).toBeInTheDocument();
    // Local value uses HK$ prefix for the HKD-denominated market_cap.
    expect(screen.getByText('HK$3.9T')).toBeInTheDocument();
  });

  it('renders the ADV column in USD regardless of toggle', () => {
    renderWithProviders(<ResultsTable {...defaultProps} results={[tencentHk]} />);
    // $800M shows regardless of toggle state
    expect(screen.getByText('$800.0M')).toBeInTheDocument();
    fireEvent.click(screen.getByTestId('mcap-display-toggle'));
    expect(screen.getByText('$800.0M')).toBeInTheDocument();
  });

  it('renders nothing for the market badge when market is null (legacy rows)', () => {
    const legacyUsRow = { ...tencentHk, market: null, exchange: null, currency: null };
    renderWithProviders(<ResultsTable {...defaultProps} results={[legacyUsRow]} />);
    expect(screen.queryByTestId('market-badge-HK')).not.toBeInTheDocument();
    expect(screen.queryByTestId('market-badge-US')).not.toBeInTheDocument();
    // Price cell falls back to $ prefix when currency is null.
    expect(screen.getByText('$410.50')).toBeInTheDocument();
  });
});
