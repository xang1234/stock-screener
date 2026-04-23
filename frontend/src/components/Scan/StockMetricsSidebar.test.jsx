import { screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import { renderWithProviders } from '../../test/renderWithProviders';
import StockMetricsSidebar from './StockMetricsSidebar';

describe('StockMetricsSidebar market-cap display', () => {
  it('falls back to scan-row market cap when fundamentals market cap is missing', () => {
    renderWithProviders(
      <StockMetricsSidebar
        stockData={{
          symbol: '0700.HK',
          company_name: 'Tencent Holdings',
          currency: 'HKD',
          market_cap: 3_900_000_000_000,
        }}
        fundamentals={{
          symbol: '0700.HK',
          market_cap: null,
        }}
      />
    );

    expect(screen.getByText('Mkt Cap (local)')).toBeInTheDocument();
    expect(screen.getByText('HK$3.9T')).toBeInTheDocument();
  });

  it('prefers USD-normalized market cap when available', () => {
    renderWithProviders(
      <StockMetricsSidebar
        stockData={{
          symbol: '0700.HK',
          company_name: 'Tencent Holdings',
          currency: 'HKD',
          market_cap: 3_900_000_000_000,
          market_cap_usd: 500_000_000_000,
        }}
        fundamentals={{
          symbol: '0700.HK',
          market_cap: null,
        }}
      />
    );

    expect(screen.getByText('Mkt Cap (USD)')).toBeInTheDocument();
    expect(screen.getByText('$500.0B')).toBeInTheDocument();
    expect(screen.queryByText('HK$3.9T')).not.toBeInTheDocument();
  });

  it('uses native-currency formatting in fundamentals-only mode when USD cap is absent', () => {
    renderWithProviders(
      <StockMetricsSidebar
        stockData={null}
        fundamentals={{
          symbol: '2330.TW',
          currency: 'TWD',
          market_cap: 30_000_000_000_000,
        }}
      />
    );

    expect(screen.getByText('Mkt Cap (local)')).toBeInTheDocument();
    expect(screen.getByText('NT$30.0T')).toBeInTheDocument();
  });
});
