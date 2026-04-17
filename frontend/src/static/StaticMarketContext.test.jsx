import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { MemoryRouter, Routes, Route, useSearchParams } from 'react-router-dom';

import {
  STATIC_MARKET_STORAGE_KEY,
  StaticMarketProvider,
  useStaticMarket,
} from './StaticMarketContext';

function MarketHarness() {
  const { selectedMarket, setSelectedMarket } = useStaticMarket();
  const [searchParams, setSearchParams] = useSearchParams();

  return (
    <>
      <div data-testid="selected-market">{selectedMarket}</div>
      <div data-testid="search-params">{searchParams.toString()}</div>
      <button type="button" onClick={() => setSelectedMarket('HK')}>
        Switch to HK
      </button>
      <button
        type="button"
        onClick={() => {
          const next = new URLSearchParams(searchParams);
          next.set('foo', 'bar');
          setSearchParams(next, { replace: true });
        }}
      >
        Set Foo
      </button>
    </>
  );
}

function renderProvider(initialEntry = '/scan') {
  return render(
    <MemoryRouter initialEntries={[initialEntry]}>
      <Routes>
        <Route
          path="*"
          element={(
            <StaticMarketProvider supportedMarkets={['US', 'HK']} defaultMarket="US">
              <MarketHarness />
            </StaticMarketProvider>
          )}
        />
      </Routes>
    </MemoryRouter>
  );
}

describe('StaticMarketProvider', () => {
  beforeEach(() => {
    window.localStorage.clear();
  });

  afterEach(() => {
    window.localStorage.clear();
  });

  it('persists a query-selected market as the new default', async () => {
    const { unmount } = renderProvider('/breadth?market=HK');

    expect(screen.getByTestId('selected-market')).toHaveTextContent('HK');
    await waitFor(() => {
      expect(window.localStorage.getItem(STATIC_MARKET_STORAGE_KEY)).toBe('HK');
    });

    unmount();
    renderProvider('/breadth');

    expect(screen.getByTestId('selected-market')).toHaveTextContent('HK');
  });

  it('preserves existing query params when switching markets', async () => {
    renderProvider('/scan');

    fireEvent.click(screen.getByRole('button', { name: 'Set Foo' }));
    fireEvent.click(screen.getByRole('button', { name: 'Switch to HK' }));

    await waitFor(() => {
      expect(screen.getByTestId('selected-market')).toHaveTextContent('HK');
    });
    expect(screen.getByTestId('search-params')).toHaveTextContent('foo=bar&market=HK');
    expect(window.localStorage.getItem(STATIC_MARKET_STORAGE_KEY)).toBe('HK');
  });
});
