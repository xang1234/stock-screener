import { act, screen, waitFor } from '@testing-library/react';
import { useEffect } from 'react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { renderWithProviders } from '../../test/renderWithProviders';
import { fetchPriceHistoryBatch } from '../../api/priceHistory';
import GroupChartsGrid from './GroupChartsGrid';

// Chart mounts/unmounts, to check construction rather than DOM visibility.
const chartLifecycle = vi.hoisted(() => ({ mounts: [], unmounts: [] }));

vi.mock('./CandlestickChart', () => ({
  default: function MockCandlestickChart({ symbol, priceData }) {
    useEffect(() => {
      chartLifecycle.mounts.push(symbol);
      return () => chartLifecycle.unmounts.push(symbol);
    }, [symbol]);
    return (
      <div data-testid="group-candlestick-chart" data-symbol={symbol}>
        {symbol}:{priceData?.length || 0}
      </div>
    );
  },
}));

vi.mock('../../api/priceHistory', () => ({
  fetchPriceHistoryBatch: vi.fn(async (symbols) => ({
    data: Object.fromEntries(
      symbols.map((symbol) => [
        symbol,
        [{ date: '2026-06-26', open: 1, high: 2, low: 0.5, close: 1.5, volume: 100 }],
      ]),
    ),
    missing: [],
  })),
  priceHistoryKeys: {
    batch: (symbols, period = '6mo') => ['priceHistory', 'batch', period, symbols.join(',')],
  },
  PRICE_HISTORY_STALE_TIME: 300000,
}));

class MockIntersectionObserver {
  static instances = [];

  constructor(callback, options) {
    this.callback = callback;
    this.options = options;
    this.observed = new Set();
    this.disconnected = false;
    MockIntersectionObserver.instances.push(this);
  }

  observe(element) {
    this.observed.add(element);
  }

  unobserve(element) {
    this.observed.delete(element);
  }

  disconnect() {
    this.disconnected = true;
    this.observed.clear();
  }

  // Deliver an intersection for the given cells, as the browser would.
  reveal(elements) {
    act(() => {
      this.callback(elements.map((target) => ({ target, isIntersecting: true })));
    });
  }
}

const SYMBOLS_40 = Array.from({ length: 40 }, (_, i) => `S${String(i).padStart(2, '0')}`);

const cells = () => screen.getAllByTestId('group-chart-cell');
const charts = () => screen.queryAllByTestId('group-candlestick-chart');
const latestObserver = () => MockIntersectionObserver.instances.at(-1);

const renderGrid = (props = {}) =>
  renderWithProviders(<GroupChartsGrid symbols={['NVDA', 'AAPL', 'MSFT', 'META']} {...props} />);

describe('GroupChartsGrid', () => {
  beforeEach(() => {
    chartLifecycle.mounts.length = 0;
    chartLifecycle.unmounts.length = 0;
    MockIntersectionObserver.instances = [];
    fetchPriceHistoryBatch.mockClear();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('lays out chart cards as two columns on desktop widths', async () => {
    // Without IntersectionObserver every capped chart renders (fallback).
    vi.stubGlobal('IntersectionObserver', undefined);
    renderGrid();

    await waitFor(() => {
      expect(charts()).toHaveLength(4);
    });

    const chartGrid = screen.getByTestId('group-charts-grid');
    expect(chartGrid).toHaveStyle({
      display: 'grid',
    });
    const generatedCss = document.head.textContent.replace(/\s/g, '');
    expect(generatedCss).toContain('grid-template-columns:1fr');
    expect(generatedCss).toContain('grid-template-columns:repeat(2,minmax(0,1fr))');
  });

  describe('with IntersectionObserver', () => {
    beforeEach(() => {
      vi.stubGlobal('IntersectionObserver', MockIntersectionObserver);
    });

    it('creates charts only for cells near the visible area', async () => {
      renderGrid({ symbols: SYMBOLS_40, height: 200 });
      await waitFor(() => expect(cells()).toHaveLength(40));

      // Labels and reserved height are present before any chart exists.
      expect(charts()).toHaveLength(0);
      expect(screen.getByText('S39')).toBeInTheDocument();
      expect(MockIntersectionObserver.instances).toHaveLength(1);
      const observer = latestObserver();
      // About one row (chart + card header + gap) above and below.
      expect(observer.options.rootMargin).toBe('248px 0px');
      expect(observer.observed.size).toBe(40);

      observer.reveal(cells().slice(0, 4));
      expect(charts().map((el) => el.dataset.symbol)).toEqual(['S00', 'S01', 'S02', 'S03']);

      observer.reveal(cells().slice(4, 6));
      expect(charts()).toHaveLength(6);
      expect(observer.observed.size).toBe(34);
      expect(fetchPriceHistoryBatch).toHaveBeenCalledTimes(1);
    });

    it('keeps revealed charts mounted when scrolled away and back', async () => {
      renderGrid({ symbols: SYMBOLS_40 });
      await waitFor(() => expect(cells()).toHaveLength(40));
      const observer = latestObserver();

      observer.reveal(cells().slice(0, 2));
      // Scrolling back re-reports the same cells; nothing is rebuilt.
      observer.reveal(cells().slice(0, 2));

      expect(chartLifecycle.mounts).toEqual(['S00', 'S01']);
      expect(chartLifecycle.unmounts).toEqual([]);
    });

    it('disconnects the observer and ignores late callbacks on unmount', async () => {
      const { unmount } = renderGrid({ symbols: SYMBOLS_40 });
      await waitFor(() => expect(cells()).toHaveLength(40));
      const observer = latestObserver();
      const lateCells = cells().slice(0, 3);
      observer.reveal(lateCells.slice(0, 1));

      unmount();

      expect(observer.disconnected).toBe(true);
      expect(chartLifecycle.unmounts).toEqual(['S00']);
      observer.reveal(lateCells);
      expect(chartLifecycle.mounts).toEqual(['S00']);
    });

    it('starts fresh observations when the group changes', async () => {
      const { rerender } = renderGrid({ symbols: SYMBOLS_40 });
      await waitFor(() => expect(cells()).toHaveLength(40));
      const first = latestObserver();
      first.reveal(cells().slice(0, 2));

      rerender(<GroupChartsGrid symbols={['NVDA', 'AAPL']} />);
      await waitFor(() => expect(cells()).toHaveLength(2));

      expect(first.disconnected).toBe(true);
      expect(latestObserver()).not.toBe(first);
      expect(charts()).toHaveLength(0);
      expect(latestObserver().observed.size).toBe(2);
    });

    it('still shows missing-data cards without a chart', async () => {
      fetchPriceHistoryBatch.mockImplementationOnce(async () => ({
        data: { NVDA: [{ date: '2026-06-26', open: 1, high: 2, low: 0.5, close: 1.5, volume: 1 }] },
        missing: ['AAPL'],
      }));
      renderGrid({ symbols: ['NVDA', 'AAPL'] });
      await waitFor(() => expect(cells()).toHaveLength(2));

      expect(screen.getByText('No price data')).toBeInTheDocument();
      latestObserver().reveal(cells());
      expect(charts().map((el) => el.dataset.symbol)).toEqual(['NVDA']);
    });
  });
});
