import { describe, it, expect, vi, beforeEach } from 'vitest';
import { renderWithProviders } from '../../test/renderWithProviders';
import CandlestickChart from './CandlestickChart';

// One entry per chart the component creates: the live set of visible-range
// handlers on its time scale. A removed chart throws if touched, like the real one.
const charts = vi.hoisted(() => []);

vi.mock('./createPriceChartSeries', () => ({
  createPriceChartSeries: () => {
    const handlers = new Set();
    let removed = false;
    // Returns void, like lightweight-charts 5.x (no unsubscribe function).
    const live = (fn) => (...args) => {
      if (removed) throw new Error('chart disposed');
      fn(...args);
    };
    const timeScale = {
      subscribeVisibleTimeRangeChange: live((h) => handlers.add(h)),
      unsubscribeVisibleTimeRangeChange: live((h) => handlers.delete(h)),
      fitContent: () => {},
      setVisibleRange: () => {},
      getVisibleRange: () => null,
    };
    const chart = {
      timeScale: () => timeScale,
      subscribeCrosshairMove: () => {},
      applyOptions: () => {},
      resize: () => {},
      remove: () => { removed = true; },
    };
    const series = () => ({ setData: () => {}, priceScale: () => ({ applyOptions: () => {} }) });
    charts.push({ handlers });
    return {
      chart,
      volumeSeries: series(),
      candlestickSeries: series(),
      ema10Series: series(),
      ema20Series: series(),
      ema50Series: series(),
      rsLineSeries: series(),
      rsMarkers: { setMarkers: () => {} },
    };
  },
}));

const onRange = () => {};
const chart = (props) => (
  <CandlestickChart priceData={[]} onVisibleRangeChange={onRange} height={400} {...props} />
);

describe('CandlestickChart visible-range subscription', () => {
  beforeEach(() => {
    charts.length = 0;
  });

  it('keeps a single range handler while stepping through symbols', () => {
    const { rerender } = renderWithProviders(chart({ symbol: 'AAA' }));
    rerender(chart({ symbol: 'BBB' }));
    rerender(chart({ symbol: 'CCC' }));

    expect(charts).toHaveLength(1);
    expect(charts[0].handlers.size).toBe(1);
  });

  it('unsubscribes when the range callback is cleared', () => {
    const { rerender } = renderWithProviders(chart({ symbol: 'AAA' }));
    rerender(chart({ symbol: 'AAA', onVisibleRangeChange: null }));

    expect(charts[0].handlers.size).toBe(0);
  });

  it('subscribes the rebuilt chart when a visual input recreates it', () => {
    const { rerender, unmount } = renderWithProviders(chart({ symbol: 'AAA' }));
    rerender(chart({ symbol: 'AAA', height: 500 }));

    expect(charts).toHaveLength(2);
    expect(charts[1].handlers.size).toBe(1);
    unmount();
  });
});
