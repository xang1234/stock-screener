import { describe, it, expect, vi, beforeEach } from 'vitest';
import { ThemeProvider, createTheme } from '@mui/material';
import { renderWithProviders } from '../../test/renderWithProviders';
import { getDesignTokens } from '../../theme';
import CandlestickChart from './CandlestickChart';

// One entry per chart the component creates: the live set of visible-range
// handlers on its time scale, its visible logical range, and the series it
// drew. A removed chart throws if touched, like the real one.
const charts = vi.hoisted(() => []);

vi.mock('./createPriceChartSeries', () => ({
  createPriceChartSeries: () => {
    const handlers = new Set();
    let removed = false;
    const live = (fn) => (...args) => {
      if (removed) throw new Error('chart disposed');
      return fn(...args);
    };
    const series = () => {
      const s = { data: null, margins: null };
      s.setData = (data) => { s.data = data; };
      s.priceScale = () => ({ applyOptions: (o) => { s.margins = o.scaleMargins; } });
      return s;
    };
    const record = {
      handlers,
      range: null,
      candles: series(),
      volume: series(),
      rs: series(),
      markers: null,
    };
    // Returns void, like lightweight-charts 5.x (no unsubscribe function).
    const timeScale = {
      subscribeVisibleTimeRangeChange: live((h) => { handlers.add(h); }),
      unsubscribeVisibleTimeRangeChange: live((h) => { handlers.delete(h); }),
      fitContent: live(() => { record.range = 'fit'; }),
      setVisibleRange: () => {},
      getVisibleRange: () => null,
      getVisibleLogicalRange: live(() => record.range),
      setVisibleLogicalRange: live((range) => { record.range = range; }),
    };
    const chart = {
      timeScale: () => timeScale,
      subscribeCrosshairMove: () => {},
      applyOptions: () => {},
      resize: () => {},
      remove: () => { removed = true; },
    };
    charts.push(record);
    return {
      chart,
      volumeSeries: record.volume,
      candlestickSeries: record.candles,
      ema10Series: series(),
      ema20Series: series(),
      ema50Series: series(),
      rsLineSeries: record.rs,
      rsMarkers: { setMarkers: (m) => { record.markers = m; } },
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

describe('CandlestickChart rebuild on a theme toggle', () => {
  const bars = ['2026-01-05', '2026-01-06', '2026-01-07'].map((date, i) => ({
    date, open: 10 + i, high: 11 + i, low: 9 + i, close: 10.5 + i, volume: 1000,
  }));
  const rsLine = bars.map((b, i) => ({ time: b.date, value: 1 + i / 10 }));
  const inTheme = (mode) => (
    <ThemeProvider theme={createTheme(getDesignTokens(mode))}>
      {chart({ symbol: 'AAA', priceData: bars, rsLineData: rsLine, blueDots: ['2026-01-07'] })}
    </ThemeProvider>
  );

  beforeEach(() => {
    charts.length = 0;
  });

  it('draws the data, RS line and RS band on the rebuilt chart', () => {
    const { rerender } = renderWithProviders(inTheme('dark'));
    rerender(inTheme('light'));

    expect(charts).toHaveLength(2);
    const rebuilt = charts[1];
    expect(rebuilt.candles.data.map((c) => c.time)).toEqual(bars.map((b) => b.date));
    expect(rebuilt.volume.data).toHaveLength(bars.length);
    expect(rebuilt.rs.data).toEqual(rsLine);
    expect(rebuilt.markers).toEqual([expect.objectContaining({ time: '2026-01-07' })]);
    expect(rebuilt.rs.margins).toEqual(expect.objectContaining({ bottom: 0.22 }));
  });

  it('keeps the visible range instead of re-fitting', () => {
    const { rerender } = renderWithProviders(inTheme('dark'));
    charts[0].range = { from: 1, to: 2 }; // the user zoomed in
    rerender(inTheme('light'));

    expect(charts[1].range).toEqual({ from: 1, to: 2 });
  });
});
