import { useRef, useEffect, useLayoutEffect, useState, useMemo } from 'react';
import { Box, CircularProgress, Alert, AlertTitle, Button, ToggleButtonGroup, ToggleButton, useTheme, Typography } from '@mui/material';
import { createPriceChartSeries } from './createPriceChartSeries';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { fetchPriceHistory, fetchRSLine, priceHistoryKeys, PRICE_HISTORY_STALE_TIME } from '../../api/priceHistory';
import { rsBandForRange } from './rsBand';
import ChartSkeleton from './ChartSkeleton';
import { transformToCandlestickData } from './candlestickData';

// Debounce utility
const debounce = (fn, ms) => {
  let timer;
  const debounced = (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), ms);
  };
  debounced.cancel = () => {
    clearTimeout(timer);
  };
  return debounced;
};

/**
 * TradingView-style candlestick chart component
 *
 * @param {Object} props
 * @param {string} props.symbol - Stock symbol to display
 * @param {string} props.period - Time period (default: '6mo')
 * @param {number} props.height - Chart height in pixels
 * @param {Object} props.visibleRange - Optional visible time range to restore { from: timestamp, to: timestamp }
 * @param {Function} props.onVisibleRangeChange - Callback when visible range changes
 * @param {Array|null} props.priceData - Optional static OHLCV payload to render without API calls
 * @param {number|null} props.dataUpdatedAtOverride - Optional timestamp (ms) for static bundles
 * @param {boolean} props.compact - When true, hides overlays (Daily/Weekly toggle, OHLC legend, updated-at indicator) for dense grid layouts
 * @param {boolean} props.hideTimeframeToggle - When true, hides only the Daily/Weekly toggle (other overlays stay) and forces the daily timeframe
 * @param {boolean} props.interactive - When false, disables time-axis pan/zoom (mouse wheel, drag, pinch) until re-enabled
 */
function CandlestickChart({
  symbol,
  period = '6mo',
  height = 600,
  visibleRange = null,
  onVisibleRangeChange = null,
  priceData = null,
  rsLineData = null,
  blueDots = null,
  dataUpdatedAtOverride = null,
  compact = false,
  hideTimeframeToggle = false,
  interactive = true,
}) {
  const chartContainerRef = useRef(null);
  const chartRef = useRef(null);
  const candlestickSeriesRef = useRef(null);
  const volumeSeriesRef = useRef(null);
  const ema10SeriesRef = useRef(null);
  const ema20SeriesRef = useRef(null);
  const ema50SeriesRef = useRef(null);
  const rsLineSeriesRef = useRef(null); // RS line (stock / benchmark) overlay
  const rsMarkersRef = useRef(null); // Blue-dot markers primitive on the RS line
  const prevSymbolRef = useRef(null); // Track previous symbol
  const shouldRestoreRangeRef = useRef(false); // Flag to restore range on next data update
  const isFirstDataLoadRef = useRef(true); // Track first data load
  const prevCloseMapRef = useRef(new Map()); // Map of date -> previous close for % change calculation
  const latestCandleRef = useRef(null); // Store latest candle for default display

  const [timeframe, setTimeframe] = useState('daily');
  const [showRSLine, setShowRSLine] = useState(true); // RS line overlay toggle
  const [legendData, setLegendData] = useState(null); // OHLC legend data on hover
  const [rsBandTop, setRsBandTop] = useState(0.66); // top margin of the live RS band
  const theme = useTheme();
  const isDarkMode = theme.palette.mode === 'dark';

  const queryClient = useQueryClient();

  // Get any existing cached data for this symbol to use as placeholder
  const getCachedData = () => {
    return queryClient.getQueryData(priceHistoryKeys.symbol(symbol, period));
  };

  // Fetch price history data (uses shared query keys for cache consistency)
  const {
    data: fetchedApiData,
    isLoading,
    isFetching,
    error,
    refetch,
    dataUpdatedAt,
  } = useQuery({
    queryKey: priceHistoryKeys.symbol(symbol, period),
    queryFn: () => fetchPriceHistory(symbol, period),
    enabled: !!symbol && !priceData,
    staleTime: PRICE_HISTORY_STALE_TIME,
    keepPreviousData: true,
    // Show stale/cached data immediately while fetching fresh data
    placeholderData: getCachedData,
  });

  // Live RS line + blue-dot dates (interactive surfaces only). Static charts
  // carry the RS payload in their bundle instead, so the query stays disabled.
  const { data: fetchedRsData } = useQuery({
    queryKey: priceHistoryKeys.rsLine(symbol, period),
    queryFn: () => fetchRSLine(symbol, period),
    enabled: !!symbol && !priceData && !compact && showRSLine,
    staleTime: PRICE_HISTORY_STALE_TIME,
    keepPreviousData: true,
  });

  // RS data source: bundled payload in static mode, live query otherwise.
  const rsData = useMemo(() => {
    if (priceData) {
      return Array.isArray(rsLineData) && rsLineData.length > 0
        ? { rs_line: rsLineData, blue_dots: blueDots || [] }
        : null;
    }
    return fetchedRsData;
  }, [priceData, rsLineData, blueDots, fetchedRsData]);

  // Whether the RS overlay can render at all here (drives the toggle's visibility).
  const rsAvailable = !priceData || (Array.isArray(rsLineData) && rsLineData.length > 0);

  const apiData = priceData ?? fetchedApiData;
  const effectiveDataUpdatedAt = dataUpdatedAtOverride ?? dataUpdatedAt;
  const effectiveIsLoading = Boolean(!priceData && isLoading);
  const effectiveIsFetching = Boolean(!priceData && isFetching);
  const effectiveError = priceData ? null : error;
  const effectiveRefetch = priceData ? () => Promise.resolve({ data: priceData }) : refetch;

  // Format last updated time
  const lastUpdatedText = useMemo(() => {
    if (!effectiveDataUpdatedAt) return null;
    const now = Date.now();
    const diffMs = now - effectiveDataUpdatedAt;
    const diffSec = Math.floor(diffMs / 1000);
    const diffMin = Math.floor(diffSec / 60);
    const diffHr = Math.floor(diffMin / 60);

    if (diffSec < 60) return 'just now';
    if (diffMin < 60) return `${diffMin}m ago`;
    if (diffHr < 24) return `${diffHr}h ago`;
    return new Date(effectiveDataUpdatedAt).toLocaleDateString();
  }, [effectiveDataUpdatedAt]);

  // When the timeframe toggle is hidden, force daily so the chart can't
  // remain on a stale weekly aggregation chosen before the toggle disappeared.
  const effectiveTimeframe = hideTimeframeToggle ? 'daily' : timeframe;

  // Single source of truth for "is the RS line actually drawn right now". The
  // RS series is daily-only and toggleable, so it's hidden on weekly or when
  // toggled off. This drives BOTH the reserved-strip layout (price/volume
  // reclaim the strip's space when RS is hidden) and the "RS" label.
  const rsStripShown =
    showRSLine &&
    effectiveTimeframe === 'daily' &&
    Array.isArray(rsData?.rs_line) &&
    rsData.rs_line.length > 0;

  // Transform data - memoized to avoid expensive EMA recalculations on every render
  const chartData = useMemo(() => {
    if (!apiData) return null;
    return transformToCandlestickData(apiData, effectiveTimeframe);
  }, [apiData, effectiveTimeframe]);

  // Initialize chart on mount using useLayoutEffect for synchronous DOM access
  useLayoutEffect(() => {
    if (!chartContainerRef.current) {
      return;
    }

    const containerWidth = chartContainerRef.current.clientWidth;
    const containerHeight = chartContainerRef.current.clientHeight;

    // Use provided height if container doesn't have dimensions yet
    const chartWidth = containerWidth > 0 ? containerWidth : 800;
    const chartHeight = containerHeight > 0 ? containerHeight : height;

    const {
      chart,
      volumeSeries,
      candlestickSeries,
      ema10Series,
      ema20Series,
      ema50Series,
      rsLineSeries,
      rsMarkers,
    } = createPriceChartSeries(chartContainerRef.current, {
      width: chartWidth,
      height: chartHeight,
      isDarkMode,
      interactive,
    });
    chartRef.current = chart;
    volumeSeriesRef.current = volumeSeries;
    candlestickSeriesRef.current = candlestickSeries;
    ema10SeriesRef.current = ema10Series;
    ema20SeriesRef.current = ema20Series;
    ema50SeriesRef.current = ema50Series;
    rsLineSeriesRef.current = rsLineSeries;
    rsMarkersRef.current = rsMarkers;

    // Subscribe to crosshair move for OHLC legend (skip in compact mode — legend is hidden)
    if (!compact) chart.subscribeCrosshairMove((param) => {
      if (!param.time || !param.seriesData || !candlestickSeriesRef.current) {
        // Mouse left the chart or no data - fall back to latest candle
        if (latestCandleRef.current) {
          setLegendData(latestCandleRef.current);
        }
        return;
      }

      const candleData = param.seriesData.get(candlestickSeriesRef.current);
      if (candleData) {
        const prevClose = prevCloseMapRef.current.get(candleData.time);
        let changePercent = null;
        if (prevClose !== undefined && prevClose !== null && prevClose !== 0) {
          changePercent = ((candleData.close - prevClose) / prevClose) * 100;
        }
        setLegendData({
          open: candleData.open,
          high: candleData.high,
          low: candleData.low,
          close: candleData.close,
          changePercent,
        });
      }
    });

    // Handle container resize (including Modal fade completion)
    // Use ResizeObserver to detect when container becomes visible/changes size
    const resizeObserver = new ResizeObserver((entries) => {
      if (chartRef.current && entries[0]) {
        const { width, height } = entries[0].contentRect;
        if (width > 0 && height > 0) {
          chartRef.current.resize(width, height);
        }
      }
    });

    resizeObserver.observe(chartContainerRef.current);

    // Cleanup on unmount
    return () => {
      resizeObserver.disconnect();
      if (chartRef.current) {
        chartRef.current.remove();
        chartRef.current = null;
      }
      candlestickSeriesRef.current = null;
      volumeSeriesRef.current = null;
      ema10SeriesRef.current = null;
      ema20SeriesRef.current = null;
      ema50SeriesRef.current = null;
      rsLineSeriesRef.current = null;
      rsMarkersRef.current = null;
    };
    // `interactive` is intentionally not in the deps: it's only used as the
    // chart's initial handleScroll/handleScale value here, and the dedicated
    // applyOptions effect below picks up subsequent changes without remounting
    // the chart (which would reset visible range / EMAs).
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [height, isDarkMode, symbol, compact]); // Re-initialize only when required visual inputs change

  // Track symbol changes - set flag to restore range when symbol changes
  useEffect(() => {
    if (prevSymbolRef.current !== null && prevSymbolRef.current !== symbol) {
      // Symbol changed - flag that we should restore range on next data update
      shouldRestoreRangeRef.current = true;
    }
    prevSymbolRef.current = symbol;
  }, [symbol]);

  // Toggle pan/zoom handlers without re-initializing the chart so user state
  // (visible range, EMAs) is preserved when interactivity is enabled/disabled.
  useEffect(() => {
    if (!chartRef.current) return;
    chartRef.current.applyOptions({
      handleScroll: interactive,
      handleScale: interactive,
    });
  }, [interactive]);

  // Subscribe to visible time range changes
  useEffect(() => {
    if (!chartRef.current || !onVisibleRangeChange) return;

    const debouncedRangeChange = debounce((range) => {
      if (range) {
        onVisibleRangeChange(range);
      }
    }, 100);

    const timeScale = chartRef.current.timeScale();
    const unsubscribe = timeScale.subscribeVisibleTimeRangeChange((range) => {
      if (range) {
        debouncedRangeChange(range);
      }
    });

    return () => {
      debouncedRangeChange.cancel();
      if (unsubscribe) unsubscribe();
    };
  }, [onVisibleRangeChange, symbol]);

  // Update chart data when data changes
  useEffect(() => {
    if (!chartData || !chartRef.current) {
      return;
    }

    // Update volume data
    if (volumeSeriesRef.current && chartData.volume.length > 0) {
      volumeSeriesRef.current.setData(chartData.volume);
    }

    // Update candlestick data
    if (candlestickSeriesRef.current && chartData.candlesticks.length > 0) {
      candlestickSeriesRef.current.setData(chartData.candlesticks);
    }

    // Update EMAs
    if (ema10SeriesRef.current && chartData.ema10.length > 0) {
      ema10SeriesRef.current.setData(chartData.ema10);
    }

    if (ema20SeriesRef.current && chartData.ema20.length > 0) {
      ema20SeriesRef.current.setData(chartData.ema20);
    }

    if (ema50SeriesRef.current && chartData.ema50.length > 0) {
      ema50SeriesRef.current.setData(chartData.ema50);
    }

    // Build previous close map for % change calculation
    const newPrevCloseMap = new Map();
    for (let i = 1; i < chartData.candlesticks.length; i++) {
      const currentCandle = chartData.candlesticks[i];
      const prevCandle = chartData.candlesticks[i - 1];
      newPrevCloseMap.set(currentCandle.time, prevCandle.close);
    }
    prevCloseMapRef.current = newPrevCloseMap;

    // Set latest candle as default legend data
    if (chartData.candlesticks.length > 0) {
      const latestCandle = chartData.candlesticks[chartData.candlesticks.length - 1];
      const prevClose = newPrevCloseMap.get(latestCandle.time);
      let changePercent = null;
      if (prevClose !== undefined && prevClose !== null && prevClose !== 0) {
        changePercent = ((latestCandle.close - prevClose) / prevClose) * 100;
      }
      const latestLegend = {
        open: latestCandle.open,
        high: latestCandle.high,
        low: latestCandle.low,
        close: latestCandle.close,
        changePercent,
      };
      latestCandleRef.current = latestLegend;
      setLegendData(latestLegend);
    }

    // Check if we should restore the range (symbol changed and new data loaded)
    if (shouldRestoreRangeRef.current) {
      shouldRestoreRangeRef.current = false; // Clear the flag

      if (visibleRange && visibleRange.from && visibleRange.to) {
        // Use setTimeout to ensure data is fully rendered before setting range
        setTimeout(() => {
          if (chartRef.current) {
            chartRef.current.timeScale().setVisibleRange(visibleRange);
          }
        }, 0);
      } else {
        // No saved range - fit content
        chartRef.current.timeScale().fitContent();
      }
    } else if (isFirstDataLoadRef.current) {
      // First load - fit content
      isFirstDataLoadRef.current = false;
      chartRef.current.timeScale().fitContent();
    }
    // Otherwise, don't touch the zoom - let user adjust freely
  }, [chartData, visibleRange]);

  // Update the RS line overlay + blue-dot markers.
  // Only rendered on the daily timeframe (the RS series is daily); cleared
  // otherwise so stale points never linger under weekly candles.
  useEffect(() => {
    const series = rsLineSeriesRef.current;
    const markers = rsMarkersRef.current;
    if (!series || !chartRef.current) return;

    if (!rsStripShown) {
      series.setData([]);
      if (markers) markers.setMarkers([]);
      return;
    }

    const points = rsData.rs_line;
    series.setData(points.map((p) => ({ time: p.time, value: p.value })));

    const timesInSeries = new Set(points.map((p) => p.time));
    const markerList = (rsData.blue_dots || [])
      .filter((t) => timesInSeries.has(t))
      .map((t) => ({ time: t, position: 'inBar', color: '#2196f3', shape: 'circle' }));
    if (markers) markers.setMarkers(markerList);
  }, [rsData, rsStripShown]);

  // RS strip layout: when the RS line is shown, compress price to a 0.66 floor
  // so the [0.66, 0.78] band below it is always empty (the RS scale floats in
  // [rTop, 0.78], sized dynamically by the effect below); when hidden, expand
  // price to 0.78 to reclaim that space so the chart doesn't carry an empty
  // band. Runs after the init layout effect (same commit, ordered after) and
  // re-runs on chart re-creation so fresh series get the right bands.
  // useLayoutEffect keeps the resize off-screen (no flash).
  useLayoutEffect(() => {
    const candle = candlestickSeriesRef.current;
    const volume = volumeSeriesRef.current;
    if (!candle || !volume || !chartRef.current) return;

    // RS shown -> compress price to a 0.66 floor (bottom 0.34) so [0.66, 0.78] is
    // an always-empty strip the RS band lives in; hidden -> full height (0.78).
    const candleBottom = rsStripShown ? 0.34 : 0.22;
    candle.priceScale().applyOptions({ scaleMargins: { top: 0.05, bottom: candleBottom } });
    volume.priceScale().applyOptions({ scaleMargins: { top: 0.8, bottom: 0 } });
  }, [rsStripShown, symbol, height, isDarkMode, compact]);

  // Dynamic RS band: size the RS overlay scale so the line fills the empty space
  // below the candles without overlapping them. Recomputes on data change and on
  // pan/zoom (price re-auto-scales to the visible window, so the safe band moves).
  // Debounced; the 12%-38% clamp lives in computeRsBand. Skipped when RS is hidden.
  useEffect(() => {
    const chart = chartRef.current;
    if (!chart || !rsLineSeriesRef.current || !rsStripShown) return;

    // candles/rsLine are captured per effect run. They stay fresh because the
    // effect re-subscribes (and the cleanup cancels the pending debounce) whenever
    // chartData/rsData change, so a stale debounced callback can never fire.
    const candles = chartData?.candlesticks || [];
    const rsLine = rsData?.rs_line || [];

    const apply = () => {
      const liveChart = chartRef.current;
      const rsSeries = rsLineSeriesRef.current;
      if (!liveChart || !rsSeries) return; // guard against teardown mid-debounce
      const rTop = rsBandForRange(candles, rsLine, liveChart.timeScale().getVisibleRange());
      rsSeries.priceScale().applyOptions({ scaleMargins: { top: rTop, bottom: 0.22 } });
      setRsBandTop(rTop);
    };

    apply();
    const debouncedApply = debounce(apply, 80);
    const timeScale = chart.timeScale();
    timeScale.subscribeVisibleTimeRangeChange(debouncedApply);
    return () => {
      debouncedApply.cancel();
      // Only unsubscribe if this exact chart is still mounted. On unmount or a
      // symbol-change recreate, the old chart (and its time scale) is already
      // disposed, and calling unsubscribe on it would throw.
      if (chartRef.current === chart) {
        timeScale.unsubscribeVisibleTimeRangeChange(debouncedApply);
      }
    };
  }, [chartData, rsData, rsStripShown]);

  // Determine overlay state
  // Only show full loading state if we have no data at all (not even placeholder)
  const hasData = chartData && chartData.candlesticks.length > 0;
  const showLoading = effectiveIsLoading && !hasData;
  const showError = !effectiveIsLoading && effectiveError && !hasData;
  const showNoData = !effectiveIsLoading && !effectiveError && !hasData;
  // Show refresh indicator when fetching but we have data to display
  const showRefreshIndicator = effectiveIsFetching && hasData;

  // The "RS" label rides the strip, so it shows whenever the strip is drawn —
  // except in compact mode, where (like the OHLC legend/toggles) overlays are
  // suppressed for dense grid tiles.
  const rsLineVisible = !compact && rsStripShown;

  return (
    <Box
      sx={{
        width: '100%',
        height: height,
        position: 'relative',
        display: 'flex',
        flexDirection: 'column',
      }}
    >
      {/* Top-left overlay row: OHLC legend with the timeframe/RS toggles
          immediately to its right, so the buttons never cover the price. */}
      {!compact && !showLoading && !showError && !showNoData && (
        <Box
          sx={{
            position: 'absolute',
            top: 10,
            left: 10,
            zIndex: 10,
            display: 'flex',
            alignItems: 'center',
            flexWrap: 'wrap',
            gap: 1,
          }}
        >
          {/* OHLC Legend - tracks the hovered candle */}
          {legendData && (
            <Box
              sx={{
                bgcolor: 'rgba(30, 30, 30, 0.85)',
                borderRadius: 1,
                px: 1.5,
                py: 0.5,
                display: 'flex',
                gap: 2,
                fontFamily: 'monospace',
                fontSize: '0.8rem',
              }}
            >
              <span style={{ color: '#999' }}>
                O <span style={{ color: '#fff' }}>{legendData.open.toFixed(2)}</span>
              </span>
              <span style={{ color: '#999' }}>
                H <span style={{ color: '#fff' }}>{legendData.high.toFixed(2)}</span>
              </span>
              <span style={{ color: '#999' }}>
                L <span style={{ color: '#fff' }}>{legendData.low.toFixed(2)}</span>
              </span>
              <span style={{ color: '#999' }}>
                C <span style={{ color: '#fff' }}>{legendData.close.toFixed(2)}</span>
              </span>
              {legendData.changePercent !== null && (
                <span
                  style={{
                    color: legendData.changePercent >= 0 ? '#4CF64D' : '#E619CD',
                    fontWeight: 500,
                  }}
                >
                  {legendData.changePercent >= 0 ? '+' : ''}{legendData.changePercent.toFixed(2)}%
                </span>
              )}
            </Box>
          )}

          {/* Timeframe toggle */}
          {!hideTimeframeToggle && (
            <Box
              sx={{
                display: 'flex',
                gap: 0.5,
                bgcolor: 'background.paper',
                borderRadius: 1,
                boxShadow: 1,
              }}
            >
              <ToggleButtonGroup
                value={timeframe}
                exclusive
                onChange={(e, newTimeframe) => {
                  if (newTimeframe !== null) {
                    setTimeframe(newTimeframe);
                  }
                }}
                size="small"
              >
                <ToggleButton value="daily">Daily</ToggleButton>
                <ToggleButton value="weekly">Weekly</ToggleButton>
              </ToggleButtonGroup>
              {/* RS line overlay toggle — shown only where RS data can load
                  (live charts, or static charts whose bundle carries rs_line). */}
              {rsAvailable && (
                <ToggleButtonGroup size="small">
                  <ToggleButton
                    value="rs"
                    selected={showRSLine}
                    disabled={effectiveTimeframe !== 'daily'}
                    onClick={() => setShowRSLine((prev) => !prev)}
                    title="RS line (stock vs. benchmark) with blue-dot leadership signals"
                  >
                    RS
                  </ToggleButton>
                </ToggleButtonGroup>
              )}
            </Box>
          )}
        </Box>
      )}

      {/* Chart Container - always rendered so useLayoutEffect can initialize */}
      <div
        ref={chartContainerRef}
        style={{
          width: '100%',
          height: '100%',
        }}
      />

      {/* RS strip label - pinned to the top of the RS band (scaleMargins.top
          of the 'rs' scale) so the lower line is clearly the relative-strength
          overlay, not another moving average. */}
      {rsLineVisible && (
        <Typography
          variant="caption"
          sx={{
            position: 'absolute',
            top: `${(rsBandTop * 100).toFixed(1)}%`,
            left: 8,
            zIndex: 10,
            color: '#FFA726',
            fontFamily: 'monospace',
            fontWeight: 600,
            fontSize: '0.65rem',
            letterSpacing: '0.05em',
            pointerEvents: 'none',
            textShadow: '0 0 3px rgba(0, 0, 0, 0.6)',
          }}
        >
          RS
        </Typography>
      )}

      {/* Loading skeleton overlay */}
      {showLoading && (
        <Box
          sx={{
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
          }}
        >
          <ChartSkeleton height={height} isDarkMode={isDarkMode} />
        </Box>
      )}

      {/* Refresh indicator - shows when fetching fresh data while displaying cached data */}
      {showRefreshIndicator && (
        <Box
          sx={{
            position: 'absolute',
            bottom: 10,
            right: 10,
            zIndex: 10,
            display: 'flex',
            alignItems: 'center',
            gap: 1,
            bgcolor: 'rgba(30, 30, 30, 0.85)',
            borderRadius: 1,
            px: 1.5,
            py: 0.5,
          }}
        >
          <CircularProgress size={14} sx={{ color: '#87FBFB' }} />
          <Typography variant="caption" sx={{ color: '#87FBFB', fontSize: '0.7rem' }}>
            Refreshing...
          </Typography>
        </Box>
      )}

      {/* Last updated indicator */}
      {!compact && !showLoading && !showError && !showNoData && lastUpdatedText && !showRefreshIndicator && (
        <Box
          sx={{
            position: 'absolute',
            bottom: 10,
            right: 10,
            zIndex: 10,
            bgcolor: 'rgba(30, 30, 30, 0.7)',
            borderRadius: 1,
            px: 1,
            py: 0.25,
          }}
        >
          <Typography variant="caption" sx={{ color: '#999', fontSize: '0.65rem' }}>
            Updated {lastUpdatedText}
          </Typography>
        </Box>
      )}

      {/* Error overlay */}
      {showError && (
        <Box
          sx={{
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            bgcolor: 'background.paper',
            p: 3,
          }}
        >
          <Alert severity="error" sx={{ maxWidth: '100%' }}>
            <AlertTitle>Failed to load chart data</AlertTitle>
            {effectiveError.message || 'An error occurred while fetching the chart data'}
            <Button onClick={() => effectiveRefetch()} variant="outlined" size="small" sx={{ mt: 1 }}>
              Retry
            </Button>
          </Alert>
        </Box>
      )}

      {/* No data overlay */}
      {showNoData && (
        <Box
          sx={{
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'center',
            bgcolor: 'background.paper',
          }}
        >
          <Alert severity="info">No historical data available for {symbol}</Alert>
        </Box>
      )}
    </Box>
  );
}

export default CandlestickChart;
