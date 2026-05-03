import { useRef, useEffect, useLayoutEffect, useState, useMemo } from 'react';
import { createChart, CrosshairMode, CandlestickSeries, LineSeries, HistogramSeries } from 'lightweight-charts';
import { Box, CircularProgress, Alert, AlertTitle, Button, ToggleButtonGroup, ToggleButton, useTheme, Skeleton, Typography } from '@mui/material';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { fetchPriceHistory, priceHistoryKeys, PRICE_HISTORY_STALE_TIME } from '../../api/priceHistory';

/**
 * Chart skeleton placeholder that shows chart structure while loading
 */
const ChartSkeleton = ({ height, isDarkMode }) => {
  const bgColor = isDarkMode ? '#1e1e1e' : '#ffffff';
  const lineColor = isDarkMode ? '#363a45' : '#e0e0e0';
  const skeletonColor = isDarkMode ? '#2a2a2a' : '#f0f0f0';

  return (
    <Box
      sx={{
        width: '100%',
        height: height,
        bgcolor: bgColor,
        position: 'relative',
        display: 'flex',
        flexDirection: 'column',
        p: 2,
      }}
    >
      {/* Y-axis area */}
      <Box sx={{ display: 'flex', flex: 1 }}>
        {/* Price axis labels (left side simulation) */}
        <Box sx={{ width: 60, display: 'flex', flexDirection: 'column', justifyContent: 'space-between', pr: 1 }}>
          {[...Array(6)].map((_, i) => (
            <Skeleton
              key={i}
              variant="text"
              width={40}
              height={16}
              sx={{ bgcolor: skeletonColor }}
            />
          ))}
        </Box>

        {/* Chart area */}
        <Box sx={{ flex: 1, position: 'relative', borderLeft: `1px solid ${lineColor}`, borderBottom: `1px solid ${lineColor}` }}>
          {/* Horizontal grid lines */}
          {[...Array(5)].map((_, i) => (
            <Box
              key={i}
              sx={{
                position: 'absolute',
                left: 0,
                right: 0,
                top: `${(i + 1) * 16.67}%`,
                borderTop: `1px dashed ${lineColor}`,
                opacity: 0.5,
              }}
            />
          ))}

          {/* Candlestick skeleton bars */}
          <Box sx={{
            display: 'flex',
            alignItems: 'flex-end',
            height: '70%',
            gap: '2px',
            px: 1,
            pt: 2,
          }}>
            {[...Array(40)].map((_, i) => {
              // Create varied heights for realistic look
              const minHeight = 20;
              const height = minHeight + Math.sin(i * 0.3) * 30 + Math.random() * 20;

              return (
                <Box
                  key={i}
                  sx={{
                    flex: 1,
                    maxWidth: 12,
                    height: `${height}%`,
                    bgcolor: skeletonColor,
                    borderRadius: 0.5,
                    animation: 'pulse 1.5s ease-in-out infinite',
                    animationDelay: `${i * 0.02}s`,
                    '@keyframes pulse': {
                      '0%, 100%': { opacity: 0.4 },
                      '50%': { opacity: 0.7 },
                    },
                  }}
                />
              );
            })}
          </Box>

          {/* Volume skeleton bars at bottom */}
          <Box sx={{
            display: 'flex',
            alignItems: 'flex-end',
            height: '25%',
            gap: '2px',
            px: 1,
            borderTop: `1px solid ${lineColor}`,
            pt: 0.5,
          }}>
            {[...Array(40)].map((_, i) => {
              const height = 20 + Math.random() * 60;
              return (
                <Box
                  key={i}
                  sx={{
                    flex: 1,
                    maxWidth: 12,
                    height: `${height}%`,
                    bgcolor: skeletonColor,
                    borderRadius: 0.5,
                    opacity: 0.5,
                    animation: 'pulse 1.5s ease-in-out infinite',
                    animationDelay: `${i * 0.02}s`,
                  }}
                />
              );
            })}
          </Box>
        </Box>
      </Box>

      {/* X-axis area (time labels) */}
      <Box sx={{ display: 'flex', justifyContent: 'space-around', pt: 1, pl: 8 }}>
        {[...Array(6)].map((_, i) => (
          <Skeleton
            key={i}
            variant="text"
            width={50}
            height={16}
            sx={{ bgcolor: skeletonColor }}
          />
        ))}
      </Box>

      {/* Loading indicator */}
      <Box
        sx={{
          position: 'absolute',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: 1,
        }}
      >
        <CircularProgress size={32} sx={{ color: isDarkMode ? '#666' : '#bbb' }} />
        <Typography variant="caption" color="text.secondary">
          Loading chart...
        </Typography>
      </Box>
    </Box>
  );
};

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
 * Calculate EMA (Exponential Moving Average)
 */
const calculateEMA = (data, period) => {
  if (!data || data.length < period) return [];

  const k = 2 / (period + 1);
  const emaData = [];

  // Calculate initial SMA as first EMA value
  let ema = 0;
  for (let i = 0; i < period; i++) {
    ema += data[i].close;
  }
  ema = ema / period;
  emaData.push({ time: data[period - 1].date, value: ema });

  // Calculate EMA for remaining data
  for (let i = period; i < data.length; i++) {
    ema = data[i].close * k + ema * (1 - k);
    emaData.push({ time: data[i].date, value: ema });
  }

  return emaData;
};

/**
 * Aggregate daily data to weekly
 */
const aggregateToWeekly = (dailyData) => {
  if (!dailyData || dailyData.length === 0) return [];

  const weeklyData = [];
  let currentWeek = null;

  dailyData.forEach((day) => {
    const date = new Date(day.date);
    const weekStart = new Date(date);
    weekStart.setDate(date.getDate() - date.getDay()); // Start of week (Sunday)
    const weekKey = weekStart.toISOString().split('T')[0];

    if (!currentWeek || currentWeek.weekKey !== weekKey) {
      if (currentWeek) {
        weeklyData.push(currentWeek.data);
      }
      currentWeek = {
        weekKey,
        data: {
          date: day.date,
          open: day.open,
          high: day.high,
          low: day.low,
          close: day.close,
          volume: day.volume,
        }
      };
    } else {
      currentWeek.data.high = Math.max(currentWeek.data.high, day.high);
      currentWeek.data.low = Math.min(currentWeek.data.low, day.low);
      currentWeek.data.close = day.close;
      currentWeek.data.volume += day.volume;
      currentWeek.data.date = day.date; // Use latest date for the week
    }
  });

  if (currentWeek) {
    weeklyData.push(currentWeek.data);
  }

  return weeklyData;
};

/**
 * Transform API data to TradingView Lightweight Charts format
 */
const transformToCandlestickData = (apiData, timeframe = 'daily') => {
  if (!apiData || apiData.length === 0) {
    return { candlesticks: [], volume: [], ema10: [], ema20: [], ema50: [] };
  }

  // Aggregate to weekly if needed
  const processedData = timeframe === 'weekly' ? aggregateToWeekly(apiData) : apiData;

  const candlesticks = [];
  const volume = [];

  processedData.forEach((d) => {
    // Candlestick data
    candlesticks.push({
      time: d.date,
      open: d.open,
      high: d.high,
      low: d.low,
      close: d.close,
    });

    // Volume data
    volume.push({
      time: d.date,
      value: d.volume,
      color: d.close >= d.open ? 'rgba(33, 150, 243, 0.5)' : 'rgba(230, 25, 205, 0.5)',
    });
  });

  // Calculate EMAs
  const ema10 = calculateEMA(processedData, 10);
  const ema20 = calculateEMA(processedData, 20);
  const ema50 = calculateEMA(processedData, 50);

  return { candlesticks, volume, ema10, ema20, ema50 };
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
  const prevSymbolRef = useRef(null); // Track previous symbol
  const shouldRestoreRangeRef = useRef(false); // Flag to restore range on next data update
  const isFirstDataLoadRef = useRef(true); // Track first data load
  const prevCloseMapRef = useRef(new Map()); // Map of date -> previous close for % change calculation
  const latestCandleRef = useRef(null); // Store latest candle for default display

  const [timeframe, setTimeframe] = useState('daily');
  const [legendData, setLegendData] = useState(null); // OHLC legend data on hover
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

  // Transform data - memoized to avoid expensive EMA recalculations on every render
  const chartData = useMemo(() => {
    if (!apiData) return null;
    return transformToCandlestickData(apiData, timeframe);
  }, [apiData, timeframe]);

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

    const chart = createChart(chartContainerRef.current, {
      width: chartWidth,
      height: chartHeight,
      layout: {
        background: { type: 'solid', color: isDarkMode ? '#1e1e1e' : '#ffffff' },
        textColor: isDarkMode ? '#d1d4dc' : '#333333',
      },
      grid: {
        vertLines: { color: isDarkMode ? '#363a45' : '#e0e0e0' },
        horzLines: { color: isDarkMode ? '#363a45' : '#e0e0e0' },
      },
      crosshair: {
        mode: CrosshairMode.Normal,
      },
      rightPriceScale: {
        borderColor: isDarkMode ? '#485263' : '#cccccc',
        mode: 1, // Logarithmic scale
      },
      timeScale: {
        borderColor: isDarkMode ? '#485263' : '#cccccc',
        timeVisible: true,
        secondsVisible: false,
      },
      handleScroll: interactive,
      handleScale: interactive,
    });

    chartRef.current = chart;

    // Create volume series (at bottom)
    const volumeSeries = chart.addSeries(HistogramSeries, {
      priceFormat: {
        type: 'volume',
      },
      priceScaleId: 'volume',
    });
    volumeSeries.priceScale().applyOptions({
      scaleMargins: {
        top: 0.7, // Volume takes bottom 30% of chart
        bottom: 0,
      },
    });
    volumeSeriesRef.current = volumeSeries;

    // Create candlestick series
    const candlestickSeries = chart.addSeries(CandlestickSeries, {
      upColor: '#2196f3',
      downColor: '#E619CD',
      borderVisible: false,
      wickUpColor: '#2196f3',
      wickDownColor: '#E619CD',
      priceScaleId: 'right',
    });
    candlestickSeries.priceScale().applyOptions({
      scaleMargins: {
        top: 0.1,
        bottom: 0.3, // Leave room for volume
      },
    });
    candlestickSeriesRef.current = candlestickSeries;

    // Create EMA 10 line series (Bright Green)
    const ema10Series = chart.addSeries(LineSeries, {
      color: '#4CF64D',
      lineWidth: 2,
      priceScaleId: 'right',
    });
    ema10SeriesRef.current = ema10Series;

    // Create EMA 20 line series (Cyan)
    const ema20Series = chart.addSeries(LineSeries, {
      color: '#87FBFB',
      lineWidth: 2,
      priceScaleId: 'right',
    });
    ema20SeriesRef.current = ema20Series;

    // Create EMA 50 line series (Green)
    const ema50Series = chart.addSeries(LineSeries, {
      color: '#38CD07',
      lineWidth: 2,
      priceScaleId: 'right',
    });
    ema50SeriesRef.current = ema50Series;

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

  // Determine overlay state
  // Only show full loading state if we have no data at all (not even placeholder)
  const hasData = chartData && chartData.candlesticks.length > 0;
  const showLoading = effectiveIsLoading && !hasData;
  const showError = !effectiveIsLoading && effectiveError && !hasData;
  const showNoData = !effectiveIsLoading && !effectiveError && !hasData;
  // Show refresh indicator when fetching but we have data to display
  const showRefreshIndicator = effectiveIsFetching && hasData;

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
      {/* Timeframe Toggle - only show when chart has data */}
      {!compact && !hideTimeframeToggle && !showLoading && !showError && !showNoData && (
        <Box
          sx={{
            position: 'absolute',
            top: 10,
            right: 10,
            zIndex: 10,
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
        </Box>
      )}

      {/* OHLC Legend - show when hovering over chart */}
      {!compact && !showLoading && !showError && !showNoData && legendData && (
        <Box
          sx={{
            position: 'absolute',
            top: 10,
            left: 10,
            zIndex: 10,
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

      {/* Chart Container - always rendered so useLayoutEffect can initialize */}
      <div
        ref={chartContainerRef}
        style={{
          width: '100%',
          height: '100%',
        }}
      />

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
