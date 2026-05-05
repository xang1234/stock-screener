import { useMemo, memo } from 'react';
import { AreaChart, Area, ResponsiveContainer, YAxis } from 'recharts';
import { Box, Tooltip, Typography } from '@mui/material';

/**
 * Price Sparkline Component
 *
 * Renders an area chart sparkline showing the price trend
 * over the last 30 trading days. Uses normalized prices for
 * consistent visual comparison across stocks.
 *
 * Color coding:
 * - Green: Price up overall (30-day change positive)
 * - Red: Price down overall (30-day change negative)
 *
 * Displays 1-day change as text badge
 */
function PriceSparkline({
  data,
  trend,
  change1d,
  industry,  // Industry to show in tooltip
  width = 100,
  height = 28,
  showChange = true,  // Whether to show the 1-day change text
  sparklineWidth = 60,  // Width of the inner sparkline chart when showChange is true
}) {
  // Transform data for chart
  const { chartData, domain, color, fillColor } = useMemo(() => {
    if (!data || !Array.isArray(data) || data.length === 0) {
      return { chartData: [], domain: [0, 1], color: '#9e9e9e', fillColor: '#9e9e9e' };
    }

    // Convert to chart format
    const chartData = data.map((value, index) => ({
      index,
      value,
    }));

    const minVal = Math.min(...data);
    const maxVal = Math.max(...data);

    // Add 5% padding to domain
    const range = maxVal - minVal || 0.01;
    const padding = range * 0.05;

    // Determine color based on trend
    // trend: 1 = up, -1 = down, 0 = flat
    const isUp = trend === 1;
    const color = isUp ? '#4caf50' : '#f44336';
    const fillColor = isUp ? 'rgba(76, 175, 80, 0.3)' : 'rgba(244, 67, 54, 0.3)';

    return {
      chartData,
      domain: [minVal - padding, maxVal + padding],
      color,
      fillColor,
    };
  }, [data, trend]);

  // Format 1-day change for display
  const changeText = useMemo(() => {
    if (change1d === null || change1d === undefined) return null;
    const sign = change1d >= 0 ? '+' : '';
    return `${sign}${change1d.toFixed(1)}%`;
  }, [change1d]);

  const changeColor = useMemo(() => {
    if (change1d === null || change1d === undefined) return 'text.secondary';
    return change1d >= 0 ? 'success.main' : 'error.main';
  }, [change1d]);

  // Tooltip content
  const tooltipText = useMemo(() => {
    const parts = [];

    // Add industry if available
    if (industry) {
      parts.push(industry);
    }

    // Add 30-day trend description
    const trendText = trend === 1 ? 'Up' : trend === -1 ? 'Down' : 'Flat';
    if (data && data.length > 0) {
      const overallChange = ((data[data.length - 1] - data[0]) / data[0]) * 100;
      parts.push(`30d: ${overallChange >= 0 ? '+' : ''}${overallChange.toFixed(1)}% (${trendText})`);
    }

    return parts.join(' | ') || 'No data';
  }, [industry, trend, data]);

  // No data - show placeholder
  if (!chartData || chartData.length === 0) {
    return (
      <Box
        sx={{
          width,
          height,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: 'text.disabled',
          fontSize: 10,
        }}
      >
        -
      </Box>
    );
  }

  return (
    <Tooltip title={tooltipText} arrow placement="top">
      <Box
        sx={{
          width,
          height,
          cursor: 'pointer',
          display: 'flex',
          alignItems: 'center',
          gap: 0.5,
        }}
      >
        {/* Sparkline Chart */}
        <Box
          sx={{
            width: showChange ? sparklineWidth : '100%',
            height: '100%',
            flex: showChange ? '1 1 auto' : 1,
            minWidth: 0,
          }}
        >
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart
              data={chartData}
              margin={{ top: 2, right: 0, left: 0, bottom: 2 }}
            >
              <YAxis domain={domain} hide />
              <Area
                type="monotone"
                dataKey="value"
                stroke={color}
                strokeWidth={1.5}
                fill={fillColor}
              />
            </AreaChart>
          </ResponsiveContainer>
        </Box>

        {/* 1-Day Change Badge */}
        {showChange && changeText && (
          <Typography
            sx={{
              fontSize: 10,
              fontWeight: 600,
              fontFamily: 'monospace',
              color: changeColor,
              whiteSpace: 'nowrap',
              flexShrink: 0,
            }}
          >
            {changeText}
          </Typography>
        )}
      </Box>
    </Tooltip>
  );
}

// Memoize component - only re-render when data or key props change
export default memo(PriceSparkline, (prevProps, nextProps) => {
  // Deep compare data arrays
  if (prevProps.data === nextProps.data) {
    return (
      prevProps.trend === nextProps.trend &&
      prevProps.change1d === nextProps.change1d &&
      prevProps.width === nextProps.width &&
      prevProps.height === nextProps.height &&
      prevProps.sparklineWidth === nextProps.sparklineWidth
    );
  }

  if (!prevProps.data || !nextProps.data) return false;
  if (prevProps.data.length !== nextProps.data.length) return false;

  // For sparkline data, compare first, last, and length (sufficient for visual comparison)
  return (
    prevProps.data[0] === nextProps.data[0] &&
    prevProps.data[prevProps.data.length - 1] === nextProps.data[nextProps.data.length - 1] &&
    prevProps.trend === nextProps.trend &&
    prevProps.change1d === nextProps.change1d &&
    prevProps.width === nextProps.width &&
    prevProps.height === nextProps.height &&
    prevProps.sparklineWidth === nextProps.sparklineWidth
  );
});
