/**
 * Market Breadth Chart with SPY Overlay
 *
 * Displays daily breadth movers (stocks up/down 4%+)
 * with S&P 500 price overlay on secondary axis
 */
import { useMemo } from 'react';
import {
  ComposedChart,
  Area,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import {
  Box,
  Card,
  CardContent,
  Typography,
  CircularProgress,
  Alert,
  ToggleButton,
  ToggleButtonGroup,
} from '@mui/material';
import { format, parseISO } from 'date-fns';

/**
 * Custom tooltip component for breadth chart
 */
const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload || !payload.length) return null;

  return (
    <Box
      sx={{
        backgroundColor: 'background.paper',
        border: '1px solid',
        borderColor: 'divider',
        borderRadius: 1,
        p: 1.5,
        boxShadow: 2,
      }}
    >
      <Typography variant="body2" sx={{ fontWeight: 600, mb: 1 }}>
        {format(parseISO(label), 'MMM dd, yyyy')}
      </Typography>
      {payload.map((entry, index) => (
        <Typography key={index} variant="body2" sx={{ color: entry.color }}>
          {entry.name}:{' '}
          {entry.name === 'SPY' ? `$${entry.value?.toFixed(2)}` : entry.value}
        </Typography>
      ))}
    </Box>
  );
};

/**
 * BreadthChart Component
 *
 * @param {Object} props
 * @param {Array} props.breadthData - Historical breadth data array
 * @param {Array} props.spyData - SPY historical price data array
 * @param {boolean} props.isLoading - Loading state
 * @param {Error} props.error - Error object if any
 * @param {string} props.timeRange - Selected time range (6M, 1Y, 2Y)
 * @param {Function} props.onTimeRangeChange - Callback when time range changes
 * @param {number} props.height - Chart height in pixels
 */
function BreadthChart({
  breadthData,
  spyData,
  isLoading,
  error,
  timeRange = '1Y',
  onTimeRangeChange,
  availableRanges = ['1M', '3M', '6M', '1Y'],
  height = 400,
  fillContainer = false,
}) {
  // Merge breadth and SPY data by date
  const chartData = useMemo(() => {
    if (!breadthData || breadthData.length === 0) return [];

    // Create map of SPY prices by date
    const spyByDate = {};
    if (spyData && spyData.length > 0) {
      spyData.forEach((d) => {
        spyByDate[d.date] = d.close;
      });
    }

    // Merge data (breadth data is primary, add SPY where available)
    return breadthData
      .map((b) => ({
        date: b.date,
        stocksUp: b.stocks_up_4pct,
        stocksDown: b.stocks_down_4pct,
        spy: spyByDate[b.date] || null,
      }))
      .sort((a, b) => new Date(a.date) - new Date(b.date));
  }, [breadthData, spyData]);

  if (isLoading) {
    return (
      <Card sx={{ mb: 2 }}>
        <CardContent
          sx={{ display: 'flex', justifyContent: 'center', py: 6 }}
        >
          <CircularProgress />
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card sx={{ mb: 2 }}>
        <CardContent>
          <Alert severity="error">
            Error loading breadth chart: {error.message}
          </Alert>
        </CardContent>
      </Card>
    );
  }

  if (!chartData || chartData.length === 0) {
    return (
      <Card sx={{ mb: 2 }}>
        <CardContent>
          <Alert severity="info">
            No breadth data available. Consider running a backfill.
          </Alert>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card sx={{ mb: fillContainer ? 0 : 2, height: fillContainer ? '100%' : 'auto', display: 'flex', flexDirection: 'column' }}>
      <CardContent sx={{ flex: 1, display: 'flex', flexDirection: 'column', p: fillContainer ? 1.5 : 2, '&:last-child': { pb: fillContainer ? 1.5 : 2 } }}>
        <Box
          sx={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            mb: 1,
          }}
        >
          <Typography variant="subtitle1" sx={{ fontWeight: 600, fontSize: fillContainer ? '14px' : '16px' }}>
            Market Breadth (4%+ Movers) with SPY Overlay
          </Typography>
          <ToggleButtonGroup
            value={timeRange}
            exclusive
            onChange={(e, newRange) => {
              if (newRange !== null) {
                onTimeRangeChange?.(newRange);
              }
            }}
            size="small"
          >
            {availableRanges.map((range) => (
              <ToggleButton key={range} value={range}>
                {range}
              </ToggleButton>
            ))}
          </ToggleButtonGroup>
        </Box>

        <Box sx={{ flex: 1, minHeight: 0 }}>
          <ResponsiveContainer width="100%" height={fillContainer ? '100%' : height}>
          <ComposedChart
            data={chartData}
            margin={{ top: 10, right: 60, left: 20, bottom: 10 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />

            <XAxis
              dataKey="date"
              tick={{ fontSize: 11 }}
              tickFormatter={(date) => format(parseISO(date), 'MMM yy')}
              interval="preserveStartEnd"
            />

            {/* Left Y-axis: Breadth counts */}
            <YAxis
              yAxisId="left"
              tick={{ fontSize: 11 }}
              tickFormatter={(value) => value}
              domain={[0, 'auto']}
              label={{
                value: 'Stock Count',
                angle: -90,
                position: 'insideLeft',
                style: { fontSize: 12 },
              }}
            />

            {/* Right Y-axis: SPY price */}
            <YAxis
              yAxisId="right"
              orientation="right"
              tick={{ fontSize: 11 }}
              tickFormatter={(value) => `$${value}`}
              domain={['auto', 'auto']}
              label={{
                value: 'SPY Price',
                angle: 90,
                position: 'insideRight',
                style: { fontSize: 12 },
              }}
            />

            <Tooltip content={<CustomTooltip />} />
            <Legend />

            {/* Stocks Up 4%+ - Green Area */}
            <Area
              yAxisId="left"
              type="monotone"
              dataKey="stocksUp"
              name="Up 4%+"
              fill="rgba(76, 175, 80, 0.3)"
              stroke="#4caf50"
              strokeWidth={1.5}
            />

            {/* Stocks Down 4%+ - Red Area */}
            <Area
              yAxisId="left"
              type="monotone"
              dataKey="stocksDown"
              name="Down 4%+"
              fill="rgba(244, 67, 54, 0.3)"
              stroke="#f44336"
              strokeWidth={1.5}
            />

            {/* SPY Price - Blue Line */}
            <Line
              yAxisId="right"
              type="monotone"
              dataKey="spy"
              name="SPY"
              stroke="#2196f3"
              strokeWidth={2}
              dot={false}
              connectNulls
            />
          </ComposedChart>
        </ResponsiveContainer>
        </Box>

        {/* Legend explanation */}
        <Box
          sx={{
            mt: 1,
            display: 'flex',
            gap: 3,
            flexWrap: 'wrap',
            justifyContent: 'center',
          }}
        >
          <Typography variant="caption" color="text.secondary">
            <span style={{ color: '#4caf50', fontWeight: 600 }}>Green:</span>{' '}
            Stocks up 4%+ (daily count)
          </Typography>
          <Typography variant="caption" color="text.secondary">
            <span style={{ color: '#f44336', fontWeight: 600 }}>Red:</span>{' '}
            Stocks down 4%+ (daily count)
          </Typography>
          <Typography variant="caption" color="text.secondary">
            <span style={{ color: '#2196f3', fontWeight: 600 }}>Blue:</span>{' '}
            S&P 500 (SPY) price
          </Typography>
        </Box>
      </CardContent>
    </Card>
  );
}

export default BreadthChart;
