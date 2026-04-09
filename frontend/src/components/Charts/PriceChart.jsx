/**
 * Price Chart Component with Moving Averages
 *
 * Displays stock price history with 50-day, 150-day, and 200-day moving averages
 */
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { Card, CardContent, Typography, Box, CircularProgress } from '@mui/material';
import { useQuery } from '@tanstack/react-query';
import apiClient from '../../api/client';

/**
 * Fetch historical price data with moving averages
 */
const fetchPriceHistory = async (symbol, period = '6mo') => {
  const response = await apiClient.get(`/v1/stocks/${symbol}/history`, {
    params: { period },
  });
  return response.data;
};

/**
 * Custom tooltip for the chart
 */
const CustomTooltip = ({ active, payload, label }) => {
  if (active && payload && payload.length) {
    return (
      <Box
        sx={{
          backgroundColor: 'white',
          border: '1px solid #ccc',
          borderRadius: 1,
          p: 1.5,
        }}
      >
        <Typography variant="body2" sx={{ mb: 1 }}>
          <strong>{label}</strong>
        </Typography>
        {payload.map((entry, index) => (
          <Typography
            key={`item-${index}`}
            variant="body2"
            sx={{ color: entry.color }}
          >
            {entry.name}: ${entry.value?.toFixed(2)}
          </Typography>
        ))}
      </Box>
    );
  }
  return null;
};

/**
 * PriceChart Component
 */
const PriceChart = ({ symbol, period = '6mo', height = 400 }) => {
  const { data, isLoading, error } = useQuery({
    queryKey: ['priceHistory', symbol, period],
    queryFn: () => fetchPriceHistory(symbol, period),
    enabled: !!symbol,
  });

  if (!symbol) {
    return (
      <Card>
        <CardContent>
          <Typography color="text.secondary">
            Enter a stock symbol to view price chart
          </Typography>
        </CardContent>
      </Card>
    );
  }

  if (isLoading) {
    return (
      <Card>
        <CardContent sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
          <CircularProgress />
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card>
        <CardContent>
          <Typography color="error">
            Error loading price chart: {error.message}
          </Typography>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          {symbol} - Price Chart with Moving Averages
        </Typography>
        <ResponsiveContainer width="100%" height={height}>
          <LineChart
            data={data}
            margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis
              dataKey="date"
              tick={{ fontSize: 12 }}
              tickFormatter={(date) => {
                const d = new Date(date);
                return `${d.getMonth() + 1}/${d.getDate()}`;
              }}
            />
            <YAxis
              tick={{ fontSize: 12 }}
              tickFormatter={(value) => `$${value.toFixed(0)}`}
              domain={['auto', 'auto']}
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend />

            {/* Price line */}
            <Line
              type="monotone"
              dataKey="close"
              name="Price"
              stroke="#2196f3"
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 6 }}
            />

            {/* 50-day MA */}
            <Line
              type="monotone"
              dataKey="ma_50"
              name="50-day MA"
              stroke="#4caf50"
              strokeWidth={1.5}
              dot={false}
              strokeDasharray="5 5"
            />

            {/* 150-day MA */}
            <Line
              type="monotone"
              dataKey="ma_150"
              name="150-day MA"
              stroke="#ff9800"
              strokeWidth={1.5}
              dot={false}
              strokeDasharray="5 5"
            />

            {/* 200-day MA */}
            <Line
              type="monotone"
              dataKey="ma_200"
              name="200-day MA"
              stroke="#f44336"
              strokeWidth={1.5}
              dot={false}
              strokeDasharray="5 5"
            />
          </LineChart>
        </ResponsiveContainer>

        {/* Legend explanation */}
        <Box sx={{ mt: 2, display: 'flex', gap: 3, flexWrap: 'wrap' }}>
          <Typography variant="caption" color="text.secondary">
            <strong>Blue:</strong> Current Price
          </Typography>
          <Typography variant="caption" color="text.secondary">
            <strong>Green:</strong> 50-day MA
          </Typography>
          <Typography variant="caption" color="text.secondary">
            <strong>Orange:</strong> 150-day MA
          </Typography>
          <Typography variant="caption" color="text.secondary">
            <strong>Red:</strong> 200-day MA
          </Typography>
        </Box>
      </CardContent>
    </Card>
  );
};

export default PriceChart;
