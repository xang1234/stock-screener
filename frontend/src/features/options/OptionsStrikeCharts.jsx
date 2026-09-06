import { Box, Grid, Paper, Typography } from '@mui/material';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

function ChartFrame({ title, children }) {
  return (
    <Paper variant="outlined" sx={{ p: 1.5, height: 280 }}>
      <Typography variant="subtitle2" sx={{ mb: 1 }}>{title}</Typography>
      <Box sx={{ height: 225 }}>{children}</Box>
    </Paper>
  );
}

export default function OptionsStrikeCharts({ points, spotPrice }) {
  return (
    <Grid container spacing={2}>
      <Grid item xs={12} lg={4}>
        <ChartFrame title="Open interest and volume">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={points}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="strike" />
              <YAxis />
              <Tooltip />
              <ReferenceLine x={spotPrice} stroke="#fff" strokeDasharray="3 3" />
              <Bar dataKey="call_open_interest" name="Call OI" fill="#2e7d32" />
              <Bar dataKey="put_open_interest" name="Put OI" fill="#d32f2f" />
              <Bar dataKey="call_volume" name="Call volume" fill="#66bb6a" />
              <Bar dataKey="put_volume" name="Put volume" fill="#ef5350" />
            </BarChart>
          </ResponsiveContainer>
        </ChartFrame>
      </Grid>
      <Grid item xs={12} lg={4}>
        <ChartFrame title="Estimated GEX by strike">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={points}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="strike" />
              <YAxis />
              <Tooltip />
              <ReferenceLine y={0} stroke="#fff" />
              <Bar dataKey="estimated_call_gex" name="Estimated call GEX" fill="#1976d2" />
              <Bar dataKey="estimated_put_gex" name="Estimated put GEX" fill="#dc004e" />
            </BarChart>
          </ResponsiveContainer>
        </ChartFrame>
      </Grid>
      <Grid item xs={12} lg={4}>
        <ChartFrame title="Implied volatility smile">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={points}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="strike" />
              <YAxis tickFormatter={(value) => `${Math.round(value * 100)}%`} />
              <Tooltip formatter={(value) => `${(value * 100).toFixed(1)}%`} />
              <ReferenceLine x={spotPrice} stroke="#fff" strokeDasharray="3 3" />
              <Line dataKey="call_iv" name="Call IV" stroke="#2e7d32" connectNulls={false} />
              <Line dataKey="put_iv" name="Put IV" stroke="#d32f2f" connectNulls={false} />
            </LineChart>
          </ResponsiveContainer>
        </ChartFrame>
      </Grid>
    </Grid>
  );
}
