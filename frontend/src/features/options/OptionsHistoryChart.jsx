import { Box, Paper, Typography } from '@mui/material';
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

export default function OptionsHistoryChart({ history }) {
  return (
    <Paper variant="outlined" sx={{ p: 1.5, mt: 2 }}>
      <Typography variant="subtitle2">Observed history</Typography>
      <Typography variant="caption" color="text.secondary">
        History preserves gaps; missing sessions are not filled or treated as zero.
      </Typography>
      <Box sx={{ height: 260, mt: 1 }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={history}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="as_of_date" />
            <YAxis yAxisId="price" />
            <YAxis yAxisId="gex" orientation="right" />
            <Tooltip />
            <Line yAxisId="price" dataKey="max_pain" name="Max Pain" stroke="#ffb300" connectNulls={false} />
            <Line yAxisId="gex" dataKey="net_gex" name="Estimated Net GEX" stroke="#1976d2" connectNulls={false} />
          </LineChart>
        </ResponsiveContainer>
      </Box>
    </Paper>
  );
}
