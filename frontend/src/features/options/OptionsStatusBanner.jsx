import { Alert, Box, Chip, Typography } from '@mui/material';

const dateFormatter = new Intl.DateTimeFormat('en-US', {
  month: 'short',
  day: 'numeric',
  year: 'numeric',
  timeZone: 'UTC',
});

const timeFormatter = new Intl.DateTimeFormat('en-US', {
  month: 'short',
  day: 'numeric',
  year: 'numeric',
  hour: 'numeric',
  minute: '2-digit',
  timeZone: 'UTC',
  timeZoneName: 'short',
});

const formatDate = (value) => (value ? dateFormatter.format(new Date(`${value}T00:00:00Z`)) : 'Unknown');
const formatTime = (value) => (value ? timeFormatter.format(new Date(value)) : 'Unavailable');

export default function OptionsStatusBanner({ data }) {
  const buildingCount = data.items?.filter((item) => item.state === 'building_history').length || 0;
  return (
    <Box sx={{ mb: 2 }}>
      {data.stale && (
        <Alert severity="warning" sx={{ mb: 1.5 }}>
          Showing the previous published options snapshot. Equity data may be newer; use the options observation time below.
        </Alert>
      )}
      <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap', alignItems: 'center' }}>
        <Typography variant="body2">
          Equity source: {formatDate(data.source_as_of_date)} · run {data.source_feature_run_id ?? 'unknown'}
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Options observed: {formatTime(data.latest_observation_at)}
        </Typography>
        <Typography variant="body2" color="text.secondary">
          {data.provider === 'yahoo' ? 'Yahoo' : data.provider} · {data.calculation_version} · {Math.round(data.coverage * 100)}% coverage
        </Typography>
        {buildingCount > 0 && (
          <Chip size="small" color="info" variant="outlined" label={`${buildingCount} building history`} />
        )}
      </Box>
    </Box>
  );
}
