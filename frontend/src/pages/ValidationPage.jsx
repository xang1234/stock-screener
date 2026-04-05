import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  Alert,
  Box,
  CircularProgress,
  Paper,
  Stack,
  ToggleButton,
  ToggleButtonGroup,
  Typography,
} from '@mui/material';

import { getValidationOverview } from '../api/validation';
import { ValidationSection } from '../components/Validation/ValidationPanels';

const LOOKBACK_OPTIONS = [30, 90, 180];

function ValidationPage() {
  const [sourceKind, setSourceKind] = useState('scan_pick');
  const [lookbackDays, setLookbackDays] = useState(90);

  const { data, isLoading, error } = useQuery({
    queryKey: ['validationOverview', sourceKind, lookbackDays],
    queryFn: () => getValidationOverview(sourceKind, lookbackDays),
    staleTime: 60_000,
    placeholderData: (previousData) => previousData,
  });

  if (isLoading && !data) {
    return (
      <Box display="flex" justifyContent="center" py={6}>
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return <Alert severity="error">Failed to load validation overview: {error.message}</Alert>;
  }

  return (
    <Stack spacing={2}>
      <Paper sx={{ p: 2 }}>
        <Stack spacing={1.5}>
          <Typography variant="h5" sx={{ fontWeight: 700 }}>
            Validation
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Deterministic scorecard for published scan picks and theme alerts using cached price history only.
          </Typography>
          <Stack direction={{ xs: 'column', md: 'row' }} spacing={2}>
            <Box>
              <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.5 }}>
                Source
              </Typography>
              <ToggleButtonGroup
                exclusive
                size="small"
                value={sourceKind}
                onChange={(_, value) => value && setSourceKind(value)}
              >
                <ToggleButton value="scan_pick">Scan Picks</ToggleButton>
                <ToggleButton value="theme_alert">Theme Alerts</ToggleButton>
              </ToggleButtonGroup>
            </Box>
            <Box>
              <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.5 }}>
                Lookback
              </Typography>
              <ToggleButtonGroup
                exclusive
                size="small"
                value={lookbackDays}
                onChange={(_, value) => value && setLookbackDays(value)}
              >
                {LOOKBACK_OPTIONS.map((value) => (
                  <ToggleButton key={value} value={value}>{value}D</ToggleButton>
                ))}
              </ToggleButtonGroup>
            </Box>
          </Stack>
          {isLoading && data && (
            <Box display="flex" justifyContent="center">
              <CircularProgress size={24} />
            </Box>
          )}
        </Stack>
      </Paper>

      {data && (
        <ValidationSection
          degradedReasons={data.degraded_reasons}
          horizons={data.horizons}
          recentEvents={data.recent_events}
          failureClusters={data.failure_clusters}
        />
      )}
    </Stack>
  );
}

export default ValidationPage;
