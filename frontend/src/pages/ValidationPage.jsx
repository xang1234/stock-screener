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
import { useMarket } from '../contexts/MarketContext';

const LOOKBACK_OPTIONS = [30, 90, 180];

function ValidationPage() {
  const [sourceKind, setSourceKind] = useState('scan_pick');
  const [lookbackDays, setLookbackDays] = useState(90);
  const { selectedMarket } = useMarket();
  const market = selectedMarket || 'US';

  const { data, isLoading, error } = useQuery({
    queryKey: ['validationOverview', sourceKind, lookbackDays, market],
    queryFn: () => getValidationOverview(sourceKind, lookbackDays, market),
    staleTime: 60_000,
    // Keep the page while a toggle refetches, never across a market switch.
    placeholderData: (previousData) => (previousData?.market === market ? previousData : undefined),
  });
  // Placeholder data from another source must not be shown under this source's label.
  const overview = data?.source_kind === sourceKind ? data : null;

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
            Backtest
          </Typography>
          <Typography variant="body2" color="text.secondary">
            Deterministic scorecard for published scan picks, theme alerts, and replayed Minervini picks using cached price history only.
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
                <ToggleButton value="replay_pick">Replay (Minervini)</ToggleButton>
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
          {sourceKind === 'replay_pick' && (
            <Alert severity="info">
              Price-only Minervini replay: the top 10 per session among stocks passing the template.
              Survivor-biased (today&apos;s universe) and not the live composite score.
              {overview && (
                <>
                  {' '}
                  {overview.freshness?.latest_replay_as_of_date
                    ? `Replayed through ${overview.freshness.latest_replay_as_of_date}.`
                    : 'No replay picks are stored for this market yet.'}
                </>
              )}
            </Alert>
          )}
          {((isLoading && data) || (data && !overview)) && (
            <Box display="flex" justifyContent="center">
              <CircularProgress size={24} />
            </Box>
          )}
        </Stack>
      </Paper>

      {overview && (
        <ValidationSection
          degradedReasons={overview.degraded_reasons}
          horizons={overview.horizons}
          recentEvents={overview.recent_events}
          groups={overview.groups}
          benchmarkSymbol={overview.benchmark_symbol}
          stopLossPct={overview.stop_loss_pct}
        />
      )}
    </Stack>
  );
}

export default ValidationPage;
