import { useEffect, useMemo, useState } from 'react';
import {
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  Checkbox,
  Chip,
  Divider,
  FormControl,
  FormControlLabel,
  FormGroup,
  InputLabel,
  MenuItem,
  Select,
  Stack,
  Typography,
} from '@mui/material';

function normalizeEnabled(primaryMarket, enabledMarkets) {
  const next = enabledMarkets.includes(primaryMarket)
    ? enabledMarkets
    : [primaryMarket, ...enabledMarkets];
  return Array.from(new Set(next));
}

export default function BootstrapSetupScreen({
  primaryMarket,
  enabledMarkets,
  supportedMarkets,
  bootstrapState,
  isStartingBootstrap,
  bootstrapError,
  onStartBootstrap,
}) {
  const [selectedPrimary, setSelectedPrimary] = useState(primaryMarket || 'US');
  const [selectedMarkets, setSelectedMarkets] = useState(() => (
    normalizeEnabled(primaryMarket || 'US', enabledMarkets?.length ? enabledMarkets : ['US'])
  ));

  useEffect(() => {
    const nextPrimary = primaryMarket || 'US';
    setSelectedPrimary(nextPrimary);
    setSelectedMarkets(
      normalizeEnabled(nextPrimary, enabledMarkets?.length ? enabledMarkets : [nextPrimary])
    );
  }, [enabledMarkets, primaryMarket]);

  const normalizedSelection = useMemo(
    () => normalizeEnabled(selectedPrimary, selectedMarkets),
    [selectedMarkets, selectedPrimary]
  );

  const toggleMarket = (market) => {
    if (market === selectedPrimary) {
      return;
    }
    setSelectedMarkets((previous) => (
      previous.includes(market)
        ? previous.filter((value) => value !== market)
        : [...previous, market]
    ));
  };

  const handlePrimaryChange = (event) => {
    const nextPrimary = event.target.value;
    setSelectedPrimary(nextPrimary);
    setSelectedMarkets((previous) => normalizeEnabled(nextPrimary, previous));
  };

  const handleStart = async () => {
    await onStartBootstrap({
      primaryMarket: selectedPrimary,
      enabledMarkets: normalizedSelection,
    });
  };

  const running = bootstrapState === 'running';

  return (
    <Box
      sx={{
        minHeight: '100vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        px: 2,
        py: 4,
      }}
    >
      <Card sx={{ width: '100%', maxWidth: 720 }}>
        <CardContent sx={{ p: 4 }}>
          <Stack spacing={3}>
            <Box>
              <Chip size="small" color="primary" label="Local Setup" sx={{ mb: 1 }} />
              <Typography variant="h4" sx={{ fontWeight: 700, mb: 1 }}>
                First-run market bootstrap
              </Typography>
              <Typography color="text.secondary">
                Pick the primary market to hydrate first. Additional enabled markets will queue after
                the primary market is usable.
              </Typography>
            </Box>

            {bootstrapError && (
              <Alert severity="error">
                {typeof bootstrapError === 'string' ? bootstrapError : 'Failed to start bootstrap.'}
              </Alert>
            )}

            {running && (
              <Alert severity="info">
                Initial sync is running for {primaryMarket}. The app will switch to the main workspace
                as soon as that market has core data.
              </Alert>
            )}

            <Divider />

            <FormControl fullWidth>
              <InputLabel id="bootstrap-primary-market-label">Primary market</InputLabel>
              <Select
                labelId="bootstrap-primary-market-label"
                value={selectedPrimary}
                label="Primary market"
                onChange={handlePrimaryChange}
                disabled={running || isStartingBootstrap}
              >
                {supportedMarkets.map((market) => (
                  <MenuItem key={market} value={market}>
                    {market}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>

            <Box>
              <Typography variant="subtitle2" sx={{ mb: 1 }}>
                Enabled markets
              </Typography>
              <FormGroup>
                {supportedMarkets.map((market) => (
                  <FormControlLabel
                    key={market}
                    control={(
                      <Checkbox
                        checked={normalizedSelection.includes(market)}
                        disabled={running || isStartingBootstrap || market === selectedPrimary}
                        onChange={() => toggleMarket(market)}
                      />
                    )}
                    label={market === selectedPrimary ? `${market} (primary)` : market}
                  />
                ))}
              </FormGroup>
            </Box>

            <Divider />

            <Box>
              <Typography variant="subtitle2" sx={{ mb: 1 }}>
                Bootstrap order
              </Typography>
              <Typography color="text.secondary">
                1. Universe refresh
              </Typography>
              <Typography color="text.secondary">
                2. Benchmark and price refresh
              </Typography>
              <Typography color="text.secondary">
                3. Fundamentals refresh
              </Typography>
              <Typography color="text.secondary">
                4. Breadth and group rankings
              </Typography>
              <Typography color="text.secondary">
                5. Initial autoscan snapshot
              </Typography>
            </Box>

            <Box sx={{ display: 'flex', justifyContent: 'flex-end' }}>
              <Button
                variant="contained"
                onClick={handleStart}
                disabled={running || isStartingBootstrap}
              >
                {isStartingBootstrap ? 'Starting...' : 'Start bootstrap'}
              </Button>
            </Box>
          </Stack>
        </CardContent>
      </Card>
    </Box>
  );
}
