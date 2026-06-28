import { useQuery } from '@tanstack/react-query';
import { Alert, Box, Skeleton, Chip, Typography } from '@mui/material';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import TrendingDownIcon from '@mui/icons-material/TrendingDown';
import PauseCircleIcon from '@mui/icons-material/PauseCircle';
import apiClient from '../../api/client';

const fetchRegime = () => apiClient.get('/v1/market/regime').then(r => r.data);

export default function MarketRegimeBanner() {
  const { data, isLoading, error } = useQuery({
    queryKey: ['market-regime'],
    queryFn: fetchRegime,
    staleTime: 5 * 60 * 1000,
  });

  if (isLoading) return <Skeleton variant="rectangular" height={40} sx={{ borderRadius: 1, mb: 1 }} />;
  if (error) return null;
  if (!data) return null;

  const { buy_allowed, spy_phase, pct_stocks_phase2 } = data;
  const phasePct = pct_stocks_phase2 != null ? `${(pct_stocks_phase2 * 100).toFixed(0)}%` : '—';

  const severity = buy_allowed ? 'success' : spy_phase === 2 ? 'warning' : 'error';
  const Icon = buy_allowed ? TrendingUpIcon : spy_phase === 2 ? PauseCircleIcon : TrendingDownIcon;
  const message = buy_allowed
    ? `Market Regime: BUY — SPY Phase ${spy_phase}, ${phasePct} stocks in Phase 2`
    : spy_phase === 2
      ? `Market Regime: CAUTION — SPY Phase ${spy_phase} but only ${phasePct} stocks in Phase 2 (need ≥15%)`
      : `Market Regime: DEFENSIVE — SPY Phase ${spy_phase}, ${phasePct} stocks in Phase 2`;

  return (
    <Alert
      severity={severity}
      icon={<Icon fontSize="small" />}
      sx={{ mb: 1, py: 0.5, '& .MuiAlert-message': { display: 'flex', alignItems: 'center', gap: 1 } }}
    >
      <Typography variant="body2" fontWeight={600}>{message}</Typography>
      <Chip
        label={buy_allowed ? 'BUYS ON' : 'BUYS OFF'}
        color={buy_allowed ? 'success' : 'default'}
        size="small"
        sx={{ fontWeight: 700, fontSize: '0.7rem' }}
      />
    </Alert>
  );
}
