import { useQuery } from '@tanstack/react-query';
import { Alert, Box, Chip, CircularProgress, Paper, Stack, Typography } from '@mui/material';

import { getSocialThemePulse } from '../../api/socialSignals';
import { formatSocialScore } from './socialSignalPresentation';

const STATUS = {
  confirmed: 'Confirmed', discovering: 'Discovering',
  insufficient_market_data: 'Insufficient market data',
};

export default function SocialThemePulse({ enabled, market }) {
  const query = useQuery({
    queryKey: ['socialSignals', 'themePulse', market],
    queryFn: () => getSocialThemePulse(market),
    enabled,
    staleTime: 60_000,
  });
  if (!enabled) return null;
  return (
    <Paper variant="outlined" sx={{ p: 1.5, my: 2 }}>
      <Typography variant="h6">Social Pulse</Typography>
      <Typography variant="body2" color="text.secondary">
        Display-only evidence from the published Social run; it does not change legacy Theme ordering.
      </Typography>
      {query.isLoading ? <CircularProgress size={20} aria-label="Loading Theme Social Pulse" /> : null}
      {query.isError ? <Alert severity="warning">Theme Social Pulse could not be loaded.</Alert> : null}
      {query.data?.available === false ? <Alert severity="info" sx={{ mt: 1 }}>
        {query.data.reason_code === 'no_published_run'
          ? 'The first Social Signal run is still warming up.'
          : 'Theme Social Pulse is unavailable.'}
      </Alert> : null}
      <Stack direction="row" gap={1} flexWrap="wrap" sx={{ mt: 1 }}>
        {(query.data?.items || []).map((item) => (
          <Box key={item.theme_key} sx={{ p: 1, minWidth: 220, border: 1, borderColor: 'divider', borderRadius: 1 }}>
            <Stack direction="row" justifyContent="space-between" gap={1}>
              <Typography variant="subtitle2">{item.name}</Typography>
              <Chip size="small" label={STATUS[item.status] || item.status} />
            </Stack>
            <Typography variant="body2">Social strength {formatSocialScore(item.social_strength)}</Typography>
            <Typography variant="body2">Market strength {formatSocialScore(item.market_strength)}</Typography>
            <Typography variant="caption" color="text.secondary">
              {item.measured_company_count ?? 0}/{item.accepted_company_count ?? 0} companies
              {item.benchmark_symbol ? ` · ${item.benchmark_symbol}` : ''}
            </Typography>
          </Box>
        ))}
      </Stack>
    </Paper>
  );
}
