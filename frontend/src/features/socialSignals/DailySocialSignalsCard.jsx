import { useQuery } from '@tanstack/react-query';
import {
  Alert, Box, Button, Chip, CircularProgress, Paper, Stack, Typography,
} from '@mui/material';

import { getSocialSummary } from '../../api/socialSignals';
import { formatSocialScore } from './socialSignalPresentation';

const unavailableCopy = {
  no_published_run: 'The first Social Signal run is still warming up.',
  social_signals_disabled: 'Social Signals are not enabled.',
};

const themeLabel = (value) => String(value || '').replaceAll('_', ' ')
  .replace(/^./, (letter) => letter.toUpperCase());

export default function DailySocialSignalsCard({ market, onOpen }) {
  const query = useQuery({
    queryKey: ['socialSignals', 'summary', market],
    queryFn: () => getSocialSummary(market),
    staleTime: 60_000,
  });
  const data = query.data;
  const signals = (data?.top_signals || []).slice(0, 5);
  const exposure = signals.find((item) => item.explanation?.market_exposure != null)
    ?.explanation.market_exposure;

  return (
    <Paper variant="outlined" sx={{ p: 1.5, mb: 2 }}>
      <Stack direction="row" justifyContent="space-between" alignItems="center" gap={1}>
        <Box>
          <Typography variant="subtitle1" fontWeight={700}>Social Signals</Typography>
          <Typography variant="caption" color="text.secondary">Published X-list attention for {market}</Typography>
        </Box>
        <Button size="small" onClick={onOpen}>Open Social Signals</Button>
      </Stack>
      {query.isLoading ? <CircularProgress size={20} aria-label="Loading Social summary" /> : null}
      {query.isError ? <Alert severity="warning">Social summary could not be loaded.</Alert> : null}
      {data?.available === false ? (
        <Alert severity="info" sx={{ mt: 1 }}>
          {unavailableCopy[data.reason_code] || 'Social Signals are temporarily unavailable.'}
        </Alert>
      ) : null}
      {data?.available ? (
        <>
          <Stack direction="row" gap={0.75} flexWrap="wrap" sx={{ my: 1 }}>
            <Chip size="small" color={data.stale ? 'warning' : 'success'} label={data.stale ? 'Stale' : 'Fresh'} />
            <Chip size="small" label={`${data.participating_source_count ?? 0}/${data.enabled_source_count ?? 0} sources`} />
            <Chip size="small" label={`Exposure ${formatSocialScore(exposure)} · ${exposure == null
              ? 'Unavailable' : Number(exposure) >= 50 ? 'Risk-on' : 'Risk-off'}`} />
            <Typography variant="caption" sx={{ alignSelf: 'center' }}>
              Last success {data.published_at ? new Date(data.published_at).toLocaleString() : '—'}
            </Typography>
          </Stack>
          <Stack direction="row" gap={1.5} flexWrap="wrap">
            {signals.map((item) => (
              <Typography key={item.candidate_key} variant="body2">
                <strong>{item.canonical_symbol}</strong> {formatSocialScore(item.queue_score)}
              </Typography>
            ))}
          </Stack>
          <Typography variant="caption" color="text.secondary" display="block" sx={{ mt: 1 }}>
            {(data.dominant_themes || []).slice(0, 3).map((theme) => (
              `${themeLabel(theme.theme_key)} · ${theme.accepted_company_count} companies`
            )).join(' · ') || 'No dominant theme yet'}
          </Typography>
        </>
      ) : null}
    </Paper>
  );
}
