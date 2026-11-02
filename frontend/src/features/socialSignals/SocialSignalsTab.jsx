import { useMemo, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { useNavigate } from 'react-router-dom';
import {
  Alert, Box, Button, Chip, CircularProgress, Paper, Stack, TextField,
  Typography,
} from '@mui/material';

import {
  getSocialContext, getSocialEvidence, getSocialQueue, socialQueueKey,
} from '../../api/socialSignals';
import ChartViewerModal from '../../components/Scan/ChartViewerModalLazy';
import { useMarket } from '../../contexts/MarketContext';
import SocialEvidenceDrawer from './SocialEvidenceDrawer';
import { isSocialSignalMarket, SOCIAL_SIGNAL_MARKETS_COPY } from './socialSignalMarkets';
import SocialSignalsTable from './SocialSignalsTable';
import {
  ATTEMPT_TITLES, explanationMap, featureLagLabel, noActionableReason, socialFreshness,
  socialStateLabel, visibleSocialRows,
} from './socialSignalPresentation';

const EMPTY = [];
const VIEWS = [
  ['actionable', 'Actionable'], ['watch', 'Watch'], ['risk_off', 'Risk-off'], ['all', 'All Signals'],
];
const FILTERS = [
  ['source', 'Source'], ['theme', 'Theme'], ['instrument', 'Instrument'],
  ['state', 'State'], ['ticker', 'Ticker'],
];

const unavailableCopy = {
  no_published_run: 'The first Social Signal run is still warming up. Saved results will appear after publication.',
  social_signals_disabled: 'Social Signals are not enabled for this installation.',
  reauthentication_required: 'X access needs attention. An administrator can reauthenticate the provider in Operations.',
};

export default function SocialSignalsTab() {
  const navigate = useNavigate();
  const { selectedMarket } = useMarket();
  const market = selectedMarket || 'US';
  const marketSupported = isSocialSignalMarket(market);
  const [window, setWindow] = useState('7d');
  const [view, setView] = useState('actionable');
  const [rankMode, setRankMode] = useState('blended');
  const [page, setPage] = useState(1);
  const [contextPage, setContextPage] = useState(1);
  const [filters, setFilters] = useState({});
  const [selected, setSelected] = useState(null);
  const [chartOpen, setChartOpen] = useState(false);
  const controls = { market, window, view, rankMode, page, pageSize: 50, filters };
  const queueQuery = useQuery({
    queryKey: socialQueueKey(controls), queryFn: () => getSocialQueue(controls),
    staleTime: 60_000, enabled: marketSupported,
  });
  const actionableEmpty = view === 'actionable' && queueQuery.data?.available === true
    && queueQuery.data.total === 0;
  const watchControls = { ...controls, view: 'watch' };
  const watchQuery = useQuery({
    queryKey: socialQueueKey(watchControls), queryFn: () => getSocialQueue(watchControls),
    staleTime: 60_000, enabled: marketSupported && actionableEmpty,
  });
  const fallback = actionableEmpty && watchQuery.data?.total > 0;
  const shown = fallback ? watchQuery.data : queueQuery.data;
  const contextQuery = useQuery({
    queryKey: ['socialSignals', 'context', market, window, contextPage],
    queryFn: () => getSocialContext({ market, window, page: contextPage, pageSize: 50 }),
    enabled: marketSupported && view === 'all',
  });
  const evidenceQuery = useQuery({
    queryKey: ['socialSignals', 'evidence', selected?.candidate_key, window],
    queryFn: () => getSocialEvidence(selected.candidate_key, window),
    enabled: marketSupported && Boolean(selected),
  });
  const rows = useMemo(
    () => visibleSocialRows(shown?.items || EMPTY, filters),
    [filters, shown?.items],
  );
  const visibleSymbols = useMemo(
    () => [...new Set(rows.filter((row) => row.market && row.canonical_symbol)
      .map((row) => row.canonical_symbol))], [rows],
  );
  const attempt = queueQuery.data?.latest_attempt;
  const freshness = socialFreshness(queueQuery.data);
  const featureLag = rows.map((row) => featureLagLabel(explanationMap(row))).find(Boolean);
  const filtered = Object.values(filters).some((value) => String(value || '').trim());
  const emptyCopy = filtered ? 'No signals match these filters.'
    : view === 'all' ? 'No signals in this window.'
      : `No ${socialStateLabel(view)} signals in this window.`;
  const selectControl = (setter, value) => { setter(value); setPage(1); };
  const scanVisible = () => {
    const params = new URLSearchParams({ market, symbols: visibleSymbols.join(',') });
    navigate({ pathname: '/scan', search: `?${params.toString()}` });
  };

  return (
    <Box sx={{ height: '100%', overflow: 'auto', p: 1 }}>
      <Paper variant="outlined" sx={{ p: 1.5, mb: 1.5 }}>
        <Stack direction={{ xs: 'column', lg: 'row' }} justifyContent="space-between" gap={1}>
          <Box>
            <Typography variant="h6">Social Signal Queue</Typography>
            <Typography variant="body2" color="text.secondary">
              Stocks getting attention on tracked X lists, ranked by how well their setup, strength, theme, and the market back it up.
            </Typography>
          </Box>
          <Stack direction="row" gap={0.5} alignItems="center" flexWrap="wrap">
            {['1d', '7d', '14d'].map((value) => <Button key={value}
              variant={window === value ? 'contained' : 'outlined'}
              onClick={() => selectControl(setWindow, value)}>{value.toUpperCase()}</Button>)}
            {VIEWS.map(([value, label]) => <Button key={value}
              variant={view === value ? 'contained' : 'outlined'}
              onClick={() => selectControl(setView, value)}>{label}</Button>)}
            <Button variant={rankMode === 'blended' ? 'contained' : 'outlined'}
              onClick={() => selectControl(setRankMode, 'blended')}>Blended</Button>
            <Button variant={rankMode === 'pure_social' ? 'contained' : 'outlined'}
              onClick={() => selectControl(setRankMode, 'pure_social')}>Pure Social</Button>
          </Stack>
        </Stack>
        <Stack direction="row" gap={1} sx={{ mt: 1.25 }} flexWrap="wrap">
          {FILTERS.map(([key, label]) => (
            <TextField key={key} size="small" label={label} value={filters[key] || ''}
              onChange={(event) => {
                setFilters((current) => ({ ...current, [key]: event.target.value }));
                setPage(1);
              }}
              sx={{ width: 135 }} />
          ))}
        </Stack>
      </Paper>

      {!marketSupported ? <Alert severity="info">
        {SOCIAL_SIGNAL_MARKETS_COPY}
      </Alert> : null}
      {marketSupported && queueQuery.isLoading ? <Box textAlign="center" py={8}><CircularProgress aria-label="Loading Social Signals" /></Box> : null}
      {queueQuery.isError ? <Alert severity="error">
        {queueQuery.error?.response?.status === 401 ? 'Your session expired. Sign in again.' : 'Social Signals could not be loaded.'}
      </Alert> : null}
      {queueQuery.data?.available === false ? (
        <Alert severity={['collection_failed', 'failed'].includes(attempt?.status) ? 'error' : 'info'}
          action={attempt ? <Button size="small" onClick={() => navigate('/operations#sources')}>
            Open Operations</Button> : null}>
          {attempt
            ? `${ATTEMPT_TITLES[attempt.status] || 'Collection is in progress'}.`
            : unavailableCopy[queueQuery.data.reason_code] || 'Social Signals are temporarily unavailable.'}
        </Alert>
      ) : null}
      {queueQuery.data?.available ? (
        <>
          <Stack direction="row" gap={1} alignItems="center" sx={{ mb: 1 }}>
            <Typography variant="caption">{shown.total} published candidates</Typography>
            <Chip size="small" variant="outlined" color={freshness.color} label={freshness.label} />
            {featureLag ? <Chip size="small" variant="outlined" color="warning" label={featureLag} /> : null}
            <Button size="small" disabled={!visibleSymbols.length} onClick={scanVisible}>Send visible to Scan</Button>
          </Stack>
          {fallback ? <Alert severity="info" sx={{ mb: 1 }}>
            {noActionableReason(rows)} Showing Watch.
          </Alert> : null}
          {!rows.length && !(actionableEmpty && watchQuery.isLoading) ? <Alert severity="info">{emptyCopy}</Alert> : null}
          <SocialSignalsTable items={rows} onSelect={setSelected} />
          {view === 'all' ? <>
            <SocialSignalsTable title="Context" items={contextQuery.data?.items || EMPTY}
              onSelect={setSelected} />
            <SectionPager page={contextPage} total={contextQuery.data?.total || 0}
              onPage={setContextPage} label="Context" />
          </> : null}
          <Stack direction="row" justifyContent="flex-end" gap={1}>
            <Button disabled={page === 1} onClick={() => setPage((value) => value - 1)}>Previous</Button>
            <Typography variant="caption" sx={{ alignSelf: 'center' }}>Page {page}</Typography>
            <Button disabled={page * 50 >= shown.total}
              onClick={() => setPage((value) => value + 1)}>Next</Button>
          </Stack>
        </>
      ) : null}
      <SocialEvidenceDrawer open={Boolean(selected)} onClose={() => setSelected(null)}
        selected={selected} query={evidenceQuery} onOpenChart={() => setChartOpen(true)}
        onScan={scanVisible} />
      <ChartViewerModal open={chartOpen} onClose={() => setChartOpen(false)}
        initialSymbol={selected?.canonical_symbol} scanId={null}
        navigationSymbolsOverride={visibleSymbols}
        currentPageResults={rows.map((row) => ({ ...row, symbol: row.canonical_symbol }))} />
    </Box>
  );
}

function SectionPager({ page, total, onPage, label }) {
  if (total <= 50 && page === 1) return null;
  return (
    <Stack direction="row" justifyContent="flex-end" gap={1} sx={{ mt: -1, mb: 2 }}>
      <Button size="small" disabled={page === 1} onClick={() => onPage(page - 1)}>
        Previous {label}
      </Button>
      <Typography variant="caption" sx={{ alignSelf: 'center' }}>Page {page}</Typography>
      <Button size="small" disabled={page * 50 >= total} onClick={() => onPage(page + 1)}>
        Next {label}
      </Button>
    </Stack>
  );
}
