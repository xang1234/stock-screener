import { useMemo, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { useNavigate } from 'react-router-dom';
import {
  Alert, Box, Button, CircularProgress, Paper, Stack, TextField,
  Typography,
} from '@mui/material';

import {
  getSocialContext, getSocialEvidence, getSocialQueue, getSocialUnresolved,
  socialQueueKey,
} from '../../api/socialSignals';
import ChartViewerModal from '../../components/Scan/ChartViewerModalLazy';
import { useMarket } from '../../contexts/MarketContext';
import SocialEvidenceDrawer from './SocialEvidenceDrawer';
import SocialSignalsTable from './SocialSignalsTable';
import { socialFreshnessLabel, visibleSocialRows } from './socialSignalPresentation';

const EMPTY = [];
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
  const [window, setWindow] = useState('7d');
  const [view, setView] = useState('actionable');
  const [rankMode, setRankMode] = useState('blended');
  const [page, setPage] = useState(1);
  const [contextPage, setContextPage] = useState(1);
  const [unresolvedPage, setUnresolvedPage] = useState(1);
  const [filters, setFilters] = useState({});
  const [selected, setSelected] = useState(null);
  const [chartOpen, setChartOpen] = useState(false);
  const controls = { market, window, view, rankMode, page, pageSize: 50, filters };
  const queueQuery = useQuery({
    queryKey: socialQueueKey(controls), queryFn: () => getSocialQueue(controls),
    staleTime: 60_000,
  });
  const contextQuery = useQuery({
    queryKey: ['socialSignals', 'context', market, window, contextPage],
    queryFn: () => getSocialContext({ market, window, page: contextPage, pageSize: 50 }),
    enabled: view === 'all',
  });
  const unresolvedQuery = useQuery({
    queryKey: ['socialSignals', 'unresolved', 'global', window, unresolvedPage],
    queryFn: () => getSocialUnresolved({ market, window, page: unresolvedPage, pageSize: 50 }),
    enabled: view === 'all',
  });
  const evidenceQuery = useQuery({
    queryKey: ['socialSignals', 'evidence', selected?.candidate_key, window],
    queryFn: () => getSocialEvidence(selected.candidate_key, window),
    enabled: Boolean(selected),
  });
  const rows = useMemo(
    () => visibleSocialRows(queueQuery.data?.items || EMPTY, filters),
    [filters, queueQuery.data?.items],
  );
  const visibleSymbols = useMemo(
    () => [...new Set(rows.filter((row) => row.market && row.canonical_symbol)
      .map((row) => row.canonical_symbol))], [rows],
  );
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
              X-list attention ranked with frozen setup, relative-strength, group, Theme, and Market confirmation.
            </Typography>
          </Box>
          <Stack direction="row" gap={0.5} alignItems="center" flexWrap="wrap">
            {['1d', '7d', '14d'].map((value) => <Button key={value}
              variant={window === value ? 'contained' : 'outlined'}
              onClick={() => selectControl(setWindow, value)}>{value.toUpperCase()}</Button>)}
            <Button variant={view === 'actionable' ? 'contained' : 'outlined'}
              onClick={() => selectControl(setView, 'actionable')}>Actionable</Button>
            <Button variant={view === 'all' ? 'contained' : 'outlined'}
              onClick={() => selectControl(setView, 'all')}>All Signals</Button>
            <Button variant={rankMode === 'blended' ? 'contained' : 'outlined'}
              onClick={() => selectControl(setRankMode, 'blended')}>Blended</Button>
            <Button variant={rankMode === 'pure_social' ? 'contained' : 'outlined'}
              onClick={() => selectControl(setRankMode, 'pure_social')}>Pure Social</Button>
          </Stack>
        </Stack>
        <Stack direction="row" gap={1} sx={{ mt: 1.25 }} flexWrap="wrap">
          {FILTERS.map(([key, label]) => (
            <TextField key={key} size="small" label={label} value={filters[key] || ''}
              onChange={(event) => setFilters((current) => ({ ...current, [key]: event.target.value }))}
              sx={{ width: 135 }} />
          ))}
        </Stack>
      </Paper>

      {queueQuery.isLoading ? <Box textAlign="center" py={8}><CircularProgress aria-label="Loading Social Signals" /></Box> : null}
      {queueQuery.isError ? <Alert severity="error">
        {queueQuery.error?.response?.status === 401 ? 'Your session expired. Sign in again.' : 'Social Signals could not be loaded.'}
      </Alert> : null}
      {queueQuery.data?.available === false ? <Alert severity="info">
        {unavailableCopy[queueQuery.data.reason_code] || 'Social Signals are temporarily unavailable.'}
      </Alert> : null}
      {queueQuery.data?.available ? (
        <>
          <Stack direction="row" gap={1} alignItems="center" sx={{ mb: 1 }}>
            <Typography variant="caption">{queueQuery.data.total} published candidates</Typography>
            <Typography variant="caption" color={queueQuery.data.stale ? 'warning.main' : 'success.main'}>
              {socialFreshnessLabel(queueQuery.data)}
            </Typography>
            <Typography variant="caption">Formula {queueQuery.data.items?.[0]?.formula_version || '—'}</Typography>
            <Button size="small" disabled={!visibleSymbols.length} onClick={scanVisible}>Send visible to Scan</Button>
          </Stack>
          {!rows.length ? <Alert severity="info">No signals match these controls.</Alert> : null}
          <SocialSignalsTable items={rows} window={window} onSelect={setSelected} />
          {view === 'all' ? <>
            <SocialSignalsTable title="Context" items={contextQuery.data?.items || EMPTY}
              window={window} onSelect={setSelected} />
            <SectionPager page={contextPage} total={contextQuery.data?.total || 0}
              onPage={setContextPage} label="Context" />
            <SocialSignalsTable title="Needs resolution" items={unresolvedQuery.data?.items || EMPTY}
              window={window} onSelect={setSelected} />
            <SectionPager page={unresolvedPage} total={unresolvedQuery.data?.total || 0}
              onPage={setUnresolvedPage} label="Needs resolution" />
          </> : null}
          <Stack direction="row" justifyContent="flex-end" gap={1}>
            <Button disabled={page === 1} onClick={() => setPage((value) => value - 1)}>Previous</Button>
            <Typography variant="caption" sx={{ alignSelf: 'center' }}>Page {page}</Typography>
            <Button disabled={page * 50 >= queueQuery.data.total}
              onClick={() => setPage((value) => value + 1)}>Next</Button>
          </Stack>
        </>
      ) : null}
      <SocialEvidenceDrawer open={Boolean(selected)} onClose={() => setSelected(null)}
        selected={selected} query={evidenceQuery} onOpenChart={() => setChartOpen(true)}
        onOpenTheme={(themeId) => navigate(`/themes?theme=${encodeURIComponent(themeId)}`)}
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
