import { useState } from 'react';
import {
  Accordion,
  AccordionDetails,
  AccordionSummary,
  Alert,
  Box,
  CircularProgress,
  Paper,
  Stack,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  ToggleButton,
  ToggleButtonGroup,
  Typography,
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';

import { formatPercent } from '../../utils/formatUtils';
import { useStaticMarket } from '../StaticMarketContext';
import { resolveStaticMarketEntry, useStaticManifest, useStaticScoreboard } from '../dataClient';

const HORIZONS = [5, 20, 60];
const ORIGIN_SECTIONS = [
  { origin: 'live', title: 'Recorded live', collapsed: false },
  {
    origin: 'artifact',
    title: 'Recovered from retained build artifacts, not recorded live',
    collapsed: true,
  },
  {
    origin: 'artifact_pre_stage_fix',
    title: 'Recovered from retained build artifacts, not recorded live, from before the Stage calculation fix (#671)',
    collapsed: true,
  },
];

const MINIMUM_SAMPLE = 20;

const formatRate = (rate) => (rate == null ? '-' : `${Math.round(rate * 100)}%`);

const STAT_COLUMNS = [
  { label: 'Median', value: (horizon) => formatPercent(horizon.median_return_pct) },
  { label: 'Positive', value: (horizon) => formatRate(horizon.positive_rate) },
  { label: 'Beat benchmark', value: (horizon) => formatRate(horizon.beat_benchmark_rate) },
  { label: 'Median excess', value: (horizon) => formatPercent(horizon.median_excess_return_pct) },
];

const STICKY_SCREEN_SX = { position: 'sticky', left: 0, zIndex: 1, bgcolor: 'background.paper' };

const count = (value) => value.toLocaleString('en-US');
const plural = (value, noun) => `${count(value)} ${noun}${value === 1 ? '' : 's'}`;

const horizonOf = (cohort, sessions) => cohort.horizons.find((horizon) => horizon.horizon_sessions === sessions);

const sampleText = (cohort, horizon) => {
  if (cohort.events === 0) return 'No picks';
  return [
    `${count(horizon.sample_size)} closed`,
    horizon.sample_size > 0
      ? `${plural(horizon.signal_dates, 'date')}, effective ${horizon.effective_dates.toFixed(1)}`
      : plural(horizon.signal_dates, 'date'),
    `${count(horizon.pending_windows)} pending`,
    horizon.sample_size === 0 && horizon.oldest_open_sessions != null
      && `oldest ${horizon.oldest_open_sessions} of ${horizon.horizon_sessions} sessions`,
    horizon.skipped_missing_history > 0 && `${count(horizon.skipped_missing_history)} no price`,
    horizon.suppressed_overlapping > 0 && `${count(horizon.suppressed_overlapping)} overlapping`,
    horizon.excess_sample_size !== horizon.sample_size && `excess on ${count(horizon.excess_sample_size)}`,
  ].filter(Boolean).join(' · ');
};

const noResultNote = (cohorts, sessions) => {
  if (cohorts.some((cohort) => horizonOf(cohort, sessions).sample_size > 0)) return null;
  const closedElsewhere = cohorts.some((cohort) => cohort.horizons.some((horizon) => horizon.sample_size > 0));
  return closedElsewhere
    ? `No ${sessions}-session window has closed yet. Nothing here is a result.`
    : 'No window has closed yet. Nothing here is a result.';
};

function CohortRow({ cohort, sessions }) {
  const horizon = horizonOf(cohort, sessions);
  const toneSx = { color: horizon.thin ? 'text.secondary' : 'text.primary' };
  const captions = [
    `${cohort.first_date} to ${cohort.last_date}`,
    cohort.retired && 'Retired',
    horizon.thin && horizon.sample_size > 0 && (horizon.sample_size < MINIMUM_SAMPLE ? 'Small sample' : 'Few dates'),
  ].filter(Boolean);

  return (
    <TableRow>
      <TableCell sx={{ ...toneSx, ...STICKY_SCREEN_SX, minWidth: 120 }}>
        <Box component="span" sx={{ display: 'block', fontWeight: 600 }}>{cohort.name}</Box>
        {captions.map((caption) => (
          <Typography key={caption} variant="caption" component="span" sx={{ display: 'block', color: 'text.secondary' }}>{caption}</Typography>
        ))}
      </TableCell>
      <TableCell sx={{ ...toneSx, minWidth: 180 }}>{sampleText(cohort, horizon)}</TableCell>
      {STAT_COLUMNS.map((column) => (
        <TableCell key={column.label} align="right" sx={{ ...toneSx, fontFamily: 'monospace', whiteSpace: 'nowrap' }}>
          {column.value(horizon)}
        </TableCell>
      ))}
    </TableRow>
  );
}

const hasClosedWindow = (cohort) => cohort.horizons.some((horizon) => horizon.sample_size > 0);

const defaultView = (cohorts) => {
  const closedIn = (origin) => cohorts.some((cohort) => cohort.origin === origin && hasClosedWindow(cohort));
  const openOrigin = closedIn('live')
    ? null
    : ORIGIN_SECTIONS.find((section) => section.collapsed && closedIn(section.origin))?.origin;
  const visible = cohorts.filter((cohort) => cohort.origin === 'live' || cohort.origin === openOrigin);
  const onlyShorterClosed = visible.some(hasClosedWindow) && !visible.some((cohort) => horizonOf(cohort, 20).sample_size > 0);
  return { openOrigin, sessions: onlyShorterClosed ? 5 : 20 };
};

function OriginSection({ section, cohorts, sessions, defaultExpanded }) {
  const note = noResultNote(cohorts, sessions);
  const body = (
    <Stack spacing={1}>
      {note && <Alert severity="info">{note}</Alert>}
      {cohorts.length > 0 && (
        <TableContainer>
          <Table size="small" sx={{ '& caption': { captionSide: 'top', px: 1, pt: 0, pb: 0.5 } }}>
            <caption>{`${sessions}-session windows`}</caption>
            <TableHead>
              <TableRow>
                <TableCell sx={STICKY_SCREEN_SX}>Screen</TableCell>
                <TableCell>Sample</TableCell>
                {STAT_COLUMNS.map((column) => <TableCell key={column.label} align="right">{column.label}</TableCell>)}
              </TableRow>
            </TableHead>
            <TableBody>
              {cohorts.map((cohort) => (
                <CohortRow key={`${cohort.preset}-${cohort.definition}`} cohort={cohort} sessions={sessions} />
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      )}
    </Stack>
  );

  if (!section.collapsed) {
    return (
      <Paper component="section" elevation={0} sx={{ p: 1.5, border: '1px solid', borderColor: 'divider' }}>
        <Typography component="h2" variant="subtitle1" sx={{ fontWeight: 600, mb: 1 }}>{section.title}</Typography>
        {body}
      </Paper>
    );
  }
  if (cohorts.length === 0) return null;
  return (
    <Box component="section">
      <Accordion defaultExpanded={defaultExpanded} disableGutters elevation={0} sx={{ border: '1px solid', borderColor: 'divider' }}>
        <Box component="h2" sx={{ m: 0 }}>
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Typography component="span" variant="subtitle1" sx={{ fontWeight: 600 }}>{section.title}</Typography>
          </AccordionSummary>
        </Box>
        <AccordionDetails sx={{ p: 1.5 }}>{body}</AccordionDetails>
      </Accordion>
    </Box>
  );
}

export default function StaticScoreboardPage() {
  const manifestQuery = useStaticManifest();
  const { selectedMarket } = useStaticMarket();
  const marketEntry = resolveStaticMarketEntry(manifestQuery.data, selectedMarket);
  const scoreboardQuery = useStaticScoreboard(marketEntry);
  const [pickedSessions, setPickedSessions] = useState(null);
  const scoreboard = scoreboardQuery.data;

  if (manifestQuery.isLoading || scoreboardQuery.isLoading) {
    return <Box sx={{ display: 'flex', justifyContent: 'center', p: 6 }}><CircularProgress /></Box>;
  }
  if (scoreboardQuery.isError || !scoreboard) {
    return <Alert severity="info">The scoreboard is not available for {marketEntry.market} in this snapshot.</Alert>;
  }

  const { openOrigin, sessions: defaultSessions } = defaultView(scoreboard.cohorts);
  const sessions = pickedSessions ?? defaultSessions;

  return (
    <Stack spacing={1.5}>
      <Paper elevation={0} sx={{ p: 1.5, border: '1px solid', borderColor: 'divider' }}>
        <Typography component="h1" variant="h6">{marketEntry.display_name} scoreboard</Typography>
        <Typography variant="body2" sx={{ mb: 1 }}>How each preset screen&apos;s picks moved after they were published.</Typography>
        <Typography variant="body2" sx={{ fontWeight: 600, mb: 0.5 }}>
          {[
            `Recording since ${scoreboard.recording_since ?? '-'}`,
            scoreboard.scored_since && scoreboard.scored_since !== scoreboard.recording_since && `Scored since ${scoreboard.scored_since}`,
            `Ledger through ${scoreboard.ledger_through ?? '-'}`,
            plural(scoreboard.missing_sessions, 'missing session'),
          ].filter(Boolean).join(' · ')}
        </Typography>
        <Box component="ul" sx={{ m: 0, pl: 2.5, typography: 'body2', color: 'text.secondary' }}>
          <li>Entry is the first open after publication. Exit is the close of the Nth session.</li>
          <li>A pick is a stock newly entering the screen. Stocks on the list on the first recorded day, or on a day the screen&apos;s definition or the Stage calculation changed, are not counted.</li>
          <li>A stock that returns after its window ends counts again.</li>
          <li>{`Returns are price only, in local currency, with no costs.${scoreboard.benchmark ? ` Excess is against ${scoreboard.benchmark}.` : ''}`}</li>
          <li>Prices are as the provider reports them, so an unadjusted split can appear as a false move. Medians limit the effect.</li>
          <li>Funds that match a screen are included.</li>
          <li>A stock counts once per screen until its window ends. A repeat that enters earlier is left out and shown as overlapping.</li>
          <li>Effective dates count each date by how many picks it holds, so one busy day counts as about one.</li>
          <li>Stocks that later delist drop out as no price, which can flatter results.</li>
        </Box>
      </Paper>
      <ToggleButtonGroup
        exclusive
        size="small"
        value={sessions}
        onChange={(_, next) => next && setPickedSessions(next)}
        aria-label="Horizon"
      >
        {HORIZONS.map((horizon) => <ToggleButton key={horizon} value={horizon}>{horizon} sessions</ToggleButton>)}
      </ToggleButtonGroup>
      {ORIGIN_SECTIONS.map((section) => (
        <OriginSection
          // A market already loaded skips the spinner, and an Accordion reads defaultExpanded only on mount.
          key={`${marketEntry.market}-${section.origin}`}
          section={section}
          cohorts={scoreboard.cohorts.filter((cohort) => cohort.origin === section.origin)}
          sessions={sessions}
          defaultExpanded={section.origin === openOrigin}
        />
      ))}
    </Stack>
  );
}
