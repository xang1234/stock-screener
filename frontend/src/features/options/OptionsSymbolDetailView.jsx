import {
  Accordion,
  AccordionDetails,
  AccordionSummary,
  Alert,
  Box,
  Button,
  Chip,
  Grid,
  Paper,
  Typography,
} from '@mui/material';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';

import OptionsHistoryChart from './OptionsHistoryChart';
import OptionsMetricTable from './OptionsMetricTable';
import OptionsQualityBadge from './OptionsQualityBadge';
import OptionsSourceBadges from './OptionsSourceBadges';
import OptionsStatusBanner from './OptionsStatusBanner';
import OptionsStrikeCharts from './OptionsStrikeCharts';

const humanize = (value) => String(value || '').replaceAll('_', ' ');

export default function OptionsSymbolDetailView({ data, onBack }) {
  const { item } = data;
  return (
    <Box>
      <Button startIcon={<ArrowBackIcon />} onClick={onBack} sx={{ mb: 1 }}>Back to Command Center</Button>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap', mb: 1 }}>
        <Typography variant="h4" component="h1">{item.symbol} options</Typography>
        <OptionsSourceBadges
          badges={item.source_badges}
          candidateRank={item.candidate_rank}
          leaderRank={item.leader_rank}
        />
        <OptionsQualityBadge state={item.state} reasonCodes={item.reason_codes} />
      </Box>
      <OptionsStatusBanner data={{ ...data, items: [item] }} />
      {item.warnings.length > 0 && <Alert severity="warning" sx={{ mb: 2 }}>{item.warnings.join(' · ')}</Alert>}
      <Grid container spacing={2} sx={{ mb: 2 }}>
        <Grid item xs={12} md={4}>
          <Paper variant="outlined" sx={{ p: 2, height: '100%' }}>
            <Typography variant="subtitle2">Contract snapshot</Typography>
            <Typography variant="h6">${item.spot_price?.toLocaleString()}</Typography>
            <Typography variant="body2">Expiration {item.expiration || 'Unavailable'}</Typography>
            <Typography variant="caption" color="text.secondary">
              {item.call_open_interest?.toLocaleString() ?? '—'} call OI · {item.put_open_interest?.toLocaleString() ?? '—'} put OI
            </Typography>
          </Paper>
        </Grid>
        <Grid item xs={12} md={4}>
          <Paper variant="outlined" sx={{ p: 2, height: '100%' }}>
            <Typography variant="subtitle2">Observation depth</Typography>
            <Typography>{item.lifetime_observation_count} lifetime observations</Typography>
            <Typography variant="body2">{item.short_history_observation_count} short-history observations</Typography>
            <Typography variant="body2">{item.iv_history_observation_count} IV-history observations</Typography>
          </Paper>
        </Grid>
        <Grid item xs={12} md={4}>
          <Paper variant="outlined" sx={{ p: 2, height: '100%' }}>
            <Typography variant="subtitle2">Publication quality</Typography>
            <Typography>{Math.round(data.coverage * 100)}% current coverage</Typography>
            <Typography variant="body2">Provider: {data.provider}</Typography>
            <Typography variant="body2">Retries: {item.retry_count}</Typography>
          </Paper>
        </Grid>
      </Grid>
      <OptionsStrikeCharts points={data.strike_points} spotPrice={item.spot_price} />
      <Grid container spacing={2} sx={{ mt: 0 }}>
        <Grid item xs={12} md={5}>
          <Paper variant="outlined" sx={{ mt: 2 }}>
            <Typography variant="subtitle2" sx={{ px: 2, pt: 1.5 }}>Metrics</Typography>
            <OptionsMetricTable metrics={item.metrics} />
          </Paper>
        </Grid>
        <Grid item xs={12} md={7}>
          <OptionsHistoryChart history={data.history} />
        </Grid>
      </Grid>
      <Accordion sx={{ mt: 2 }}>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography>Method and data quality</Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Typography variant="subtitle2">Assumptions</Typography>
          {Object.entries(item.assumptions).map(([key, value]) => (
            <Chip key={key} size="small" variant="outlined" label={`${humanize(key)}: ${String(value)}`} sx={{ mr: 1, mt: 1 }} />
          ))}
          <Typography variant="subtitle2" sx={{ mt: 2 }}>Quality evidence</Typography>
          {Object.entries(item.quality_evidence).map(([key, value]) => (
            <Chip
              key={key}
              size="small"
              variant="outlined"
              label={`${humanize(key)}: ${value ?? 'Unavailable'}`}
              sx={{ mr: 1, mt: 1 }}
            />
          ))}
          <Typography variant="subtitle2" sx={{ mt: 2 }}>Reasons and warnings</Typography>
          <Typography variant="body2">
            {[...item.reason_codes, ...item.warnings].map(humanize).join(' · ') || 'No additional warnings'}
          </Typography>
          <Typography variant="caption" color="text.secondary" display="block" sx={{ mt: 1 }}>
            GEX, gamma flip, and wall values are model estimates, not observed dealer positions.
          </Typography>
        </AccordionDetails>
      </Accordion>
    </Box>
  );
}
