import {
  Box, Chip, Paper, Table, TableBody, TableCell, TableContainer, TableHead,
  TableRow, Typography,
} from '@mui/material';

import {
  explanationMap, formatSocialScore, socialCoverage, socialStateLabel,
} from './socialSignalPresentation';

const HEADINGS = [
  'Symbol', 'Market', 'Queue', 'Social', 'Confirmation', 'Mentions', 'Authors', 'Setup',
  'RS', 'Group', 'Theme', 'State', 'Freshness',
];

export default function SocialSignalsTable({ title, items = [], window, onSelect }) {
  if (!items.length) return null;
  return (
    <Box sx={{ mb: 2 }}>
      {title ? <Typography variant="subtitle2" sx={{ mb: 0.75 }}>{title}</Typography> : null}
      <TableContainer component={Paper} variant="outlined">
        <Table size="small" aria-label={title || 'Ranked Social Signals'}>
          <TableHead><TableRow>{HEADINGS.map((heading) => (
            <TableCell key={heading}>{heading}</TableCell>
          ))}</TableRow></TableHead>
          <TableBody>
            {items.map((item) => {
              const explanation = explanationMap(item);
              const coverage = socialCoverage(item, window);
              return (
                <TableRow
                  hover key={item.candidate_key} tabIndex={0}
                  onClick={() => onSelect?.(item)}
                  onKeyDown={(event) => event.key === 'Enter' && onSelect?.(item)}
                  sx={{ cursor: 'pointer' }}
                >
                  <TableCell><Typography fontWeight={700}>{item.canonical_symbol}</Typography></TableCell>
                  <TableCell>{item.market || 'Global'}</TableCell>
                  <TableCell>{formatSocialScore(item.queue_score)}</TableCell>
                  <TableCell>{formatSocialScore(item.social_score)}</TableCell>
                  <TableCell>{formatSocialScore(item.confirmation_score)}</TableCell>
                  <TableCell>{item.mention_count}</TableCell>
                  <TableCell>{explanation.author_count ?? '—'}</TableCell>
                  <TableCell>{explanation.readiness || formatSocialScore(explanation.setup_score)}</TableCell>
                  <TableCell>{formatSocialScore(explanation.rs_rating_3m ?? explanation.rs_rating_1m)}</TableCell>
                  <TableCell>{explanation.group_rank ?? '—'}</TableCell>
                  <TableCell>{explanation.theme || explanation.linked_theme || '—'}</TableCell>
                  <TableCell><Chip size="small" label={socialStateLabel(item.state)} /></TableCell>
                  <TableCell>
                    <Typography variant="caption" display="block">{coverage.label}</Typography>
                    <Typography variant="caption" color={`${coverage.tone}.main`}>{coverage.detail}</Typography>
                  </TableCell>
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      </TableContainer>
    </Box>
  );
}
