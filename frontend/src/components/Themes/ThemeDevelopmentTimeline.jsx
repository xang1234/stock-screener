import { useQuery } from '@tanstack/react-query';
import { Alert, Box, Chip, Link, Stack, Typography } from '@mui/material';
import { getThemeDevelopments } from '../../api/themes';

const labels = { new_event: 'New event', repeated_coverage: 'Repeated coverage', material_update: 'Substantive update', contradiction: 'Conflicting report', additional_detail: 'Additional detail', uncertain: 'Event identity uncertain' };
function safeUrl(url) {
  try { const value = new URL(url); return ['http:', 'https:'].includes(value.protocol) ? value.href : undefined; } catch { return undefined; }
}
export default function ThemeDevelopmentTimeline({ themeId }) {
  const { data, isLoading, error } = useQuery({ queryKey: ['themeDevelopments', themeId], queryFn: () => getThemeDevelopments(themeId), enabled: !!themeId });
  return <Stack spacing={1} sx={{ mb: 3 }}>
    <Typography variant="h6">Developments across posts</Typography>
    {isLoading && <Typography>Loading developments…</Typography>}
    {error && <Alert severity="error">Unable to load development history.</Alert>}
    {data && <>
      {!data.tracking_enabled && <Alert severity="info">Automatic development preparation is disabled. Existing observations remain available.</Alert>}
      <Typography>{data.event_count} distinct events · {data.material_update_count} substantive updates</Typography>
      {!!(data.work_counts?.pending || data.work_counts?.processing || data.work_counts?.retry) && <Typography variant="body2">Some sources are awaiting preparation or retry.</Typography>}
      {!!data.work_counts?.failed && <Alert severity="warning">{data.work_counts.failed} source revisions could not be prepared. Earlier observations are retained.</Alert>}
      {!data.observations.length && <Typography>No prepared developments for this theme yet.</Typography>}
      {data.observations.map((row) => <Box key={row.id} sx={{ borderLeft: 2, borderColor: 'divider', pl: 2, py: 1, opacity: row.superseded ? 0.65 : 1 }}>
        <Stack direction="row" spacing={1}><Chip size="small" label={labels[row.classification] || row.classification} /><Chip size="small" label={`Event ${row.event_id}`} />{row.superseded && <Chip size="small" label="Superseded revision" />}</Stack>
        <Typography>{row.facts.summary}</Typography>
        <Typography variant="caption">Source reports: {row.facts.status}. Published {row.published_at ? new Date(row.published_at).toLocaleString() : 'unknown'} · Prepared {new Date(row.available_at).toLocaleString()}</Typography>
        {safeUrl(row.url) && <Box><Link href={safeUrl(row.url)} target="_blank" rel="noopener noreferrer">{row.source_name || 'Source post'}</Link></Box>}
        {row.citations.map((citation, i) => <Typography key={i} component="blockquote" variant="body2" sx={{ ml: 0 }}>{citation.quote} <Typography component="span" variant="caption">({safeUrl(citation.url) ? <Link href={safeUrl(citation.url)} target="_blank" rel="noopener noreferrer">{citation.source_id === 'primary' ? 'Post evidence' : 'Attached evidence'}</Link> : citation.source_id})</Typography></Typography>)}
      </Box>)}
    </>}
  </Stack>;
}
