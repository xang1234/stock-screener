import { useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { Alert, Autocomplete, Box, Button, MenuItem, Stack, TextField, Typography } from '@mui/material';
import { searchThemeEquivalence, previewThemeEquivalence, applyThemeEquivalence, getThemeEquivalenceHistory, undoThemeEquivalence } from '../../api/themes';

export default function ThemeEquivalencePanel() {
  const client = useQueryClient();
  const [pipeline, setPipeline] = useState('technical');
  const [source, setSource] = useState(null);
  const [target, setTarget] = useState(null);
  const [search, setSearch] = useState('');
  const [actor, setActor] = useState('');
  const [reason, setReason] = useState('');
  const [preview, setPreview] = useState(null);
  const [message, setMessage] = useState('');
  const choices = useQuery({ queryKey: ['themeEquivalenceSearch', pipeline, search], queryFn: () => searchThemeEquivalence(pipeline, search) });
  const history = useQuery({ queryKey: ['themeEquivalenceHistory', pipeline], queryFn: () => getThemeEquivalenceHistory(pipeline) });
  const mutation = useMutation({
    mutationFn: async ({ action, id }) => {
      if (action === 'preview') return { preview: await previewThemeEquivalence(source.id, target.id) };
      const body = { actor: actor.trim(), reason: reason.trim() };
      if (action === 'undo') return undoThemeEquivalence(id, body);
      return applyThemeEquivalence({ ...body, source_id: source.id, target_id: target.id,
        expected_version: preview.version, operation_key: preview.operationKey });
    },
    onSuccess: (data) => {
      if (data.preview) { setPreview({ ...data.preview, operationKey: crypto.randomUUID?.() || Array.from(crypto.getRandomValues(new Uint8Array(16)), (v) => v.toString(16).padStart(2, '0')).join('') }); return; }
      setPreview(null); setSource(null); setTarget(null);
      setMessage(data.refresh_status === 'complete' ? 'Grouping saved. Current results refreshed.' : 'Grouping saved. Current results are awaiting a refresh.');
      client.invalidateQueries();
    },
  });
  const error = mutation.error || choices.error || history.error;
  const canReview = actor.trim() && reason.trim() && !mutation.isPending;
  return <Stack spacing={2} sx={{ p: 2 }}>
    <Alert severity="info">Group only equivalent investment exposures. Keep broader and narrower themes, such as Memory and HBM, separate. Original names and evidence are preserved; grouping can be undone.</Alert>
    {error && <Alert severity="error">{error.response?.data?.detail || error.message || 'Unable to load grouping'}</Alert>}
    {message && <Alert severity="success">{message}</Alert>}
    <TextField disabled={mutation.isPending} select label="Pipeline" value={pipeline} onChange={(e) => { setPipeline(e.target.value); setSource(null); setTarget(null); setPreview(null); }}>
      <MenuItem value="technical">Technical</MenuItem><MenuItem value="fundamental">Fundamental</MenuItem>
    </TextField>
    {[['Theme to group', source, setSource], ['Display under theme', target, setTarget]].map(([label, value, setter]) =>
      <Autocomplete disabled={mutation.isPending} filterOptions={(options) => options} key={label} options={choices.data?.themes || []} value={value} loading={choices.isLoading}
        getOptionLabel={(option) => option.name} isOptionEqualToValue={(a, b) => a.id === b.id}
        onInputChange={(_, text, why) => { if (why === 'input') setSearch(text); }}
        onChange={(_, next) => { setter(next); setPreview(null); }}
        renderInput={(params) => <TextField {...params} label={label} />} />)}
    <Button disabled={!source || !target || source.id === target.id || mutation.isPending} onClick={() => mutation.mutate({ action: 'preview' })}>Preview grouping</Button>
    {preview && <Alert severity="info">{preview.aliases.map((a) => a.name).join(' + ')}: {preview.parent_posts} distinct source posts. Display under {target?.name}.</Alert>}
    <TextField label="Reviewer" value={actor} onChange={(e) => setActor(e.target.value)} inputProps={{ maxLength: 120 }} />
    <TextField label="Reason for grouping or undo" value={reason} onChange={(e) => setReason(e.target.value)} multiline inputProps={{ maxLength: 2000 }} />
    <Button variant="contained" disabled={!preview || !canReview} onClick={() => mutation.mutate({ action: 'apply' })}>Apply reviewed grouping</Button>
    <Typography variant="h6">Grouping history</Typography>
    {!history.isLoading && !history.data?.operations.length && <Typography>No reviewed groups yet.</Typography>}
    {history.data?.operations.map((op) => <Box key={op.id} sx={{ borderTop: 1, borderColor: 'divider', pt: 1 }}>
      <Typography>{op.aliases.map((a) => a.name).join(' + ')} — {op.active ? 'Active' : 'Undone'}</Typography>
      <Typography variant="body2">{op.actor}: {op.reason}</Typography>
      {op.refresh_pending && <Typography variant="body2">Current rankings are awaiting refresh.</Typography>}
      {!op.active && <Typography variant="body2">Undo: {op.undo_reason}</Typography>}
      {op.active && <Button disabled={!canReview} onClick={() => mutation.mutate({ action: 'undo', id: op.id })}>Undo grouping {op.id}</Button>}
    </Box>)}
  </Stack>;
}
