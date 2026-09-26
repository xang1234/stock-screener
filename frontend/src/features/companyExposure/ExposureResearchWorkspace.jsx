import { useState } from 'react';
import { useMutation, useQuery } from '@tanstack/react-query';
import {
  Alert, Box, Button, MenuItem, Paper, Stack, TextField, Typography,
} from '@mui/material';

import {
  getResearchJob, getResearchJobPreview, isResearchJobSettled,
  requestExposureResearch, researchJobKey, researchPreviewKey,
} from '../../api/companyExposures';
import ExposureResearchPanel from './ExposureResearchPanel';

const errorCode = (error) => error?.response?.data?.detail?.code
  || (typeof error?.response?.data?.detail === 'string' ? error.response.data.detail : null)
  || error?.message || 'Request failed';

const newIdempotencyKey = () => `ui-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;

export default function ExposureResearchWorkspace() {
  const [keyDraft, setKeyDraft] = useState('');
  const [adminKey, setAdminKey] = useState('');
  const [symbol, setSymbol] = useState('');
  const [themeId, setThemeId] = useState('');
  const [kind, setKind] = useState('verify');
  const [jobId, setJobId] = useState(null);
  const [message, setMessage] = useState(null);

  const job = useQuery({
    queryKey: researchJobKey(jobId),
    queryFn: () => getResearchJob(adminKey, jobId),
    enabled: Boolean(adminKey && jobId),
    refetchInterval: (query) => (isResearchJobSettled(query.state.data) ? false : 5_000),
  });
  const revisionId = job.data?.assessment_revision_id;
  const preview = useQuery({
    queryKey: researchPreviewKey(jobId, revisionId),
    queryFn: () => getResearchJobPreview(adminKey, jobId),
    enabled: Boolean(adminKey && jobId && revisionId),
    staleTime: Infinity,
  });

  const request = useMutation({
    mutationFn: () => requestExposureResearch(adminKey, {
      kind,
      symbol: symbol.trim().toUpperCase(),
      economicThemeId: themeId.trim(),
      idempotencyKey: newIdempotencyKey(),
    }),
    onSuccess: (data) => {
      setJobId(data.job_id);
      setMessage({
        severity: 'success',
        text: data.dispatch === 'not_dispatched'
          ? 'Research queued; the research worker has not picked it up yet.'
          : 'Research queued.',
      });
    },
    onError: (error) => setMessage({ severity: 'error', text: errorCode(error) }),
  });

  const canSubmit = Boolean(adminKey && symbol.trim() && themeId.trim()) && !request.isPending;

  return (
    <Paper variant="outlined" sx={{ p: 2, mt: 3 }} data-testid="exposure-research-workspace">
      <Typography variant="h6">Company exposure research (shadow)</Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 1.5 }}>
        Verify one US listing against one economic theme from primary filings. Results are
        previews for review and never change theme membership.
      </Typography>
      {!adminKey ? (
        <Stack direction={{ xs: 'column', sm: 'row' }} spacing={1}>
          <TextField
            size="small"
            type="password"
            label="Admin key"
            value={keyDraft}
            onChange={(event) => setKeyDraft(event.target.value)}
            autoComplete="off"
          />
          <Button
            variant="outlined"
            disabled={!keyDraft}
            onClick={() => { setAdminKey(keyDraft); setKeyDraft(''); }}
          >
            Unlock
          </Button>
        </Stack>
      ) : (
        <Box
          component="form"
          onSubmit={(event) => { event.preventDefault(); if (canSubmit) request.mutate(); }}
        >
          <Stack direction={{ xs: 'column', md: 'row' }} spacing={1}>
            <TextField
              size="small"
              label="US symbol"
              value={symbol}
              onChange={(event) => setSymbol(event.target.value)}
            />
            <TextField
              size="small"
              label="Economic theme ID"
              value={themeId}
              onChange={(event) => setThemeId(event.target.value)}
              sx={{ minWidth: 320 }}
            />
            <TextField
              select
              size="small"
              label="Kind"
              value={kind}
              onChange={(event) => setKind(event.target.value)}
            >
              <MenuItem value="verify">Verify</MenuItem>
              <MenuItem value="refresh">Refresh</MenuItem>
            </TextField>
            <Button type="submit" variant="contained" disabled={!canSubmit}>
              Request research
            </Button>
          </Stack>
        </Box>
      )}
      {message && <Alert severity={message.severity} sx={{ mt: 1.5 }}>{message.text}</Alert>}
      {job.isError && <Alert severity="error" sx={{ mt: 1.5 }}>{errorCode(job.error)}</Alert>}
      {job.data && (
        <Box sx={{ mt: 2 }}>
          <ExposureResearchPanel job={job.data} preview={preview.data} />
        </Box>
      )}
    </Paper>
  );
}
