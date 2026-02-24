import { useMemo, useState } from 'react';
import {
  Alert,
  Box,
  Button,
  Chip,
  Dialog,
  DialogContent,
  DialogTitle,
  Divider,
  IconButton,
  Stack,
  TextField,
  Tooltip,
  Typography,
} from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import RestoreIcon from '@mui/icons-material/Restore';
import PublishIcon from '@mui/icons-material/Publish';
import PreviewIcon from '@mui/icons-material/Preview';
import SaveIcon from '@mui/icons-material/Save';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import {
  getThemePolicyConfig,
  promoteStagedThemePolicy,
  revertThemePolicy,
  updateThemePolicy,
} from '../../api/themes';

const MATCHER_FIELDS = [
  ['match_default_threshold', 'Match Default Threshold'],
  ['fuzzy_attach_threshold', 'Fuzzy Attach Threshold'],
  ['fuzzy_review_threshold', 'Fuzzy Review Threshold'],
  ['fuzzy_ambiguity_margin', 'Fuzzy Ambiguity Margin'],
  ['embedding_attach_threshold', 'Embedding Attach Threshold'],
  ['embedding_review_threshold', 'Embedding Review Threshold'],
  ['embedding_ambiguity_margin', 'Embedding Ambiguity Margin'],
];

const LIFECYCLE_FIELDS = [
  ['promotion_min_mentions_7d', 'Promotion Min Mentions (7D)'],
  ['promotion_min_source_diversity_7d', 'Promotion Min Source Diversity (7D)'],
  ['promotion_min_avg_confidence_30d', 'Promotion Min Avg Confidence (30D)'],
  ['promotion_min_persistence_days', 'Promotion Min Persistence Days'],
  ['dormancy_inactivity_days', 'Dormancy Inactivity Days'],
  ['dormancy_min_mentions_30d', 'Dormancy Min Mentions (30D)'],
  ['dormancy_min_silence_days', 'Dormancy Min Silence Days'],
  ['reactivation_min_mentions_7d', 'Reactivation Min Mentions (7D)'],
  ['reactivation_min_source_diversity_7d', 'Reactivation Min Source Diversity (7D)'],
  ['reactivation_min_avg_confidence_30d', 'Reactivation Min Avg Confidence (30D)'],
];

function toNumberOrNull(value) {
  if (value === '' || value == null) return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function ThemePolicySettingsModal({ open, onClose, pipeline = 'technical' }) {
  const queryClient = useQueryClient();
  const [adminKey, setAdminKey] = useState('');
  const [adminActor, setAdminActor] = useState('themes_ui');
  const [note, setNote] = useState('');
  const [matcherDraft, setMatcherDraft] = useState({});
  const [lifecycleDraft, setLifecycleDraft] = useState({});
  const [message, setMessage] = useState(null);
  const [messageType, setMessageType] = useState('info');

  const { data, isLoading, error } = useQuery({
    queryKey: ['themePolicyConfig', pipeline, adminKey],
    queryFn: () => getThemePolicyConfig(pipeline, adminKey || null),
    enabled: open && !!adminKey.trim(),
  });

  const effective = data?.effective || { matcher: {}, lifecycle: {} };
  const activeVersion = data?.active_version_id || 'none';

  const buildPayload = (mode) => {
    const matcher = {};
    const lifecycle = {};
    Object.entries(matcherDraft).forEach(([key, value]) => {
      const num = toNumberOrNull(value);
      if (num !== null) matcher[key] = num;
    });
    Object.entries(lifecycleDraft).forEach(([key, value]) => {
      const num = toNumberOrNull(value);
      if (num !== null) lifecycle[key] = Number.isInteger(effective.lifecycle?.[key]) ? Math.round(num) : num;
    });
    return {
      pipeline,
      mode,
      note: note || null,
      matcher,
      lifecycle,
    };
  };

  const refresh = () => {
    queryClient.invalidateQueries({ queryKey: ['themePolicyConfig', pipeline, adminKey] });
  };

  const updateMutation = useMutation({
    mutationFn: (mode) => updateThemePolicy(buildPayload(mode), adminKey || null, adminActor || null),
    onSuccess: (result) => {
      setMessageType('success');
      setMessage(`${result.status.toUpperCase()}: ${result.diff_keys?.length || 0} policy fields changed`);
      if (result.mode !== 'preview') {
        setMatcherDraft({});
        setLifecycleDraft({});
      }
      refresh();
    },
    onError: (err) => {
      setMessageType('error');
      setMessage(err?.response?.data?.detail || err.message || 'Policy update failed');
    },
  });

  const promoteMutation = useMutation({
    mutationFn: () => promoteStagedThemePolicy(pipeline, note || null, adminKey || null, adminActor || null),
    onSuccess: () => {
      setMessageType('success');
      setMessage('Staged policy promoted to active');
      refresh();
    },
    onError: (err) => {
      setMessageType('error');
      setMessage(err?.response?.data?.detail || err.message || 'Failed to promote staged policy');
    },
  });

  const revertMutation = useMutation({
    mutationFn: (versionId) => revertThemePolicy(
      { pipeline, version_id: versionId, note: note || null },
      adminKey || null,
      adminActor || null,
    ),
    onSuccess: () => {
      setMessageType('success');
      setMessage('Policy reverted successfully');
      refresh();
    },
    onError: (err) => {
      setMessageType('error');
      setMessage(err?.response?.data?.detail || err.message || 'Failed to revert policy');
    },
  });

  const historyRows = useMemo(() => data?.history || [], [data?.history]);

  return (
    <Dialog open={open} onClose={onClose} maxWidth="lg" fullWidth>
      <DialogTitle>
        <Box display="flex" justifyContent="space-between" alignItems="center">
          <Box>
            <Typography variant="h6">Theme Policy Controls</Typography>
            <Typography variant="body2" color="text.secondary">
              Safe-by-default controls with preview, staged rollout, and versioned audit history.
            </Typography>
          </Box>
          <IconButton onClick={onClose} size="small">
            <CloseIcon />
          </IconButton>
        </Box>
      </DialogTitle>
      <DialogContent>
        {!adminKey.trim() && (
          <Alert severity="info" sx={{ mb: 2 }}>
            Enter admin key to load policy settings.
          </Alert>
        )}
        {(message || error) && (
          <Alert severity={message ? messageType : 'error'} sx={{ mb: 2 }}>
            {message || error?.message || 'Failed to load policy config'}
          </Alert>
        )}

        <Stack direction={{ xs: 'column', md: 'row' }} spacing={2} sx={{ mb: 2 }}>
          <TextField
            size="small"
            label="Admin Key"
            type="password"
            value={adminKey}
            onChange={(e) => setAdminKey(e.target.value)}
            fullWidth
          />
          <TextField
            size="small"
            label="Actor"
            value={adminActor}
            onChange={(e) => setAdminActor(e.target.value)}
            fullWidth
          />
          <TextField
            size="small"
            label="Change Note"
            value={note}
            onChange={(e) => setNote(e.target.value)}
            fullWidth
          />
        </Stack>

        <Box sx={{ display: 'flex', gap: 1, mb: 2, flexWrap: 'wrap' }}>
          <Chip label={`Pipeline: ${pipeline}`} color="primary" size="small" />
          <Chip label={`Active Version: ${activeVersion}`} size="small" variant="outlined" />
          {data?.staged?.version_id && <Chip label={`Staged: ${data.staged.version_id}`} color="warning" size="small" />}
        </Box>

        <Divider sx={{ mb: 2 }} />

        {isLoading ? (
          <Typography color="text.secondary">Loading policy config...</Typography>
        ) : (
          <Stack spacing={2}>
            <Typography variant="subtitle2">Matcher Thresholds</Typography>
            <Stack direction={{ xs: 'column', md: 'row' }} spacing={1} flexWrap="wrap">
              {MATCHER_FIELDS.map(([key, label]) => (
                <TextField
                  key={key}
                  size="small"
                  type="number"
                  label={label}
                  value={matcherDraft[key] ?? ''}
                  onChange={(e) => setMatcherDraft((prev) => ({ ...prev, [key]: e.target.value }))}
                  placeholder={String(effective.matcher?.[key] ?? '')}
                  sx={{ minWidth: 230 }}
                />
              ))}
            </Stack>

            <Typography variant="subtitle2">Lifecycle Thresholds</Typography>
            <Stack direction={{ xs: 'column', md: 'row' }} spacing={1} flexWrap="wrap">
              {LIFECYCLE_FIELDS.map(([key, label]) => (
                <TextField
                  key={key}
                  size="small"
                  type="number"
                  label={label}
                  value={lifecycleDraft[key] ?? ''}
                  onChange={(e) => setLifecycleDraft((prev) => ({ ...prev, [key]: e.target.value }))}
                  placeholder={String(effective.lifecycle?.[key] ?? '')}
                  sx={{ minWidth: 250 }}
                />
              ))}
            </Stack>

            <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
              <Tooltip title="Preview changes without writing to storage">
                <span>
                  <Button
                    variant="outlined"
                    startIcon={<PreviewIcon />}
                    onClick={() => updateMutation.mutate('preview')}
                    disabled={updateMutation.isPending}
                  >
                    Preview
                  </Button>
                </span>
              </Tooltip>
              <Tooltip title="Stage changes for later promotion">
                <span>
                  <Button
                    variant="outlined"
                    color="warning"
                    startIcon={<SaveIcon />}
                    onClick={() => updateMutation.mutate('stage')}
                    disabled={updateMutation.isPending}
                  >
                    Stage
                  </Button>
                </span>
              </Tooltip>
              <Tooltip title="Apply immediately and create a versioned audit record">
                <span>
                  <Button
                    variant="contained"
                    startIcon={<PublishIcon />}
                    onClick={() => updateMutation.mutate('apply')}
                    disabled={updateMutation.isPending}
                  >
                    Apply
                  </Button>
                </span>
              </Tooltip>
              <Button
                variant="contained"
                color="warning"
                startIcon={<PublishIcon />}
                onClick={() => promoteMutation.mutate()}
                disabled={!data?.staged?.version_id || promoteMutation.isPending}
              >
                Promote Staged
              </Button>
            </Box>

            <Divider />
            <Typography variant="subtitle2">Version History (Revert)</Typography>
            <Stack spacing={1}>
              {historyRows.slice(0, 10).map((row) => (
                <Box
                  key={row.version_id}
                  sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', p: 1, border: 1, borderColor: 'divider', borderRadius: 1 }}
                >
                  <Box>
                    <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>{row.version_id}</Typography>
                    <Typography variant="caption" color="text.secondary">
                      {row.updated_at} by {row.updated_by}{row.note ? ` - ${row.note}` : ''}
                    </Typography>
                  </Box>
                  <Button
                    size="small"
                    variant="outlined"
                    color="error"
                    startIcon={<RestoreIcon />}
                    onClick={() => revertMutation.mutate(row.version_id)}
                    disabled={revertMutation.isPending}
                  >
                    Revert
                  </Button>
                </Box>
              ))}
              {!historyRows.length && (
                <Typography variant="body2" color="text.secondary">
                  No policy history yet.
                </Typography>
              )}
            </Stack>
          </Stack>
        )}
      </DialogContent>
    </Dialog>
  );
}

export default ThemePolicySettingsModal;
