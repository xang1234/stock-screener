import { useEffect, useMemo, useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import {
  Alert, Box, Button, Chip, Collapse, Divider, FormControlLabel, MenuItem,
  Paper, Select, Stack, Switch, Table, TableBody, TableCell, TableHead,
  TableRow, TextField, Typography,
} from '@mui/material';

import {
  createSocialSource, decideSocialAssociation, getSocialAdminHealth,
  getSocialAdminRuntime, getSocialAdminSources, getSocialAnalysis,
  getSocialAssociations, getSocialCompanyIdentities, getSocialRuns,
  getSocialValidation, refreshSocialSignals, renameSocialSource,
  retrySocialAnalysis, testSocialSource, transitionSocialSource,
  updateSocialAdminRuntime, updateSocialCompanyIdentities,
} from '../../api/socialSignals';

const money = (value) => `US$${value ?? '—'}`;
const errorCode = (error) => error?.response?.data?.detail?.code
  || error?.response?.data?.detail || error?.message || 'Request failed';

export default function SocialSignalHealthPanel() {
  const queryClient = useQueryClient();
  const [keyDraft, setKeyDraft] = useState('');
  const [adminKey, setAdminKey] = useState('');
  const [keyGeneration, setKeyGeneration] = useState(0);
  const [showArchived, setShowArchived] = useState(false);
  const [newName, setNewName] = useState('');
  const [newRef, setNewRef] = useState('');
  const [renameDrafts, setRenameDrafts] = useState({});
  const [reasons, setReasons] = useState({});
  const [message, setMessage] = useState(null);
  const [identityDraft, setIdentityDraft] = useState('');
  const [validation, setValidation] = useState(null);
  const enabled = Boolean(adminKey);
  const refresh = () => queryClient.invalidateQueries({ queryKey: ['socialAdmin'] });

  const runtime = useQuery({
    queryKey: ['socialAdmin', 'runtime', keyGeneration],
    queryFn: () => getSocialAdminRuntime(adminKey), enabled,
  });
  const health = useQuery({
    queryKey: ['socialAdmin', 'health', keyGeneration],
    queryFn: () => getSocialAdminHealth(adminKey), enabled,
  });
  const sources = useQuery({
    queryKey: ['socialAdmin', 'sources', keyGeneration, showArchived],
    queryFn: () => getSocialAdminSources(adminKey, showArchived), enabled,
    refetchInterval: (query) => (query.state.data || []).some(
      (source) => ['queued', 'running'].includes(source.test_progress)
    ) ? 5_000 : 30_000,
  });
  const analysis = useQuery({
    queryKey: ['socialAdmin', 'analysis', keyGeneration],
    queryFn: () => getSocialAnalysis(adminKey), enabled,
  });
  const associations = useQuery({
    queryKey: ['socialAdmin', 'associations', keyGeneration],
    queryFn: () => getSocialAssociations(adminKey), enabled,
  });
  const identities = useQuery({
    queryKey: ['socialAdmin', 'identities', keyGeneration],
    queryFn: () => getSocialCompanyIdentities(adminKey), enabled,
  });
  const runs = useQuery({
    queryKey: ['socialAdmin', 'runs', keyGeneration],
    queryFn: () => getSocialRuns(adminKey), enabled,
  });
  const mutation = useMutation({
    mutationFn: async (work) => work(),
    onSuccess: () => { setMessage({ severity: 'success', text: 'Social administration updated.' }); refresh(); },
    onError: (error) => {
      refresh();
      setMessage({ severity: 'error', text: errorCode(error) === 'version_conflict'
        ? 'This record changed elsewhere. Reloaded the latest version.' : String(errorCode(error)) });
    },
  });
  const identityMutation = useMutation({
    mutationFn: (payload) => updateSocialCompanyIdentities(adminKey, payload),
    onSuccess: () => {
      setMessage({ severity: 'success', text: 'Social administration updated.' });
      refresh();
    },
    onError: async (error) => {
      if (errorCode(error) === 'version_conflict') {
        const latest = await identities.refetch();
        if (latest.data) {
          setIdentityDraft(JSON.stringify(latest.data.entries || [], null, 2));
        }
        setMessage({ severity: 'error', text: 'This record changed elsewhere. Reloaded the latest version.' });
        return;
      }
      setMessage({ severity: 'error', text: String(errorCode(error)) });
    },
  });
  const runtimeDraft = useMemo(() => runtime.data || { mode: 'off', provider: 'disabled' }, [runtime.data]);
  const [modeDraft, setModeDraft] = useState(null);
  const [providerDraft, setProviderDraft] = useState(null);
  useEffect(() => {
    if (identities.data) {
      setIdentityDraft((current) => current || JSON.stringify(identities.data.entries || [], null, 2));
    }
  }, [identities.data]);

  const doRefresh = async () => {
    try {
      await refreshSocialSignals(adminKey);
      setMessage({ severity: 'success', text: 'Social refresh queued.' });
      refresh();
    } catch (error) {
      if (error?.response?.status === 429) {
        setMessage({ severity: 'warning', text: `Refresh is cooling down. Try again in ${error.response.headers?.['retry-after'] || 'a few'} seconds.` });
      } else setMessage({ severity: 'error', text: String(errorCode(error)) });
    }
  };

  const loadValidation = async (runId) => {
    try { setValidation(await getSocialValidation(adminKey, runId)); }
    catch (error) { setMessage({ severity: 'error', text: String(errorCode(error)) }); }
  };

  const saveIdentities = () => {
    try {
      const entries = JSON.parse(identityDraft || '[]');
      identityMutation.mutate({
        expected_version: identities.data.registry_version, entries,
      });
    } catch {
      setMessage({ severity: 'error', text: 'Identity entries must be a valid JSON list.' });
    }
  };

  return (
    <Paper variant="outlined" sx={{ p: 2, mt: 3 }}>
      <Typography variant="h6">Social Signals administration</Typography>
      <Typography variant="body2" color="text.secondary">
        Shared provider, source, analysis budget, and published-run health. The key stays in this page only.
      </Typography>
      <Stack direction={{ xs: 'column', sm: 'row' }} gap={1} sx={{ my: 2 }}>
        <TextField size="small" type="password" label="Social admin key" value={keyDraft}
          onChange={(event) => setKeyDraft(event.target.value)} />
        <Button variant="contained" disabled={!keyDraft.trim()}
          onClick={() => {
            setIdentityDraft(''); setAdminKey(keyDraft.trim());
            setKeyGeneration((value) => value + 1); setMessage(null);
          }}>
          Load Social administration
        </Button>
      </Stack>
      {!enabled ? <Alert severity="info">Enter the admin key to load Social health and controls.</Alert> : null}
      {message ? <Alert severity={message.severity} sx={{ mb: 2 }}>{message.text}</Alert> : null}
      {enabled && [runtime, health, sources].some((query) => query.isError) ? (
        <Alert severity="error">Admin key rejected or Social administration could not be loaded.</Alert>
      ) : null}
      <Collapse in={enabled}>
        {health.data ? <HealthSummary health={health.data} /> : null}
        <Divider sx={{ my: 2 }} />
        <Typography variant="subtitle1" fontWeight={700}>Runtime</Typography>
        <Alert severity="warning" sx={{ my: 1 }}>
          Validation can read X and spend the shared LLM allowance, but cannot publish. Live can publish after every safety check passes.
        </Alert>
        <Stack direction="row" gap={1} flexWrap="wrap">
          <Select size="small" aria-label="Social mode" value={modeDraft ?? runtimeDraft.mode}
            onChange={(event) => setModeDraft(event.target.value)}>
            {['off', 'validation', 'live'].map((value) => <MenuItem key={value} value={value}>{value}</MenuItem>)}
          </Select>
          <Select size="small" aria-label="Social provider" value={providerDraft ?? runtimeDraft.provider}
            onChange={(event) => setProviderDraft(event.target.value)}>
            {['disabled', 'official', 'xui'].map((value) => <MenuItem key={value} value={value}>{value}</MenuItem>)}
          </Select>
          <Button onClick={() => mutation.mutate(() => updateSocialAdminRuntime(adminKey, {
            mode: modeDraft ?? runtimeDraft.mode, provider: providerDraft ?? runtimeDraft.provider,
            expected_version: runtimeDraft.version,
          }))}>Save runtime</Button>
          <Button variant="outlined" onClick={doRefresh}>Refresh Social Signals</Button>
        </Stack>
        <Divider sx={{ my: 2 }} />
        <Stack direction="row" justifyContent="space-between" alignItems="center">
          <Typography variant="subtitle1" fontWeight={700}>Social Sources</Typography>
          <FormControlLabel control={<Switch checked={showArchived}
            onChange={(event) => setShowArchived(event.target.checked)} />} label="Show archived" />
        </Stack>
        <Alert severity="info" sx={{ mb: 1 }}>Test List reads at most five posts. A pending list cannot be enabled until the test passes.</Alert>
        <Stack direction="row" gap={1} flexWrap="wrap" sx={{ mb: 1 }}>
          <TextField size="small" label="List name" value={newName} onChange={(event) => setNewName(event.target.value)} />
          <TextField size="small" label="List ID or URL" value={newRef} onChange={(event) => setNewRef(event.target.value)} />
          <Button disabled={!newName.trim() || !newRef.trim()} onClick={() => mutation.mutate(() =>
            createSocialSource(adminKey, { name: newName.trim(), list_ref: newRef.trim() }))}>Add pending list</Button>
        </Stack>
        <SourceTable rows={sources.data || []} renameDrafts={renameDrafts} setRenameDrafts={setRenameDrafts}
          run={(work) => mutation.mutate(work)} adminKey={adminKey} />
        <Divider sx={{ my: 2 }} />
        <Typography variant="subtitle1" fontWeight={700}>Saved analysis</Typography>
        <Typography variant="body2" color="text.secondary">
          Budget-paused work resumes after reset. Outside-window work is retained and only retried when you request it.
        </Typography>
        {(analysis.data || []).map((work) => <Stack key={work.work_id} direction="row" gap={1} alignItems="center">
          <Typography variant="body2">Work {work.work_id} · {work.state}</Typography>
          {['waiting_budget', 'failed_retryable', 'failed_terminal', 'outside_window'].includes(work.state) ? (
            <Button size="small" aria-label={`Retry work ${work.work_id}`}
              onClick={() => mutation.mutate(() => retrySocialAnalysis(adminKey, work.work_id))}>Retry</Button>
          ) : null}
        </Stack>)}
        <Divider sx={{ my: 2 }} />
        <Typography variant="subtitle1" fontWeight={700}>Theme association review</Typography>
        <Alert severity="info" sx={{ my: 1 }}>
          Social-linked Theme merges are not available in this version; ordinary legacy Theme merges are unchanged.
        </Alert>
        {(associations.data || []).map((row) => <Stack key={row.association_id} direction="row" gap={1} alignItems="center" flexWrap="wrap">
          <Typography variant="body2">{row.theme_name} · {row.market}:{row.canonical_symbol} · {row.state}</Typography>
          <TextField size="small" label={`Decision reason ${row.association_id}`}
            value={reasons[row.association_id] || ''}
            onChange={(event) => setReasons((current) => ({ ...current, [row.association_id]: event.target.value }))} />
          {['accepted', 'rejected'].map((target) => <Button key={target} size="small"
            aria-label={`${target === 'accepted' ? 'Accept' : 'Reject'} ${row.canonical_symbol}`}
            disabled={!reasons[row.association_id]?.trim()}
            onClick={() => mutation.mutate(() => decideSocialAssociation(adminKey, row.association_id, {
              target, reason: reasons[row.association_id].trim(), expected_version: row.version,
            }))}>{target}</Button>)}
        </Stack>)}
        <Divider sx={{ my: 2 }} />
        <Typography variant="subtitle1" fontWeight={700}>Verified company identities</Typography>
        <Alert severity="info" sx={{ my: 1 }}>
          You supply and verify issuer links. The LLM and this editor do not verify them; unknown companies cannot satisfy automatic thresholds.
        </Alert>
        <TextField fullWidth multiline minRows={3} label="Identity entries JSON"
          placeholder='[{"symbol":"0700","company_id":"issuer-id","verification_reference":"...","verified_at":"...+00:00"}]'
          value={identityDraft} onChange={(event) => setIdentityDraft(event.target.value)} />
        <Button sx={{ mt: 1 }} onClick={saveIdentities}>Save verified identities</Button>
        <Divider sx={{ my: 2 }} />
        <Typography variant="subtitle1" fontWeight={700}>Validation previews</Typography>
        {(runs.data || []).filter((run) => run.mode === 'validation').map((run) => (
          <Button key={run.run_id} size="small" onClick={() => loadValidation(run.run_id)}>
            Preview {run.run_id}
          </Button>
        ))}
        {validation ? <Typography variant="body2">
          {validation.candidates?.length || 0} staged candidates · {validation.associations?.length || 0} association proposals · not published
        </Typography> : null}
      </Collapse>
    </Paper>
  );
}

function HealthSummary({ health }) {
  const budget = health.budget || {};
  const reauth = (health.reason_codes || []).includes('reauthentication_required');
  return <Box sx={{ mt: 2 }}>
    <Stack direction="row" gap={0.75} flexWrap="wrap">
      <Chip label={`${health.enabled_source_count ?? 0}/${health.source_count ?? 0} enabled sources`} />
      <Chip label={`${health.participating_source_count ?? 0}/${health.enabled_source_count ?? 0} source coverage`} />
      <Chip label={health.social_fresh ? 'Fresh' : 'Stale or degraded'} color={health.social_fresh ? 'success' : 'warning'} />
      <Chip label={`${health.collection_status || 'unknown'} collection`} />
      <Chip label={`${health.processing_status || 'unknown'} processing`} />
    </Stack>
    <Typography variant="body2" sx={{ mt: 1 }}>
      Budget: {money(budget.limit_usd || '2')} per day · {money(budget.remaining_usd)} remaining · {money(budget.spent_usd)} spent · {money(budget.reserved_usd)} reserved
    </Typography>
    <Typography variant="caption" color="text.secondary">
      Reset {budget.next_reset_at ? new Date(budget.next_reset_at).toLocaleString() : '—'} · {budget.timezone || 'Asia/Singapore'} · pricing {budget.pricing_status || 'not configured'}{budget.pricing_version ? ` (${budget.pricing_version})` : ''}{budget.blocked_models?.length ? ` · blocked models: ${budget.blocked_models.join(', ')}` : ''}
    </Typography>
    <Typography variant="body2">Backlog: {health.backlog?.waiting ?? 0} waiting · {health.backlog?.failed ?? 0} failed · {health.backlog?.outside_window ?? 0} outside window</Typography>
    <Typography variant="body2">Last successful collection {health.last_collection_at ? new Date(health.last_collection_at).toLocaleString() : '—'}</Typography>
    {reauth ? <Alert severity="error" sx={{ mt: 1 }}>
      {health.provider === 'xui'
        ? 'Reauthenticate the private worker outside this app, then run Test List.'
        : 'Update the official X API credentials on the worker, then run Test List.'}
    </Alert> : null}
    {health.unknown_company_identity_count > 0 ? <Alert severity="warning" sx={{ mt: 1 }}>
      {health.unknown_company_identity_count} company links need administrator verification before they can count toward automatic Theme thresholds.
    </Alert> : null}
  </Box>;
}

function SourceTable({ rows, renameDrafts, setRenameDrafts, run, adminKey }) {
  return <Table size="small"><TableHead><TableRow>
    {['Name', 'List', 'State', 'Test', 'Last success', 'Actions'].map((value) => <TableCell key={value}>{value}</TableCell>)}
  </TableRow></TableHead><TableBody>{rows.map((source) => {
    const passed = source.test_outcome?.status === 'passed';
    const testState = source.test_progress || source.test_outcome?.status || 'untested';
    const testDetail = source.test_outcome
      ? `${source.test_outcome.provider} ${testState} · ${source.test_outcome.sample_count}/5${source.test_outcome.tested_at ? ` · ${new Date(source.test_outcome.tested_at).toLocaleString()}` : ''}`
      : testState;
    return <TableRow key={source.source_id}>
      <TableCell><TextField size="small" aria-label={`Rename ${source.name}`}
        value={renameDrafts[source.source_id] ?? source.name}
        onChange={(event) => setRenameDrafts((current) => ({ ...current, [source.source_id]: event.target.value }))} /></TableCell>
      <TableCell><Typography variant="body2">{source.list_id}</Typography><Typography variant="caption">{source.canonical_url}</Typography></TableCell>
      <TableCell><Chip size="small" label={source.lifecycle} /></TableCell>
      <TableCell>{testDetail}</TableCell>
      <TableCell>{source.collected_at ? new Date(source.collected_at).toLocaleString() : 'Never'}</TableCell>
      <TableCell><Stack direction="row" gap={0.5} flexWrap="wrap">
        <Button size="small" disabled={(renameDrafts[source.source_id] ?? source.name) === source.name}
          onClick={() => run(() => renameSocialSource(adminKey, source.source_id,
            renameDrafts[source.source_id], source.version))}>Rename</Button>
        {source.lifecycle !== 'archived' ? <Button size="small" aria-label={`Test ${source.name}`}
          onClick={() => run(() => testSocialSource(adminKey, source.source_id, source.version))}>Test</Button> : null}
        {source.lifecycle !== 'enabled' && source.lifecycle !== 'archived' && passed ? <Button size="small"
          aria-label={`Enable ${source.name}`} onClick={() => run(() => transitionSocialSource(
            adminKey, source.source_id, 'enabled', source.version))}>Enable</Button> : null}
        {source.lifecycle === 'enabled' ? <Button size="small" onClick={() => run(() => transitionSocialSource(
          adminKey, source.source_id, 'disabled', source.version))}>Disable</Button> : null}
        {source.lifecycle !== 'archived' ? <Button size="small" color="warning" onClick={() => {
          if (window.confirm(`Archive ${source.name}? Its audit history will be retained.`)) run(() => transitionSocialSource(
            adminKey, source.source_id, 'archived', source.version));
        }}>Archive</Button> : null}
      </Stack><Typography variant="caption" display="block">
        {(source.audit || []).map((event) => `${event.action} · ${event.occurred_at}`).join(' | ')}
      </Typography></TableCell>
    </TableRow>;
  })}</TableBody></Table>;
}
