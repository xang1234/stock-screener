import { useEffect, useMemo, useState } from 'react';
import { keepPreviousData, useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import {
  Alert, AlertTitle, Box, Button, Chip, Collapse, Divider, FormControlLabel, MenuItem,
  Paper, Select, Stack, Switch, Table, TableBody, TableCell, TableHead,
  TablePagination, TableRow, TextField, Typography,
} from '@mui/material';

import {
  createSocialSource, decideEconomicSocialAssociation, decideSocialAssociation, getSocialAdminHealth,
  getSocialAdminRuntime, getSocialAdminSources, getSocialAnalysis,
  getSocialAssociations, getSocialCompanyIdentities, getSocialRuns,
  getSocialUnresolved, getSocialValidation, refreshSocialSignals, renameSocialSource,
  retrySocialAnalysis, testSocialSource, transitionSocialSource,
  updateSocialAdminRuntime, updateSocialCompanyIdentities,
} from '../../api/socialSignals';
import { ATTEMPT_TITLES } from './socialSignalPresentation';

const UNRESOLVED_PAGE_SIZE = 50;
const money = (value) => `US$${value ?? '—'}`;
const errorCode = (error) => error?.response?.data?.detail?.code
  || error?.response?.data?.detail || error?.message || 'Request failed';

export default function SocialSignalHealthPanel() {
  const queryClient = useQueryClient();
  const [keyDraft, setKeyDraft] = useState('');
  const [adminKey, setAdminKey] = useState('');
  const [keyGeneration, setKeyGeneration] = useState(0);
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
            // A new key must not see another administrator's cached responses.
            queryClient.removeQueries({ queryKey: ['socialAdmin'] });
            setAdminKey(keyDraft.trim()); setKeyGeneration((value) => value + 1);
          }}>
          Load Social administration
        </Button>
      </Stack>
      {/* Remounting per key drops every draft and late response from the previous key. */}
      <SocialAdminBody key={keyGeneration} adminKey={adminKey} keyGeneration={keyGeneration} />
    </Paper>
  );
}

function SocialAdminBody({ adminKey, keyGeneration }) {
  const queryClient = useQueryClient();
  const [showArchived, setShowArchived] = useState(false);
  const [newName, setNewName] = useState('');
  const [newRef, setNewRef] = useState('');
  const [renameDrafts, setRenameDrafts] = useState({});
  const [reasons, setReasons] = useState({});
  const [message, setMessage] = useState(null);
  const [identityDraft, setIdentityDraft] = useState('');
  const [validation, setValidation] = useState(null);
  const [modeDraft, setModeDraft] = useState(null);
  const [providerDraft, setProviderDraft] = useState(null);
  const [unresolvedPage, setUnresolvedPage] = useState(1);
  const enabled = Boolean(adminKey);
  const refresh = () => queryClient.invalidateQueries({ queryKey: ['socialAdmin'] });

  const runtime = useQuery({
    queryKey: ['socialAdmin', 'runtime', keyGeneration],
    queryFn: () => getSocialAdminRuntime(adminKey), enabled,
  });
  const health = useQuery({
    queryKey: ['socialAdmin', 'health', keyGeneration],
    queryFn: () => getSocialAdminHealth(adminKey), enabled,
    refetchInterval: (query) => (
      ['collecting', 'processing'].includes(query.state.data?.latest_live_attempt?.status) ? 5_000 : 30_000
    ),
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
  const unresolved = useQuery({
    queryKey: ['socialAdmin', 'unresolved', keyGeneration, unresolvedPage],
    queryFn: () => getSocialUnresolved({
      window: '7d', page: unresolvedPage, pageSize: UNRESOLVED_PAGE_SIZE,
    }),
    enabled,
    placeholderData: keepPreviousData,
  });
  const unresolvedLastPage = Math.max(1, Math.ceil((unresolved.data?.total ?? 0) / UNRESOLVED_PAGE_SIZE));
  if (unresolved.isSuccess && !unresolved.isPlaceholderData && unresolvedPage > unresolvedLastPage) {
    setUnresolvedPage(unresolvedLastPage);
  }
  const mutation = useMutation({
    mutationFn: async (work) => work(),
    onSuccess: () => { setMessage({ severity: 'success', text: 'Social administration updated.' }); refresh(); },
    onError: async (error) => {
      if (errorCode(error) === 'version_conflict') {
        setModeDraft(null);
        setProviderDraft(null);
        setRenameDrafts({});
        await Promise.all([runtime.refetch(), sources.refetch()]);
        setMessage({ severity: 'error', text: 'This record changed elsewhere. Reloaded the latest version.' });
        return;
      }
      refresh();
      setMessage({ severity: 'error', text: String(errorCode(error)) });
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
    <>
      {!enabled ? <Alert severity="info">Enter the admin key to load Social health and controls.</Alert> : null}
      {message ? <Alert severity={message.severity} sx={{ mb: 2 }}>{message.text}</Alert> : null}
      {enabled && [runtime, health, sources].some((query) => query.isError) ? (
        <Alert severity="error">Admin key rejected or Social administration could not be loaded.</Alert>
      ) : null}
      <Collapse in={enabled}>
        {health.data ? <HealthSummary health={health.data} /> : null}
        {health.data?.latest_live_attempt ? <LatestAttempt attempt={health.data.latest_live_attempt} /> : null}
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
          <Button onClick={() => mutation.mutate(
            () => updateSocialAdminRuntime(adminKey, {
              mode: modeDraft ?? runtimeDraft.mode, provider: providerDraft ?? runtimeDraft.provider,
              expected_version: runtimeDraft.version,
            }),
            { onSuccess: () => queryClient.invalidateQueries({ queryKey: ['appCapabilities'] }) },
          )}>Save runtime</Button>
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
          <Typography variant="body2">
            Work {work.work_id} · {work.error_code === 'skipped:no_signal' ? 'skipped: repost, or no ticker or company' : work.state}
          </Typography>
          {['waiting_budget', 'failed_retryable', 'failed_terminal', 'outside_window'].includes(work.state) ? (
            <Button size="small" aria-label={`Retry work ${work.work_id}`}
              onClick={() => mutation.mutate(() => retrySocialAnalysis(adminKey, work.work_id))}>Retry</Button>
          ) : null}
        </Stack>)}
        <Divider sx={{ my: 2 }} />
        <Typography variant="subtitle1" fontWeight={700}>Tickers that need resolution</Typography>
        <Typography variant="body2" color="text.secondary">
          Mentioned in the last 7 days but not matched to a listing, so they are never ranked.
        </Typography>
        <UnresolvedTable query={unresolved} page={unresolvedPage} onPage={setUnresolvedPage} />
        <Divider sx={{ my: 2 }} />
        <Typography variant="subtitle1" fontWeight={700}>Theme association review</Typography>
        <Alert severity="info" sx={{ my: 1 }}>
          Social-linked Theme merges are not available in this version; ordinary legacy Theme merges are unchanged.
        </Alert>
        {(associations.data || []).map((row) => {
          // An economic row (economic authority) is decided by its economic id
          // and revision; a legacy row by its id and version (#515).
          const key = row.economic_association_id ?? row.association_id;
          const decide = (target) => (row.economic_association_id
            ? decideEconomicSocialAssociation(adminKey, row.economic_association_id, {
              target, reason: reasons[key].trim(), expected_revision: row.economic_revision,
            })
            : decideSocialAssociation(adminKey, row.association_id, {
              target, reason: reasons[key].trim(), expected_version: row.version,
            }));
          return <Stack key={key} direction="row" gap={1} alignItems="center" flexWrap="wrap">
            <Typography variant="body2">{row.theme_name} · {row.market}:{row.canonical_symbol} · {row.state}</Typography>
            <TextField size="small" label={`Decision reason ${key}`}
              value={reasons[key] || ''}
              onChange={(event) => setReasons((current) => ({ ...current, [key]: event.target.value }))} />
            {['accepted', 'rejected'].map((target) => <Button key={target} size="small"
              aria-label={`${target === 'accepted' ? 'Accept' : 'Reject'} ${row.canonical_symbol}`}
              disabled={!reasons[key]?.trim()}
              onClick={() => mutation.mutate(() => decide(target))}>{target}</Button>)}
          </Stack>;
        })}
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
    </>
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
      <Chip label={`Formula ${health.formula_version || '—'}`} />
    </Stack>
    <Typography variant="body2" sx={{ mt: 1 }}>
      Budget: {money(budget.limit_usd || '2')} per day · {money(budget.remaining_usd)} remaining · {money(budget.spent_usd)} spent · {money(budget.reserved_usd)} reserved
    </Typography>
    <Typography variant="caption" color="text.secondary">
      Reset {budget.next_reset_at ? new Date(budget.next_reset_at).toLocaleString() : '—'} · {budget.timezone || 'Asia/Singapore'} · pricing {budget.pricing_status || 'not configured'}{budget.pricing_version ? ` (${budget.pricing_version})` : ''}{budget.blocked_models?.length ? ` · blocked models: ${budget.blocked_models.join(', ')}` : ''}
    </Typography>
    <Typography variant="body2">Backlog: {health.backlog?.waiting ?? 0} waiting · {health.backlog?.failed ?? 0} failed · {health.backlog?.outside_window ?? 0} outside window · {health.backlog?.no_signal ?? 0} skipped (repost, or no ticker or company)</Typography>
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

const attemptReasonCopy = {
  daily_limit_exhausted: 'Daily X read budget used up',
  incremental_gap_suspected: 'More posts than one read covers; some older posts were not read',
  provider_error: 'Provider error',
  provider_network_error: 'Network error after one retry',
  provider_timeout: 'Provider timed out after one retry',
  provider_unavailable: 'Provider unavailable',
  rate_limited: 'Rate limited',
  reauthentication_required: 'X authentication needs attention',
};

const failureCopy = {
  analysis_failed: 'No collected post in the scoring window could be scored, so nothing was published. Later runs retry queued posts. Retry failed work under Saved analysis.',
  publication_failed: 'Analysis finished, but publishing failed. The next scheduled run collects again.',
  generation_superseded: 'A newer run published first, so this one was closed.',
  social_runtime_changed: 'The Social runtime changed during collection, so this run was closed.',
  source_participation_failed: 'An enabled list could not take part, so this run was closed.',
  not_published: 'Analysis finished, but the run was not published.',
};

function LatestAttempt({ attempt }) {
  const failed = ['collection_failed', 'failed'].includes(attempt.status);
  return (
    <Alert severity={failed ? 'error' : 'info'} sx={{ mt: 2 }}>
      <AlertTitle>{ATTEMPT_TITLES[attempt.status] || attempt.status}</AlertTitle>
      <Typography variant="body2" sx={{ mb: 1 }}>
        Live run started {new Date(attempt.started_at).toLocaleString()}
      </Typography>
      <Stack gap={0.75}>
        {(attempt.sources || []).map((source) => {
          const collected = source.read_status === 'success';
          const status = collected ? 'Collected' : source.read_status === 'failed' ? 'Failed' : 'Waiting';
          const count = source.received_count == null ? '—' : source.received_count;
          const limited = source.history_status === 'warming_up'
            || (source.reason_codes || []).includes('bounded_provider_read');
          const reasons = (source.reason_codes || [])
            .filter((reason) => reason !== 'bounded_provider_read')
            .map((reason) => attemptReasonCopy[reason] || reason.replaceAll('_', ' '));
          return (
            <Box key={source.name} sx={{ border: 1, borderColor: 'divider', borderRadius: 1, p: 1 }}>
              <Typography variant="subtitle2">{source.name}</Typography>
              <Typography variant="body2">{status} · {count} posts</Typography>
              {limited ? <Typography variant="caption" color="warning.main">Limited history</Typography> : null}
              {reasons.map((reason) => <Typography key={reason} variant="caption" display="block"
                color={source.read_status === 'failed' ? 'error.main' : 'text.secondary'}>{reason}</Typography>)}
            </Box>
          );
        })}
      </Stack>
      {attempt.status === 'published' && attempt.unanalyzed_posts > 0 ? <Typography variant="body2" sx={{ mt: 1 }}>
        {`${attempt.unanalyzed_posts} collected ${attempt.unanalyzed_posts === 1 ? 'post was' : 'posts were'} not analysed `
          + "and left out of this run's scores. Later runs retry the ones still queued."}
      </Typography> : null}
      {attempt.status === 'collection_failed' ? <Typography variant="body2" sx={{ mt: 1 }}>
        Analysis did not start because no enabled list collected successfully.
      </Typography> : null}
      {attempt.status === 'failed' ? <Typography variant="body2" sx={{ mt: 1 }}>
        {failureCopy[attempt.failure_reason] || failureCopy.not_published}
      </Typography> : null}
    </Alert>
  );
}

function UnresolvedTable({ query, page, onPage }) {
  const { data } = query;
  const note = (text) => <Typography variant="body2" sx={{ mt: 1 }}>{text}</Typography>;
  if (query.isPending) return note('Loading unresolved tickers…');
  if (query.isError) return <Alert severity="error" sx={{ mt: 1 }}>Unresolved tickers could not be loaded.</Alert>;
  if (!data.available) return note(`Unresolved tickers are unavailable (${data.reason_code}).`);
  if (!data.total) return note('No unresolved tickers in the latest run.');
  return <>
    <Table size="small" aria-label="Tickers that need resolution"><TableHead><TableRow>
      {['Ticker', 'Mentions', 'Lists', 'Latest mention'].map((value) => <TableCell key={value}>{value}</TableCell>)}
    </TableRow></TableHead><TableBody>{data.items.map((row) => (
      <TableRow key={row.candidate_key}>
        <TableCell>{row.canonical_symbol || row.candidate_key.replace(/^unresolved:/, '')}</TableCell>
        <TableCell>{row.mention_count}</TableCell>
        <TableCell>{row.observed_list_count ?? 0}/{row.enabled_list_count ?? 0}</TableCell>
        <TableCell>{row.latest_mention ? new Date(row.latest_mention).toLocaleString() : '—'}</TableCell>
      </TableRow>
    ))}</TableBody></Table>
    <TablePagination component="div" count={data.total} page={page - 1} rowsPerPage={UNRESOLVED_PAGE_SIZE}
      rowsPerPageOptions={[]} onPageChange={(_, next) => onPage(next + 1)} />
  </>;
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
