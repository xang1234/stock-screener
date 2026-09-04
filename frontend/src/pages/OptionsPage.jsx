import { useEffect, useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { Alert, Box, Button, CircularProgress, Typography } from '@mui/material';
import RefreshIcon from '@mui/icons-material/Refresh';
import { useNavigate } from 'react-router-dom';

import { getOptionsCommandCenter, refreshOptionsAnalytics } from '../api/optionsAnalytics';
import { getTaskStatus } from '../api/tasks';
import OptionsCommandCenterView from '../features/options/OptionsCommandCenterView';
import { optionsCommandCenterQueryKey } from '../features/options/optionsContract';

const commandKey = optionsCommandCenterQueryKey({ mode: 'live', runId: 'published' });
const refreshTaskName = 'daily-us-options-analytics';

export default function OptionsPage() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [accepted, setAccepted] = useState(null);
  const [refreshOutcome, setRefreshOutcome] = useState(null);
  const commandQuery = useQuery({
    queryKey: commandKey,
    queryFn: getOptionsCommandCenter,
    refetchInterval: accepted ? 5000 : false,
    retry: (failureCount, error) => error?.response?.status !== 404 && failureCount < 2,
  });
  const taskQuery = useQuery({
    queryKey: ['options-analytics', 'refresh-task', accepted?.taskId ?? null],
    queryFn: () => getTaskStatus(refreshTaskName, accepted.taskId),
    enabled: Boolean(accepted?.taskId),
    retry: false,
    refetchInterval: (query) => (
      ['completed', 'failed'].includes(query.state.data?.status) ? false : 2000
    ),
  });
  const refreshMutation = useMutation({
    mutationFn: () => refreshOptionsAnalytics({
      sourceRunId: null,
      force: true,
    }),
    onMutate: () => setRefreshOutcome(null),
    onSuccess: (result) => {
      setAccepted({
        taskId: result.task_id,
        baselineRunId: commandQuery.data?.run_id ?? null,
      });
    },
  });

  useEffect(() => {
    if (!accepted || !commandQuery.data || commandQuery.data.run_id === accepted.baselineRunId) return;
    setAccepted(null);
    queryClient.invalidateQueries({ queryKey: ['options-analytics', 'symbol', 'live'] });
  }, [accepted, commandQuery.data, queryClient]);

  useEffect(() => {
    if (!accepted) return;
    if (taskQuery.isError) {
      setRefreshOutcome('Could not confirm refresh status. You can try again.');
      setAccepted(null);
      return;
    }
    const task = taskQuery.data;
    if (!task) return;
    const taskStatus = String(task.status || '').toLowerCase();
    const resultStatus = String(task.result?.status || '').toLowerCase();
    if (taskStatus === 'failed') {
      setRefreshOutcome(`Refresh failed${task.error ? `: ${task.error}` : '.'}`);
      setAccepted(null);
      return;
    }
    if (taskStatus !== 'completed') return;
    if (resultStatus === 'published') {
      queryClient.invalidateQueries({ queryKey: commandKey });
      return;
    }
    const displayStatus = (resultStatus || 'without publication').replaceAll('_', ' ');
    const reasons = task.result?.reason_codes?.join(', ');
    setRefreshOutcome(
      `Refresh ended as ${displayStatus}${reasons ? ` (${reasons})` : ''}. The published snapshot was not changed.`,
    );
    setAccepted(null);
  }, [accepted, queryClient, taskQuery.data, taskQuery.isError]);

  if (commandQuery.isLoading) {
    return <Box sx={{ display: 'flex', justifyContent: 'center', p: 6 }}><CircularProgress /></Box>;
  }
  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'flex-end', mb: 1 }}>
        <Button
          size="small"
          variant="outlined"
          startIcon={<RefreshIcon />}
          aria-label="Refresh options analytics"
          disabled={refreshMutation.isPending || Boolean(accepted)}
          onClick={() => refreshMutation.mutate()}
        >
          Refresh
        </Button>
      </Box>
      {accepted && (
        <Alert severity="info" sx={{ mb: 1 }}>
          Accepted as task {accepted.taskId}. The published view will update only after the new run passes quality checks.
        </Alert>
      )}
      {refreshMutation.isError && (
        <Alert severity="error" sx={{ mb: 1 }}>Refresh could not be queued.</Alert>
      )}
      {refreshOutcome && (
        <Alert severity="error" sx={{ mb: 1 }}>{refreshOutcome}</Alert>
      )}
      {commandQuery.isError ? (
        <Alert severity={commandQuery.error?.response?.status === 404 ? 'info' : 'error'}>
          {commandQuery.error?.response?.status === 404
            ? 'No published options snapshot yet. Refresh to create one.'
            : 'Could not load the published options snapshot.'}
        </Alert>
      ) : commandQuery.data.items.length === 0 ? (
        <Typography>No current Candidates or Leaders have published options analytics.</Typography>
      ) : (
        <OptionsCommandCenterView
          data={commandQuery.data}
          onOpenSymbol={(symbol) => navigate(`/options/${encodeURIComponent(symbol)}`)}
        />
      )}
    </Box>
  );
}
