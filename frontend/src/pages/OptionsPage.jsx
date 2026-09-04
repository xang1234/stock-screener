import { useEffect, useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { Alert, Box, Button, CircularProgress, Typography } from '@mui/material';
import RefreshIcon from '@mui/icons-material/Refresh';
import { useNavigate } from 'react-router-dom';

import { getOptionsCommandCenter, refreshOptionsAnalytics } from '../api/optionsAnalytics';
import OptionsCommandCenterView from '../features/options/OptionsCommandCenterView';
import { optionsCommandCenterQueryKey } from '../features/options/optionsContract';

const commandKey = optionsCommandCenterQueryKey({ mode: 'live', runId: 'published' });

export default function OptionsPage() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [accepted, setAccepted] = useState(null);
  const commandQuery = useQuery({
    queryKey: commandKey,
    queryFn: getOptionsCommandCenter,
    refetchInterval: accepted ? 5000 : false,
  });
  const refreshMutation = useMutation({
    mutationFn: () => refreshOptionsAnalytics({
      sourceRunId: commandQuery.data?.source_feature_run_id ?? null,
      force: true,
    }),
    onSuccess: (result) => setAccepted({
      taskId: result.task_id,
      baselineRunId: commandQuery.data?.run_id ?? null,
    }),
  });

  useEffect(() => {
    if (!accepted || !commandQuery.data || commandQuery.data.run_id === accepted.baselineRunId) return;
    setAccepted(null);
    queryClient.invalidateQueries({ queryKey: ['options-analytics', 'symbol', 'live'] });
  }, [accepted, commandQuery.data, queryClient]);

  if (commandQuery.isLoading) {
    return <Box sx={{ display: 'flex', justifyContent: 'center', p: 6 }}><CircularProgress /></Box>;
  }
  if (commandQuery.isError) {
    return <Alert severity="error">Could not load the published options snapshot.</Alert>;
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
      {commandQuery.data.items.length === 0 ? (
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
