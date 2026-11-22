import { useState, useEffect, useCallback, useMemo } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { getPipelineStatus } from '../api/themes';
import { PipelineContext } from './pipelineContextStore';

// "skipped": a run queued before cutover to economic authority (#472).
const TERMINAL_PIPELINE_STATUSES = new Set(['completed', 'failed', 'skipped']);

const QUEUED_STATUS = {
  status: 'queued',
  current_step: null,
  step_number: 0,
  total_steps: 5,
  percent: 0,
  message: 'Pipeline queued...',
};

// A 4xx never changes on retry: the run is gone, so stop polling and let the card close.
const isClientError = (error) => {
  const status = error?.response?.status;
  return status >= 400 && status < 500;
};

const RUN_NOT_FOUND_STATUS = {
  ...QUEUED_STATUS,
  status: 'failed',
  message: 'Pipeline status unavailable',
  error_message: 'Pipeline run not found',
};

export function PipelineProvider({ children }) {
  const queryClient = useQueryClient();
  const [pipelineRunId, setPipelineRunId] = useState(null);
  const [isMinimized, setIsMinimized] = useState(true);

  // One request in flight per run, so a slow response cannot land after a newer one.
  const { data, error } = useQuery({
    queryKey: ['pipelineStatus', pipelineRunId],
    queryFn: () => getPipelineStatus(pipelineRunId),
    enabled: Boolean(pipelineRunId),
    retry: false, // The next poll is the retry.
    refetchInterval: (query) => (
      TERMINAL_PIPELINE_STATUSES.has(query.state.data?.status) || isClientError(query.state.error)
        ? false
        : 2000
    ),
  });

  const isCardVisible = Boolean(pipelineRunId);
  let pipelineStatus = null;
  if (isCardVisible) {
    pipelineStatus = isClientError(error) ? RUN_NOT_FOUND_STATUS : (data ?? QUEUED_STATUS);
  }
  const isPipelineRunning = isCardVisible && !TERMINAL_PIPELINE_STATUSES.has(pipelineStatus.status);

  const isCompleted = data?.status === 'completed';
  useEffect(() => {
    if (!isCompleted) return;
    queryClient.invalidateQueries({ queryKey: ['themeRankings'] });
    queryClient.invalidateQueries({ queryKey: ['emergingThemes'] });
    queryClient.invalidateQueries({ queryKey: ['themeAlerts'] });
    queryClient.invalidateQueries({ queryKey: ['failedItemsCount'] });
  }, [isCompleted, queryClient]);

  const startPipeline = useCallback((runId) => {
    setPipelineRunId(runId || null);
    setIsMinimized(true);
  }, []);

  const closePipelineCard = useCallback(() => {
    setPipelineRunId(null);
  }, []);

  const toggleMinimize = useCallback(() => {
    setIsMinimized(prev => !prev);
  }, []);

  const value = useMemo(() => ({
    pipelineRunId,
    pipelineStatus,
    isPipelineRunning,
    isCardVisible,
    isMinimized,
    startPipeline,
    closePipelineCard,
    toggleMinimize,
  }), [
    pipelineRunId,
    pipelineStatus,
    isPipelineRunning,
    isCardVisible,
    isMinimized,
    startPipeline,
    closePipelineCard,
    toggleMinimize,
  ]);

  return (
    <PipelineContext.Provider value={value}>
      {children}
    </PipelineContext.Provider>
  );
}
