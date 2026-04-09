import { useState, useEffect, useCallback } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { getPipelineStatus } from '../api/themes';
import { PipelineContext } from './pipelineContextStore';

export function PipelineProvider({ children }) {
  const queryClient = useQueryClient();
  const [pipelineRunId, setPipelineRunId] = useState(null);
  const [pipelineStatus, setPipelineStatus] = useState(null);
  const [isPipelineRunning, setIsPipelineRunning] = useState(false);
  const [isMinimized, setIsMinimized] = useState(true);

  // Pipeline polling effect
  useEffect(() => {
    let pollInterval;

    if (pipelineRunId && isPipelineRunning) {
      pollInterval = setInterval(async () => {
        try {
          const status = await getPipelineStatus(pipelineRunId);
          setPipelineStatus(status);

          if (status.status === 'completed' || status.status === 'failed') {
            setIsPipelineRunning(false);
            clearInterval(pollInterval);

            // Refresh theme data on completion
            if (status.status === 'completed') {
              queryClient.invalidateQueries({ queryKey: ['themeRankings'] });
              queryClient.invalidateQueries({ queryKey: ['emergingThemes'] });
              queryClient.invalidateQueries({ queryKey: ['themeAlerts'] });
              queryClient.invalidateQueries({ queryKey: ['failedItemsCount'] });
            }
          }
        } catch (error) {
          console.error('Error polling pipeline status:', error);
        }
      }, 2000); // Poll every 2 seconds
    }

    return () => {
      if (pollInterval) clearInterval(pollInterval);
    };
  }, [pipelineRunId, isPipelineRunning, queryClient]);

  // Start a pipeline run
  const startPipeline = useCallback((runId) => {
    setPipelineRunId(runId);
    setIsPipelineRunning(true);
    setIsMinimized(true); // Start minimized by default
    setPipelineStatus({
      status: 'queued',
      current_step: null,
      step_number: 0,
      total_steps: 5,
      percent: 0,
      message: 'Pipeline queued...'
    });
  }, []);

  // Close/dismiss the pipeline card
  const closePipelineCard = useCallback(() => {
    setPipelineRunId(null);
    setPipelineStatus(null);
  }, []);

  // Toggle minimize state
  const toggleMinimize = useCallback(() => {
    setIsMinimized(prev => !prev);
  }, []);

  // Check if the card should be visible
  const isCardVisible = isPipelineRunning ||
    (pipelineStatus?.status === 'completed' && pipelineRunId !== null) ||
    (pipelineStatus?.status === 'failed' && pipelineRunId !== null);

  const value = {
    pipelineRunId,
    pipelineStatus,
    isPipelineRunning,
    isCardVisible,
    isMinimized,
    startPipeline,
    closePipelineCard,
    toggleMinimize,
  };

  return (
    <PipelineContext.Provider value={value}>
      {children}
    </PipelineContext.Provider>
  );
}
