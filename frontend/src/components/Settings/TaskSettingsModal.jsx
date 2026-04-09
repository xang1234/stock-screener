import { useState, useEffect, useCallback } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Typography,
  Box,
  CircularProgress,
  IconButton,
  Tooltip,
  Chip,
  Alert,
} from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import ScheduleIcon from '@mui/icons-material/Schedule';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import ErrorIcon from '@mui/icons-material/Error';
import HourglassEmptyIcon from '@mui/icons-material/HourglassEmpty';
import { getScheduledTasks, triggerTask, getTaskStatus } from '../../api/tasks';

/**
 * Format a relative time string (e.g., "2h ago", "5d ago")
 */
function formatRelativeTime(dateString) {
  if (!dateString) return 'Never';

  const date = new Date(dateString);
  const now = new Date();
  const diffMs = now - date;
  const diffSec = Math.floor(diffMs / 1000);
  const diffMin = Math.floor(diffSec / 60);
  const diffHour = Math.floor(diffMin / 60);
  const diffDay = Math.floor(diffHour / 24);

  if (diffDay > 0) return `${diffDay}d ago`;
  if (diffHour > 0) return `${diffHour}h ago`;
  if (diffMin > 0) return `${diffMin}m ago`;
  return 'Just now';
}

/**
 * Get status chip color and icon
 */
function getStatusDisplay(status) {
  switch (status) {
    case 'completed':
      return { color: 'success', icon: <CheckCircleIcon fontSize="small" />, label: 'Completed' };
    case 'running':
      return { color: 'info', icon: <HourglassEmptyIcon fontSize="small" />, label: 'Running' };
    case 'queued':
      return { color: 'warning', icon: <ScheduleIcon fontSize="small" />, label: 'Queued' };
    case 'failed':
      return { color: 'error', icon: <ErrorIcon fontSize="small" />, label: 'Failed' };
    default:
      return { color: 'default', icon: null, label: status || 'Unknown' };
  }
}

function TaskSettingsModal({ open, onClose }) {
  const queryClient = useQueryClient();
  const [runningTasks, setRunningTasks] = useState({}); // taskName -> { taskId, status }
  const [error, setError] = useState(null);

  // Query for scheduled tasks
  const { data: tasksData, isLoading } = useQuery({
    queryKey: ['scheduledTasks'],
    queryFn: getScheduledTasks,
    enabled: open,
    refetchInterval: open ? 10000 : false, // Poll every 10s when open
  });

  // Mutation for triggering tasks
  const runTaskMutation = useMutation({
    mutationFn: triggerTask,
    onSuccess: (data, taskName) => {
      setRunningTasks(prev => ({
        ...prev,
        [taskName]: { taskId: data.task_id, status: 'queued' }
      }));
      setError(null);
    },
    onError: (err) => {
      setError(err.response?.data?.detail || 'Failed to trigger task');
    },
  });

  // Poll running task statuses
  const pollTaskStatus = useCallback(async (taskName, taskId) => {
    try {
      const status = await getTaskStatus(taskName, taskId);

      setRunningTasks(prev => {
        const updated = { ...prev };
        if (status.status === 'completed' || status.status === 'failed') {
          // Task is done - remove from running tasks and refresh list
          delete updated[taskName];
          queryClient.invalidateQueries({ queryKey: ['scheduledTasks'] });
        } else {
          updated[taskName] = { taskId, status: status.status };
        }
        return updated;
      });
    } catch (err) {
      console.error('Error polling task status:', err);
    }
  }, [queryClient]);

  // Effect to poll running tasks
  useEffect(() => {
    if (!open || Object.keys(runningTasks).length === 0) return;

    const interval = setInterval(() => {
      Object.entries(runningTasks).forEach(([taskName, { taskId }]) => {
        pollTaskStatus(taskName, taskId);
      });
    }, 2000); // Poll every 2s for running tasks

    return () => clearInterval(interval);
  }, [open, runningTasks, pollTaskStatus]);

  // Reset running tasks when modal closes
  useEffect(() => {
    if (!open) {
      setRunningTasks({});
      setError(null);
    }
  }, [open]);

  const handleRunTask = (taskName) => {
    runTaskMutation.mutate(taskName);
  };

  const isTaskRunning = (taskName) => {
    return !!runningTasks[taskName] || runTaskMutation.isPending;
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
      <DialogTitle>
        <Box display="flex" justifyContent="space-between" alignItems="center">
          <Box display="flex" alignItems="center" gap={1}>
            <ScheduleIcon />
            <Typography variant="h6">Scheduled Tasks</Typography>
          </Box>
          <IconButton onClick={onClose} size="small">
            <CloseIcon />
          </IconButton>
        </Box>
      </DialogTitle>

      <DialogContent dividers>
        {error && (
          <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
            {error}
          </Alert>
        )}

        {isLoading ? (
          <Box display="flex" justifyContent="center" p={4}>
            <CircularProgress />
          </Box>
        ) : (
          <TableContainer component={Paper} variant="outlined">
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell sx={{ fontWeight: 'bold' }}>Task</TableCell>
                  <TableCell sx={{ fontWeight: 'bold' }}>Schedule</TableCell>
                  <TableCell sx={{ fontWeight: 'bold' }}>Last Run</TableCell>
                  <TableCell sx={{ fontWeight: 'bold' }}>Status</TableCell>
                  <TableCell align="center" sx={{ fontWeight: 'bold' }}>Action</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {tasksData?.tasks.map((task) => {
                  const running = isTaskRunning(task.name);
                  const lastRunStatus = task.last_run?.status || null;
                  const statusDisplay = lastRunStatus
                    ? getStatusDisplay(running ? runningTasks[task.name]?.status || 'running' : lastRunStatus)
                    : null;

                  return (
                    <TableRow key={task.name} hover>
                      <TableCell>
                        <Box>
                          <Typography variant="body2" fontWeight="medium">
                            {task.display_name}
                          </Typography>
                          <Typography variant="caption" color="text.secondary">
                            {task.description}
                          </Typography>
                        </Box>
                      </TableCell>
                      <TableCell>
                        <Typography variant="body2">
                          {task.schedule_description}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Typography variant="body2">
                          {task.last_run
                            ? formatRelativeTime(task.last_run.started_at)
                            : 'Never'
                          }
                        </Typography>
                        {task.last_run?.duration_seconds && (
                          <Typography variant="caption" color="text.secondary">
                            ({Math.round(task.last_run.duration_seconds)}s)
                          </Typography>
                        )}
                      </TableCell>
                      <TableCell>
                        {running ? (
                          <Chip
                            size="small"
                            color="info"
                            icon={<CircularProgress size={12} color="inherit" />}
                            label="Running"
                          />
                        ) : statusDisplay ? (
                          <Chip
                            size="small"
                            color={statusDisplay.color}
                            icon={statusDisplay.icon}
                            label={statusDisplay.label}
                          />
                        ) : (
                          <Typography variant="body2" color="text.secondary">
                            -
                          </Typography>
                        )}
                      </TableCell>
                      <TableCell align="center">
                        <Tooltip title={running ? 'Task is running...' : 'Run this task now'}>
                          <span>
                            <IconButton
                              size="small"
                              color="primary"
                              onClick={() => handleRunTask(task.name)}
                              disabled={running}
                            >
                              {running ? (
                                <CircularProgress size={20} />
                              ) : (
                                <PlayArrowIcon />
                              )}
                            </IconButton>
                          </span>
                        </Tooltip>
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          </TableContainer>
        )}

        {tasksData && (
          <Box mt={2}>
            <Typography variant="caption" color="text.secondary">
              {tasksData.total_tasks} scheduled tasks configured.
              Tasks run automatically based on their schedule when Celery Beat is active.
            </Typography>
          </Box>
        )}
      </DialogContent>

      <DialogActions>
        <Button onClick={onClose}>Close</Button>
      </DialogActions>
    </Dialog>
  );
}

export default TaskSettingsModal;
