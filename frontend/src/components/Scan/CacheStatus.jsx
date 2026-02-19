/**
 * CacheStatus Component
 *
 * Displays cache health status for fundamentals and price data.
 * Provides manual refresh buttons to trigger cache warmup tasks.
 *
 * REFACTORED: Now uses unified cache health endpoint with 6 states.
 *
 * Features:
 * - Auto-refresh stats every 60 seconds (5s during updates)
 * - Relative time display ("2 hours ago")
 * - Color-coded freshness indicators
 * - Dropdown menu for Full Refresh option
 * - Completion notification when task finishes
 * - Confirmation dialog for expensive operations
 */
import React, { useState, useEffect, useRef } from 'react';
import {
  Chip,
  CircularProgress,
  Box,
  Tooltip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogContentText,
  DialogActions,
  Snackbar,
  Alert,
  Menu,
  MenuItem,
  ListItemIcon,
  ListItemText,
  Button
} from '@mui/material';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
  getCacheStats,
  triggerFundamentalsRefresh,
  getCacheHealth,
  refreshCache,
  forceCancelRefresh
} from '../../api/cache';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import WarningIcon from '@mui/icons-material/Warning';
import ErrorIcon from '@mui/icons-material/Error';
import RefreshIcon from '@mui/icons-material/Refresh';
import CancelIcon from '@mui/icons-material/Cancel';
import UpdateIcon from '@mui/icons-material/Update';
import ArrowDropDownIcon from '@mui/icons-material/ArrowDropDown';

/**
 * Calculate relative time string from ISO timestamp.
 * Examples: "2 hours ago", "3 days ago", "just now"
 */
const getRelativeTime = (isoTimestamp) => {
  if (!isoTimestamp || isoTimestamp === 'N/A') {
    return 'Never';
  }

  try {
    const date = new Date(isoTimestamp);
    const now = new Date();
    const diffMs = now - date;
    const diffMinutes = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);

    if (diffMinutes < 1) return 'Just now';
    if (diffMinutes < 60) return `${diffMinutes} minute${diffMinutes > 1 ? 's' : ''} ago`;
    if (diffHours < 24) return `${diffHours} hour${diffHours > 1 ? 's' : ''} ago`;
    return `${diffDays} day${diffDays > 1 ? 's' : ''} ago`;
  } catch (error) {
    return 'Unknown';
  }
};

/**
 * Get freshness status and color for fundamentals cache.
 * - Green (Fresh): < 3 days
 * - Yellow (Stale): 3-7 days
 * - Red (Very Stale): > 7 days
 */
const getFundamentalsFreshness = (lastUpdate) => {
  if (!lastUpdate || lastUpdate === 'Never') {
    return { status: 'No Data', color: 'error', icon: <ErrorIcon /> };
  }

  try {
    const date = new Date(lastUpdate);
    const now = new Date();
    const diffDays = Math.floor((now - date) / 86400000);

    if (diffDays < 3) {
      return { status: 'Fresh', color: 'success', icon: <CheckCircleIcon /> };
    } else if (diffDays <= 7) {
      return { status: 'Stale', color: 'warning', icon: <WarningIcon /> };
    } else {
      return { status: 'Very Stale', color: 'error', icon: <ErrorIcon /> };
    }
  } catch (error) {
    return { status: 'Error', color: 'error', icon: <ErrorIcon /> };
  }
};

/**
 * Get chip props (icon, label, color) based on cache health status.
 *
 * 6 states:
 * - fresh: Cache is up to date
 * - updating: Refresh task is running
 * - stuck: Task running but no progress
 * - partial: Last warmup incomplete
 * - stale: Missing expected trading date
 * - error: Redis unavailable
 */
const getCacheChipProps = (health) => {
  if (!health) {
    return {
      icon: <ErrorIcon />,
      label: 'Cache',
      color: 'default',
      tooltip: 'Loading cache status...'
    };
  }

  switch (health.status) {
    case 'fresh':
      return {
        icon: <CheckCircleIcon />,
        label: 'Cache',
        color: 'success',
        tooltip: `Cache up to date (${health.spy_last_date || 'SPY cached'})`
      };
    case 'updating':
      const progress = health.task_running?.progress;
      return {
        icon: <CircularProgress size={10} sx={{ color: 'white' }} />,
        label: progress ? `${Math.round(progress)}%` : 'Updating',
        color: 'info',
        tooltip: health.message
      };
    case 'stuck':
      return {
        icon: <WarningIcon />,
        label: 'Stuck',
        color: 'warning',
        tooltip: health.message
      };
    case 'partial':
      return {
        icon: <WarningIcon />,
        label: 'Partial',
        color: 'warning',
        tooltip: health.message
      };
    case 'stale':
      return {
        icon: <WarningIcon />,
        label: 'Stale',
        color: 'warning',
        tooltip: health.message
      };
    case 'error':
      return {
        icon: <ErrorIcon />,
        label: 'Error',
        color: 'error',
        tooltip: health.message
      };
    default:
      return {
        icon: <WarningIcon />,
        label: 'Unknown',
        color: 'default',
        tooltip: 'Unknown cache status'
      };
  }
};

export default function CacheStatus() {
  const queryClient = useQueryClient();

  // State for dropdown menu
  const [menuAnchorEl, setMenuAnchorEl] = useState(null);

  // State for confirmation dialog
  const [confirmDialog, setConfirmDialog] = useState({
    open: false,
    type: null
  });

  // State for notifications
  const [notification, setNotification] = useState({
    open: false,
    message: '',
    severity: 'success'
  });

  // Track previous status for completion detection
  const prevStatusRef = useRef(null);

  // Query cache stats for fundamentals (every 60 seconds)
  const { data: stats, isLoading: statsLoading } = useQuery({
    queryKey: ['cacheStats'],
    queryFn: getCacheStats,
    refetchInterval: 60000,
    staleTime: 30000,
    retry: 2
  });

  // Query cache health status (NEW unified endpoint)
  // Dynamic polling: 5s during updates, 60s otherwise
  const { data: health, isLoading: healthLoading, error: healthError } = useQuery({
    queryKey: ['cacheHealth'],
    queryFn: getCacheHealth,
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      return status === 'updating' ? 5000 : 60000;
    },
    staleTime: 3000,
    retry: 2
  });

  // Detect when refresh completes and show notification
  useEffect(() => {
    if (prevStatusRef.current === 'updating' && health?.status === 'fresh') {
      setNotification({
        open: true,
        message: 'Cache refresh completed successfully',
        severity: 'success'
      });
      // Refresh the stats too
      queryClient.invalidateQueries(['cacheStats']);
    }
    prevStatusRef.current = health?.status;
  }, [health?.status, queryClient]);

  // Mutation for fundamental refresh
  const fundamentalsMutation = useMutation({
    mutationFn: triggerFundamentalsRefresh,
    onSuccess: (data) => {
      setNotification({
        open: true,
        message: 'Fundamentals refresh started',
        severity: 'success'
      });
      setConfirmDialog({ open: false, type: null });
      setTimeout(() => queryClient.invalidateQueries(['cacheStats']), 5000);
    },
    onError: (error) => {
      setNotification({
        open: true,
        message: `Error: ${error.message}`,
        severity: 'error'
      });
      setConfirmDialog({ open: false, type: null });
    }
  });

  // Mutation for smart refresh (NEW)
  const refreshMutation = useMutation({
    mutationFn: (mode) => refreshCache(mode),
    onSuccess: (data) => {
      if (data.status === 'already_running') {
        setNotification({
          open: true,
          message: data.message || 'Refresh already in progress',
          severity: 'info'
        });
      } else {
        setNotification({
          open: true,
          message: data.message || 'Cache refresh started',
          severity: 'success'
        });
      }
      setMenuAnchorEl(null);
      // Immediately invalidate to show "updating" state
      queryClient.invalidateQueries(['cacheHealth']);
    },
    onError: (error) => {
      setNotification({
        open: true,
        message: `Error: ${error.message}`,
        severity: 'error'
      });
      setMenuAnchorEl(null);
    }
  });

  // Mutation for force cancel
  const cancelMutation = useMutation({
    mutationFn: forceCancelRefresh,
    onSuccess: (data) => {
      if (data.status === 'cancelled') {
        setNotification({
          open: true,
          message: data.message || 'Task cancelled',
          severity: 'success'
        });
      } else {
        setNotification({
          open: true,
          message: data.message,
          severity: 'info'
        });
      }
      setMenuAnchorEl(null);
      queryClient.invalidateQueries(['cacheHealth']);
    },
    onError: (error) => {
      setNotification({
        open: true,
        message: `Error: ${error.message}`,
        severity: 'error'
      });
      setMenuAnchorEl(null);
    }
  });

  // Handle cache chip click
  const handleCacheChipClick = (event) => {
    // If stale/partial, directly trigger auto refresh
    if (health?.status === 'stale' || health?.status === 'partial') {
      refreshMutation.mutate('auto');
    } else if (health?.status === 'fresh' || health?.status === 'error') {
      // For fresh/error, open menu for options
      setMenuAnchorEl(event.currentTarget);
    }
    // For updating/stuck, do nothing on chip click
  };

  // Handle menu open (dropdown arrow or right-click)
  const handleMenuOpen = (event) => {
    event.stopPropagation();
    setMenuAnchorEl(event.currentTarget);
  };

  // Handle menu close
  const handleMenuClose = () => {
    setMenuAnchorEl(null);
  };

  // Handle auto refresh
  const handleAutoRefresh = () => {
    refreshMutation.mutate('auto');
    handleMenuClose();
  };

  // Handle full refresh (with confirmation for long operation)
  const handleFullRefresh = () => {
    setConfirmDialog({ open: true, type: 'full' });
    handleMenuClose();
  };

  // Handle force cancel
  const handleForceCancel = () => {
    cancelMutation.mutate();
  };

  // Handle fundamentals refresh click
  const handleFundamentalsClick = (e) => {
    e.stopPropagation();
    setConfirmDialog({ open: true, type: 'fundamentals' });
  };

  // Handle confirmation dialog
  const handleConfirm = () => {
    if (confirmDialog.type === 'fundamentals') {
      fundamentalsMutation.mutate();
    } else if (confirmDialog.type === 'full') {
      refreshMutation.mutate('full');
    }
    setConfirmDialog({ open: false, type: null });
  };

  const handleCancel = () => {
    setConfirmDialog({ open: false, type: null });
  };

  const handleNotificationClose = () => {
    setNotification({ ...notification, open: false });
  };

  // Loading state
  if (statsLoading && healthLoading) {
    return <CircularProgress size={16} />;
  }

  // Error state
  if (healthError) {
    return (
      <Tooltip title="Cache error" arrow>
        <Chip icon={<ErrorIcon />} label="Cache" color="error" size="small" sx={{ height: 20, fontSize: '10px' }} />
      </Tooltip>
    );
  }

  // Extract data
  const fundamentals = stats?.fundamentals || {};
  const fundamentalsFreshness = getFundamentalsFreshness(fundamentals.last_update);
  const cacheProps = getCacheChipProps(health);

  // Determine if menu should show force cancel
  const canForceCancel = health?.status === 'stuck' || health?.can_force_cancel;

  return (
    <>
      {/* Compact inline cache indicators */}
      <Box sx={{ display: 'flex', gap: 0.5, alignItems: 'center' }}>
        {/* Fundamentals Chip (unchanged) */}
        <Tooltip
          title={
            <Box sx={{ fontSize: '11px' }}>
              <Box sx={{ fontWeight: 600, mb: 0.5 }}>Fundamentals: {fundamentalsFreshness.status}</Box>
              <Box>Updated: {getRelativeTime(fundamentals.last_update)}</Box>
              <Box>{fundamentals.cached_count || 0} cached, {fundamentals.fresh_count || 0} fresh</Box>
              <Box sx={{ mt: 0.5, fontStyle: 'italic' }}>Click to refresh</Box>
            </Box>
          }
          arrow
        >
          <Chip
            icon={fundamentalsMutation.isPending ? <CircularProgress size={10} /> : fundamentalsFreshness.icon}
            label="Fund"
            color={fundamentalsFreshness.color}
            size="small"
            onClick={handleFundamentalsClick}
            sx={{ height: 20, fontSize: '10px', cursor: 'pointer', '& .MuiChip-icon': { fontSize: 12 } }}
          />
        </Tooltip>

        {/* Cache Chip (NEW unified status) */}
        <Tooltip
          title={
            <Box sx={{ fontSize: '11px' }}>
              <Box sx={{ fontWeight: 600, mb: 0.5 }}>
                Cache: {health?.status?.charAt(0).toUpperCase() + health?.status?.slice(1) || 'Unknown'}
              </Box>
              <Box>{cacheProps.tooltip}</Box>
              {health?.spy_last_date && (
                <Box>SPY data through: {health.spy_last_date}</Box>
              )}
              {health?.task_running?.progress && (
                <Box>Progress: {Math.round(health.task_running.progress)}%</Box>
              )}
              {health?.last_warmup && (
                <Box>Last warmup: {health.last_warmup.status} ({health.last_warmup.count}/{health.last_warmup.total})</Box>
              )}
              {health?.status !== 'updating' && (
                <Box sx={{ mt: 0.5, fontStyle: 'italic' }}>
                  {health?.status === 'stale' || health?.status === 'partial'
                    ? 'Click to refresh'
                    : 'Click for options'}
                </Box>
              )}
            </Box>
          }
          arrow
        >
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <Chip
              icon={refreshMutation.isPending || cancelMutation.isPending
                ? <CircularProgress size={10} sx={{ color: cacheProps.color === 'info' ? 'white' : undefined }} />
                : cacheProps.icon}
              label={cacheProps.label}
              color={cacheProps.color}
              size="small"
              onClick={health?.status !== 'updating' ? handleCacheChipClick : undefined}
              sx={{
                height: 20,
                fontSize: '10px',
                cursor: health?.status === 'updating' ? 'default' : 'pointer',
                borderTopRightRadius: 0,
                borderBottomRightRadius: 0,
                '& .MuiChip-icon': { fontSize: 12 },
                // Pulse animation for stale state
                ...(health?.status === 'stale' && {
                  animation: 'pulse 2s infinite',
                  '@keyframes pulse': {
                    '0%': { opacity: 1 },
                    '50%': { opacity: 0.7 },
                    '100%': { opacity: 1 }
                  }
                })
              }}
            />
            <Chip
              icon={<ArrowDropDownIcon />}
              size="small"
              color={cacheProps.color}
              onClick={handleMenuOpen}
              sx={{
                height: 20,
                minWidth: 20,
                borderTopLeftRadius: 0,
                borderBottomLeftRadius: 0,
                marginLeft: '-1px',
                cursor: 'pointer',
                '& .MuiChip-icon': { fontSize: 14, margin: 0 },
                '& .MuiChip-label': { display: 'none' }
              }}
            />
          </Box>
        </Tooltip>

        {/* Dropdown Menu */}
        <Menu
          anchorEl={menuAnchorEl}
          open={Boolean(menuAnchorEl)}
          onClose={handleMenuClose}
          anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
          transformOrigin={{ vertical: 'top', horizontal: 'right' }}
        >
          <MenuItem onClick={handleAutoRefresh} disabled={refreshMutation.isPending || health?.status === 'updating'}>
            <ListItemIcon><RefreshIcon fontSize="small" /></ListItemIcon>
            <ListItemText
              primary="Refresh Cache"
              secondary={`${health?.universe_count?.toLocaleString() || '~9,000'} symbols (skips fresh)`}
              primaryTypographyProps={{ fontSize: '12px' }}
              secondaryTypographyProps={{ fontSize: '10px' }}
            />
          </MenuItem>
          <MenuItem onClick={handleFullRefresh} disabled={refreshMutation.isPending || health?.status === 'updating'}>
            <ListItemIcon><UpdateIcon fontSize="small" /></ListItemIcon>
            <ListItemText
              primary="Force Full Refresh"
              secondary={`All ${health?.universe_count?.toLocaleString() || '~9,000'} symbols (~2 hours)`}
              primaryTypographyProps={{ fontSize: '12px' }}
              secondaryTypographyProps={{ fontSize: '10px' }}
            />
          </MenuItem>
          {canForceCancel && (
            <MenuItem onClick={handleForceCancel} disabled={cancelMutation.isPending}>
              <ListItemIcon><CancelIcon fontSize="small" color="error" /></ListItemIcon>
              <ListItemText
                primary="Force Cancel"
                secondary="Cancel stuck task"
                primaryTypographyProps={{ fontSize: '12px', color: 'error.main' }}
                secondaryTypographyProps={{ fontSize: '10px' }}
              />
            </MenuItem>
          )}
        </Menu>
      </Box>

      {/* Confirmation Dialog */}
      <Dialog open={confirmDialog.open} onClose={handleCancel}>
        <DialogTitle sx={{ fontSize: '14px', pb: 1 }}>
          {confirmDialog.type === 'fundamentals' ? 'Refresh Fundamentals?' : 'Full Cache Refresh?'}
        </DialogTitle>
        <DialogContent>
          <DialogContentText sx={{ fontSize: '12px' }}>
            {confirmDialog.type === 'fundamentals'
              ? 'This will refresh data for ~7,000 stocks. Takes ~1 hour.'
              : 'This will force re-fetch price data for ALL symbols regardless of freshness. Takes approximately 2 hours.'}
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCancel} size="small">Cancel</Button>
          <Button onClick={handleConfirm} variant="contained" size="small">Refresh</Button>
        </DialogActions>
      </Dialog>

      {/* Notification */}
      <Snackbar
        open={notification.open}
        autoHideDuration={4000}
        onClose={handleNotificationClose}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
      >
        <Alert onClose={handleNotificationClose} severity={notification.severity} sx={{ fontSize: '11px' }}>
          {notification.message}
        </Alert>
      </Snackbar>
    </>
  );
}
