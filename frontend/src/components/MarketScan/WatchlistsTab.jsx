/**
 * Watchlists Tab Component
 *
 * Main tab for user-defined watchlists display.
 * Features:
 * - Watchlist toggle buttons to switch between watchlists
 * - Settings icon to open WatchlistManager modal
 * - Renders WatchlistTable for selected watchlist
 */
import { useState, useEffect, useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  Box,
  Typography,
  IconButton,
  CircularProgress,
  ToggleButtonGroup,
  ToggleButton,
  Tooltip,
} from '@mui/material';
import SettingsIcon from '@mui/icons-material/Settings';
import RefreshIcon from '@mui/icons-material/Refresh';
import AddIcon from '@mui/icons-material/Add';
import DownloadIcon from '@mui/icons-material/Download';
import { getWatchlists, getWatchlistData } from '../../api/userWatchlists';
import WatchlistTable from './WatchlistTable';
import UserWatchlistManager from './UserWatchlistManager';
import WatchlistChartModal from './WatchlistChartModal';
import { useRuntime } from '../../contexts/RuntimeContext';

function WatchlistsTab() {
  const { bootstrap, bootstrapIncomplete } = useRuntime();
  const [selectedWatchlistId, setSelectedWatchlistId] = useState(null);
  const [managerOpen, setManagerOpen] = useState(false);
  const [chartModalOpen, setChartModalOpen] = useState(false);
  const [selectedSymbol, setSelectedSymbol] = useState(null);

  // Fetch list of watchlists for toggle
  const {
    data: watchlistsData,
    isLoading: watchlistsLoading,
    refetch: refetchWatchlists,
  } = useQuery({
    queryKey: ['userWatchlists'],
    queryFn: getWatchlists,
  });

  const watchlists = useMemo(
    () => watchlistsData?.watchlists || [],
    [watchlistsData]
  );

  // Auto-select first watchlist if none selected
  useEffect(() => {
    if (!selectedWatchlistId && watchlists.length > 0) {
      setSelectedWatchlistId(watchlists[0].id);
    }
  }, [watchlists, selectedWatchlistId]);

  // Fetch selected watchlist data with sparklines
  const {
    data: watchlistData,
    isLoading: dataLoading,
    refetch: refetchData,
  } = useQuery({
    queryKey: ['userWatchlistData', selectedWatchlistId],
    queryFn: () => getWatchlistData(selectedWatchlistId),
    enabled: !!selectedWatchlistId,
  });

  const handleRefresh = () => {
    refetchWatchlists();
    if (selectedWatchlistId) {
      refetchData();
    }
  };

  const handleWatchlistChange = (event, newWatchlistId) => {
    if (newWatchlistId !== null) {
      setSelectedWatchlistId(newWatchlistId);
    }
  };

  const handleOpenChart = (symbol) => {
    setSelectedSymbol(symbol);
    setChartModalOpen(true);
  };

  const handleDownloadWatchlist = () => {
    if (!watchlistData?.items?.length) return;

    const csvContent = watchlistData.items
      .map((item) => item.symbol)
      .join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `${watchlistData.name}.csv`;
    link.click();
    URL.revokeObjectURL(url);
  };

  // Extract symbols array from watchlist items for navigation
  const watchlistSymbols = watchlistData?.items?.map((item) => item.symbol) || [];

  if (watchlistsLoading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" height="100%">
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* Header with watchlist toggle and settings */}
      <Box
        sx={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          mb: 1,
          pb: 1,
          borderBottom: 1,
          borderColor: 'divider',
        }}
      >
        {/* Watchlist Toggle */}
        <Box display="flex" alignItems="center" gap={2}>
          <Typography variant="subtitle2" color="text.secondary">
            Watchlist:
          </Typography>
          {watchlists.length > 0 ? (
            <ToggleButtonGroup
              value={selectedWatchlistId}
              exclusive
              onChange={handleWatchlistChange}
              size="small"
            >
              {watchlists.map((watchlist) => (
                <ToggleButton
                  key={watchlist.id}
                  value={watchlist.id}
                  sx={{ px: 2, py: 0.5, textTransform: 'none' }}
                >
                  {watchlist.name}
                </ToggleButton>
              ))}
            </ToggleButtonGroup>
          ) : (
            <Typography variant="body2" color="text.secondary">
              No watchlists created yet
            </Typography>
          )}
        </Box>

        {/* Actions */}
        <Box display="flex" alignItems="center" gap={0.5}>
          <Tooltip title="Download watchlist as CSV">
            <span>
              <IconButton
                onClick={handleDownloadWatchlist}
                size="small"
                disabled={!watchlistData?.items?.length || dataLoading}
              >
                <DownloadIcon fontSize="small" />
              </IconButton>
            </span>
          </Tooltip>
          <Tooltip title="Refresh data">
            <IconButton onClick={handleRefresh} size="small">
              <RefreshIcon fontSize="small" />
            </IconButton>
          </Tooltip>
          <Tooltip title="Manage watchlists">
            <IconButton onClick={() => setManagerOpen(true)} size="small">
              <SettingsIcon fontSize="small" />
            </IconButton>
          </Tooltip>
        </Box>
      </Box>

      {/* Watchlist Table */}
      <Box sx={{ flex: 1, overflow: 'auto' }}>
        {dataLoading ? (
          <Box display="flex" justifyContent="center" py={4}>
            <CircularProgress />
          </Box>
        ) : watchlistData ? (
          <WatchlistTable watchlistData={watchlistData} onRefresh={handleRefresh} onOpenChart={handleOpenChart} />
        ) : watchlists.length === 0 ? (
          <Box textAlign="center" py={4}>
            <Typography color="text.secondary" gutterBottom>
              {bootstrapIncomplete
                ? (bootstrap?.message || 'Desktop setup is still warming market data. You can create watchlists now and the data will fill in as setup completes.')
                : 'No watchlists yet. Create your first watchlist to get started.'}
            </Typography>
            <Tooltip title="Manage watchlists">
              <IconButton color="primary" onClick={() => setManagerOpen(true)} size="large">
                <AddIcon />
              </IconButton>
            </Tooltip>
          </Box>
        ) : null}
      </Box>

      {/* Watchlist Manager Modal */}
      <UserWatchlistManager open={managerOpen} onClose={() => setManagerOpen(false)} onUpdate={handleRefresh} />

      {/* Chart Viewer Modal */}
      <WatchlistChartModal
        open={chartModalOpen}
        onClose={() => setChartModalOpen(false)}
        initialSymbol={selectedSymbol}
        symbols={watchlistSymbols}
      />
    </Box>
  );
}

export default WatchlistsTab;
