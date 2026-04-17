/**
 * Watchlists Tab Component
 *
 * Main tab for user-defined watchlists display.
 */
import { useState, useEffect, useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  Alert,
  Box,
  Chip,
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
import { getWatchlists, getWatchlistData, getWatchlistStewardship } from '../../api/userWatchlists';
import WatchlistTable from './WatchlistTable';
import UserWatchlistManager from './UserWatchlistManager';
import WatchlistChartModal from './WatchlistChartModal';
import { useStrategyProfile } from '../../contexts/StrategyProfileContext';

function WatchlistsTab() {
  const { activeProfile } = useStrategyProfile();
  const [selectedWatchlistId, setSelectedWatchlistId] = useState(null);
  const [managerOpen, setManagerOpen] = useState(false);
  const [chartModalOpen, setChartModalOpen] = useState(false);
  const [selectedSymbol, setSelectedSymbol] = useState(null);
  const [statusFilter, setStatusFilter] = useState('all');

  const {
    data: watchlistsData,
    isLoading: watchlistsLoading,
    error: watchlistsError,
    refetch: refetchWatchlists,
  } = useQuery({
    queryKey: ['userWatchlists'],
    queryFn: getWatchlists,
  });

  const watchlists = useMemo(() => watchlistsData?.watchlists || [], [watchlistsData]);

  useEffect(() => {
    if (!selectedWatchlistId && watchlists.length > 0) {
      setSelectedWatchlistId(watchlists[0].id);
    }
  }, [watchlists, selectedWatchlistId]);

  const {
    data: watchlistData,
    isLoading: dataLoading,
    error: dataError,
    refetch: refetchData,
  } = useQuery({
    queryKey: ['userWatchlistData', selectedWatchlistId],
    queryFn: () => getWatchlistData(selectedWatchlistId),
    enabled: !!selectedWatchlistId,
  });

  const {
    data: stewardshipData,
    isLoading: stewardshipLoading,
    error: stewardshipError,
    refetch: refetchStewardship,
  } = useQuery({
    queryKey: ['userWatchlistStewardship', selectedWatchlistId, activeProfile],
    queryFn: () => getWatchlistStewardship(selectedWatchlistId, activeProfile),
    enabled: !!selectedWatchlistId && !!watchlistData,
  });

  const stewardshipBySymbol = useMemo(
    () => Object.fromEntries((stewardshipData?.items || []).map((item) => [item.symbol, item])),
    [stewardshipData]
  );
  const stewardshipOrder = useMemo(
    () => Object.fromEntries((stewardshipData?.items || []).map((item, index) => [item.symbol, index])),
    [stewardshipData]
  );

  const filteredWatchlistData = useMemo(() => {
    if (!watchlistData) {
      return null;
    }
    const filteredItems = [...(watchlistData.items || [])]
      .filter((item) => statusFilter === 'all' || stewardshipBySymbol[item.symbol]?.status === statusFilter)
      .sort((left, right) => {
        const leftIndex = stewardshipOrder[left.symbol] ?? Number.MAX_SAFE_INTEGER;
        const rightIndex = stewardshipOrder[right.symbol] ?? Number.MAX_SAFE_INTEGER;
        return leftIndex - rightIndex;
      });
    return { ...watchlistData, items: filteredItems };
  }, [statusFilter, stewardshipBySymbol, stewardshipOrder, watchlistData]);

  const handleRefresh = () => {
    refetchWatchlists();
    if (selectedWatchlistId) {
      refetchData();
      refetchStewardship();
    }
  };

  const handleWatchlistChange = (_, newWatchlistId) => {
    if (newWatchlistId !== null) {
      setSelectedWatchlistId(newWatchlistId);
      setStatusFilter('all');
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

  const watchlistSymbols = watchlistData?.items?.map((item) => item.symbol) || [];

  if (watchlistsLoading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" height="100%">
        <CircularProgress />
      </Box>
    );
  }

  if (watchlistsError) {
    return <Alert severity="error">Unable to load watchlists: {watchlistsError.message}</Alert>;
  }

  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
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

      <Box sx={{ flex: 1, overflow: 'auto' }}>
        {dataLoading ? (
          <Box display="flex" justifyContent="center" py={4}>
            <CircularProgress />
          </Box>
        ) : dataError ? (
          <Alert severity="error">Unable to load watchlist data: {dataError.message}</Alert>
        ) : watchlistData ? (
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
            {stewardshipError && (
              <Alert severity="warning">
                Stewardship context is unavailable: {stewardshipError.message}
              </Alert>
            )}
            {stewardshipData && (
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                <Box sx={{ display: 'flex', gap: 0.75, flexWrap: 'wrap' }}>
                  {['strengthening', 'unchanged', 'deteriorating', 'exit_risk', 'missing_from_run'].map((status) => (
                    <Chip
                      key={status}
                      size="small"
                      variant="outlined"
                      label={`${status.replaceAll('_', ' ')}: ${stewardshipData.summary_counts?.[status] ?? 0}`}
                    />
                  ))}
                </Box>
                <ToggleButtonGroup
                  size="small"
                  value={statusFilter}
                  exclusive
                  onChange={(_, value) => value && setStatusFilter(value)}
                >
                  <ToggleButton value="all">All</ToggleButton>
                  <ToggleButton value="strengthening">Strengthening</ToggleButton>
                  <ToggleButton value="unchanged">Unchanged</ToggleButton>
                  <ToggleButton value="deteriorating">Deteriorating</ToggleButton>
                  <ToggleButton value="exit_risk">Exit Risk</ToggleButton>
                  <ToggleButton value="missing_from_run">Missing</ToggleButton>
                </ToggleButtonGroup>
              </Box>
            )}
            <WatchlistTable
              watchlistData={filteredWatchlistData}
              stewardshipBySymbol={stewardshipBySymbol}
              stewardshipLoading={stewardshipLoading}
              onRefresh={handleRefresh}
              onOpenChart={handleOpenChart}
            />
          </Box>
        ) : watchlists.length === 0 ? (
          <Box textAlign="center" py={4}>
            <Typography color="text.secondary" gutterBottom>
              No watchlists yet. Create your first watchlist to get started.
            </Typography>
            <Tooltip title="Manage watchlists">
              <IconButton color="primary" onClick={() => setManagerOpen(true)} size="large">
                <AddIcon />
              </IconButton>
            </Tooltip>
          </Box>
        ) : null}
      </Box>

      <UserWatchlistManager open={managerOpen} onClose={() => setManagerOpen(false)} onUpdate={handleRefresh} />

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
