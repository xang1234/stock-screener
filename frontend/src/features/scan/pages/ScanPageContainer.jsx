import { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { Box, CircularProgress, Container, Paper, Typography } from '@mui/material';
import {
  cancelScan,
  createScan,
  exportScanResults,
  getFilterOptions,
  getScanBootstrap,
  getScanResults,
  getScans,
  getScanStatus,
  getUniverseStats,
} from '../../../api/scans';
import FilterPanel from '../components/FilterPanelContainer';
import ChartViewerModal from '../../../components/Scan/ChartViewerModal';
import { buildFilterParams, getStableFilterKey } from '../../../utils/filterUtils';
import { fetchPriceHistory, priceHistoryKeys, PRICE_HISTORY_STALE_TIME } from '../../../api/priceHistory';
import { useFilterPresets } from '../../../hooks/useFilterPresets';
import { useRuntime } from '../../../contexts/RuntimeContext';
import { useStrategyProfile } from '../../../contexts/StrategyProfileContext';
import { DEFAULT_SCAN_DEFAULTS } from '../../../constants/scanDefaults';
import { buildDefaultScanFilters } from '../defaultFilters';
import { normalizeScanFilterOptions } from '../filterOptions';
import { DEFAULT_FILTER_KEY, TEST_SYMBOLS } from '../constants';
import ScanControlBar from '../components/ScanControlBar';
import ScanResultsSection from '../components/ScanResultsSection';
import { useScanFilterPresets } from '../hooks/useScanFilterPresets';

function ScanPage() {
  const { runtimeReady, uiSnapshots, scanDefaults } = useRuntime();
  const { activeProfileDetail } = useStrategyProfile();
  const scanDefaultsAppliedRef = useRef(null);
  const queryClient = useQueryClient();

  const [currentScanId, setCurrentScanId] = useState(null);
  const [scanStatus, setScanStatus] = useState(null);
  const [initialBootstrapSettled, setInitialBootstrapSettled] = useState(false);
  const [bootstrappedScanId, setBootstrappedScanId] = useState(null);
  const [universe, setUniverse] = useState(DEFAULT_SCAN_DEFAULTS.universe);
  const [includeVcp, setIncludeVcp] = useState(DEFAULT_SCAN_DEFAULTS.criteria.include_vcp);
  const [selectedScreeners, setSelectedScreeners] = useState(DEFAULT_SCAN_DEFAULTS.screeners);
  const [compositeMethod, setCompositeMethod] = useState(DEFAULT_SCAN_DEFAULTS.composite_method);
  const [customFilters, setCustomFilters] = useState(DEFAULT_SCAN_DEFAULTS.criteria.custom_filters);
  const [page, setPage] = useState(1);
  const [perPage, setPerPage] = useState(50);
  const [sortBy, setSortBy] = useState('composite_score');
  const [sortOrder, setSortOrder] = useState('desc');
  const [filters, setFilters] = useState(buildDefaultScanFilters);
  const [debouncedFilters, setDebouncedFilters] = useState(filters);
  const [chartModalOpen, setChartModalOpen] = useState(false);
  const [selectedSymbol, setSelectedSymbol] = useState(null);
  const [showFilters, setShowFilters] = useState(false);

  const snapshotEnabled = runtimeReady && Boolean(uiSnapshots?.scan);
  const initialQueriesEnabled = runtimeReady && (!snapshotEnabled || initialBootstrapSettled);

  useEffect(() => {
    if (!runtimeReady) {
      return;
    }
    const nextDefaults = activeProfileDetail?.scan_defaults ?? scanDefaults ?? DEFAULT_SCAN_DEFAULTS;
    const profileKey = activeProfileDetail?.profile || 'runtime-default';
    if (scanDefaultsAppliedRef.current === profileKey) {
      return;
    }

    setUniverse(nextDefaults.universe ?? DEFAULT_SCAN_DEFAULTS.universe);
    setIncludeVcp(nextDefaults.criteria?.include_vcp ?? DEFAULT_SCAN_DEFAULTS.criteria.include_vcp);
    setSelectedScreeners(nextDefaults.screeners ?? DEFAULT_SCAN_DEFAULTS.screeners);
    setCompositeMethod(nextDefaults.composite_method ?? DEFAULT_SCAN_DEFAULTS.composite_method);
    setCustomFilters(nextDefaults.criteria?.custom_filters ?? DEFAULT_SCAN_DEFAULTS.criteria.custom_filters);
    scanDefaultsAppliedRef.current = profileKey;
  }, [activeProfileDetail, runtimeReady, scanDefaults]);

  const applyScanBootstrapSnapshot = useCallback(
    (snapshot, requestedScanId = null) => {
      const payload = snapshot?.payload ?? {};
      queryClient.setQueryData(['universeStats'], payload.universe_stats ?? null);
      queryClient.setQueryData(['scanHistory'], payload.recent_scans ?? { scans: [] });

      const selectedScanId =
        payload.selected_scan?.scan_id ??
        payload.results_page?.scan_id ??
        requestedScanId ??
        null;

      if (!selectedScanId) {
        return;
      }

      queryClient.setQueryData(['filterOptions', selectedScanId], payload.filter_options ?? null);
      queryClient.setQueryData(
        ['scanResults', selectedScanId, 1, 50, 'composite_score', 'desc', DEFAULT_FILTER_KEY],
        payload.results_page ?? null
      );
      if (payload.selected_scan_status) {
        queryClient.setQueryData(['scanStatus', selectedScanId], payload.selected_scan_status);
      }
      setCurrentScanId(selectedScanId);
      setBootstrappedScanId(selectedScanId);
      setScanStatus(payload.selected_scan_status?.status ?? payload.selected_scan?.status ?? null);
    },
    [queryClient]
  );

  const {
    presets,
    isLoading: presetsLoading,
    createPresetAsync,
    updatePresetAsync,
    deletePreset,
    isCreating: presetIsCreating,
    isUpdating: presetIsUpdating,
  } = useFilterPresets();

  const presetState = useScanFilterPresets({
    presets,
    createPresetAsync,
    updatePresetAsync,
    deletePreset,
    filters,
    sortBy,
    sortOrder,
    setFilters,
    setSortBy,
    setSortOrder,
    setPage,
  });

  const scanBootstrapQuery = useQuery({
    queryKey: ['scanBootstrap', 'latest'],
    queryFn: () => getScanBootstrap(),
    enabled: snapshotEnabled && !currentScanId && !initialBootstrapSettled,
    retry: false,
    staleTime: 60_000,
  });

  useEffect(() => {
    if (!snapshotEnabled) {
      return;
    }
    if (scanBootstrapQuery.isError) {
      setInitialBootstrapSettled(true);
      return;
    }
    if (!scanBootstrapQuery.isSuccess) {
      return;
    }
    if (scanBootstrapQuery.data?.is_stale) {
      setInitialBootstrapSettled(true);
      return;
    }
    applyScanBootstrapSnapshot(scanBootstrapQuery.data);
    setInitialBootstrapSettled(true);
  }, [
    applyScanBootstrapSnapshot,
    scanBootstrapQuery.data,
    scanBootstrapQuery.isError,
    scanBootstrapQuery.isSuccess,
    snapshotEnabled,
  ]);

  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedFilters(filters);
    }, 300);
    return () => clearTimeout(timer);
  }, [filters]);

  const handleLoadScan = useCallback(
    async (scanId) => {
      setCurrentScanId(scanId);
      setBootstrappedScanId(null);
      setPage(1);

      if (snapshotEnabled) {
        try {
          const snapshot = await getScanBootstrap(scanId);
          if (!snapshot?.is_stale) {
            applyScanBootstrapSnapshot(snapshot, scanId);
            return;
          }
        } catch (error) {
          console.error('Scan bootstrap unavailable, falling back to live endpoints:', error);
        }
      }

      try {
        const status = await getScanStatus(scanId);
        setScanStatus(status.status);
      } catch (error) {
        console.error('Error loading scan:', error);
        setScanStatus('completed');
      }
    },
    [applyScanBootstrapSnapshot, snapshotEnabled]
  );

  const { data: universeStats, isLoading: statsLoading } = useQuery({
    queryKey: ['universeStats'],
    queryFn: getUniverseStats,
    enabled: initialQueriesEnabled,
    staleTime: 60_000,
  });

  const { data: scanHistory, refetch: refetchScans } = useQuery({
    queryKey: ['scanHistory'],
    queryFn: () => getScans({ limit: 20 }),
    enabled: initialQueriesEnabled || scanStatus === 'running' || scanStatus === 'queued',
    refetchInterval: scanStatus === 'running' ? 10000 : false,
    refetchIntervalInBackground: false,
    staleTime: 60_000,
  });

  useEffect(() => {
    if (!currentScanId && scanHistory?.scans?.length > 0) {
      const latestCompletedScan = scanHistory.scans.find(
        (scan) => scan.status === 'completed' || scan.status === 'cancelled'
      );
      if (latestCompletedScan) {
        handleLoadScan(latestCompletedScan.scan_id);
      }
    }
  }, [currentScanId, handleLoadScan, scanHistory]);

  const createScanMutation = useMutation({
    mutationFn: createScan,
    onSuccess: (data) => {
      setCurrentScanId(data.scan_id);
      setBootstrappedScanId(null);
      setScanStatus(data.status);
      refetchScans();
    },
  });

  const cancelScanMutation = useMutation({
    mutationFn: cancelScan,
    onSuccess: () => {
      setScanStatus('cancelled');
      refetchScans();
    },
  });

  const { data: statusData } = useQuery({
    queryKey: ['scanStatus', currentScanId],
    queryFn: () => getScanStatus(currentScanId),
    enabled: Boolean(currentScanId) && (scanStatus === 'running' || scanStatus === 'queued'),
    refetchInterval: (data) => {
      if (data?.status && data.status !== 'running' && data.status !== 'queued') {
        return false;
      }
      return 2000;
    },
    refetchIntervalInBackground: false,
    staleTime: 0,
    gcTime: 0,
  });

  const getApiFilterParams = useCallback(
    () => buildFilterParams(debouncedFilters, { page, perPage, sortBy, sortOrder }),
    [debouncedFilters, page, perPage, sortBy, sortOrder]
  );

  const stableFilterKey = useMemo(() => getStableFilterKey(debouncedFilters), [debouncedFilters]);

  const {
    data: resultsData,
    isLoading: resultsLoading,
    refetch: refetchResults,
  } = useQuery({
    queryKey: ['scanResults', currentScanId, page, perPage, sortBy, sortOrder, stableFilterKey],
    queryFn: () => getScanResults(currentScanId, getApiFilterParams()),
    enabled: Boolean(currentScanId) && (scanStatus === 'completed' || scanStatus === 'cancelled'),
    staleTime: 10 * 60 * 1000,
    gcTime: 30 * 60 * 1000,
    placeholderData: (previousData) => previousData,
  });

  useEffect(() => {
    if (!statusData) {
      return;
    }
    const previousStatus = scanStatus;
    setScanStatus(statusData.status);

    if (previousStatus !== 'completed' && statusData.status === 'completed') {
      setTimeout(() => refetchResults(), 500);
    }
  }, [refetchResults, scanStatus, statusData]);

  const { data: filterOptionsData } = useQuery({
    queryKey: ['filterOptions', currentScanId],
    queryFn: () => getFilterOptions(currentScanId),
    enabled: Boolean(currentScanId) && (scanStatus === 'completed' || scanStatus === 'cancelled'),
    staleTime: 60_000,
  });

  const handleStartScan = () => {
    const criteria = { include_vcp: includeVcp };
    if (selectedScreeners.includes('custom')) {
      criteria.custom_filters = customFilters;
    }
    const scanRequest = {
      universe,
      screeners: selectedScreeners,
      composite_method: compositeMethod,
      criteria,
    };
    if (universe === 'test') {
      scanRequest.symbols = TEST_SYMBOLS;
    }
    createScanMutation.mutate(scanRequest);
  };

  const handleScreenerToggle = (screener) => {
    setSelectedScreeners((previous) => {
      if (previous.includes(screener)) {
        if (previous.length === 1) {
          return previous;
        }
        return previous.filter((item) => item !== screener);
      }
      return [...previous, screener];
    });
  };

  const handleSortChange = (field, nextOrder) => {
    setSortBy(field);
    setSortOrder(nextOrder);
    setPage(1);
  };

  const handlePerPageChange = (nextPerPage) => {
    setPerPage(nextPerPage);
    setPage(1);
  };

  const handleFilterChange = (newFilters) => {
    setFilters(newFilters);
    setPage(1);
  };

  const handleResetFilters = () => {
    setFilters(buildDefaultScanFilters());
    setPage(1);
    presetState.clearActivePreset();
  };

  const handleCancelScan = () => {
    if (currentScanId && window.confirm('Are you sure you want to cancel this scan?')) {
      cancelScanMutation.mutate(currentScanId);
    }
  };

  const handleExport = async () => {
    try {
      const exportParams = buildFilterParams(debouncedFilters, { sortBy, sortOrder });
      const blob = await exportScanResults(currentScanId, exportParams);
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `scan_results_${new Date().toISOString().slice(0, 10)}.csv`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Export failed:', error);
      alert('Failed to export results. Please try again.');
    }
  };

  const handleOpenChart = (symbol) => {
    setSelectedSymbol(symbol);
    setChartModalOpen(true);
  };

  const handleRowHover = useCallback(
    (symbol) => {
      queryClient.prefetchQuery({
        queryKey: priceHistoryKeys.symbol(symbol, '6mo'),
        queryFn: () => fetchPriceHistory(symbol, '6mo'),
        staleTime: PRICE_HISTORY_STALE_TIME,
      });
    },
    [queryClient]
  );

  useEffect(() => {
    if (!resultsData?.results || resultsData.results.length === 0) {
      return;
    }
    if (
      bootstrappedScanId === currentScanId &&
      page === 1 &&
      perPage === 50 &&
      sortBy === 'composite_score' &&
      sortOrder === 'desc' &&
      stableFilterKey === DEFAULT_FILTER_KEY
    ) {
      return;
    }

    const prioritySymbols = resultsData.results.slice(0, 5);
    prioritySymbols.forEach((result, index) => {
      setTimeout(() => {
        queryClient.prefetchQuery({
          queryKey: priceHistoryKeys.symbol(result.symbol, '6mo'),
          queryFn: () => fetchPriceHistory(result.symbol, '6mo'),
          staleTime: PRICE_HISTORY_STALE_TIME,
          retry: false,
        });
      }, index * 200);
    });

    const remainingSymbols = resultsData.results.slice(5, 20);
    if (remainingSymbols.length === 0) {
      return;
    }
    const prefetchRemaining = () => {
      remainingSymbols.forEach((result, index) => {
        setTimeout(() => {
          queryClient.prefetchQuery({
            queryKey: priceHistoryKeys.symbol(result.symbol, '6mo'),
            queryFn: () => fetchPriceHistory(result.symbol, '6mo'),
            staleTime: PRICE_HISTORY_STALE_TIME,
            retry: false,
          });
        }, index * 300);
      });
    };

    if ('requestIdleCallback' in window) {
      requestIdleCallback(prefetchRemaining);
    } else {
      setTimeout(prefetchRemaining, 2000);
    }
  }, [
    bootstrappedScanId,
    currentScanId,
    page,
    perPage,
    queryClient,
    resultsData?.results,
    sortBy,
    sortOrder,
    stableFilterKey,
  ]);

  if (!runtimeReady) {
    return (
      <Container maxWidth="xl" sx={{ mt: 2, mb: 2 }}>
        <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
          <CircularProgress />
        </Box>
      </Container>
    );
  }

  return (
    <Container maxWidth="xl" sx={{ pt: 1 }}>
      <ScanControlBar
        currentScanId={currentScanId}
        scanHistory={scanHistory}
        onLoadScan={handleLoadScan}
        universe={universe}
        onUniverseChange={setUniverse}
        universeStats={universeStats}
        statsLoading={statsLoading}
        selectedScreeners={selectedScreeners}
        onScreenerToggle={handleScreenerToggle}
        includeVcp={includeVcp}
        onIncludeVcpChange={setIncludeVcp}
        compositeMethod={compositeMethod}
        onCompositeMethodChange={setCompositeMethod}
        createScanPending={createScanMutation.isPending}
        scanStatus={scanStatus}
        onStartScan={handleStartScan}
        onCancelScan={handleCancelScan}
        cancelScanPending={cancelScanMutation.isPending}
        statusData={statusData}
        customFilters={customFilters}
        onCustomFiltersChange={setCustomFilters}
        createScanError={createScanMutation.error}
        cancelScanError={cancelScanMutation.error}
        testSymbolsCount={TEST_SYMBOLS.length}
      />

      {(scanStatus === 'completed' || scanStatus === 'cancelled') && (
        <FilterPanel
          filters={filters}
          onFilterChange={handleFilterChange}
          onReset={handleResetFilters}
          filterOptions={normalizeScanFilterOptions(filterOptionsData)}
          expanded={showFilters}
          onToggle={() => setShowFilters((previous) => !previous)}
          presets={presets}
          activePresetId={presetState.activePresetId}
          hasUnsavedChanges={presetState.hasUnsavedChanges()}
          presetsLoading={presetsLoading}
          presetsSaving={presetIsCreating || presetIsUpdating}
          onLoadPreset={presetState.handleLoadPreset}
          onSavePreset={presetState.handleOpenSaveDialog}
          onUpdatePreset={presetState.handleUpdatePreset}
          onRenamePreset={presetState.handleRenamePreset}
          onDeletePreset={presetState.handleDeletePreset}
          saveDialogOpen={presetState.saveDialogOpen}
          saveDialogMode={presetState.saveDialogMode}
          saveDialogInitialName={presetState.saveDialogInitialName}
          saveDialogInitialDescription={presetState.saveDialogInitialDescription}
          saveDialogError={presetState.saveDialogError}
          onSaveDialogClose={presetState.handleSaveDialogClose}
          onSaveDialogSave={presetState.handleSaveDialogSave}
        />
      )}

      {(scanStatus === 'completed' || scanStatus === 'cancelled') && (
        <ScanResultsSection
          resultsLoading={resultsLoading}
          resultsData={resultsData}
          filters={filters}
          onExport={handleExport}
          page={page}
          perPage={perPage}
          sortBy={sortBy}
          sortOrder={sortOrder}
          onPageChange={setPage}
          onPerPageChange={handlePerPageChange}
          onSortChange={handleSortChange}
          onOpenChart={handleOpenChart}
          onRowHover={handleRowHover}
          onRetry={refetchResults}
        />
      )}

      {!currentScanId && (
        <Paper sx={{ p: 5, textAlign: 'center' }}>
          <Typography variant="body1" color="text.secondary">
            Click &quot;Start Scan&quot; to begin scanning all stocks in your universe
          </Typography>
        </Paper>
      )}

      <ChartViewerModal
        open={chartModalOpen}
        onClose={() => setChartModalOpen(false)}
        initialSymbol={selectedSymbol}
        scanId={currentScanId}
        filters={debouncedFilters}
        sortBy={sortBy}
        sortOrder={sortOrder}
        currentPageResults={resultsData?.results || []}
      />
    </Container>
  );
}

export default ScanPage;
