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
  refreshScanCache,
} from '../../../api/scans';
import FilterPanel from '../components/FilterPanelContainer';
import ChartViewerModal from '../../../components/Scan/ChartViewerModal';
import { buildFilterParams, getStableFilterKey } from '../../../utils/filterUtils';
import {
  fetchPriceHistory,
  prefetchPriceHistoryBatch,
  priceHistoryKeys,
  PRICE_HISTORY_STALE_TIME,
} from '../../../api/priceHistory';
import { useFilterPresets } from '../../../hooks/useFilterPresets';
import { useRuntimeActivity } from '../../../hooks/useRuntimeActivity';
import { useRuntime } from '../../../contexts/RuntimeContext';
import { useStrategyProfile } from '../../../contexts/StrategyProfileContext';
import { DEFAULT_SCAN_DEFAULTS } from '../../../constants/scanDefaults';
import { buildDefaultScanFilters } from '../defaultFilters';
import { normalizeScanFilterOptions } from '../filterOptions';
import { DEFAULT_FILTER_KEY } from '../constants';
import ScanControlBar from '../components/ScanControlBar';
import ScanResultsSection from '../components/ScanResultsSection';
import { useScanFilterPresets } from '../hooks/useScanFilterPresets';
import { buildUniverseDef, parseLegacyUniverseDefault } from '../universeSelection';

const INITIAL_UNIVERSE_SELECTION = parseLegacyUniverseDefault(DEFAULT_SCAN_DEFAULTS.universe);
const SCAN_BLOCKING_ACTIVITY_STAGES = new Set(['prices', 'fundamentals']);
const SCAN_BLOCKING_ACTIVITY_STATUSES = new Set(['queued', 'running']);

function buildRefreshConflict(activity, market) {
  if (!market || market === 'TEST') {
    return null;
  }

  const marketActivity = (activity?.markets ?? []).find((item) => (
    item?.market === market
    && SCAN_BLOCKING_ACTIVITY_STAGES.has(item.stage_key)
    && SCAN_BLOCKING_ACTIVITY_STATUSES.has(item.status)
  ));
  if (!marketActivity) {
    return null;
  }

  const stageLabel = (marketActivity.stage_label || marketActivity.stage_key || 'refresh').toLowerCase();
  const statusLabel = marketActivity.status === 'queued' ? 'queued' : 'running';
  return {
    market,
    stageKey: marketActivity.stage_key,
    status: marketActivity.status,
    lifecycle: marketActivity.lifecycle ?? null,
    message: `${market} ${stageLabel} is ${statusLabel}. Wait for it to finish before starting a scan.`,
  };
}

function getMutationErrorMessage(error) {
  if (!error) {
    return null;
  }
  return error?.response?.data?.detail?.message
    || error?.response?.data?.message
    || error?.message
    || 'Failed to start scan.';
}

function getMutationErrorDetail(error) {
  const detail = error?.response?.data?.detail;
  return detail && typeof detail === 'object' ? detail : null;
}

function ScanPage() {
  const { runtimeReady, uiSnapshots, scanDefaults } = useRuntime();
  const { activeProfileDetail } = useStrategyProfile();
  const scanDefaultsAppliedRef = useRef(null);
  const hasAutoLoadedScanRef = useRef(false);
  const scanHistoryRef = useRef([]);
  const queryClient = useQueryClient();

  const [currentScanId, setCurrentScanId] = useState(null);
  const [scanStatus, setScanStatus] = useState(null);
  const [initialBootstrapSettled, setInitialBootstrapSettled] = useState(false);
  const [bootstrappedScanId, setBootstrappedScanId] = useState(null);
  const [universeMarket, setUniverseMarket] = useState(INITIAL_UNIVERSE_SELECTION.market);
  const [universeScope, setUniverseScope] = useState(INITIAL_UNIVERSE_SELECTION.scope);
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
  const runtimeActivityQuery = useRuntimeActivity({ enabled: runtimeReady });

  useEffect(() => {
    if (!runtimeReady) {
      return;
    }
    const nextDefaults = activeProfileDetail?.scan_defaults ?? scanDefaults ?? DEFAULT_SCAN_DEFAULTS;
    const profileKey = activeProfileDetail?.profile || 'runtime-default';
    if (scanDefaultsAppliedRef.current === profileKey) {
      return;
    }

    const parsed = parseLegacyUniverseDefault(nextDefaults.universe ?? DEFAULT_SCAN_DEFAULTS.universe);
    setUniverseMarket(parsed.market);
    setUniverseScope(parsed.scope);
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
      if (!scanId) {
        setCurrentScanId(null);
        setBootstrappedScanId(null);
        setScanStatus(null);
        setPage(1);
        hasAutoLoadedScanRef.current = true;
        return;
      }

      const knownStatus = scanHistoryRef.current.find((scan) => scan.scan_id === scanId)?.status ?? null;
      setCurrentScanId(scanId);
      setBootstrappedScanId(null);
      setScanStatus(knownStatus);
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
        setScanStatus(knownStatus);
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
    scanHistoryRef.current = scanHistory?.scans ?? [];
  }, [scanHistory?.scans]);

  useEffect(() => {
    if (hasAutoLoadedScanRef.current) {
      return;
    }
    if (!currentScanId && scanHistory?.scans?.length > 0) {
      const latestCompletedScan = scanHistory.scans.find(
        (scan) => scan.status === 'completed' || scan.status === 'cancelled'
      );
      if (latestCompletedScan) {
        hasAutoLoadedScanRef.current = true;
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
      setPage(1);
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

  const refreshScanCacheMutation = useMutation({
    mutationFn: (params) => refreshScanCache(params),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['runtimeActivity'] });
    },
  });

  const { data: statusData } = useQuery({
    queryKey: ['scanStatus', currentScanId],
    queryFn: () => getScanStatus(currentScanId),
    enabled: Boolean(currentScanId) && (scanStatus === 'running' || scanStatus === 'queued'),
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      if (status && status !== 'running' && status !== 'queued') {
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
  const normalizedFilterOptions = useMemo(
    () => normalizeScanFilterOptions(filterOptionsData),
    [filterOptionsData]
  );
  const refreshConflict = useMemo(
    () => buildRefreshConflict(runtimeActivityQuery.data, universeMarket),
    [runtimeActivityQuery.data, universeMarket]
  );
  const createScanError = useMemo(() => {
    const message = getMutationErrorMessage(createScanMutation.error);
    const detail = getMutationErrorDetail(createScanMutation.error);
    return message ? { message, detail } : null;
  }, [createScanMutation.error]);

  const handleStartScan = () => {
    if (refreshConflict) {
      return;
    }
    const universeDef = buildUniverseDef(universeMarket, universeScope);
    if (!universeDef) {
      return;
    }
    const criteria = { include_vcp: includeVcp };
    if (selectedScreeners.includes('custom')) {
      criteria.custom_filters = customFilters;
    }
    createScanMutation.mutate({
      universe_def: universeDef,
      screeners: selectedScreeners,
      composite_method: compositeMethod,
      criteria,
    });
  };

  const handleRefreshStaleData = useCallback((market) => {
    if (!market) {
      return;
    }
    refreshScanCacheMutation.mutate({ market, mode: 'full' });
  }, [refreshScanCacheMutation]);

  const handleUniverseMarketChange = useCallback((nextMarket) => {
    if (nextMarket === universeMarket) {
      return;
    }
    setUniverseMarket(nextMarket);
    setUniverseScope(null);
  }, [universeMarket]);

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

    const visibleSymbols = resultsData.results
      .slice(0, 20)
      .map((result) => result.symbol)
      .filter(Boolean);
    if (visibleSymbols.length === 0) {
      return;
    }

    let cancelled = false;
    const run = () => {
      if (cancelled) return;
      prefetchPriceHistoryBatch(queryClient, visibleSymbols, '6mo');
    };

    if ('requestIdleCallback' in window) {
      const handle = window.requestIdleCallback(run, { timeout: 1000 });
      return () => {
        cancelled = true;
        if (window.cancelIdleCallback) window.cancelIdleCallback(handle);
      };
    }
    const timer = setTimeout(run, 0);
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
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
        universeMarket={universeMarket}
        universeScope={universeScope}
        onUniverseMarketChange={handleUniverseMarketChange}
        onUniverseScopeChange={setUniverseScope}
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
        createScanError={createScanError}
        cancelScanError={cancelScanMutation.error}
        refreshConflict={refreshConflict}
        onRefreshStaleData={handleRefreshStaleData}
        refreshStaleDataPending={refreshScanCacheMutation.isPending}
        refreshStaleDataError={refreshScanCacheMutation.error}
      />

      {(scanStatus === 'completed' || scanStatus === 'cancelled') && (
        <FilterPanel
          filters={filters}
          onFilterChange={handleFilterChange}
          onReset={handleResetFilters}
          filterOptions={normalizedFilterOptions}
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
