import { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { Alert, Box, Button, CircularProgress, Container, Paper, Typography } from '@mui/material';
import {
  cancelScan,
  createScan,
  exportScanResultsQuery,
  getFilterOptions,
  getScanBootstrap,
  getScans,
  getScanStatus,
  getUniverseStats,
  refreshScanCache,
} from '../../../api/scans';
import FilterPanel from '../components/FilterPanelContainer';
import ChartViewerModal from '../../../components/Scan/ChartViewerModalLazy';
import {
  fetchPriceHistory,
  prefetchPriceHistoryBatch,
  priceHistoryKeys,
  PRICE_HISTORY_STALE_TIME,
} from '../../../api/priceHistory';
import { useFilterPresets } from '../../../hooks/useFilterPresets';
import { useRuntimeActivity } from '../../../hooks/useRuntimeActivity';
import { useRuntime } from '../../../contexts/RuntimeContext';
import { useMarket } from '../../../contexts/MarketContext';
import { useStrategyProfileData } from '../../../contexts/StrategyProfileContext';
import { DEFAULT_SCAN_DEFAULTS } from '../../../constants/scanDefaults';
import { buildDefaultScanFilters } from '../defaultFilters';
import { normalizeScanFilterOptions } from '../filterOptions';
import { DEFAULT_FILTER_KEY } from '../constants';
import ScanControlBar from '../components/ScanControlBar';
import ScanResultsSection from '../components/ScanResultsSection';
import GuidedFilterBuilderDialog from '../components/GuidedFilterBuilderDialog';
import {
  buildScanQueryRequest,
} from '../filterExpressionModel';
import {
  legacyFiltersToExpression,
} from '../legacyFilterExpression';
import {
  filterPresetsForOpportunityCapability,
  resolveLiveOpportunityCapability,
} from '../opportunityCapabilityPolicy';
import { useScanFilterPresets } from '../hooks/useScanFilterPresets';
import { useOpportunityCapabilityTransition } from '../hooks/useOpportunityCapabilityTransition';
import {
  createScanFilterQuery,
  stableScanFilterQueryKey,
} from '../hooks/useScanFilterQueryState';
import { useScanResultsController } from '../hooks/useScanResultsController';
import { canOfferLastPublished, describePublishedSource } from '../publishedSource';
import {
  buildUniverseDef,
  parseLegacyUniverseDefault,
} from '../universeSelection';
import {
  buildRuntimeUniverseSelections,
  getMarketScanBlocker,
  resolveUniverseScopeValue,
} from '../runtimeUniverseSelections';

const INITIAL_UNIVERSE_SELECTION = parseLegacyUniverseDefault(DEFAULT_SCAN_DEFAULTS.universe);
const DEFAULT_SCAN_FILTERS = buildDefaultScanFilters();
const DEFAULT_SCAN_EXPRESSION = legacyFiltersToExpression(DEFAULT_SCAN_FILTERS);
const DEFAULT_SCAN_QUERY = createScanFilterQuery(DEFAULT_SCAN_EXPRESSION);
const DEFAULT_SCAN_QUERY_KEY = stableScanFilterQueryKey(DEFAULT_SCAN_QUERY);

const ROW_HOVER_PREFETCH_DELAY_MS = 150;

// Create-scan rejections that describe the submitted request, not the system.
const REQUEST_SCOPED_ERROR_CODES = new Set(['snapshot_unavailable', 'market_data_stale']);

// "No market auto-loaded yet" marker for the scan auto-load ref.
const NO_MARKET_AUTOLOADED = Symbol('no-market-autoloaded');

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

function normalizeScanWarnings(warnings) {
  return Array.isArray(warnings) ? warnings : [];
}

function importedSymbolList(value) {
  return [...new Set(String(value || '').split(',')
    .map((symbol) => symbol.trim().replace(/^\$+/, '').toUpperCase())
    .filter((symbol) => /^[A-Z0-9][A-Z0-9.-]{0,19}$/.test(symbol)))]
    .slice(0, 500);
}

function ScanPage() {
  const { runtimeReady, uiSnapshots, scanDefaults, universeOptions, features } = useRuntime();
  const { selectedMarket: globalMarket } = useMarket();
  const { activeProfileDetail } = useStrategyProfileData();
  const scanDefaultsAppliedRef = useRef(null);
  // Market whose latest scan is already auto-loaded. Changing the global
  // market re-arms the auto-load for the new market. The sentinel can never
  // collide with a market code or with null (the no-provider market value).
  const autoLoadedMarketRef = useRef(NO_MARKET_AUTOLOADED);
  const globalMarketRef = useRef(globalMarket);
  globalMarketRef.current = globalMarket;
  const scanHistoryRef = useRef([]);
  const queryClient = useQueryClient();
  const [importedSymbols, setImportedSymbols] = useState(() => {
    if (typeof window === 'undefined') return [];
    return importedSymbolList(new URLSearchParams(window.location.search).get('symbols'));
  });

  const [currentScanId, setCurrentScanId] = useState(null);
  const [scanStatus, setScanStatus] = useState(null);
  // The last finished scan the user saw. While a newer scan is queued or
  // running, its results stay on screen instead of an empty table.
  const [lastFinishedScan, setLastFinishedScan] = useState(null);
  const currentScanIdRef = useRef(null);
  currentScanIdRef.current = currentScanId;
  // Bumped whenever the user picks a scan, so a create response that lands
  // afterwards cannot replace their choice. Auto-loading does not count.
  const scanSelectionRef = useRef(0);
  // A scan whose create response arrived after the user moved on; offered
  // via a notice so it never runs unseen.
  const [lateCreatedScanId, setLateCreatedScanId] = useState(null);
  // published_source per scan, kept with the scan's identity: some scans
  // (custom symbols, other markets) never appear in the market's history.
  const [scanSources, setScanSources] = useState({});
  const rememberScanSource = useCallback((scanId, source) => {
    if (!scanId || !source) {
      return;
    }
    setScanSources((previous) => (
      previous[scanId] === source ? previous : { ...previous, [scanId]: source }
    ));
  }, []);
  const scanPending = Boolean(currentScanId) && (scanStatus === 'queued' || scanStatus === 'running');
  const showingPreviousResults = scanPending
    && Boolean(lastFinishedScan)
    && lastFinishedScan.scanId !== currentScanId;
  const viewScanId = showingPreviousResults ? lastFinishedScan.scanId : currentScanId;
  const viewScanStatus = showingPreviousResults ? lastFinishedScan.status : scanStatus;
  const viewScanFinished = viewScanStatus === 'completed' || viewScanStatus === 'cancelled';
  const [initialBootstrapSettled, setInitialBootstrapSettled] = useState(false);
  const [bootstrappedScanId, setBootstrappedScanId] = useState(null);
  const [universeMarket, setUniverseMarket] = useState(INITIAL_UNIVERSE_SELECTION.market);
  const [universeScope, setUniverseScope] = useState(INITIAL_UNIVERSE_SELECTION.scope);
  const [includeVcp, setIncludeVcp] = useState(DEFAULT_SCAN_DEFAULTS.criteria.include_vcp);
  const [selectedScreeners, setSelectedScreeners] = useState(DEFAULT_SCAN_DEFAULTS.screeners);
  const [compositeMethod, setCompositeMethod] = useState(DEFAULT_SCAN_DEFAULTS.composite_method);
  const [customFilters, setCustomFilters] = useState(DEFAULT_SCAN_DEFAULTS.criteria.custom_filters);
  const groupedFilteringEnabled = features?.grouped_scan_filters === true;
  const {
    filters,
    draftExpression,
    sortBy,
    sortOrder,
    requestExpression,
    editQuickFilter,
    resetFilters,
    requestPage,
    requestPerPage,
    requestSort,
    requestQuery,
    displayedQuery,
    displayedResultsData,
    stableFilterKey,
    resultsLoading,
    resultsFetching,
    resultsError,
    refetchResults,
  } = useScanResultsController({
    currentScanId: viewScanId,
    scanStatus: viewScanStatus,
    initialFilters: DEFAULT_SCAN_FILTERS,
    initialExpression: DEFAULT_SCAN_EXPRESSION,
  });
  const opportunityCapability = resolveLiveOpportunityCapability(
    displayedResultsData,
  );
  const opportunityStateAvailable = opportunityCapability.available;
  const opportunityStateCapabilityResolved = opportunityCapability.resolved;
  const [logicBuilderOpen, setLogicBuilderOpen] = useState(false);
  const [chartModalOpen, setChartModalOpen] = useState(false);
  const [selectedSymbol, setSelectedSymbol] = useState(null);
  const [showFilters, setShowFilters] = useState(false);

  const snapshotEnabled = runtimeReady && Boolean(uiSnapshots?.scan);
  const initialQueriesEnabled = runtimeReady && (!snapshotEnabled || initialBootstrapSettled);
  const runtimeActivityQuery = useRuntimeActivity({ enabled: runtimeReady });
  const universeSelections = useMemo(
    () => buildRuntimeUniverseSelections(universeOptions, runtimeActivityQuery.data),
    [runtimeActivityQuery.data, universeOptions]
  );

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
    setUniverseScope(resolveUniverseScopeValue(parsed.market, parsed.scope, universeSelections));
    setIncludeVcp(nextDefaults.criteria?.include_vcp ?? DEFAULT_SCAN_DEFAULTS.criteria.include_vcp);
    setSelectedScreeners(nextDefaults.screeners ?? DEFAULT_SCAN_DEFAULTS.screeners);
    setCompositeMethod(nextDefaults.composite_method ?? DEFAULT_SCAN_DEFAULTS.composite_method);
    setCustomFilters(nextDefaults.criteria?.custom_filters ?? DEFAULT_SCAN_DEFAULTS.criteria.custom_filters);
    scanDefaultsAppliedRef.current = profileKey;
  }, [activeProfileDetail, runtimeReady, scanDefaults, universeSelections]);

  useEffect(() => {
    if (!universeMarket || !universeScope) {
      return;
    }
    const resolvedScope = resolveUniverseScopeValue(universeMarket, universeScope, universeSelections);
    if (resolvedScope !== universeScope) {
      setUniverseScope(resolvedScope);
    }
  }, [universeMarket, universeScope, universeSelections]);

  // Another market's results must never stand in for this market's scan.
  const lastFinishedMarketRef = useRef(globalMarket);
  useEffect(() => {
    if (lastFinishedMarketRef.current !== globalMarket) {
      lastFinishedMarketRef.current = globalMarket;
      setLastFinishedScan(null);
    }
  }, [globalMarket]);

  useEffect(() => {
    if (currentScanId && (scanStatus === 'completed' || scanStatus === 'cancelled')) {
      setLastFinishedScan((previous) => (
        previous?.scanId === currentScanId && previous?.status === scanStatus
          ? previous
          : { scanId: currentScanId, status: scanStatus }
      ));
    }
  }, [currentScanId, scanStatus]);

  const applyScanBootstrapSnapshot = useCallback(
    (snapshot, requestedScanId = null) => {
      const payload = snapshot?.payload ?? {};
      const payloadMarket = payload.market ?? null;
      const isLatestVariant = requestedScanId == null;
      queryClient.setQueryData(['universeStats'], payload.universe_stats ?? null);
      if (isLatestVariant && payloadMarket) {
        // The market-scoped latest list is exactly GET /scans?limit=20&market=.
        // Dating it by its publish time keeps the normal staleness refetch.
        // Explicit scan variants never go stale, so their lists are not used.
        queryClient.setQueryData(
          ['scanHistory', payloadMarket],
          payload.recent_scans ?? { scans: [] },
          { updatedAt: Date.parse(snapshot.published_at) || Date.now() }
        );
      }

      const selectedScanId =
        payload.selected_scan?.scan_id ??
        payload.results_page?.scan_id ??
        requestedScanId ??
        null;

      if (!selectedScanId) {
        return;
      }

      queryClient.setQueryData(['filterOptions', selectedScanId], payload.filter_options ?? null);
      if (payload.results_page != null) {
        queryClient.setQueryData(
          ['scanResultsQuery', selectedScanId, DEFAULT_SCAN_QUERY_KEY],
          {
            data: payload.results_page,
            request: DEFAULT_SCAN_QUERY,
            requestKey: DEFAULT_SCAN_QUERY_KEY,
            scanId: selectedScanId,
          }
        );
      }
      rememberScanSource(
        selectedScanId,
        payload.selected_scan_status?.published_source ?? payload.selected_scan?.published_source,
      );
      if (payload.selected_scan_status) {
        queryClient.setQueryData(['scanStatus', selectedScanId], payload.selected_scan_status);
      } else if (payload.selected_scan) {
        queryClient.setQueryData(['scanStatus', selectedScanId], payload.selected_scan);
      }
      setCurrentScanId(selectedScanId);
      setBootstrappedScanId(selectedScanId);
      setScanStatus(payload.selected_scan_status?.status ?? payload.selected_scan?.status ?? null);
    },
    [queryClient, rememberScanSource]
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

  const availablePresets = useMemo(() => (
    filterPresetsForOpportunityCapability(
      presets,
      opportunityStateCapabilityResolved,
      opportunityStateAvailable,
    )
  ), [
    opportunityStateAvailable,
    opportunityStateCapabilityResolved,
    presets,
  ]);

  const presetState = useScanFilterPresets({
    presets: availablePresets,
    createPresetAsync,
    updatePresetAsync,
    deletePreset,
    sortBy,
    sortOrder,
    applyQuery: requestQuery,
    expression: draftExpression,
  });
  const activePresetId = presetState.activePresetId;
  const clearActivePreset = presetState.clearActivePreset;
  const activePreset = useMemo(
    () => presets.find((preset) => preset.id === activePresetId),
    [activePresetId, presets],
  );
  const opportunityStateCleanupPending = useOpportunityCapabilityTransition({
    capabilityResolved: opportunityStateCapabilityResolved,
    available: opportunityStateAvailable,
    query: displayedQuery,
    activePreset,
    onSanitizedQuery: requestQuery,
    onUnsupportedPreset: clearActivePreset,
  });

  const scanBootstrapQuery = useQuery({
    // Market-scoped so the first load selects and seeds the selected market's
    // scans, never another market's newest scan.
    queryKey: ['scanBootstrap', 'latest', globalMarket ?? null],
    queryFn: () => getScanBootstrap(null, globalMarket ?? null),
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

  const handleLoadScan = useCallback(
    async (scanId) => {
      if (!scanId) {
        currentScanIdRef.current = null;
        setCurrentScanId(null);
        setBootstrappedScanId(null);
        setScanStatus(null);
        setLastFinishedScan(null);
        requestPage(1);
        autoLoadedMarketRef.current = globalMarketRef.current;
        return;
      }

      const knownScan = scanHistoryRef.current.find((scan) => scan.scan_id === scanId);
      const knownStatus = knownScan?.status ?? null;
      // Set now, not on the next render, so a response for a scan the user
      // has since replaced is recognised as late.
      currentScanIdRef.current = scanId;
      setCurrentScanId(scanId);
      setBootstrappedScanId(null);
      setScanStatus(knownStatus);
      requestPage(1);

      if (snapshotEnabled) {
        try {
          const snapshot = await getScanBootstrap(scanId);
          // The user may have picked another scan while this one loaded.
          if (currentScanIdRef.current !== scanId) {
            return;
          }
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
        queryClient.setQueryData(['scanStatus', scanId], status);
        rememberScanSource(scanId, status.published_source);
        if (currentScanIdRef.current !== scanId) {
          return;
        }
        setScanStatus(status.status);
      } catch (error) {
        console.error('Error loading scan:', error);
        if (currentScanIdRef.current === scanId) {
          setScanStatus(knownStatus);
        }
      }
    },
    [applyScanBootstrapSnapshot, queryClient, rememberScanSource, requestPage, snapshotEnabled]
  );

  const handleUserLoadScan = useCallback((scanId) => {
    scanSelectionRef.current += 1;
    setLateCreatedScanId(null);
    handleLoadScan(scanId);
  }, [handleLoadScan]);

  const { data: universeStats, isLoading: statsLoading } = useQuery({
    queryKey: ['universeStats'],
    queryFn: getUniverseStats,
    enabled: initialQueriesEnabled,
    staleTime: 60_000,
  });

  const { data: scanHistory, refetch: refetchScans } = useQuery({
    // Scoped to the global market selector: the previous-scans list only
    // shows the selected market's scans.
    queryKey: ['scanHistory', globalMarket],
    queryFn: () => getScans({ limit: 20, market: globalMarket ?? undefined }),
    enabled: initialQueriesEnabled || scanStatus === 'running' || scanStatus === 'queued',
    refetchInterval: scanStatus === 'running' ? 10000 : false,
    refetchIntervalInBackground: false,
    staleTime: 60_000,
  });

  useEffect(() => {
    scanHistoryRef.current = scanHistory?.scans ?? [];
  }, [scanHistory?.scans]);

  // Keep the loaded scan coherent with the global market selector: on first
  // load and on every market switch, load that market's newest finished scan
  // (the history list is already market-scoped).
  useEffect(() => {
    if (autoLoadedMarketRef.current === globalMarket) {
      return;
    }
    if (scanStatus === 'running' || scanStatus === 'queued') {
      return;
    }
    const scans = scanHistory?.scans ?? [];
    if (scans.length === 0) {
      return;
    }
    const latestCompletedScan = scans.find(
      (scan) => scan.status === 'completed' || scan.status === 'cancelled'
    );
    if (!latestCompletedScan) {
      // Nothing finished yet (e.g. a scan is still running) — leave the
      // marker unset so the next history refresh can auto-load.
      return;
    }
    autoLoadedMarketRef.current = globalMarket;
    if (latestCompletedScan.scan_id !== currentScanId) {
      handleLoadScan(latestCompletedScan.scan_id);
    }
  }, [currentScanId, globalMarket, handleLoadScan, scanHistory, scanStatus]);

  const createScanMutation = useMutation({
    mutationFn: createScan,
    onMutate: () => ({
      selection: scanSelectionRef.current,
      market: globalMarketRef.current,
    }),
    onSuccess: (data, _variables, context) => {
      rememberScanSource(data.scan_id, data.published_source);
      if (
        context?.selection !== scanSelectionRef.current
        || context?.market !== globalMarketRef.current
      ) {
        // The user picked another scan or market while this was in flight.
        setLateCreatedScanId(data.scan_id);
        refetchScans();
        return;
      }
      // Set now so a late response for the previous scan is recognised.
      currentScanIdRef.current = data.scan_id;
      setCurrentScanId(data.scan_id);
      setBootstrappedScanId(null);
      setScanStatus(data.status);
      requestPage(1);
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

  useEffect(() => {
    if (!statusData) {
      return;
    }
    const previousStatus = scanStatus;
    setScanStatus(statusData.status);
    rememberScanSource(currentScanId, statusData.published_source);

    if (previousStatus !== 'completed' && statusData.status === 'completed') {
      setTimeout(() => refetchResults(), 500);
    }
  }, [currentScanId, refetchResults, rememberScanSource, scanStatus, statusData]);

  // Warnings and the source notice describe the scan whose rows are shown.
  const viewHistoryScan = useMemo(
    () => scanHistory?.scans?.find((scan) => scan.scan_id === viewScanId),
    [scanHistory?.scans, viewScanId]
  );
  const scanWarnings = useMemo(() => {
    if (!viewScanId) {
      return [];
    }
    if (viewScanId === currentScanId && Array.isArray(statusData?.warnings)) {
      return statusData.warnings;
    }
    if (createScanMutation.data?.scan_id === viewScanId) {
      return normalizeScanWarnings(createScanMutation.data.warnings);
    }
    return normalizeScanWarnings(viewHistoryScan?.warnings);
  }, [
    createScanMutation.data,
    currentScanId,
    statusData?.warnings,
    viewHistoryScan?.warnings,
    viewScanId,
  ]);
  const publishedSourceNotice = describePublishedSource(
    (viewScanId && scanSources[viewScanId]) ?? viewHistoryScan?.published_source ?? null
  );

  const { data: filterOptionsData } = useQuery({
    queryKey: ['filterOptions', viewScanId],
    queryFn: () => getFilterOptions(viewScanId),
    enabled: Boolean(viewScanId) && viewScanFinished,
    staleTime: 60_000,
  });
  const normalizedFilterOptions = useMemo(
    () => normalizeScanFilterOptions(filterOptionsData),
    [filterOptionsData]
  );
  const refreshConflict = useMemo(
    () => getMarketScanBlocker(runtimeActivityQuery.data, universeMarket),
    [runtimeActivityQuery.data, universeMarket]
  );
  const createScanError = useMemo(() => {
    const message = getMutationErrorMessage(createScanMutation.error);
    const detail = getMutationErrorDetail(createScanMutation.error);
    return message ? { message, detail } : null;
  }, [createScanMutation.error]);

  // Handlers passed to ScanControlBar / FilterPanel stay referentially stable
  // so those memoized children skip re-rendering on unrelated page updates
  // (status polls, result fetches, chart modal state).
  const scanRequest = useMemo(() => {
    const universeDef = importedSymbols.length
      ? { type: 'custom', symbols: importedSymbols }
      : buildUniverseDef(universeMarket, universeScope, universeSelections);
    if (!universeDef) {
      return null;
    }
    const criteria = { include_vcp: includeVcp };
    if (selectedScreeners.includes('custom')) {
      criteria.custom_filters = customFilters;
    }
    return {
      universe_def: universeDef,
      screeners: selectedScreeners,
      composite_method: compositeMethod,
      criteria,
    };
  }, [
    compositeMethod,
    customFilters,
    importedSymbols,
    includeVcp,
    selectedScreeners,
    universeMarket,
    universeScope,
    universeSelections,
  ]);
  // These rejections were judged on the request's own symbols, so they stop
  // applying once the universe or criteria change. scan_already_active is
  // system state and keeps applying.
  const failedRequestChanged = JSON.stringify({ ...createScanMutation.variables, data_mode: undefined })
    !== JSON.stringify({ ...scanRequest, data_mode: undefined });
  const lastPublishedAvailable = canOfferLastPublished(
    refreshConflict,
    REQUEST_SCOPED_ERROR_CODES.has(createScanError?.detail?.code) && failedRequestChanged
      ? null
      : createScanError,
  );
  const startScanRequest = createScanMutation.mutate;
  const submitScan = useCallback((dataMode) => {
    // Only a current-data scan waits for the refresh; a last-published
    // read never computes.
    if (refreshConflict && dataMode !== 'last_published') {
      return;
    }
    if (!scanRequest) {
      return;
    }
    startScanRequest({
      ...scanRequest,
      ...(dataMode ? { data_mode: dataMode } : {}),
    });
  }, [refreshConflict, scanRequest, startScanRequest]);
  const handleStartScan = useCallback(() => submitScan(null), [submitScan]);
  const handleUseLastPublished = useCallback(() => submitScan('last_published'), [submitScan]);

  const refreshScanCacheRequest = refreshScanCacheMutation.mutate;
  const handleRefreshStaleData = useCallback((market) => {
    if (!market) {
      return;
    }
    refreshScanCacheRequest({ market, mode: 'full' });
  }, [refreshScanCacheRequest]);

  const handleUniverseMarketChange = useCallback((nextMarket) => {
    if (nextMarket === universeMarket) {
      return;
    }
    setUniverseMarket(nextMarket);
    setUniverseScope(null);
  }, [universeMarket]);

  const handleScreenerToggle = useCallback((screener) => {
    setSelectedScreeners((previous) => {
      if (previous.includes(screener)) {
        if (previous.length === 1) {
          return previous;
        }
        return previous.filter((item) => item !== screener);
      }
      return [...previous, screener];
    });
  }, []);

  const handleResetFilters = useCallback(() => {
    resetFilters(DEFAULT_SCAN_FILTERS);
    clearActivePreset();
  }, [clearActivePreset, resetFilters]);

  const cancelScanRequest = cancelScanMutation.mutate;
  const handleCancelScan = useCallback(() => {
    if (currentScanId && window.confirm('Are you sure you want to cancel this scan?')) {
      cancelScanRequest(currentScanId);
    }
  }, [cancelScanRequest, currentScanId]);

  const handleExport = async () => {
    try {
      const blob = await exportScanResultsQuery(
        viewScanId,
        buildScanQueryRequest(displayedQuery.expression, {
          sortBy: displayedQuery.sortBy,
          sortOrder: displayedQuery.sortOrder,
        }),
      );
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

  // The modal pages through the shown scan; close it if that scan changes
  // (e.g. previous results hand over to the finished new scan).
  useEffect(() => {
    setChartModalOpen(false);
  }, [viewScanId]);

  const handleOpenChart = useCallback((symbol) => {
    setSelectedSymbol(symbol);
    setChartModalOpen(true);
  }, []);

  const handleToggleFilters = useCallback(() => {
    setShowFilters((previous) => !previous);
  }, []);

  const handleOpenLogicBuilder = useCallback(() => {
    setLogicBuilderOpen(true);
  }, []);

  const handleClearCustomSymbols = useCallback(() => {
    const next = new URLSearchParams(window.location.search);
    next.delete('symbols');
    const search = next.toString();
    window.history.replaceState(null, '', `${window.location.pathname}${search ? `?${search}` : ''}`);
    setImportedSymbols([]);
  }, []);

  // Prefetch only once the pointer rests on a row; sweeping across the table
  // would otherwise fire one history request per row passed over.
  const hoverPrefetchTimerRef = useRef(null);
  const handleRowHover = useCallback(
    (symbol) => {
      clearTimeout(hoverPrefetchTimerRef.current);
      hoverPrefetchTimerRef.current = setTimeout(() => {
        queryClient.prefetchQuery({
          queryKey: priceHistoryKeys.symbol(symbol, '6mo'),
          queryFn: () => fetchPriceHistory(symbol, '6mo'),
          staleTime: PRICE_HISTORY_STALE_TIME,
        });
      }, ROW_HOVER_PREFETCH_DELAY_MS);
    },
    [queryClient]
  );
  useEffect(() => () => clearTimeout(hoverPrefetchTimerRef.current), []);

  useEffect(() => {
    if (!displayedResultsData?.results || displayedResultsData.results.length === 0) {
      return;
    }
    if (
      bootstrappedScanId === viewScanId &&
      displayedQuery.page === 1 &&
      displayedQuery.perPage === 50 &&
      displayedQuery.sortBy === 'composite_score' &&
      displayedQuery.sortOrder === 'desc' &&
      stableFilterKey === DEFAULT_FILTER_KEY
    ) {
      return;
    }

    const visibleSymbols = displayedResultsData.results
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
    viewScanId,
    queryClient,
    displayedResultsData?.results,
    displayedQuery.page,
    displayedQuery.perPage,
    displayedQuery.sortBy,
    displayedQuery.sortOrder,
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
        onLoadScan={handleUserLoadScan}
        universeMarket={universeMarket}
        universeScope={universeScope}
        onUniverseMarketChange={handleUniverseMarketChange}
        onUniverseScopeChange={setUniverseScope}
        universeStats={universeStats}
        universeSelections={universeSelections}
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
        scanWarnings={scanWarnings}
        customSymbols={importedSymbols}
        onClearCustomSymbols={handleClearCustomSymbols}
        lastPublishedAvailable={lastPublishedAvailable}
        onUseLastPublished={handleUseLastPublished}
      />

      {lateCreatedScanId && (
        <Alert
          severity="info"
          role="status"
          sx={{ mb: 2 }}
          action={(
            <Button color="inherit" size="small" onClick={() => handleUserLoadScan(lateCreatedScanId)}>
              Open scan
            </Button>
          )}
        >
          A scan you started was created while you were viewing another scan.
        </Alert>
      )}
      {showingPreviousResults && (
        <Alert severity="info" role="status" sx={{ mb: 2 }}>
          Showing your previous results until the new scan finishes.
        </Alert>
      )}
      {viewScanFinished && publishedSourceNotice && (
        <Alert severity={publishedSourceNotice.severity} role="status" sx={{ mb: 2 }}>
          {publishedSourceNotice.text}
        </Alert>
      )}

      {viewScanFinished && (
        <FilterPanel
          filters={filters}
          onFilterChange={editQuickFilter}
          onReset={handleResetFilters}
          filterOptions={normalizedFilterOptions}
          expanded={showFilters}
          onToggle={handleToggleFilters}
          presets={presetState.availablePresets}
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
          groupedFilteringEnabled={groupedFilteringEnabled}
          expression={draftExpression}
          onOpenLogicBuilder={handleOpenLogicBuilder}
        />
      )}

      {viewScanFinished && (
        <ScanResultsSection
          resultsLoading={resultsLoading || opportunityStateCleanupPending}
          resultsData={displayedResultsData}
          expression={displayedQuery.expression}
          resultsFetching={resultsFetching}
          resultsError={resultsError}
          onExport={handleExport}
          page={displayedQuery.page}
          perPage={displayedQuery.perPage}
          sortBy={displayedQuery.sortBy}
          sortOrder={displayedQuery.sortOrder}
          onPageChange={requestPage}
          onPerPageChange={requestPerPage}
          onSortChange={requestSort}
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
        scanId={viewScanId}
        filters={filters}
        expression={displayedQuery.expression}
        sortBy={displayedQuery.sortBy}
        sortOrder={displayedQuery.sortOrder}
        currentPageResults={displayedResultsData?.results || []}
      />

      {groupedFilteringEnabled && (
        <GuidedFilterBuilderDialog
          open={logicBuilderOpen}
          expression={draftExpression}
          onClose={() => setLogicBuilderOpen(false)}
          onApply={(nextExpression) => {
            requestExpression(nextExpression);
            setLogicBuilderOpen(false);
            presetState.clearActivePreset();
          }}
          filterOptions={normalizedFilterOptions}
          opportunityStateAvailable={opportunityStateAvailable}
        />
      )}
    </Container>
  );
}

export default ScanPage;
