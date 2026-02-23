import { useState, useEffect, useCallback, useMemo } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
  Container,
  Typography,
  Button,
  Box,
  Alert,
  CircularProgress,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  FormControlLabel,
  Checkbox,
  Paper,
  Chip,
  TextField,
  Grid,
  Collapse,
} from '@mui/material';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import DownloadIcon from '@mui/icons-material/Download';
import StopIcon from '@mui/icons-material/Stop';
import {
  createScan,
  getScanStatus,
  getScanResults,
  getUniverseStats,
  exportScanResults,
  getScans,
  cancelScan,
  getFilterOptions,
} from '../api/scans';
import ScanProgress from '../components/Scan/ScanProgress';
import ResultsTable from '../components/Scan/ResultsTable';
import FilterPanel from '../components/Scan/FilterPanel';
import ChartViewerModal from '../components/Scan/ChartViewerModal';
import { buildFilterParams, getStableFilterKey } from '../utils/filterUtils';
import { fetchPriceHistory, priceHistoryKeys, PRICE_HISTORY_STALE_TIME } from '../api/priceHistory';
import { useFilterPresets } from '../hooks/useFilterPresets';

// Test list of 20 popular stocks for quick testing
const TEST_SYMBOLS = [
  'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA',
  'META', 'TSLA', 'BRK.B', 'V', 'JPM',
  'WMT', 'MA', 'JNJ', 'PG', 'XOM',
  'UNH', 'HD', 'CVX', 'ABBV', 'KO'
];

function ScanPage() {
  // Scan state
  const [currentScanId, setCurrentScanId] = useState(null);
  const [scanStatus, setScanStatus] = useState(null);

  // Scan creation options
  const [universe, setUniverse] = useState('all');
  const [includeVcp, setIncludeVcp] = useState(true);

  // Multi-screener options
  const [selectedScreeners, setSelectedScreeners] = useState(['minervini', 'canslim', 'ipo', 'custom', 'volume_breakthrough', 'setup_engine']);
  const [compositeMethod, setCompositeMethod] = useState('weighted_average');

  // Custom screener filters
  const [customFilters, setCustomFilters] = useState({
    price_min: 20,
    price_max: 500,
    rs_rating_min: 75,
    volume_min: 1000000,
    market_cap_min: 1000000000,
    eps_growth_min: 20,
    sales_growth_min: 15,
    ma_alignment: true,
    min_score: 70,
  });

  // Results pagination and sorting
  const [page, setPage] = useState(1);
  const [perPage, setPerPage] = useState(50);
  const [sortBy, setSortBy] = useState('composite_score');
  const [sortOrder, setSortOrder] = useState('desc');

  // Filters - new compact structure
  const [filters, setFilters] = useState({
    // Text search
    symbolSearch: '',

    // Categorical
    stage: null,
    ratings: [],
    ibdIndustries: { values: [], mode: 'include' },
    gicsSectors: { values: [], mode: 'include' },

    // Volume & Market Cap
    minVolume: null,
    minMarketCap: null,

    // Score ranges
    compositeScore: { min: null, max: null },
    minerviniScore: { min: null, max: null },
    canslimScore: { min: null, max: null },
    ipoScore: { min: null, max: null },
    customScore: { min: null, max: null },
    volBreakthroughScore: { min: null, max: null },

    // Setup Engine
    seSetupScore: { min: null, max: null },
    seDistanceToPivot: { min: null, max: null },
    seBbSqueeze: { min: null, max: null },
    seVolumeVs50d: { min: null, max: null },
    seSetupReady: null,
    seRsLineNewHigh: null,

    // RS ranges
    rsRating: { min: null, max: null },
    rs1m: { min: null, max: null },
    rs3m: { min: null, max: null },
    rs12m: { min: null, max: null },

    // Price & Growth
    price: { min: null, max: null },
    adrPercent: { min: null, max: null },
    epsGrowth: { min: null, max: null },
    salesGrowth: { min: null, max: null },

    // VCP
    vcpScore: { min: null, max: null },
    vcpPivot: { min: null, max: null },
    vcpDetected: null,
    vcpReady: null,

    // Booleans
    maAlignment: null,
    passesTemplate: null,

    // Technical Filters - Performance (price change %)
    perfDay: { min: null, max: null },
    perfWeek: { min: null, max: null },
    perfMonth: { min: null, max: null },

    // Qullamaggie extended performance
    perf3m: { min: null, max: null },
    perf6m: { min: null, max: null },

    // Episodic Pivot metrics
    gapPercent: { min: null, max: null },
    volumeSurge: { min: null, max: null },

    // Technical Filters - EMA Distances
    ema10Distance: { min: null, max: null },
    ema20Distance: { min: null, max: null },
    ema50Distance: { min: null, max: null },

    // Technical Filters - 52-Week Distances
    week52HighDistance: { min: null, max: null },
    week52LowDistance: { min: null, max: null },

    // IPO Date Filter
    ipoAfter: null,
  });

  // Debounced filters for API calls (prevents rapid API spam during filter adjustments)
  const [debouncedFilters, setDebouncedFilters] = useState(filters);

  // Chart viewer modal
  const [chartModalOpen, setChartModalOpen] = useState(false);
  const [selectedSymbol, setSelectedSymbol] = useState(null);

  // Filter panel visibility
  const [showFilters, setShowFilters] = useState(false);

  // Filter presets state
  const [activePresetId, setActivePresetId] = useState(null);
  const [presetFiltersSnapshot, setPresetFiltersSnapshot] = useState(null);
  const [presetSortSnapshot, setPresetSortSnapshot] = useState(null);

  // Save preset dialog state
  const [saveDialogOpen, setSaveDialogOpen] = useState(false);
  const [saveDialogMode, setSaveDialogMode] = useState('save');
  const [saveDialogInitialName, setSaveDialogInitialName] = useState('');
  const [saveDialogInitialDescription, setSaveDialogInitialDescription] = useState('');
  const [saveDialogError, setSaveDialogError] = useState(null);

  // Query client for prefetching
  const queryClient = useQueryClient();

  // Filter presets hook
  const {
    presets,
    isLoading: presetsLoading,
    createPresetAsync,
    updatePresetAsync,
    deletePreset,
    isCreating: presetIsCreating,
    isUpdating: presetIsUpdating,
  } = useFilterPresets();

  // Debounce filter changes to prevent rapid API calls
  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedFilters(filters);
    }, 300); // 300ms debounce delay

    return () => clearTimeout(timer);
  }, [filters]);

  // Fetch universe stats
  const { data: universeStats, isLoading: statsLoading } = useQuery({
    queryKey: ['universeStats'],
    queryFn: getUniverseStats,
  });

  // Fetch scan history - only poll when a scan is actively running
  const { data: scanHistory, refetch: refetchScans } = useQuery({
    queryKey: ['scanHistory'],
    queryFn: () => getScans({ limit: 20 }),
    refetchInterval: scanStatus === 'running' ? 10000 : false, // Only poll when scan is running
    refetchIntervalInBackground: false, // Don't poll when tab not focused
  });

  // Auto-load most recent completed scan on initial mount
  useEffect(() => {
    if (!currentScanId && scanHistory?.scans?.length > 0) {
      const latestCompletedScan = scanHistory.scans.find(
        scan => scan.status === 'completed' || scan.status === 'cancelled'
      );
      if (latestCompletedScan) {
        handleLoadScan(latestCompletedScan.scan_id);
      }
    }
  }, [scanHistory]);

  // Create scan mutation
  const createScanMutation = useMutation({
    mutationFn: createScan,
    onSuccess: (data) => {
      setCurrentScanId(data.scan_id);
      setScanStatus(data.status);
      refetchScans(); // Refresh scan history
    },
  });

  // Cancel scan mutation
  const cancelScanMutation = useMutation({
    mutationFn: cancelScan,
    onSuccess: () => {
      setScanStatus('cancelled');
      refetchScans(); // Refresh scan history
    },
  });

  // Poll scan status while running/queued - stop when complete/failed/cancelled
  const { data: statusData, error: statusError } = useQuery({
    queryKey: ['scanStatus', currentScanId],
    queryFn: () => getScanStatus(currentScanId),
    enabled: !!currentScanId && (scanStatus === 'running' || scanStatus === 'queued'),
    refetchInterval: (data) => {
      // Stop polling when scan is no longer running
      if (data?.status && data.status !== 'running' && data.status !== 'queued') {
        return false;
      }
      return 2000; // Poll every 2 seconds while running
    },
    refetchIntervalInBackground: false, // Don't poll when tab not focused to save resources
    staleTime: 0, // Always fetch fresh data
    gcTime: 0, // Don't cache (cacheTime renamed to gcTime in v5)
  });

  // Update scan status when data changes
  useEffect(() => {
    if (statusData) {
      const prevStatus = scanStatus;
      setScanStatus(statusData.status);

      // If scan just completed, trigger results fetch
      if (prevStatus !== 'completed' && statusData.status === 'completed') {
        setTimeout(() => refetchResults(), 500); // Small delay to ensure DB is updated
      }
    }
  }, [statusData]);

  // Fetch filter options (industries, sectors) for the current scan
  const { data: filterOptionsData } = useQuery({
    queryKey: ['filterOptions', currentScanId],
    queryFn: () => getFilterOptions(currentScanId),
    enabled: !!currentScanId && (scanStatus === 'completed' || scanStatus === 'cancelled'),
    staleTime: 60000, // Cache for 1 minute
  });

  // Build filter params for API using shared utility (uses debounced filters)
  const getApiFilterParams = useCallback(
    () => buildFilterParams(debouncedFilters, { page, perPage, sortBy, sortOrder }),
    [debouncedFilters, page, perPage, sortBy, sortOrder]
  );

  // Memoize stable filter key to prevent unnecessary cache key changes
  const stableFilterKey = useMemo(
    () => getStableFilterKey(debouncedFilters),
    [debouncedFilters]
  );

  // Fetch scan results (uses debounced filters to prevent rapid API calls)
  const {
    data: resultsData,
    isLoading: resultsLoading,
    refetch: refetchResults,
  } = useQuery({
    queryKey: ['scanResults', currentScanId, page, perPage, sortBy, sortOrder, stableFilterKey],
    queryFn: () => getScanResults(currentScanId, getApiFilterParams()),
    enabled: !!currentScanId && (scanStatus === 'completed' || scanStatus === 'cancelled'),
    staleTime: 10 * 60 * 1000, // Cache for 10 minutes - results don't change often
    gcTime: 30 * 60 * 1000, // Keep in cache for 30 minutes
    placeholderData: (previousData) => previousData, // Use previous data while loading for smoother UX
  });

  // Handle scan creation
  const handleStartScan = () => {
    console.log('🔍 MULTI-SCREENER VERSION - selectedScreeners:', selectedScreeners, 'compositeMethod:', compositeMethod);
    const criteria = {
      include_vcp: includeVcp,
    };

    // Add custom filters if custom screener is selected
    if (selectedScreeners.includes('custom')) {
      criteria.custom_filters = customFilters;
    }

    // Map UI universe value to API parameters
    // For 'test', we send universe='test' with TEST_SYMBOLS
    // For exchange/index values (nyse, nasdaq, amex, sp500), the backend's
    // from_legacy() parser handles the conversion to typed UniverseDefinition
    const scanRequest = {
      universe: universe,
      screeners: selectedScreeners,
      composite_method: compositeMethod,
      criteria: criteria,
    };

    // Add custom symbols if test universe is selected
    if (universe === 'test') {
      scanRequest.symbols = TEST_SYMBOLS;
    }

    console.log('📤 Sending scan request:', scanRequest);
    createScanMutation.mutate(scanRequest);
  };

  // Handle screener toggle
  const handleScreenerToggle = (screener) => {
    setSelectedScreeners(prev => {
      if (prev.includes(screener)) {
        // Don't allow unchecking if it's the last one
        if (prev.length === 1) return prev;
        return prev.filter(s => s !== screener);
      } else {
        return [...prev, screener];
      }
    });
  };

  // Handle sort change
  const handleSortChange = (field, order) => {
    setSortBy(field);
    setSortOrder(order);
    setPage(1); // Reset to first page on sort change
  };

  const handlePerPageChange = (nextPerPage) => {
    setPerPage(nextPerPage);
    setPage(1);
  };

  // Handle filter change
  const handleFilterChange = (newFilters) => {
    setFilters(newFilters);
    setPage(1); // Reset to first page on filter change
  };

  // Handle filter reset
  const handleResetFilters = () => {
    setFilters({
      symbolSearch: '',
      stage: null,
      ratings: [],
      ibdIndustries: { values: [], mode: 'include' },
      gicsSectors: { values: [], mode: 'include' },
      compositeScore: { min: null, max: null },
      minerviniScore: { min: null, max: null },
      canslimScore: { min: null, max: null },
      ipoScore: { min: null, max: null },
      customScore: { min: null, max: null },
      volBreakthroughScore: { min: null, max: null },
      // Setup Engine
      seSetupScore: { min: null, max: null },
      seDistanceToPivot: { min: null, max: null },
      seBbSqueeze: { min: null, max: null },
      seVolumeVs50d: { min: null, max: null },
      seSetupReady: null,
      seRsLineNewHigh: null,
      rsRating: { min: null, max: null },
      rs1m: { min: null, max: null },
      rs3m: { min: null, max: null },
      rs12m: { min: null, max: null },
      price: { min: null, max: null },
      adrPercent: { min: null, max: null },
      epsGrowth: { min: null, max: null },
      salesGrowth: { min: null, max: null },
      vcpScore: { min: null, max: null },
      vcpPivot: { min: null, max: null },
      vcpDetected: null,
      vcpReady: null,
      maAlignment: null,
      passesTemplate: null,
      // Technical Filters
      perfDay: { min: null, max: null },
      perfWeek: { min: null, max: null },
      perfMonth: { min: null, max: null },
      // Qullamaggie extended performance
      perf3m: { min: null, max: null },
      perf6m: { min: null, max: null },
      // Episodic Pivot metrics
      gapPercent: { min: null, max: null },
      volumeSurge: { min: null, max: null },
      ema10Distance: { min: null, max: null },
      ema20Distance: { min: null, max: null },
      ema50Distance: { min: null, max: null },
      week52HighDistance: { min: null, max: null },
      week52LowDistance: { min: null, max: null },
      // IPO Date Filter
      ipoAfter: null,
    });
    setPage(1);
    // Clear active preset when resetting filters
    setActivePresetId(null);
    setPresetFiltersSnapshot(null);
    setPresetSortSnapshot(null);
  };

  // Compute whether current filters differ from the loaded preset
  const hasUnsavedChanges = useCallback(() => {
    if (!activePresetId || !presetFiltersSnapshot) return false;

    // Compare filters
    const filtersChanged = JSON.stringify(filters) !== JSON.stringify(presetFiltersSnapshot);

    // Compare sort settings
    const sortChanged = presetSortSnapshot &&
      (sortBy !== presetSortSnapshot.sortBy || sortOrder !== presetSortSnapshot.sortOrder);

    return filtersChanged || sortChanged;
  }, [activePresetId, filters, sortBy, sortOrder, presetFiltersSnapshot, presetSortSnapshot]);

  // Handle loading a filter preset
  const handleLoadPreset = (presetId) => {
    if (!presetId) {
      // Clear preset selection
      setActivePresetId(null);
      setPresetFiltersSnapshot(null);
      setPresetSortSnapshot(null);
      return;
    }

    const preset = presets.find(p => p.id === presetId);
    if (preset) {
      // Load filters from preset
      setFilters(preset.filters);
      setSortBy(preset.sort_by);
      setSortOrder(preset.sort_order);

      // Store snapshot for comparison
      setActivePresetId(presetId);
      setPresetFiltersSnapshot(preset.filters);
      setPresetSortSnapshot({ sortBy: preset.sort_by, sortOrder: preset.sort_order });

      setPage(1);
    }
  };

  // Handle opening save preset dialog
  const handleOpenSaveDialog = () => {
    setSaveDialogMode('save');
    setSaveDialogInitialName('');
    setSaveDialogInitialDescription('');
    setSaveDialogError(null);
    setSaveDialogOpen(true);
  };

  // Handle updating the current preset
  const handleUpdatePreset = async () => {
    if (!activePresetId) return;

    try {
      await updatePresetAsync({
        presetId: activePresetId,
        updates: {
          filters: filters,
          sort_by: sortBy,
          sort_order: sortOrder,
        },
      });

      // Update snapshot after successful save
      setPresetFiltersSnapshot(filters);
      setPresetSortSnapshot({ sortBy, sortOrder });
    } catch (error) {
      console.error('Failed to update preset:', error);
      alert('Failed to update preset. Please try again.');
    }
  };

  // Handle renaming a preset
  const handleRenamePreset = (presetId) => {
    const preset = presets.find(p => p.id === presetId);
    if (preset) {
      setSaveDialogMode('rename');
      setSaveDialogInitialName(preset.name);
      setSaveDialogInitialDescription(preset.description || '');
      setSaveDialogError(null);
      setSaveDialogOpen(true);
    }
  };

  // Handle deleting a preset
  const handleDeletePreset = (presetId) => {
    deletePreset(presetId);
    if (activePresetId === presetId) {
      setActivePresetId(null);
      setPresetFiltersSnapshot(null);
      setPresetSortSnapshot(null);
    }
  };

  // Handle save dialog close
  const handleSaveDialogClose = () => {
    setSaveDialogOpen(false);
    setSaveDialogError(null);
  };

  // Handle save dialog save action
  const handleSaveDialogSave = async (name, description) => {
    setSaveDialogError(null);

    try {
      if (saveDialogMode === 'save') {
        // Create new preset
        const newPreset = await createPresetAsync({
          name,
          description: description || null,
          filters: filters,
          sort_by: sortBy,
          sort_order: sortOrder,
        });

        // Set as active preset
        setActivePresetId(newPreset.id);
        setPresetFiltersSnapshot(filters);
        setPresetSortSnapshot({ sortBy, sortOrder });
      } else {
        // Rename existing preset
        await updatePresetAsync({
          presetId: activePresetId,
          updates: { name, description: description || null },
        });
      }

      setSaveDialogOpen(false);
    } catch (error) {
      console.error('Failed to save preset:', error);
      const errorMessage = error.response?.data?.detail || 'Failed to save preset';
      setSaveDialogError(errorMessage);
    }
  };

  // Handle loading a previous scan
  const handleLoadScan = async (scanId) => {
    setCurrentScanId(scanId);
    setPage(1); // Reset to first page

    // Fetch the scan status
    try {
      const status = await getScanStatus(scanId);
      setScanStatus(status.status);
    } catch (error) {
      console.error('Error loading scan:', error);
      setScanStatus('completed'); // Fallback to completed
    }
  };

  // Handle cancel scan
  const handleCancelScan = () => {
    if (currentScanId && window.confirm('Are you sure you want to cancel this scan?')) {
      cancelScanMutation.mutate(currentScanId);
    }
  };

  // Handle export (uses debounced filters for consistency)
  const handleExport = async () => {
    try {
      const exportParams = buildFilterParams(debouncedFilters, { sortBy, sortOrder });
      const blob = await exportScanResults(currentScanId, exportParams);

      // Create download link
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

  // Handle opening chart viewer modal
  const handleOpenChart = (symbol) => {
    setSelectedSymbol(symbol);
    setChartModalOpen(true);
  };

  // Prefetch price history on row hover (Phase 2 optimization)
  const handleRowHover = useCallback((symbol) => {
    queryClient.prefetchQuery({
      queryKey: priceHistoryKeys.symbol(symbol, '6mo'),
      queryFn: () => fetchPriceHistory(symbol, '6mo'),
      staleTime: PRICE_HISTORY_STALE_TIME,
    });
  }, [queryClient]);

  // Prefetch visible page symbols when results load (Phase 3 optimization)
  useEffect(() => {
    if (!resultsData?.results || resultsData.results.length === 0) return;

    // Prefetch first 5 symbols immediately (highest priority - most likely to be clicked)
    // Reduced from 10 to avoid overwhelming yfinance API
    const prioritySymbols = resultsData.results.slice(0, 5);
    prioritySymbols.forEach((result, index) => {
      setTimeout(() => {
        queryClient.prefetchQuery({
          queryKey: priceHistoryKeys.symbol(result.symbol, '6mo'),
          queryFn: () => fetchPriceHistory(result.symbol, '6mo'),
          staleTime: PRICE_HISTORY_STALE_TIME,
          retry: false, // Don't retry failed prefetches (symbol may not exist)
        });
      }, index * 200); // Stagger by 200ms to avoid rate limiting
    });

    // Prefetch remaining symbols with lower priority (staggered to avoid network congestion)
    // Only prefetch next 15 symbols (20 total) to avoid too many requests
    const remainingSymbols = resultsData.results.slice(5, 20);
    if (remainingSymbols.length > 0) {
      const prefetchRemaining = () => {
        remainingSymbols.forEach((result, index) => {
          setTimeout(() => {
            queryClient.prefetchQuery({
              queryKey: priceHistoryKeys.symbol(result.symbol, '6mo'),
              queryFn: () => fetchPriceHistory(result.symbol, '6mo'),
              staleTime: PRICE_HISTORY_STALE_TIME,
              retry: false, // Don't retry failed prefetches
            });
          }, index * 300); // Stagger by 300ms to avoid overwhelming the network
        });
      };

      // Use requestIdleCallback for non-critical prefetching if available
      if ('requestIdleCallback' in window) {
        requestIdleCallback(prefetchRemaining);
      } else {
        setTimeout(prefetchRemaining, 2000); // Delay start by 2s
      }
    }
  }, [resultsData?.results, queryClient]);

  // Format scan label for dropdown
  const formatScanLabel = (scan) => {
    let universeLabel;

    // Prefer structured universe_type if available (post-migration scans)
    if (scan.universe_type) {
      switch (scan.universe_type) {
        case 'all':
          universeLabel = 'All';
          break;
        case 'exchange':
          universeLabel = scan.universe_exchange || 'Exchange';
          break;
        case 'index':
          universeLabel = scan.universe_index === 'SP500' ? 'S&P500' : (scan.universe_index || 'Index');
          break;
        case 'custom':
          universeLabel = `Custom (${scan.universe_symbols_count || '?'})`;
          break;
        case 'test':
          universeLabel = `Test (${scan.universe_symbols_count || '?'})`;
          break;
        default:
          universeLabel = scan.universe_type;
      }
    } else {
      // Fallback for pre-migration scans using legacy universe string
      const u = (scan.universe || '').toLowerCase();
      universeLabel = u === 'custom' ? 'Test' :
                      u === 'sp500' ? 'S&P500' :
                      u === 'all' ? 'All' :
                      u === 'all stocks' ? 'All' :
                      scan.universe ? scan.universe.toUpperCase() : 'Unknown';
    }

    const dateStr = new Date(scan.started_at).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
    return `${universeLabel} (${scan.passed_stocks}/${scan.total_stocks}) - ${dateStr}`;
  };

  return (
    <Container maxWidth="xl" sx={{ pt: 1 }}>
      {/* Compact Scan Control Bar */}
      <Paper elevation={1} sx={{ p: 1.5, mb: 2 }}>
        <Box sx={{ display: 'flex', gap: 1.5, alignItems: 'center', flexWrap: 'wrap' }}>
          {/* Previous Scans Dropdown */}
          <FormControl size="small" sx={{ minWidth: 200 }}>
            <InputLabel id="prev-scan-label">Previous Scans</InputLabel>
            <Select
              labelId="prev-scan-label"
              value={currentScanId || ''}
              label="Previous Scans"
              onChange={(e) => e.target.value && handleLoadScan(e.target.value)}
            >
              <MenuItem value="">
                <em>New Scan</em>
              </MenuItem>
              {scanHistory?.scans?.map((scan) => (
                <MenuItem key={scan.scan_id} value={scan.scan_id}>
                  {formatScanLabel(scan)}
                </MenuItem>
              ))}
            </Select>
          </FormControl>

          <Box sx={{ borderLeft: 1, borderColor: 'divider', height: 32, mx: 0.5 }} />

          {/* Universe */}
          <FormControl size="small" sx={{ minWidth: 150 }}>
            <InputLabel id="universe-label">Universe</InputLabel>
            <Select
              labelId="universe-label"
              value={universe}
              label="Universe"
              onChange={(e) => setUniverse(e.target.value)}
              disabled={createScanMutation.isPending || scanStatus === 'running'}
            >
              <MenuItem value="test">Test (20)</MenuItem>
              <MenuItem value="sp500">
                S&P 500{universeStats?.by_exchange ? ` (${universeStats.sp500 || '~500'})` : ''}
              </MenuItem>
              <MenuItem value="nyse">
                NYSE{universeStats?.by_exchange?.NYSE ? ` (${universeStats.by_exchange.NYSE})` : ''}
              </MenuItem>
              <MenuItem value="nasdaq">
                NASDAQ{universeStats?.by_exchange?.NASDAQ ? ` (${universeStats.by_exchange.NASDAQ})` : ''}
              </MenuItem>
              <MenuItem value="amex">
                AMEX{universeStats?.by_exchange?.AMEX ? ` (${universeStats.by_exchange.AMEX})` : ''}
              </MenuItem>
              <MenuItem value="all">
                All{universeStats?.active ? ` (${universeStats.active})` : ''}
              </MenuItem>
            </Select>
          </FormControl>

          {/* Screening Strategies - Compact Chips */}
          <Box sx={{ display: 'flex', gap: 0.5, alignItems: 'center', flexWrap: 'wrap' }}>
            <Box sx={{ fontSize: '11px', color: 'text.secondary', mr: 0.5 }}>Strategies:</Box>
            {[
              { id: 'minervini', label: 'Min' },
              { id: 'canslim', label: 'CAN' },
              { id: 'ipo', label: 'IPO' },
              { id: 'custom', label: 'Cust' },
              { id: 'volume_breakthrough', label: 'VolB' },
              { id: 'setup_engine', label: 'Setup' },
            ].map((screener) => (
              <Chip
                key={screener.id}
                label={screener.label}
                size="small"
                variant={selectedScreeners.includes(screener.id) ? 'filled' : 'outlined'}
                color={selectedScreeners.includes(screener.id) ? 'primary' : 'default'}
                onClick={() => handleScreenerToggle(screener.id)}
                disabled={createScanMutation.isPending || scanStatus === 'running'}
                sx={{ height: 24, fontSize: '10px' }}
              />
            ))}
          </Box>

          {/* VCP Toggle */}
          <FormControlLabel
            control={
              <Checkbox
                checked={includeVcp}
                onChange={(e) => setIncludeVcp(e.target.checked)}
                disabled={createScanMutation.isPending || scanStatus === 'running'}
                size="small"
              />
            }
            label={<Box sx={{ fontSize: '11px' }}>VCP</Box>}
            sx={{ mr: 0 }}
          />

          {/* Composite Method - only show if multiple screeners */}
          {selectedScreeners.length > 1 && (
            <FormControl size="small" sx={{ minWidth: 100 }}>
              <InputLabel id="composite-method-label">Method</InputLabel>
              <Select
                labelId="composite-method-label"
                value={compositeMethod}
                label="Method"
                onChange={(e) => setCompositeMethod(e.target.value)}
                disabled={createScanMutation.isPending || scanStatus === 'running'}
              >
                <MenuItem value="weighted_average">Avg</MenuItem>
                <MenuItem value="maximum">Max</MenuItem>
                <MenuItem value="minimum">Min</MenuItem>
              </Select>
            </FormControl>
          )}

          <Box sx={{ flexGrow: 1 }} />

          {/* Stock count */}
          <Box sx={{ fontSize: '11px', color: 'text.secondary' }}>
            {universe === 'test' ? `${TEST_SYMBOLS.length} stocks` :
             statsLoading ? '...' :
             !universeStats ? '' :
             universe === 'sp500' ? `${universeStats.sp500 || '~500'} stocks` :
             universe === 'nyse' && universeStats.by_exchange?.NYSE ? `${universeStats.by_exchange.NYSE} stocks` :
             universe === 'nasdaq' && universeStats.by_exchange?.NASDAQ ? `${universeStats.by_exchange.NASDAQ} stocks` :
             universe === 'amex' && universeStats.by_exchange?.AMEX ? `${universeStats.by_exchange.AMEX} stocks` :
             `${universeStats.active} stocks`}
          </Box>

          {/* Start/Cancel Button */}
          {scanStatus === 'running' ? (
            <Button
              variant="outlined"
              color="error"
              size="small"
              startIcon={cancelScanMutation.isPending ? <CircularProgress size={14} /> : <StopIcon />}
              onClick={handleCancelScan}
              disabled={cancelScanMutation.isPending}
            >
              Cancel
            </Button>
          ) : (
            <Button
              variant="contained"
              size="small"
              startIcon={createScanMutation.isPending ? <CircularProgress size={14} /> : <PlayArrowIcon />}
              onClick={handleStartScan}
              disabled={createScanMutation.isPending}
            >
              Scan
            </Button>
          )}

          {/* Inline Scan Progress */}
          {currentScanId && statusData && (
            <ScanProgress
              status={statusData.status}
              progress={statusData.progress}
              totalStocks={statusData.total_stocks}
              completedStocks={statusData.completed_stocks}
              passedStocks={statusData.passed_stocks}
              etaSeconds={statusData.eta_seconds}
            />
          )}
        </Box>

        {/* Custom Screener Configuration - Collapsible */}
        <Collapse in={selectedScreeners.includes('custom')}>
          <Box sx={{ mt: 1.5, pt: 1.5, borderTop: 1, borderColor: 'divider' }}>
            <Grid container spacing={1}>
              <Grid item xs={6} sm={4} md={2}>
                <TextField
                  label="Min Price"
                  type="number"
                  value={customFilters.price_min}
                  onChange={(e) => setCustomFilters({...customFilters, price_min: Number(e.target.value)})}
                  disabled={createScanMutation.isPending || scanStatus === 'running'}
                  fullWidth
                  size="small"
                />
              </Grid>
              <Grid item xs={6} sm={4} md={2}>
                <TextField
                  label="Max Price"
                  type="number"
                  value={customFilters.price_max}
                  onChange={(e) => setCustomFilters({...customFilters, price_max: Number(e.target.value)})}
                  disabled={createScanMutation.isPending || scanStatus === 'running'}
                  fullWidth
                  size="small"
                />
              </Grid>
              <Grid item xs={6} sm={4} md={2}>
                <TextField
                  label="Min RS"
                  type="number"
                  value={customFilters.rs_rating_min}
                  onChange={(e) => setCustomFilters({...customFilters, rs_rating_min: Number(e.target.value)})}
                  disabled={createScanMutation.isPending || scanStatus === 'running'}
                  fullWidth
                  size="small"
                  inputProps={{ min: 0, max: 100 }}
                />
              </Grid>
              <Grid item xs={6} sm={4} md={2}>
                <TextField
                  label="Min Vol"
                  type="number"
                  value={customFilters.volume_min}
                  onChange={(e) => setCustomFilters({...customFilters, volume_min: Number(e.target.value)})}
                  disabled={createScanMutation.isPending || scanStatus === 'running'}
                  fullWidth
                  size="small"
                />
              </Grid>
              <Grid item xs={6} sm={4} md={2}>
                <TextField
                  label="Min EPS %"
                  type="number"
                  value={customFilters.eps_growth_min}
                  onChange={(e) => setCustomFilters({...customFilters, eps_growth_min: Number(e.target.value)})}
                  disabled={createScanMutation.isPending || scanStatus === 'running'}
                  fullWidth
                  size="small"
                />
              </Grid>
              <Grid item xs={6} sm={4} md={2}>
                <TextField
                  label="Min Sales %"
                  type="number"
                  value={customFilters.sales_growth_min}
                  onChange={(e) => setCustomFilters({...customFilters, sales_growth_min: Number(e.target.value)})}
                  disabled={createScanMutation.isPending || scanStatus === 'running'}
                  fullWidth
                  size="small"
                />
              </Grid>
            </Grid>
          </Box>
        </Collapse>

        {/* Error/Warning Messages */}
        {createScanMutation.isError && (
          <Alert severity="error" sx={{ mt: 1 }}>
            Error: {createScanMutation.error.message}
          </Alert>
        )}
        {cancelScanMutation.isError && (
          <Alert severity="error" sx={{ mt: 1 }}>
            Error: {cancelScanMutation.error.message}
          </Alert>
        )}
        {scanStatus === 'cancelled' && (
          <Alert severity="warning" sx={{ mt: 1 }}>
            Scan cancelled. Showing partial results.
          </Alert>
        )}
      </Paper>

      {/* Filter Panel (only show when scan is completed or cancelled) */}
      {(scanStatus === 'completed' || scanStatus === 'cancelled') && (
        <FilterPanel
          filters={filters}
          onFilterChange={handleFilterChange}
          onReset={handleResetFilters}
          filterOptions={{
            ibdIndustries: filterOptionsData?.ibd_industries || [],
            gicsSectors: filterOptionsData?.gics_sectors || [],
            ratings: filterOptionsData?.ratings || [],
          }}
          expanded={showFilters}
          onToggle={() => setShowFilters(!showFilters)}
          // Preset props
          presets={presets}
          activePresetId={activePresetId}
          hasUnsavedChanges={hasUnsavedChanges()}
          presetsLoading={presetsLoading}
          presetsSaving={presetIsCreating || presetIsUpdating}
          onLoadPreset={handleLoadPreset}
          onSavePreset={handleOpenSaveDialog}
          onUpdatePreset={handleUpdatePreset}
          onRenamePreset={handleRenamePreset}
          onDeletePreset={handleDeletePreset}
          // Dialog props
          saveDialogOpen={saveDialogOpen}
          saveDialogMode={saveDialogMode}
          saveDialogInitialName={saveDialogInitialName}
          saveDialogInitialDescription={saveDialogInitialDescription}
          saveDialogError={saveDialogError}
          onSaveDialogClose={handleSaveDialogClose}
          onSaveDialogSave={handleSaveDialogSave}
        />
      )}

      {/* Results Table */}
      {(scanStatus === 'completed' || scanStatus === 'cancelled') && (
        <>
          {resultsLoading ? (
            <Box sx={{ display: 'flex', justifyContent: 'center', p: 5 }}>
              <CircularProgress />
              <Typography sx={{ ml: 2 }}>Loading results...</Typography>
            </Box>
          ) : resultsData && resultsData.results && resultsData.results.length > 0 ? (
            <>
              <Box sx={{ mb: 2, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Typography variant="h6">
                  Results: {resultsData.total} stocks
                  {/* Check if any filter is active */}
                  {(filters.symbolSearch ||
                    filters.stage != null ||
                    filters.ratings?.length > 0 ||
                    filters.ibdIndustries?.values?.length > 0 ||
                    filters.gicsSectors?.values?.length > 0 ||
                    filters.compositeScore?.min != null ||
                    filters.compositeScore?.max != null ||
                    filters.minerviniScore?.min != null ||
                    filters.minerviniScore?.max != null ||
                    filters.vcpDetected != null ||
                    filters.maAlignment != null ||
                    filters.passesTemplate != null) ? ' (filtered)' : ''}
                </Typography>
                <Button
                  variant="outlined"
                  startIcon={<DownloadIcon />}
                  onClick={handleExport}
                  disabled={!resultsData.total}
                >
                  Export to CSV
                </Button>
              </Box>

              <ResultsTable
                results={resultsData.results}
                total={resultsData.total}
                page={page}
                perPage={perPage}
                sortBy={sortBy}
                sortOrder={sortOrder}
                onPageChange={setPage}
                onPerPageChange={handlePerPageChange}
                onSortChange={handleSortChange}
                onOpenChart={handleOpenChart}
                onRowHover={handleRowHover}
                loading={resultsLoading}
              />
            </>
          ) : (
            <Paper sx={{ p: 5, textAlign: 'center' }}>
              <Typography variant="body1" color="text.secondary">
                No results available. This could mean:
                <br />- The scan is still processing
                <br />- The scan failed
                <br />- All results were filtered out
              </Typography>
              <Button
                variant="outlined"
                onClick={() => refetchResults()}
                sx={{ mt: 2 }}
              >
                Retry Loading Results
              </Button>
            </Paper>
          )}
        </>
      )}

      {/* No scan yet message */}
      {!currentScanId && (
        <Paper sx={{ p: 5, textAlign: 'center' }}>
          <Typography variant="body1" color="text.secondary">
            Click "Start Scan" to begin scanning all stocks in your universe
          </Typography>
        </Paper>
      )}

      {/* Chart Viewer Modal */}
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
