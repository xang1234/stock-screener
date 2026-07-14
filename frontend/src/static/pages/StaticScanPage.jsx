import { useCallback, useEffect, useMemo, useReducer, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  Alert,
  Box,
  CircularProgress,
  Paper,
  Typography,
} from '@mui/material';
import FilterPanel from '../../components/Scan/FilterPanel';
import GuidedFilterBuilderDialog from '../../features/scan/components/GuidedFilterBuilderDialog';
import ResultsTable from '../../components/Scan/ResultsTable';
import { useStaticManifest, fetchStaticJson, resolveStaticMarketEntry } from '../dataClient';
import { useStaticChartIndex } from '../chartClient';
import {
  applyScanFilterDefaults,
  buildDefaultScanFilters,
} from '../../features/scan/defaultFilters';
import { normalizeScanFilterOptions } from '../../features/scan/filterOptions';
import { paginateStaticScanRows, sortStaticScanRows } from '../scanClient';
import StaticChartViewerModal from '../StaticChartViewerModal';
import ScreenSelector from '../components/ScreenSelector';
import { usePresetScreens, buildFiltersFromPreset } from '../hooks/usePresetScreens';
import { useStaticMarket } from '../StaticMarketContext';
import {
  annotateExpressionMatches,
} from '../../features/scan/filterExpressionEvaluator';
import {
  stableExpressionKey,
} from '../../features/scan/filterExpressionModel';
import {
  legacyFiltersToExpression,
} from '../../features/scan/legacyFilterExpression';
import {
  createFilterState,
  filterStateReducer,
  selectQuickFilters,
} from '../../features/scan/filterState';

const HYDRATION_BATCH_SIZE = 2;

function StaticScanPage() {
  const manifestQuery = useStaticManifest();
  const { selectedMarket } = useStaticMarket();
  const marketEntry = useMemo(
    () => resolveStaticMarketEntry(manifestQuery.data, selectedMarket),
    [manifestQuery.data, selectedMarket],
  );
  const scanManifestQuery = useQuery({
    queryKey: ['staticScanManifest', marketEntry.pages?.scan?.path],
    queryFn: () => fetchStaticJson(marketEntry.pages.scan.path),
    enabled: Boolean(marketEntry.pages?.scan?.path),
    staleTime: Infinity,
  });
  const chartIndexQuery = useStaticChartIndex(scanManifestQuery.data?.charts?.path);

  const [filterState, dispatchFilterState] = useReducer(
    filterStateReducer,
    { defaultFilters: buildDefaultScanFilters() },
    createFilterState,
  );
  const appliedExpression = filterState.committedExpression;
  const filters = useMemo(() => selectQuickFilters(filterState), [filterState]);
  const [logicBuilderOpen, setLogicBuilderOpen] = useState(false);
  const [showFilters, setShowFilters] = useState(true);
  const [page, setPage] = useState(1);
  const [perPage, setPerPage] = useState(50);
  const [sortBy, setSortBy] = useState('composite_score');
  const [sortOrder, setSortOrder] = useState('desc');
  const [chartModalOpen, setChartModalOpen] = useState(false);
  const [selectedChartSymbol, setSelectedChartSymbol] = useState(null);
  const [hydrationState, setHydrationState] = useState({
    status: 'idle',
    rows: [],
    loadedRows: 0,
    error: null,
  });
  const sectionDefaultExpanded = useMemo(
    () => ({
      fundamental: false,
      technical: false,
      rating: false,
    }),
    []
  );
  const manifestDefaultFilterValues = useMemo(
    () => scanManifestQuery.data?.default_filters ?? {},
    [scanManifestQuery.data?.default_filters]
  );
  const manifestDefaultFilters = useMemo(
    () => applyScanFilterDefaults(manifestDefaultFilterValues),
    [manifestDefaultFilterValues]
  );
  const manifestDefaultSortBy = scanManifestQuery.data?.sort?.field ?? 'composite_score';
  const manifestDefaultSortOrder = scanManifestQuery.data?.sort?.order ?? 'desc';
  const presetScreens = scanManifestQuery.data?.preset_screens;

  useEffect(() => {
    if (scanManifestQuery.data?.default_page_size) {
      setPerPage(scanManifestQuery.data.default_page_size);
    }
    if (scanManifestQuery.data?.sort?.field) {
      setSortBy(scanManifestQuery.data.sort.field);
      setSortOrder(scanManifestQuery.data.sort.order || 'desc');
    }
  }, [scanManifestQuery.data]);

  useEffect(() => {
    if (!scanManifestQuery.data) {
      return;
    }
    dispatchFilterState({
      type: 'reset-filters',
      defaultFilters: manifestDefaultFilters,
    });
  }, [manifestDefaultFilters, scanManifestQuery.data]);

  useEffect(() => {
    const manifest = scanManifestQuery.data;
    if (!manifest) {
      return undefined;
    }

    const initialRows = Array.isArray(manifest.initial_rows) ? manifest.initial_rows : [];
    const totalRows = manifest.rows_total || initialRows.length;
    const chunks = Array.isArray(manifest.chunks) ? manifest.chunks : [];
    const rowsBySymbol = new Map(initialRows.map((row) => [row.symbol, row]));
    const initialLoadedRows = Math.min(rowsBySymbol.size, totalRows);

    if (!chunks.length || initialLoadedRows >= totalRows) {
      setHydrationState({
        status: 'complete',
        rows: initialRows,
        loadedRows: initialLoadedRows,
        error: null,
      });
      return undefined;
    }

    setHydrationState({
      status: 'loading',
      rows: initialRows,
      loadedRows: initialLoadedRows,
      error: null,
    });

    let cancelled = false;
    const hydrateRows = async () => {
      try {
        for (let index = 0; index < chunks.length; index += HYDRATION_BATCH_SIZE) {
          const batch = chunks.slice(index, index + HYDRATION_BATCH_SIZE);
          const payloads = await Promise.all(batch.map((chunk) => fetchStaticJson(chunk.path)));
          if (cancelled) {
            return;
          }

          payloads.forEach((payload) => {
            (payload.rows || []).forEach((row) => {
              rowsBySymbol.set(row.symbol, row);
            });
          });

          setHydrationState({
            status: rowsBySymbol.size >= totalRows ? 'complete' : 'loading',
            rows: Array.from(rowsBySymbol.values()),
            loadedRows: Math.min(rowsBySymbol.size, totalRows),
            error: null,
          });
        }

        if (!cancelled) {
          setHydrationState({
            status: 'complete',
            rows: Array.from(rowsBySymbol.values()),
            loadedRows: Math.min(rowsBySymbol.size, totalRows),
            error: null,
          });
        }
      } catch (error) {
        if (!cancelled) {
          const accumulatedRows = Array.from(rowsBySymbol.values());
          setHydrationState({
            status: 'error',
            rows: accumulatedRows,
            loadedRows: Math.min(accumulatedRows.length, totalRows),
            error: error instanceof Error ? error.message : 'Unknown hydration error',
          });
        }
      }
    };

    void hydrateRows();

    return () => {
      cancelled = true;
    };
  }, [scanManifestQuery.data]);

  const hydrationComplete = hydrationState.status === 'complete';
  const hydratedRows = hydrationState.rows;
  const { activeScreenId, setActiveScreenId, matchCounts } = usePresetScreens({
    screens: presetScreens,
    allRows: hydratedRows,
    hydrationComplete,
  });

  const handleSelectScreen = useCallback((screenId) => {
    setActiveScreenId(screenId);
    if (!screenId) {
      dispatchFilterState({
        type: 'reset-filters',
        defaultFilters: manifestDefaultFilters,
      });
      setSortBy(manifestDefaultSortBy);
      setSortOrder(manifestDefaultSortOrder);
    } else {
      const screen = presetScreens?.find((s) => s.id === screenId);
      if (screen) {
        dispatchFilterState({
          type: 'apply-expression',
          expression: screen.filter_expression
            || legacyFiltersToExpression(buildFiltersFromPreset(screen)),
        });
        setSortBy(screen.sort_by);
        setSortOrder(screen.sort_order);
      }
    }
  }, [
    presetScreens,
    manifestDefaultFilters,
    manifestDefaultSortBy,
    manifestDefaultSortOrder,
    setActiveScreenId,
  ]);

  const expressionKey = useMemo(
    () => stableExpressionKey(appliedExpression),
    [appliedExpression],
  );
  useEffect(() => {
    setPage(1);
  }, [expressionKey]);
  const handleQuickFiltersChange = useCallback((nextFilters) => {
    dispatchFilterState({ type: 'apply-quick-filters', filters: nextFilters });
  }, []);
  const chartEntries = useMemo(
    () => chartIndexQuery.data?.symbols || [],
    [chartIndexQuery.data]
  );
  const chartEnabledSymbols = useMemo(
    () => new Set(chartEntries.map((entry) => entry.symbol)),
    [chartEntries]
  );
  const filteredRows = useMemo(
    () => (hydrationComplete
      ? annotateExpressionMatches(hydratedRows, appliedExpression)
      : hydratedRows),
    [appliedExpression, hydratedRows, hydrationComplete]
  );
  const sortedRows = useMemo(
    () => (
      hydrationComplete
        ? sortStaticScanRows(filteredRows, sortBy, sortOrder)
        : filteredRows
    ),
    [filteredRows, hydrationComplete, sortBy, sortOrder]
  );
  const pagedRows = useMemo(
    () => (hydrationComplete ? paginateStaticScanRows(sortedRows, page, perPage) : filteredRows),
    [filteredRows, hydrationComplete, page, perPage, sortedRows]
  );
  const chartsAvailable = chartEnabledSymbols.size > 0;
  const isChartEnabled = useCallback(
    (symbol) => chartEnabledSymbols.has(symbol),
    [chartEnabledSymbols]
  );

  const handleOpenChart = (symbol) => {
    if (!isChartEnabled(symbol)) {
      return;
    }
    setSelectedChartSymbol(symbol);
    setChartModalOpen(true);
  };
  const navigationSymbols = useMemo(() => {
    const orderedRows = hydrationComplete ? sortedRows : pagedRows;
    return orderedRows
      .map((row) => row.symbol)
      .filter((symbol) => chartEnabledSymbols.has(symbol));
  }, [chartEnabledSymbols, hydrationComplete, pagedRows, sortedRows]);

  if (manifestQuery.isLoading || scanManifestQuery.isLoading) {
    return (
      <Box display="flex" justifyContent="center" py={8}>
        <CircularProgress />
      </Box>
    );
  }

  if (manifestQuery.isError || scanManifestQuery.isError) {
    return <Alert severity="error">Failed to load the static scan dataset.</Alert>;
  }

  return (
    <Box>
      <Typography variant="h5" sx={{ fontWeight: 700, letterSpacing: '-0.5px', mb: 0.5 }}>
        Daily Scan
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 2, fontSize: '12px' }}>
        Run {scanManifestQuery.data.run_id} as of {scanManifestQuery.data.as_of_date}.
      </Typography>

      <Paper elevation={0} sx={{ p: 1.5, mb: 1.5, border: '1px solid', borderColor: 'divider' }}>
        <Box display="flex" alignItems="baseline" gap={1.5}>
          <Typography variant="body1" sx={{ fontFamily: 'monospace', fontWeight: 600 }}>
            {(hydrationComplete ? filteredRows.length : hydrationState.loadedRows).toLocaleString()}
          </Typography>
          <Typography variant="caption" color="text.disabled" sx={{ fontSize: '10px' }}>
            of {scanManifestQuery.data.rows_total.toLocaleString()} rows
            {scanManifestQuery.data.charts?.available
              ? ` · ${(scanManifestQuery.data.charts.symbols_total ?? scanManifestQuery.data.charts.limit).toLocaleString()} charts`
              : ''}
          </Typography>
        </Box>
      </Paper>

      {hydrationComplete && presetScreens?.length > 0 && (
        <ScreenSelector
          screens={presetScreens}
          activeScreenId={activeScreenId}
          onSelectScreen={handleSelectScreen}
          matchCounts={matchCounts}
        />
      )}

      {!hydrationComplete && (
        <Alert severity="info" sx={{ mb: 2 }}>
          Loading full scan dataset: {hydrationState.loadedRows.toLocaleString()} /{' '}
          {scanManifestQuery.data.rows_total.toLocaleString()} rows. Filtering and sorting unlock after hydration completes.
        </Alert>
      )}

      {hydrationState.status === 'error' && (
        <Alert severity="warning" sx={{ mb: 2 }}>
          Background hydration failed. Showing the exported first page only.
        </Alert>
      )}

      {chartIndexQuery.isError && scanManifestQuery.data.charts?.path ? (
        <Alert severity="warning" sx={{ mb: 2 }}>
          Static chart payloads failed to load. Scan results remain available without chart modals.
        </Alert>
      ) : null}

      {hydrationComplete && (
        <FilterPanel
          filters={filters}
          onFilterChange={handleQuickFiltersChange}
          onReset={() => {
            dispatchFilterState({
              type: 'reset-filters',
              defaultFilters: manifestDefaultFilters,
            });
            setSortBy(manifestDefaultSortBy);
            setSortOrder(manifestDefaultSortOrder);
            setActiveScreenId(null);
          }}
          filterOptions={normalizeScanFilterOptions(scanManifestQuery.data.filter_options)}
          expanded={showFilters}
          onToggle={() => setShowFilters((previous) => !previous)}
          presetsEnabled={false}
          sectionDefaultExpanded={sectionDefaultExpanded}
          groupedFilteringEnabled
          expression={appliedExpression}
          onOpenLogicBuilder={() => setLogicBuilderOpen(true)}
        />
      )}

      <ResultsTable
        results={pagedRows}
        total={hydrationComplete ? sortedRows.length : pagedRows.length}
        page={hydrationComplete ? page : 1}
        perPage={perPage}
        sortBy={sortBy}
        sortOrder={sortOrder}
        onPageChange={hydrationComplete ? setPage : () => setPage(1)}
        onPerPageChange={hydrationComplete ? setPerPage : () => setPage(1)}
        onSortChange={(nextSortBy, nextSortOrder) => {
          if (!hydrationComplete) {
            return;
          }
          setSortBy(nextSortBy);
          setSortOrder(nextSortOrder);
          setPage(1);
        }}
        onOpenChart={chartsAvailable ? handleOpenChart : undefined}
        loading={false}
        showActions={chartsAvailable}
        showWatchlistMenu={false}
        isChartEnabled={isChartEnabled}
        sortingEnabled={hydrationComplete}
      />

      <StaticChartViewerModal
        open={chartModalOpen}
        onClose={() => setChartModalOpen(false)}
        initialSymbol={selectedChartSymbol}
        chartIndex={chartIndexQuery.data}
        navigationSymbols={navigationSymbols}
      />

      <GuidedFilterBuilderDialog
        open={logicBuilderOpen}
        expression={appliedExpression}
        onClose={() => setLogicBuilderOpen(false)}
        onApply={(nextExpression) => {
          dispatchFilterState({ type: 'apply-expression', expression: nextExpression });
          setPage(1);
          setActiveScreenId(null);
          setLogicBuilderOpen(false);
        }}
        filterOptions={normalizeScanFilterOptions(scanManifestQuery.data.filter_options)}
      />
    </Box>
  );
}

export default StaticScanPage;
