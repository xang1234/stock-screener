import { useCallback, useEffect, useMemo, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  Alert,
  Box,
  CircularProgress,
  Paper,
  Typography,
} from '@mui/material';
import FilterPanel from '../../components/Scan/FilterPanel';
import ResultsTable from '../../components/Scan/ResultsTable';
import { useStaticManifest, fetchStaticJson } from '../dataClient';
import { useStaticChartIndex } from '../chartClient';
import { buildDefaultScanFilters } from '../../features/scan/defaultFilters';
import { normalizeScanFilterOptions } from '../../features/scan/filterOptions';
import { getStableFilterKey } from '../../utils/filterUtils';
import {
  filterStaticScanRows,
  paginateStaticScanRows,
  sortStaticScanRows,
} from '../scanClient';
import StaticChartViewerModal from '../StaticChartViewerModal';

const HYDRATION_BATCH_SIZE = 2;

function StaticScanPage() {
  const manifestQuery = useStaticManifest();
  const scanManifestQuery = useQuery({
    queryKey: ['staticScanManifest', manifestQuery.data?.pages?.scan?.path],
    queryFn: () => fetchStaticJson(manifestQuery.data.pages.scan.path),
    enabled: Boolean(manifestQuery.data?.pages?.scan?.path),
    staleTime: Infinity,
  });
  const chartIndexQuery = useStaticChartIndex(scanManifestQuery.data?.charts?.path);

  const [filters, setFilters] = useState(buildDefaultScanFilters);
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
          setHydrationState({
            status: 'error',
            rows: initialRows,
            loadedRows: initialLoadedRows,
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
  const filterKey = useMemo(() => getStableFilterKey(filters), [filters]);
  useEffect(() => {
    setPage(1);
  }, [filterKey]);

  const hydratedRows = hydrationState.rows;
  const chartEntries = useMemo(
    () => chartIndexQuery.data?.symbols || [],
    [chartIndexQuery.data]
  );
  const chartEnabledSymbols = useMemo(
    () => new Set(chartEntries.map((entry) => entry.symbol)),
    [chartEntries]
  );
  const filteredRows = useMemo(
    () => (hydrationComplete ? filterStaticScanRows(hydratedRows, filters) : hydratedRows),
    [filters, hydratedRows, hydrationComplete]
  );
  const sortedRows = useMemo(
    () => (hydrationComplete ? sortStaticScanRows(filteredRows, sortBy, sortOrder) : filteredRows),
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
      <Typography variant="h4" gutterBottom>
        Daily Scan
      </Typography>
      <Typography variant="body1" color="text.secondary" sx={{ mb: 2 }}>
        Fixed daily ranking from published run {scanManifestQuery.data.run_id} as of {scanManifestQuery.data.as_of_date}.
        The first page renders from the exported top rows, then the remaining chunks hydrate in the background.
      </Typography>

      <Paper sx={{ p: 2, mb: 2 }}>
        <Typography variant="subtitle2" color="text.secondary">
          Results
        </Typography>
        <Typography variant="h6">
          {(hydrationComplete ? filteredRows.length : hydrationState.loadedRows).toLocaleString()} rows ready
        </Typography>
        <Typography variant="body2" color="text.secondary">
          {scanManifestQuery.data.rows_total.toLocaleString()} total rows exported
        </Typography>
        {scanManifestQuery.data.charts?.available ? (
          <Typography variant="body2" color="text.secondary">
            Static charts exported for the top {scanManifestQuery.data.charts.limit.toLocaleString()} ranked symbols.
          </Typography>
        ) : null}
      </Paper>

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
          onFilterChange={setFilters}
          onReset={() => setFilters(buildDefaultScanFilters())}
          filterOptions={normalizeScanFilterOptions(scanManifestQuery.data.filter_options)}
          expanded={showFilters}
          onToggle={() => setShowFilters((previous) => !previous)}
          presetsEnabled={false}
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
    </Box>
  );
}

export default StaticScanPage;
