import { useEffect, useMemo, useState } from 'react';
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
import { buildDefaultScanFilters } from '../../features/scan/defaultFilters';
import { normalizeScanFilterOptions } from '../../features/scan/filterOptions';
import { getStableFilterKey } from '../../utils/filterUtils';
import {
  filterStaticScanRows,
  paginateStaticScanRows,
  sortStaticScanRows,
} from '../scanClient';

function StaticScanPage() {
  const manifestQuery = useStaticManifest();
  const scanManifestQuery = useQuery({
    queryKey: ['staticScanManifest', manifestQuery.data?.pages?.scan?.path],
    queryFn: () => fetchStaticJson(manifestQuery.data.pages.scan.path),
    enabled: Boolean(manifestQuery.data?.pages?.scan?.path),
    staleTime: Infinity,
  });
  const scanRowsQuery = useQuery({
    queryKey: ['staticScanRows', scanManifestQuery.data?.generated_at],
    queryFn: async () => {
      const chunks = scanManifestQuery.data?.chunks || [];
      const chunkPayloads = await Promise.all(chunks.map((chunk) => fetchStaticJson(chunk.path)));
      return chunkPayloads.flatMap((chunk) => chunk.rows || []);
    },
    enabled: Boolean(scanManifestQuery.data?.chunks?.length),
    staleTime: Infinity,
  });

  const [filters, setFilters] = useState(buildDefaultScanFilters);
  const [showFilters, setShowFilters] = useState(true);
  const [page, setPage] = useState(1);
  const [perPage, setPerPage] = useState(50);
  const [sortBy, setSortBy] = useState('composite_score');
  const [sortOrder, setSortOrder] = useState('desc');

  useEffect(() => {
    if (scanManifestQuery.data?.default_page_size) {
      setPerPage(scanManifestQuery.data.default_page_size);
    }
    if (scanManifestQuery.data?.sort?.field) {
      setSortBy(scanManifestQuery.data.sort.field);
      setSortOrder(scanManifestQuery.data.sort.order || 'desc');
    }
  }, [scanManifestQuery.data]);

  const filterKey = useMemo(() => getStableFilterKey(filters), [filters]);
  useEffect(() => {
    setPage(1);
  }, [filterKey]);

  const filteredRows = useMemo(
    () => filterStaticScanRows(scanRowsQuery.data || [], filters),
    [filters, scanRowsQuery.data]
  );
  const sortedRows = useMemo(
    () => sortStaticScanRows(filteredRows, sortBy, sortOrder),
    [filteredRows, sortBy, sortOrder]
  );
  const pagedRows = useMemo(
    () => paginateStaticScanRows(sortedRows, page, perPage),
    [page, perPage, sortedRows]
  );

  if (manifestQuery.isLoading || scanManifestQuery.isLoading || scanRowsQuery.isLoading) {
    return (
      <Box display="flex" justifyContent="center" py={8}>
        <CircularProgress />
      </Box>
    );
  }

  if (manifestQuery.isError || scanManifestQuery.isError || scanRowsQuery.isError) {
    return <Alert severity="error">Failed to load the static scan dataset.</Alert>;
  }

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Daily Scan
      </Typography>
      <Typography variant="body1" color="text.secondary" sx={{ mb: 2 }}>
        Fixed daily ranking from published run {scanManifestQuery.data.run_id} as of {scanManifestQuery.data.as_of_date}.
        Filtering, sorting, and pagination are performed entirely in the browser.
      </Typography>

      <Paper sx={{ p: 2, mb: 2 }}>
        <Typography variant="subtitle2" color="text.secondary">
          Results
        </Typography>
        <Typography variant="h6">
          {filteredRows.length.toLocaleString()} matching rows
        </Typography>
        <Typography variant="body2" color="text.secondary">
          {scanManifestQuery.data.rows_total.toLocaleString()} total rows exported
        </Typography>
      </Paper>

      <FilterPanel
        filters={filters}
        onFilterChange={setFilters}
        onReset={() => setFilters(buildDefaultScanFilters())}
        filterOptions={normalizeScanFilterOptions(scanManifestQuery.data.filter_options)}
        expanded={showFilters}
        onToggle={() => setShowFilters((previous) => !previous)}
        presetsEnabled={false}
      />

      <ResultsTable
        results={pagedRows}
        total={sortedRows.length}
        page={page}
        perPage={perPage}
        sortBy={sortBy}
        sortOrder={sortOrder}
        onPageChange={setPage}
        onPerPageChange={setPerPage}
        onSortChange={(nextSortBy, nextSortOrder) => {
          setSortBy(nextSortBy);
          setSortOrder(nextSortOrder);
          setPage(1);
        }}
        loading={false}
        showActions={false}
      />
    </Box>
  );
}

export default StaticScanPage;
