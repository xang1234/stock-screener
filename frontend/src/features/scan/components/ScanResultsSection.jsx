import { Box, Button, CircularProgress, Paper, Typography } from '@mui/material';
import DownloadIcon from '@mui/icons-material/Download';
import ResultsTable from '../../../components/Scan/ResultsTable';

function isFiltered(filters) {
  return Boolean(
    filters.symbolSearch ||
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
      filters.passesTemplate != null
  );
}

export default function ScanResultsSection({
  resultsLoading,
  resultsData,
  filters,
  onExport,
  page,
  perPage,
  sortBy,
  sortOrder,
  onPageChange,
  onPerPageChange,
  onSortChange,
  onOpenChart,
  onRowHover,
  onRetry,
}) {
  if (resultsLoading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', p: 5 }}>
        <CircularProgress />
        <Typography sx={{ ml: 2 }}>Loading results...</Typography>
      </Box>
    );
  }

  if (resultsData && resultsData.results && resultsData.results.length > 0) {
    return (
      <>
        <Box sx={{ mb: 2, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Typography variant="h6">
            Results: {resultsData.total} stocks
            {isFiltered(filters) ? ' (filtered)' : ''}
          </Typography>
          <Button variant="outlined" startIcon={<DownloadIcon />} onClick={onExport} disabled={!resultsData.total}>
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
          onPageChange={onPageChange}
          onPerPageChange={onPerPageChange}
          onSortChange={onSortChange}
          onOpenChart={onOpenChart}
          onRowHover={onRowHover}
          loading={resultsLoading}
        />
      </>
    );
  }

  return (
    <Paper sx={{ p: 5, textAlign: 'center' }}>
      <Typography variant="body1" color="text.secondary">
        No results available. This could mean:
        <br />- The scan is still processing
        <br />- The scan failed
        <br />- All results were filtered out
      </Typography>
      <Button variant="outlined" onClick={onRetry} sx={{ mt: 2 }}>
        Retry Loading Results
      </Button>
    </Paper>
  );
}
