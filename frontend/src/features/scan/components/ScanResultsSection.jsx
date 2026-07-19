import { Alert, Box, Button, Chip, CircularProgress, LinearProgress, Paper, Typography } from '@mui/material';
import DownloadIcon from '@mui/icons-material/Download';
import ResultsTable from '../../../components/Scan/ResultsTable';
import { expressionSummary } from '../filterExpressionBuilder';

export default function ScanResultsSection({
  resultsLoading,
  resultsData,
  expression,
  resultsFetching = false,
  resultsError = null,
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
  const errorMessage = resultsError?.response?.data?.detail
    || resultsError?.message
    || 'The results request failed. Your previous results are still shown.';

  if (resultsLoading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', p: 5 }}>
        <CircularProgress />
        <Typography sx={{ ml: 2 }}>Loading results...</Typography>
      </Box>
    );
  }

  if (resultsData && resultsData.results && resultsData.results.length > 0) {
    const unfilteredTotal = resultsData.unfiltered_total ?? resultsData.total;
    const hasAppliedFilter = expression
      ? expression.required.conditions.length > 0 || expression.groups.some((group) => group.enabled !== false)
      : false;
    return (
      <>
        {resultsFetching && (
          <Box sx={{ mb: 1 }} role="status" aria-live="polite">
            <LinearProgress />
            <Typography variant="caption" color="text.secondary">Updating results…</Typography>
          </Box>
        )}
        {resultsError && (
          <Alert severity="error" action={<Button color="inherit" size="small" onClick={onRetry}>Retry</Button>} sx={{ mb: 1 }}>
            {typeof errorMessage === 'string' ? errorMessage : 'The results request failed. Your previous results are still shown.'}
          </Alert>
        )}
        <Box sx={{ mb: 2, display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 2 }}>
          <Box>
            <Typography variant="h6">
              Results: {resultsData.total.toLocaleString()} stocks
              {hasAppliedFilter && unfilteredTotal !== resultsData.total
                ? ` matching of ${unfilteredTotal.toLocaleString()}`
                : ''}
            </Typography>
            {expression && (
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.75, flexWrap: 'wrap', mt: 0.5 }}>
                <Chip size="small" color={hasAppliedFilter ? 'primary' : 'default'} label="Applied" />
                <Typography variant="caption" color="text.secondary">
                  {expressionSummary(expression)}
                </Typography>
              </Box>
            )}
          </Box>
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
      {resultsFetching && (
        <Box sx={{ mb: 2 }} role="status" aria-live="polite">
          <LinearProgress />
          <Typography variant="caption" color="text.secondary">Updating results…</Typography>
        </Box>
      )}
      <Typography variant="h6">
        {resultsError ? 'Could not update scan results' : 'No stocks match the applied logic'}
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
        {resultsError
          ? (typeof errorMessage === 'string' ? errorMessage : 'The results request failed.')
          : (
              <>
                {expression ? expressionSummary(expression) : 'The current filters returned no rows.'}
                {' '}Try disabling one setup, switching “match all” to “match any,” or widening a range.
              </>
            )}
      </Typography>
      <Button variant="outlined" onClick={onRetry} sx={{ mt: 2 }}>
        Retry Loading Results
      </Button>
    </Paper>
  );
}
