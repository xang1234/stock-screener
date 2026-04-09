import {
  Alert,
  Box,
  Button,
  Checkbox,
  Chip,
  CircularProgress,
  Collapse,
  FormControl,
  FormControlLabel,
  Grid,
  InputLabel,
  MenuItem,
  Paper,
  Select,
  TextField,
} from '@mui/material';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import StopIcon from '@mui/icons-material/Stop';
import ScanProgress from '../../../components/Scan/ScanProgress';
import { formatScanDropdownLabel } from '../../../utils/scanLabel';
import { SCREENER_OPTIONS } from '../constants';

function stockCountLabel(universe, universeStats, statsLoading, testSymbolsCount) {
  if (universe === 'test') {
    return `${testSymbolsCount} stocks`;
  }
  if (statsLoading) {
    return '...';
  }
  if (!universeStats) {
    return '';
  }
  if (universe === 'sp500') {
    return `${universeStats.sp500 || '~500'} stocks`;
  }
  if (universe === 'nyse' && universeStats.by_exchange?.NYSE) {
    return `${universeStats.by_exchange.NYSE} stocks`;
  }
  if (universe === 'nasdaq' && universeStats.by_exchange?.NASDAQ) {
    return `${universeStats.by_exchange.NASDAQ} stocks`;
  }
  if (universe === 'amex' && universeStats.by_exchange?.AMEX) {
    return `${universeStats.by_exchange.AMEX} stocks`;
  }
  return `${universeStats.active} stocks`;
}

export default function ScanControlBar({
  currentScanId,
  scanHistory,
  onLoadScan,
  universe,
  onUniverseChange,
  universeStats,
  statsLoading,
  selectedScreeners,
  onScreenerToggle,
  includeVcp,
  onIncludeVcpChange,
  compositeMethod,
  onCompositeMethodChange,
  createScanPending,
  scanStatus,
  onStartScan,
  onCancelScan,
  cancelScanPending,
  statusData,
  customFilters,
  onCustomFiltersChange,
  createScanError,
  cancelScanError,
  testSymbolsCount,
}) {
  const controlsDisabled = createScanPending || scanStatus === 'running';

  return (
    <Paper elevation={1} sx={{ p: 1.5, mb: 2 }}>
      <Box sx={{ display: 'flex', gap: 1.5, alignItems: 'center', flexWrap: 'wrap' }}>
        <FormControl size="small" sx={{ minWidth: 200 }}>
          <InputLabel id="prev-scan-label">Previous Scans</InputLabel>
          <Select
            labelId="prev-scan-label"
            value={currentScanId || ''}
            label="Previous Scans"
            onChange={(event) => event.target.value && onLoadScan(event.target.value)}
          >
            <MenuItem value="">
              <em>New Scan</em>
            </MenuItem>
            {scanHistory?.scans?.map((scan) => (
              <MenuItem key={scan.scan_id} value={scan.scan_id}>
                {formatScanDropdownLabel(scan)}
              </MenuItem>
            ))}
          </Select>
        </FormControl>

        <Box sx={{ borderLeft: 1, borderColor: 'divider', height: 32, mx: 0.5 }} />

        <FormControl size="small" sx={{ minWidth: 150 }}>
          <InputLabel id="universe-label">Universe</InputLabel>
          <Select
            labelId="universe-label"
            value={universe}
            label="Universe"
            onChange={(event) => onUniverseChange(event.target.value)}
            disabled={controlsDisabled}
          >
            <MenuItem value="test">Test (20)</MenuItem>
            <MenuItem value="sp500">S&amp;P 500{universeStats?.by_exchange ? ` (${universeStats.sp500 || '~500'})` : ''}</MenuItem>
            <MenuItem value="nyse">NYSE{universeStats?.by_exchange?.NYSE ? ` (${universeStats.by_exchange.NYSE})` : ''}</MenuItem>
            <MenuItem value="nasdaq">NASDAQ{universeStats?.by_exchange?.NASDAQ ? ` (${universeStats.by_exchange.NASDAQ})` : ''}</MenuItem>
            <MenuItem value="amex">AMEX{universeStats?.by_exchange?.AMEX ? ` (${universeStats.by_exchange.AMEX})` : ''}</MenuItem>
            <MenuItem value="all">All{universeStats?.active ? ` (${universeStats.active})` : ''}</MenuItem>
          </Select>
        </FormControl>

        <Box sx={{ display: 'flex', gap: 0.5, alignItems: 'center', flexWrap: 'wrap' }}>
          <Box sx={{ fontSize: '11px', color: 'text.secondary', mr: 0.5 }}>Strategies:</Box>
          {SCREENER_OPTIONS.map((screener) => (
            <Chip
              key={screener.id}
              label={screener.label}
              size="small"
              variant={selectedScreeners.includes(screener.id) ? 'filled' : 'outlined'}
              color={selectedScreeners.includes(screener.id) ? 'primary' : 'default'}
              onClick={() => onScreenerToggle(screener.id)}
              disabled={controlsDisabled}
              sx={{ height: 24, fontSize: '10px' }}
            />
          ))}
        </Box>

        <FormControlLabel
          control={
            <Checkbox
              checked={includeVcp}
              onChange={(event) => onIncludeVcpChange(event.target.checked)}
              disabled={controlsDisabled}
              size="small"
            />
          }
          label={<Box sx={{ fontSize: '11px' }}>VCP</Box>}
          sx={{ mr: 0 }}
        />

        {selectedScreeners.length > 1 && (
          <FormControl size="small" sx={{ minWidth: 100 }}>
            <InputLabel id="composite-method-label">Method</InputLabel>
            <Select
              labelId="composite-method-label"
              value={compositeMethod}
              label="Method"
              onChange={(event) => onCompositeMethodChange(event.target.value)}
              disabled={controlsDisabled}
            >
              <MenuItem value="weighted_average">Avg</MenuItem>
              <MenuItem value="maximum">Max</MenuItem>
              <MenuItem value="minimum">Min</MenuItem>
            </Select>
          </FormControl>
        )}

        <Box sx={{ flexGrow: 1 }} />

        <Box sx={{ fontSize: '11px', color: 'text.secondary' }}>
          {stockCountLabel(universe, universeStats, statsLoading, testSymbolsCount)}
        </Box>

        {scanStatus === 'running' ? (
          <Button
            variant="outlined"
            color="error"
            size="small"
            startIcon={cancelScanPending ? <CircularProgress size={14} /> : <StopIcon />}
            onClick={onCancelScan}
            disabled={cancelScanPending}
          >
            Cancel
          </Button>
        ) : (
          <Button
            variant="contained"
            size="small"
            startIcon={createScanPending ? <CircularProgress size={14} /> : <PlayArrowIcon />}
            onClick={onStartScan}
            disabled={createScanPending}
          >
            Scan
          </Button>
        )}

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

      <Collapse in={selectedScreeners.includes('custom')}>
        <Box sx={{ mt: 1.5, pt: 1.5, borderTop: 1, borderColor: 'divider' }}>
          <Grid container spacing={1}>
            <Grid item xs={6} sm={4} md={2}>
              <TextField
                label="Min Price"
                type="number"
                value={customFilters.price_min}
                onChange={(event) => onCustomFiltersChange({ ...customFilters, price_min: Number(event.target.value) })}
                disabled={controlsDisabled}
                fullWidth
                size="small"
              />
            </Grid>
            <Grid item xs={6} sm={4} md={2}>
              <TextField
                label="Max Price"
                type="number"
                value={customFilters.price_max}
                onChange={(event) => onCustomFiltersChange({ ...customFilters, price_max: Number(event.target.value) })}
                disabled={controlsDisabled}
                fullWidth
                size="small"
              />
            </Grid>
            <Grid item xs={6} sm={4} md={2}>
              <TextField
                label="Min RS"
                type="number"
                value={customFilters.rs_rating_min}
                onChange={(event) => onCustomFiltersChange({ ...customFilters, rs_rating_min: Number(event.target.value) })}
                disabled={controlsDisabled}
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
                onChange={(event) => onCustomFiltersChange({ ...customFilters, volume_min: Number(event.target.value) })}
                disabled={controlsDisabled}
                fullWidth
                size="small"
              />
            </Grid>
            <Grid item xs={6} sm={4} md={2}>
              <TextField
                label="Min EPS %"
                type="number"
                value={customFilters.eps_growth_min}
                onChange={(event) => onCustomFiltersChange({ ...customFilters, eps_growth_min: Number(event.target.value) })}
                disabled={controlsDisabled}
                fullWidth
                size="small"
              />
            </Grid>
            <Grid item xs={6} sm={4} md={2}>
              <TextField
                label="Min Sales %"
                type="number"
                value={customFilters.sales_growth_min}
                onChange={(event) => onCustomFiltersChange({ ...customFilters, sales_growth_min: Number(event.target.value) })}
                disabled={controlsDisabled}
                fullWidth
                size="small"
              />
            </Grid>
          </Grid>
        </Box>
      </Collapse>

      {createScanError && (
        <Alert severity="error" sx={{ mt: 1 }}>
          Error: {createScanError.message}
        </Alert>
      )}
      {cancelScanError && (
        <Alert severity="error" sx={{ mt: 1 }}>
          Error: {cancelScanError.message}
        </Alert>
      )}
      {scanStatus === 'cancelled' && (
        <Alert severity="warning" sx={{ mt: 1 }}>
          Scan cancelled. Showing partial results.
        </Alert>
      )}
    </Paper>
  );
}
