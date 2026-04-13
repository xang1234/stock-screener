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
import { SCREENER_OPTIONS, UNIVERSE_MARKETS, UNIVERSE_SCOPES_BY_MARKET } from '../constants';
import { getSelectionCount } from '../universeSelection';

function stockCountLabel(universeMarket, universeScope, universeStats, statsLoading) {
  if (!universeMarket) {
    return 'Pick a market to start';
  }
  if (universeMarket !== 'TEST' && !universeScope) {
    return '...';
  }
  if (statsLoading) {
    return '...';
  }
  const count = getSelectionCount(universeMarket, universeScope, universeStats);
  if (count === null || count === undefined) {
    return '';
  }
  return `${count} stocks`;
}

function toOptionalNumber(rawValue) {
  return rawValue === '' ? undefined : Number(rawValue);
}

export default function ScanControlBar({
  currentScanId,
  scanHistory,
  onLoadScan,
  universeMarket,
  universeScope,
  onUniverseMarketChange,
  onUniverseScopeChange,
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
}) {
  const controlsDisabled = createScanPending || scanStatus === 'running';
  const scopeOptions = universeMarket ? UNIVERSE_SCOPES_BY_MARKET[universeMarket] ?? [] : [];
  const needsScope = universeMarket && universeMarket !== 'TEST';
  const startDisabled =
    createScanPending
    || !universeMarket
    || (needsScope && !universeScope);

  return (
    <Paper elevation={1} sx={{ p: 1.5, mb: 2 }}>
      <Box sx={{ display: 'flex', gap: 1.5, alignItems: 'center', flexWrap: 'wrap' }}>
        <FormControl size="small" sx={{ minWidth: 200 }}>
          <InputLabel id="prev-scan-label">Previous Scans</InputLabel>
          <Select
            labelId="prev-scan-label"
            value={currentScanId || ''}
            label="Previous Scans"
            onChange={(event) => onLoadScan(event.target.value)}
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

        <FormControl size="small" sx={{ minWidth: 140 }}>
          <InputLabel id="universe-market-label">Market</InputLabel>
          <Select
            labelId="universe-market-label"
            value={universeMarket ?? ''}
            label="Market"
            onChange={(event) => onUniverseMarketChange(event.target.value || null)}
            disabled={controlsDisabled}
          >
            {UNIVERSE_MARKETS.map((option) => {
              const count = option.value === 'TEST'
                ? null
                : getSelectionCount(option.value, 'market', universeStats);
              return (
                <MenuItem key={option.value} value={option.value}>
                  {option.label}{count ? ` (${count})` : ''}
                </MenuItem>
              );
            })}
          </Select>
        </FormControl>

        {needsScope && (
          <FormControl size="small" sx={{ minWidth: 150 }}>
            <InputLabel id="universe-scope-label">Universe</InputLabel>
            <Select
              labelId="universe-scope-label"
              value={universeScope ?? ''}
              label="Universe"
              onChange={(event) => onUniverseScopeChange(event.target.value || null)}
              disabled={controlsDisabled}
            >
              {scopeOptions.map((option) => {
                const count = getSelectionCount(universeMarket, option.value, universeStats);
                return (
                  <MenuItem key={option.value} value={option.value}>
                    {option.label}{count ? ` (${count})` : ''}
                  </MenuItem>
                );
              })}
            </Select>
          </FormControl>
        )}

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
          {stockCountLabel(universeMarket, universeScope, universeStats, statsLoading)}
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
            disabled={startDisabled}
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
                value={customFilters.price_min ?? ''}
                onChange={(event) => onCustomFiltersChange({ ...customFilters, price_min: toOptionalNumber(event.target.value) })}
                disabled={controlsDisabled}
                fullWidth
                size="small"
              />
            </Grid>
            <Grid item xs={6} sm={4} md={2}>
              <TextField
                label="Max Price"
                type="number"
                value={customFilters.price_max ?? ''}
                onChange={(event) => onCustomFiltersChange({ ...customFilters, price_max: toOptionalNumber(event.target.value) })}
                disabled={controlsDisabled}
                fullWidth
                size="small"
              />
            </Grid>
            <Grid item xs={6} sm={4} md={2}>
              <TextField
                label="Min RS"
                type="number"
                value={customFilters.rs_rating_min ?? ''}
                onChange={(event) => onCustomFiltersChange({ ...customFilters, rs_rating_min: toOptionalNumber(event.target.value) })}
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
                value={customFilters.volume_min ?? ''}
                onChange={(event) => onCustomFiltersChange({ ...customFilters, volume_min: toOptionalNumber(event.target.value) })}
                disabled={controlsDisabled}
                fullWidth
                size="small"
              />
            </Grid>
            <Grid item xs={6} sm={4} md={2}>
              <TextField
                label="Min EPS %"
                type="number"
                value={customFilters.eps_growth_min ?? ''}
                onChange={(event) => onCustomFiltersChange({ ...customFilters, eps_growth_min: toOptionalNumber(event.target.value) })}
                disabled={controlsDisabled}
                fullWidth
                size="small"
              />
            </Grid>
            <Grid item xs={6} sm={4} md={2}>
              <TextField
                label="Min Sales %"
                type="number"
                value={customFilters.sales_growth_min ?? ''}
                onChange={(event) => onCustomFiltersChange({ ...customFilters, sales_growth_min: toOptionalNumber(event.target.value) })}
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
