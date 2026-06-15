import {
  Box,
  CircularProgress,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Typography,
} from '@mui/material';

import PriceSparkline from '../Scan/PriceSparkline';
import RSSparkline from '../Scan/RSSparkline';
import TickerCell from '../common/TickerCell';
import { getGroupRankColor } from '../../utils/colorUtils';
import { formatLocalCurrency } from '../../utils/formatUtils';
import { resolveMarketCapDisplay } from '../../utils/marketCapUtils';

const formatNumber = (value, digits = 0) => {
  if (value == null) return '-';
  return Number(value).toLocaleString(undefined, {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  });
};

function DailyScanRowsTable({
  title,
  subtitle,
  rows,
  // null means every row opens a chart (server mode); the static site passes
  // the set of symbols that have an exported chart bundle.
  chartEnabledSymbols = null,
  navigationSymbols = [],
  onOpenChart = null,
  emptyMessage,
  action = null,
  showRs = false,
  showRating = false,
  isLoading = false,
  isError = false,
  errorMessage = 'Failed to load rows.',
  priceSparklineWidth = 137,
  priceSparklineInnerWidth = 86,
  testId,
}) {
  const isChartEnabled = (symbol) => (
    Boolean(onOpenChart) && (chartEnabledSymbols == null || chartEnabledSymbols.has(symbol))
  );
  const handleRowOpen = (symbol) => {
    if (isChartEnabled(symbol)) {
      onOpenChart(symbol, navigationSymbols);
    }
  };
  const colSpan = 8 + (showRs ? 1 : 0) + (showRating ? 1 : 0);

  return (
    <Paper
      data-testid={testId}
      elevation={0}
      sx={{ p: 1.5, mb: 2, border: '1px solid', borderColor: 'divider' }}
    >
      <Box
        sx={{
          display: 'flex',
          alignItems: 'flex-start',
          justifyContent: 'space-between',
          gap: 1,
          flexWrap: 'wrap',
          mb: 1,
        }}
      >
        <Box>
          <Typography variant="subtitle1" sx={{ fontWeight: 600, fontSize: '13px', textTransform: 'uppercase', letterSpacing: '0.5px', mb: 0.5 }}>
            {title}
          </Typography>
          <Typography variant="caption" color="text.disabled" sx={{ display: 'block', fontSize: '10px' }}>
            {subtitle}
          </Typography>
        </Box>
        {action}
      </Box>
      <TableContainer>
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell align="center">Symbol</TableCell>
              <TableCell align="center">Score</TableCell>
              {showRs ? <TableCell align="center">RS</TableCell> : null}
              <TableCell align="center">Price</TableCell>
              <TableCell align="center">MCap</TableCell>
              {showRating ? <TableCell align="center">Rating</TableCell> : null}
              <TableCell align="center">Price Trend (30d)</TableCell>
              <TableCell align="center">RS Trend (30d)</TableCell>
              <TableCell align="center">IBD Group</TableCell>
              <TableCell align="center">Grp Rank</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {isLoading && rows.length === 0 ? (
              <TableRow>
                <TableCell align="center" colSpan={colSpan}>
                  <CircularProgress size={18} />
                </TableCell>
              </TableRow>
            ) : null}
            {isError && rows.length === 0 ? (
              <TableRow>
                <TableCell align="center" colSpan={colSpan} sx={{ color: 'error.main', py: 2 }}>
                  {errorMessage}
                </TableCell>
              </TableRow>
            ) : null}
            {rows.map((row) => {
              const rowChartEnabled = isChartEnabled(row.symbol);
              return (
                <TableRow
                  key={row.symbol}
                  hover={rowChartEnabled}
                  tabIndex={rowChartEnabled ? 0 : -1}
                  onClick={() => handleRowOpen(row.symbol)}
                  onKeyDown={(event) => {
                    if (!rowChartEnabled) return;
                    if (event.key === 'Enter' || event.key === ' ') {
                      event.preventDefault();
                      handleRowOpen(row.symbol);
                    }
                  }}
                  sx={{ cursor: rowChartEnabled ? 'pointer' : 'default' }}
                >
                  <TableCell align="center">
                    <TickerCell symbol={row.symbol} companyName={row.company_name} align="center" />
                  </TableCell>
                  <TableCell align="center">{formatNumber(row.composite_score, 1)}</TableCell>
                  {showRs ? <TableCell align="center">{formatNumber(row.rs_rating, 0)}</TableCell> : null}
                  <TableCell align="center">{formatLocalCurrency(row.current_price, row.currency)}</TableCell>
                  <TableCell align="center">
                    {resolveMarketCapDisplay(row, null, { preferUsd: true }).formattedValue}
                  </TableCell>
                  {showRating ? <TableCell align="center">{row.rating}</TableCell> : null}
                  <TableCell align="center">
                    {row.price_sparkline_data ? (
                      <Box display="flex" justifyContent="center">
                        <PriceSparkline
                          data={row.price_sparkline_data}
                          trend={row.price_trend}
                          change1d={row.price_change_1d}
                          industry={row.ibd_industry_group}
                          width={priceSparklineWidth}
                          height={28}
                          sparklineWidth={priceSparklineInnerWidth}
                        />
                      </Box>
                    ) : '-'}
                  </TableCell>
                  <TableCell align="center">
                    {row.rs_sparkline_data ? (
                      <Box display="flex" justifyContent="center">
                        <RSSparkline
                          data={row.rs_sparkline_data}
                          trend={row.rs_trend}
                          width={117}
                          height={20}
                        />
                      </Box>
                    ) : '-'}
                  </TableCell>
                  <TableCell align="center" sx={{
                    color: 'text.secondary', fontSize: '12px',
                    overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: 140,
                  }}>
                    {row.ibd_industry_group || '-'}
                  </TableCell>
                  <TableCell align="center" sx={{
                    fontFamily: 'monospace', fontWeight: row.ibd_group_rank != null && row.ibd_group_rank <= 20 ? 600 : 400,
                    color: getGroupRankColor(row.ibd_group_rank),
                  }}>
                    {row.ibd_group_rank ?? '-'}
                  </TableCell>
                </TableRow>
              );
            })}
            {!isLoading && !isError && rows.length === 0 ? (
              <TableRow>
                <TableCell align="center" colSpan={colSpan}>
                  {emptyMessage}
                </TableCell>
              </TableRow>
            ) : null}
          </TableBody>
        </Table>
      </TableContainer>
    </Paper>
  );
}

export default DailyScanRowsTable;
