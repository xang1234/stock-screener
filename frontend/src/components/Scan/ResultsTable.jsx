import { useMemo, useRef, useCallback, memo } from 'react';
import { useVirtualizer } from '@tanstack/react-virtual';
import {
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TablePagination,
  TableSortLabel,
  Paper,
  Chip,
  Typography,
  Box,
  CircularProgress,
  IconButton,
} from '@mui/material';
import CheckIcon from '@mui/icons-material/Check';
import CloseIcon from '@mui/icons-material/Close';
import ShowChartIcon from '@mui/icons-material/ShowChart';
import RSSparkline from './RSSparkline';
import PriceSparkline from './PriceSparkline';
import AddToWatchlistMenu from '../common/AddToWatchlistMenu';
import {
  getStageColor,
  getRatingColor,
  getGrowthColor,
  getEpsRatingColor,
  getGroupRankColor,
} from '../../utils/colorUtils';
import { formatLargeNumber, formatIpoAge, getIpoAgeColor } from '../../utils/formatUtils';

// Row height constant for virtualization
const ROW_HEIGHT = 32;

// Column definitions with explicit widths
const columns = [
  { id: 'chart', label: '', sortable: false, width: 60 },
  { id: 'symbol', label: 'Sym', sortable: true, width: 65 },
  { id: 'rs_trend', label: 'RS Trend', sortable: true, width: 110 },
  { id: 'price_change_1d', label: 'Price', sortable: true, width: 110 },
  { id: 'gics_sector', label: 'Sector', sortable: true, width: 80 },
  { id: 'ibd_group_rank', label: 'Grp', sortable: true, width: 45 },
  { id: 'composite_score', label: 'Comp', sortable: true, width: 50 },
  { id: 'minervini_score', label: 'Min', sortable: true, width: 45 },
  { id: 'canslim_score', label: 'CAN', sortable: true, width: 45 },
  { id: 'ipo_score', label: 'IPO', sortable: true, width: 45 },
  { id: 'custom_score', label: 'Cust', sortable: true, width: 45 },
  { id: 'volume_breakthrough_score', label: 'VolB', sortable: true, width: 50 },
  { id: 'se_setup_score', label: 'SE', sortable: true, width: 45 },
  { id: 'se_pattern_primary', label: 'Pat', sortable: true, width: 55 },
  { id: 'se_distance_to_pivot_pct', label: 'Pvt%', sortable: true, width: 50 },
  { id: 'se_bb_width_pctile_252', label: 'Sqz', sortable: true, width: 45 },
  { id: 'se_volume_vs_50d', label: 'V50', sortable: true, width: 45 },
  { id: 'se_rs_line_new_high', label: 'RSH', sortable: false, width: 35 },
  { id: 'se_pivot_price', label: 'Pvt$', sortable: true, width: 55 },
  { id: 'rs_rating', label: 'RS', sortable: true, width: 40 },
  { id: 'rs_rating_1m', label: '1M', sortable: true, width: 40 },
  { id: 'rs_rating_3m', label: '3M', sortable: true, width: 40 },
  { id: 'rs_rating_12m', label: '12M', sortable: true, width: 45 },
  { id: 'beta', label: 'β', sortable: true, width: 45 },
  { id: 'beta_adj_rs', label: 'βRS', sortable: true, width: 45 },
  { id: 'eps_rating', label: 'EPS Rtg', sortable: true, width: 55 },
  { id: 'stage', label: 'Stg', sortable: true, width: 40 },
  { id: 'current_price', label: 'Price', sortable: true, width: 65 },
  { id: 'volume', label: '$Vol', sortable: true, width: 60 },
  { id: 'market_cap', label: 'MCap', sortable: true, width: 60 },
  { id: 'ipo_date', label: 'IPO', sortable: true, width: 50 },
  { id: 'eps_growth_qq', label: 'EPS', sortable: true, width: 50 },
  { id: 'sales_growth_qq', label: 'Sales', sortable: true, width: 50 },
  { id: 'adr_percent', label: 'ADR', sortable: true, width: 50 },
  { id: 'ma_alignment', label: 'MA', sortable: false, width: 35 },
  { id: 'vcp_detected', label: 'VCP', sortable: false, width: 40 },
  { id: 'vcp_score', label: 'VScr', sortable: true, width: 50 },
  { id: 'vcp_pivot', label: 'Pvt', sortable: true, width: 55 },
  { id: 'vcp_ready_for_breakout', label: 'Rdy', sortable: false, width: 35 },
  { id: 'passes_template', label: 'Pass', sortable: false, width: 40 },
  { id: 'rating', label: 'Rate', sortable: false, width: 80 },
];

/**
 * Memoized table row component to prevent unnecessary re-renders
 */
const VirtualTableRow = memo(function VirtualTableRow({ row, onRowClick, onRowHover, onOpenChart }) {
  const handleRowClick = useCallback(() => {
    onRowClick(row.symbol);
  }, [onRowClick, row.symbol]);

  const handleRowHover = useCallback(() => {
    onRowHover?.(row.symbol);
  }, [onRowHover, row.symbol]);

  const handleChartClick = useCallback((e) => {
    e.stopPropagation();
    onOpenChart(row.symbol);
  }, [onOpenChart, row.symbol]);

  return (
    <TableRow
      hover
      onClick={handleRowClick}
      onMouseEnter={handleRowHover}
      sx={{ cursor: 'pointer', height: ROW_HEIGHT }}
    >
      <TableCell align="center" onClick={(e) => e.stopPropagation()} sx={{ p: '2px', width: 60, minWidth: 60 }}>
        <IconButton
          size="small"
          onClick={handleChartClick}
          sx={{ color: 'primary.main', p: 0 }}
        >
          <ShowChartIcon sx={{ fontSize: 14 }} />
        </IconButton>
        <AddToWatchlistMenu symbols={row.symbol} size="small" />
      </TableCell>

      <TableCell sx={{ fontWeight: 600, width: 65, minWidth: 65 }}>
        {row.symbol}
      </TableCell>

      <TableCell align="center" sx={{ p: '4px', width: 110, minWidth: 110 }}>
        <RSSparkline
          data={row.rs_sparkline_data}
          trend={row.rs_trend}
          width={100}
          height={28}
        />
      </TableCell>

      <TableCell align="center" sx={{ p: '4px', width: 110, minWidth: 110 }}>
        <PriceSparkline
          data={row.price_sparkline_data}
          trend={row.price_trend}
          change1d={row.price_change_1d}
          industry={row.ibd_industry_group}
          width={100}
          height={28}
        />
      </TableCell>

      <TableCell align="center" sx={{ color: 'text.secondary', width: 80, minWidth: 80, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
        {row.gics_sector || '-'}
      </TableCell>

      <TableCell align="center" sx={{
        fontFamily: 'monospace',
        color: getGroupRankColor(row.ibd_group_rank),
        fontWeight: row.ibd_group_rank && row.ibd_group_rank <= 20 ? 600 : 400,
        width: 45, minWidth: 45
      }}>
        {row.ibd_group_rank ?? '-'}
      </TableCell>

      <TableCell align="center" sx={{ fontWeight: 600, color: 'primary.main', fontFamily: 'monospace', width: 50, minWidth: 50 }}>
        {row.composite_score?.toFixed(1) || '-'}
      </TableCell>

      <TableCell align="center" sx={{ fontFamily: 'monospace', width: 45, minWidth: 45 }}>
        {row.minervini_score != null ? row.minervini_score.toFixed(1) : '-'}
      </TableCell>

      <TableCell align="center" sx={{ fontFamily: 'monospace', width: 45, minWidth: 45 }}>
        {row.canslim_score != null ? row.canslim_score.toFixed(1) : '-'}
      </TableCell>

      <TableCell align="center" sx={{ fontFamily: 'monospace', width: 45, minWidth: 45 }}>
        {row.ipo_score != null ? row.ipo_score.toFixed(1) : '-'}
      </TableCell>

      <TableCell align="center" sx={{ fontFamily: 'monospace', width: 45, minWidth: 45 }}>
        {row.custom_score != null ? row.custom_score.toFixed(1) : '-'}
      </TableCell>

      <TableCell align="center" sx={{ fontFamily: 'monospace', width: 50, minWidth: 50 }}>
        {row.volume_breakthrough_score != null ? row.volume_breakthrough_score.toFixed(1) : '-'}
      </TableCell>

      <TableCell align="center" sx={{ fontFamily: 'monospace', width: 45, minWidth: 45 }}>
        {row.se_setup_score != null ? row.se_setup_score.toFixed(1) : '-'}
      </TableCell>

      <TableCell align="center" sx={{ color: 'text.secondary', width: 55, minWidth: 55, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
        {row.se_pattern_primary || '-'}
      </TableCell>

      <TableCell align="center" sx={{ fontFamily: 'monospace', width: 50, minWidth: 50 }}>
        {row.se_distance_to_pivot_pct != null ? `${row.se_distance_to_pivot_pct.toFixed(1)}%` : '-'}
      </TableCell>

      <TableCell align="center" sx={{ fontFamily: 'monospace', width: 45, minWidth: 45 }}>
        {row.se_bb_width_pctile_252 != null ? row.se_bb_width_pctile_252.toFixed(0) : '-'}
      </TableCell>

      <TableCell align="center" sx={{ fontFamily: 'monospace', width: 45, minWidth: 45 }}>
        {row.se_volume_vs_50d != null ? `${row.se_volume_vs_50d.toFixed(1)}x` : '-'}
      </TableCell>

      <TableCell align="center" sx={{ width: 35, minWidth: 35 }}>
        {row.se_rs_line_new_high == null ? '-' : row.se_rs_line_new_high ? (
          <CheckIcon sx={{ fontSize: 14, color: 'success.main' }} />
        ) : (
          <CloseIcon sx={{ fontSize: 14, color: 'text.disabled' }} />
        )}
      </TableCell>

      <TableCell align="right" sx={{ fontFamily: 'monospace', width: 55, minWidth: 55 }}>
        {row.se_pivot_price != null ? `$${row.se_pivot_price.toFixed(2)}` : '-'}
      </TableCell>

      <TableCell align="center" sx={{ fontFamily: 'monospace', width: 40, minWidth: 40 }}>
        {row.rs_rating?.toFixed(0) || '-'}
      </TableCell>

      <TableCell align="center" sx={{ fontFamily: 'monospace', width: 40, minWidth: 40 }}>
        {row.rs_rating_1m?.toFixed(0) || '-'}
      </TableCell>

      <TableCell align="center" sx={{ fontFamily: 'monospace', width: 40, minWidth: 40 }}>
        {row.rs_rating_3m?.toFixed(0) || '-'}
      </TableCell>

      <TableCell align="center" sx={{ fontFamily: 'monospace', width: 45, minWidth: 45 }}>
        {row.rs_rating_12m?.toFixed(0) || '-'}
      </TableCell>

      <TableCell align="center" sx={{ fontFamily: 'monospace', width: 45, minWidth: 45 }}>
        {row.beta != null ? row.beta.toFixed(2) : '-'}
      </TableCell>

      <TableCell align="center" sx={{ fontFamily: 'monospace', width: 45, minWidth: 45 }}>
        {row.beta_adj_rs != null ? row.beta_adj_rs.toFixed(0) : '-'}
      </TableCell>

      <TableCell align="center" sx={{ fontFamily: 'monospace', color: getEpsRatingColor(row.eps_rating), width: 55, minWidth: 55 }}>
        {row.eps_rating != null ? row.eps_rating : '-'}
      </TableCell>

      <TableCell align="center" sx={{ width: 40, minWidth: 40 }}>
        {row.stage != null ? (
          <Box
            component="span"
            sx={{
              backgroundColor: getStageColor(row.stage),
              color: 'white',
              padding: '1px 4px',
              borderRadius: '2px',
              fontSize: '10px',
              fontWeight: 500,
            }}
          >
            S{row.stage}
          </Box>
        ) : (
          '-'
        )}
      </TableCell>

      <TableCell align="right" sx={{ fontFamily: 'monospace', width: 65, minWidth: 65 }}>
        {row.current_price ? `$${row.current_price.toFixed(2)}` : '-'}
      </TableCell>

      <TableCell align="right" sx={{ fontFamily: 'monospace', width: 60, minWidth: 60 }}>
        {formatLargeNumber(row.volume, '$')}
      </TableCell>

      <TableCell align="right" sx={{ fontFamily: 'monospace', width: 60, minWidth: 60 }}>
        {formatLargeNumber(row.market_cap, '$')}
      </TableCell>

      <TableCell align="center" sx={{ fontFamily: 'monospace', color: getIpoAgeColor(row.ipo_date), width: 50, minWidth: 50 }}>
        {formatIpoAge(row.ipo_date)}
      </TableCell>

      <TableCell align="center" sx={{ fontFamily: 'monospace', color: getGrowthColor(row.eps_growth_qq), width: 50, minWidth: 50 }}>
        {row.eps_growth_qq != null ? `${row.eps_growth_qq.toFixed(0)}%` : '-'}
      </TableCell>

      <TableCell align="center" sx={{ fontFamily: 'monospace', color: getGrowthColor(row.sales_growth_qq), width: 50, minWidth: 50 }}>
        {row.sales_growth_qq != null ? `${row.sales_growth_qq.toFixed(0)}%` : '-'}
      </TableCell>

      <TableCell align="center" sx={{ fontFamily: 'monospace', width: 50, minWidth: 50 }}>
        {row.adr_percent != null ? `${row.adr_percent.toFixed(1)}%` : '-'}
      </TableCell>

      <TableCell align="center" sx={{ width: 35, minWidth: 35 }}>
        {row.ma_alignment ? (
          <CheckIcon sx={{ fontSize: 14, color: 'success.main' }} />
        ) : (
          <CloseIcon sx={{ fontSize: 14, color: 'error.main' }} />
        )}
      </TableCell>

      <TableCell align="center" sx={{ width: 40, minWidth: 40 }}>
        {row.vcp_detected ? (
          <CheckIcon sx={{ fontSize: 14, color: 'success.main' }} />
        ) : (
          <CloseIcon sx={{ fontSize: 14, color: 'text.disabled' }} />
        )}
      </TableCell>

      <TableCell align="center" sx={{ fontFamily: 'monospace', width: 50, minWidth: 50 }}>
        {row.vcp_score != null ? row.vcp_score.toFixed(1) : '-'}
      </TableCell>

      <TableCell align="right" sx={{ fontFamily: 'monospace', width: 55, minWidth: 55 }}>
        {row.vcp_pivot != null ? row.vcp_pivot.toFixed(2) : '-'}
      </TableCell>

      <TableCell align="center" sx={{ width: 35, minWidth: 35 }}>
        {row.vcp_ready_for_breakout ? (
          <CheckIcon sx={{ fontSize: 14, color: 'success.main' }} />
        ) : (
          <CloseIcon sx={{ fontSize: 14, color: 'text.disabled' }} />
        )}
      </TableCell>

      <TableCell align="center" sx={{ width: 40, minWidth: 40 }}>
        {row.passes_template ? (
          <CheckIcon sx={{ fontSize: 14, color: 'success.main' }} />
        ) : (
          <CloseIcon sx={{ fontSize: 14, color: 'text.disabled' }} />
        )}
      </TableCell>

      <TableCell align="center" sx={{ width: 80, minWidth: 80 }}>
        <Chip
          label={row.rating}
          color={getRatingColor(row.rating)}
          size="small"
        />
      </TableCell>
    </TableRow>
  );
}, (prevProps, nextProps) => {
  // Custom comparison - only re-render if the row data actually changed
  return prevProps.row.symbol === nextProps.row.symbol &&
         prevProps.row.composite_score === nextProps.row.composite_score &&
         prevProps.row.rs_rating === nextProps.row.rs_rating &&
         prevProps.row.current_price === nextProps.row.current_price &&
         prevProps.row.price_change_1d === nextProps.row.price_change_1d;
});

/**
 * Display scan results in a sortable, paginated table with row virtualization
 * @param {Function} onRowHover - Optional callback when hovering over a row (for prefetching)
 */
function ResultsTable({ results, total, page, perPage, sortBy, sortOrder, onPageChange, onPerPageChange, onSortChange, onOpenChart, loading, onRowHover }) {
  const parentRef = useRef(null);

  const handleChangePage = useCallback((event, newPage) => {
    onPageChange(newPage + 1); // Material-UI uses 0-based pages, API uses 1-based
  }, [onPageChange]);

  const handleRequestSort = useCallback((property) => {
    const isAsc = sortBy === property && sortOrder === 'asc';
    onSortChange(property, isAsc ? 'desc' : 'asc');
  }, [sortBy, sortOrder, onSortChange]);

  const handleRowClick = useCallback((symbol) => {
    onOpenChart(symbol);
  }, [onOpenChart]);

  // Virtualize rows - only render visible rows plus overscan
  const rowVirtualizer = useVirtualizer({
    count: results?.length || 0,
    getScrollElement: () => parentRef.current,
    estimateSize: () => ROW_HEIGHT,
    overscan: 10, // Render 10 extra rows above/below viewport
  });

  // Memoize virtual items to prevent recalculation
  const virtualRows = rowVirtualizer.getVirtualItems();

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', p: 5 }}>
        <CircularProgress />
      </Box>
    );
  }

  if (!results || results.length === 0) {
    return (
      <Paper sx={{ p: 3, textAlign: 'center' }}>
        <Typography variant="body1" color="text.secondary">
          No results found
        </Typography>
      </Paper>
    );
  }

  return (
    <Paper elevation={1}>
      <TableContainer
        ref={parentRef}
        sx={{
          maxHeight: 'calc(100vh - 280px)',
          overflow: 'auto',
        }}
      >
        <Table stickyHeader size="small" sx={{ minWidth: 2230 }}>
          <TableHead>
            <TableRow>
              {columns.map((column) => (
                <TableCell
                  key={column.id}
                  align={column.id === 'symbol' ? 'left' : 'center'}
                  sx={{
                    width: column.width,
                    minWidth: column.width,
                    maxWidth: column.width,
                    whiteSpace: 'nowrap',
                  }}
                >
                  {column.sortable ? (
                    <TableSortLabel
                      active={sortBy === column.id}
                      direction={sortBy === column.id ? sortOrder : 'asc'}
                      onClick={() => handleRequestSort(column.id)}
                    >
                      {column.label}
                    </TableSortLabel>
                  ) : (
                    column.label
                  )}
                </TableCell>
              ))}
            </TableRow>
          </TableHead>
          <TableBody>
            {/* Spacer for virtualization - pushes content down to correct position */}
            {virtualRows.length > 0 && virtualRows[0].start > 0 && (
              <tr style={{ height: virtualRows[0].start }} />
            )}
            {virtualRows.map((virtualRow) => {
              const row = results[virtualRow.index];
              return (
                <VirtualTableRow
                  key={row.symbol}
                  row={row}
                  onRowClick={handleRowClick}
                  onRowHover={onRowHover}
                  onOpenChart={onOpenChart}
                />
              );
            })}
            {/* Bottom spacer for virtualization */}
            {virtualRows.length > 0 && (
              <tr style={{ height: rowVirtualizer.getTotalSize() - (virtualRows[virtualRows.length - 1]?.end || 0) }} />
            )}
          </TableBody>
        </Table>
      </TableContainer>

      <TablePagination
        rowsPerPageOptions={[10, 25, 50, 100]}
        component="div"
        count={total}
        rowsPerPage={perPage}
        page={page - 1} // Material-UI uses 0-based pages
        onPageChange={handleChangePage}
        onRowsPerPageChange={(e) => {
          const nextPerPage = Number(e.target.value);
          onPerPageChange?.(nextPerPage);
          onPageChange(1); // Reset to first page when changing per-page
        }}
      />
    </Paper>
  );
}

// Wrap with React.memo for component-level memoization
export default memo(ResultsTable, (prevProps, nextProps) => {
  // Only re-render if these key props change
  return (
    prevProps.results === nextProps.results &&
    prevProps.total === nextProps.total &&
    prevProps.page === nextProps.page &&
    prevProps.perPage === nextProps.perPage &&
    prevProps.sortBy === nextProps.sortBy &&
    prevProps.sortOrder === nextProps.sortOrder &&
    prevProps.loading === nextProps.loading
  );
});
