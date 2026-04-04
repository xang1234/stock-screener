import { useState, useMemo } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TableSortLabel,
  Paper,
  Typography,
  Box,
  CircularProgress,
  Chip,
  IconButton,
  Tooltip,
} from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import ShowChartIcon from '@mui/icons-material/ShowChart';
import { useQuery } from '@tanstack/react-query';
import apiClient from '../../api/client';
import { getStockPeers } from '../../api/stocks';
import RSSparkline from './RSSparkline';
import AddToWatchlistMenu from '../common/AddToWatchlistMenu';
import { getStageColor } from '../../utils/colorUtils';

/**
 * Fetch peer stocks — scan-based when scanId is provided, standalone otherwise.
 */
const fetchPeers = async (scanId, symbol) => {
  if (scanId) {
    const response = await apiClient.get(`/v1/scans/${scanId}/peers/${symbol}`);
    return response.data;
  }
  return getStockPeers(symbol);
};


/**
 * Peer Comparison Modal
 * Shows all stocks in the same IBD industry group
 */
function PeerComparisonModal({ open, onClose, scanId, symbol, onOpenChart }) {
  const { data: peers, isLoading, error } = useQuery({
    queryKey: ['peers', scanId || 'standalone', symbol],
    queryFn: () => fetchPeers(scanId, symbol),
    enabled: open && !!symbol,
    staleTime: 10 * 60 * 1000,
  });

  const [sortBy, setSortBy] = useState('composite_score');
  const [sortOrder, setSortOrder] = useState('desc');

  const handleRequestSort = (property) => {
    const isAsc = sortBy === property && sortOrder === 'asc';
    setSortOrder(isAsc ? 'desc' : 'asc');
    setSortBy(property);
  };

  const sortedPeers = useMemo(() => {
    if (!peers) return [];
    return [...peers].sort((a, b) => {
      let aVal = a[sortBy];
      let bVal = b[sortBy];

      // Handle null/undefined values - push them to the end
      if (aVal == null && bVal == null) return 0;
      if (aVal == null) return 1;
      if (bVal == null) return -1;

      // String comparison for symbol
      if (sortBy === 'symbol') {
        return sortOrder === 'asc'
          ? aVal.localeCompare(bVal)
          : bVal.localeCompare(aVal);
      }

      // Numeric comparison for all other fields
      return sortOrder === 'asc' ? aVal - bVal : bVal - aVal;
    });
  }, [peers, sortBy, sortOrder]);

  const industryGroup = peers && peers.length > 0 ? peers[0].ibd_industry_group : '';

  return (
    <Dialog
      open={open}
      onClose={onClose}
      maxWidth="lg"
      fullWidth
      PaperProps={{
        sx: { minHeight: '70vh', maxHeight: '90vh' }
      }}
    >
      <DialogTitle>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Box>
            <Typography variant="h6" fontWeight="bold">
              Industry Peers: {symbol}
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5 }}>
              {industryGroup || 'Loading...'}
            </Typography>
          </Box>
          <IconButton onClick={onClose}>
            <CloseIcon />
          </IconButton>
        </Box>
      </DialogTitle>

      <DialogContent dividers>
        {isLoading && (
          <Box sx={{ display: 'flex', justifyContent: 'center', p: 5 }}>
            <CircularProgress />
          </Box>
        )}

        {error && (
          <Typography color="error">
            Error loading peers: {error.message}
          </Typography>
        )}

        {peers && peers.length > 0 && (
          <>
            <Box sx={{ mb: 2 }}>
              <Typography variant="body2" color="text.secondary">
                Found {peers.length} stocks in this industry group
              </Typography>
            </Box>

            <TableContainer component={Paper} variant="outlined">
              <Table size="small" stickyHeader>
                <TableHead>
                  <TableRow>
                    <TableCell>Chart</TableCell>
                    <TableCell>
                      <TableSortLabel
                        active={sortBy === 'symbol'}
                        direction={sortBy === 'symbol' ? sortOrder : 'asc'}
                        onClick={() => handleRequestSort('symbol')}
                      >
                        Symbol
                      </TableSortLabel>
                    </TableCell>
                    <TableCell align="center">RS Trend</TableCell>
                    <TableCell align="center">
                      <TableSortLabel
                        active={sortBy === 'composite_score'}
                        direction={sortBy === 'composite_score' ? sortOrder : 'asc'}
                        onClick={() => handleRequestSort('composite_score')}
                      >
                        Composite Score
                      </TableSortLabel>
                    </TableCell>
                    <TableCell align="center">
                      <TableSortLabel
                        active={sortBy === 'minervini_score'}
                        direction={sortBy === 'minervini_score' ? sortOrder : 'asc'}
                        onClick={() => handleRequestSort('minervini_score')}
                      >
                        Minervini
                      </TableSortLabel>
                    </TableCell>
                    <TableCell align="center">
                      <TableSortLabel
                        active={sortBy === 'rs_rating_1m'}
                        direction={sortBy === 'rs_rating_1m' ? sortOrder : 'asc'}
                        onClick={() => handleRequestSort('rs_rating_1m')}
                      >
                        RS 1M
                      </TableSortLabel>
                    </TableCell>
                    <TableCell align="center">
                      <TableSortLabel
                        active={sortBy === 'rs_rating_3m'}
                        direction={sortBy === 'rs_rating_3m' ? sortOrder : 'asc'}
                        onClick={() => handleRequestSort('rs_rating_3m')}
                      >
                        RS 3M
                      </TableSortLabel>
                    </TableCell>
                    <TableCell align="center">
                      <TableSortLabel
                        active={sortBy === 'rs_rating_12m'}
                        direction={sortBy === 'rs_rating_12m' ? sortOrder : 'asc'}
                        onClick={() => handleRequestSort('rs_rating_12m')}
                      >
                        RS 12M
                      </TableSortLabel>
                    </TableCell>
                    <TableCell align="center">
                      <TableSortLabel
                        active={sortBy === 'stage'}
                        direction={sortBy === 'stage' ? sortOrder : 'asc'}
                        onClick={() => handleRequestSort('stage')}
                      >
                        Stage
                      </TableSortLabel>
                    </TableCell>
                    <TableCell align="center">
                      <TableSortLabel
                        active={sortBy === 'current_price'}
                        direction={sortBy === 'current_price' ? sortOrder : 'asc'}
                        onClick={() => handleRequestSort('current_price')}
                      >
                        Price
                      </TableSortLabel>
                    </TableCell>
                    <TableCell align="center">
                      <TableSortLabel
                        active={sortBy === 'eps_growth_qq'}
                        direction={sortBy === 'eps_growth_qq' ? sortOrder : 'asc'}
                        onClick={() => handleRequestSort('eps_growth_qq')}
                      >
                        EPS Growth
                      </TableSortLabel>
                    </TableCell>
                    <TableCell align="center">
                      <TableSortLabel
                        active={sortBy === 'sales_growth_qq'}
                        direction={sortBy === 'sales_growth_qq' ? sortOrder : 'asc'}
                        onClick={() => handleRequestSort('sales_growth_qq')}
                      >
                        Sales Growth
                      </TableSortLabel>
                    </TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {sortedPeers.map((peer) => (
                    <TableRow
                      key={peer.symbol}
                      sx={{
                        backgroundColor: peer.symbol === symbol ? 'action.selected' : 'inherit',
                        '&:hover': { backgroundColor: 'action.hover' }
                      }}
                    >
                      <TableCell>
                        <Tooltip title="View Chart">
                          <IconButton
                            size="small"
                            onClick={() => {
                              onClose();
                              onOpenChart(peer.symbol);
                            }}
                            sx={{ color: 'primary.main' }}
                          >
                            <ShowChartIcon fontSize="small" />
                          </IconButton>
                        </Tooltip>
                        <AddToWatchlistMenu symbols={peer.symbol} size="small" />
                      </TableCell>

                      <TableCell>
                        <Typography
                          variant="body2"
                          fontWeight={peer.symbol === symbol ? 'bold' : 'normal'}
                        >
                          {peer.symbol}
                        </Typography>
                        {peer.company_name && (
                          <Typography variant="caption" color="text.secondary">
                            {peer.company_name}
                          </Typography>
                        )}
                      </TableCell>

                      <TableCell align="center" sx={{ p: '4px' }}>
                        <RSSparkline
                          data={peer.rs_sparkline_data}
                          trend={peer.rs_trend}
                          width={100}
                          height={28}
                        />
                      </TableCell>

                      <TableCell align="center">
                        <Typography variant="body2" fontWeight="bold" color="primary">
                          {peer.composite_score?.toFixed(1) || '-'}
                        </Typography>
                      </TableCell>

                      <TableCell align="center">
                        <Typography variant="body2">
                          {peer.minervini_score?.toFixed(1) || '-'}
                        </Typography>
                      </TableCell>

                      <TableCell align="center">
                        <Typography variant="body2">
                          {peer.rs_rating_1m?.toFixed(1) || '-'}
                        </Typography>
                      </TableCell>

                      <TableCell align="center">
                        <Typography variant="body2">
                          {peer.rs_rating_3m?.toFixed(1) || '-'}
                        </Typography>
                      </TableCell>

                      <TableCell align="center">
                        <Typography variant="body2">
                          {peer.rs_rating_12m?.toFixed(1) || '-'}
                        </Typography>
                      </TableCell>

                      <TableCell align="center">
                        {peer.stage && peer.stage_name ? (
                          <Chip
                            label={`S${peer.stage}`}
                            size="small"
                            sx={{
                              backgroundColor: getStageColor(peer.stage),
                              color: 'white',
                              fontWeight: 'medium',
                              fontSize: '0.75rem',
                            }}
                          />
                        ) : (
                          '-'
                        )}
                      </TableCell>

                      <TableCell align="center">
                        <Typography variant="body2">
                          {peer.current_price ? `$${peer.current_price.toFixed(2)}` : '-'}
                        </Typography>
                      </TableCell>

                      <TableCell align="center">
                        <Typography
                          variant="body2"
                          color={
                            peer.eps_growth_qq >= 20 ? 'success.main' :
                            peer.eps_growth_qq >= 0 ? 'text.primary' : 'error.main'
                          }
                        >
                          {peer.eps_growth_qq != null ? `${peer.eps_growth_qq.toFixed(1)}%` : '-'}
                        </Typography>
                      </TableCell>

                      <TableCell align="center">
                        <Typography
                          variant="body2"
                          color={
                            peer.sales_growth_qq >= 20 ? 'success.main' :
                            peer.sales_growth_qq >= 0 ? 'text.primary' : 'error.main'
                          }
                        >
                          {peer.sales_growth_qq != null ? `${peer.sales_growth_qq.toFixed(1)}%` : '-'}
                        </Typography>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </>
        )}

        {peers && peers.length === 0 && (
          <Typography color="text.secondary" textAlign="center" sx={{ p: 3 }}>
            No peer data available for this stock.
          </Typography>
        )}
      </DialogContent>

      <DialogActions>
        <Button onClick={onClose}>Close</Button>
      </DialogActions>
    </Dialog>
  );
}

export default PeerComparisonModal;
