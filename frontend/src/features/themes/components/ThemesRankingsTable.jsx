import {
  Box,
  Chip,
  CircularProgress,
  IconButton,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TablePagination,
  TableRow,
  TableSortLabel,
  Tooltip,
} from '@mui/material';
import TimelineIcon from '@mui/icons-material/Timeline';
import { THEME_STATUS_COLORS } from '../constants';
import { MomentumBar, VelocityIndicator } from './ThemeInsightsCards';

const PAGE_SIZE = 50;

export default function ThemesRankingsTable({
  isLoading,
  rankingsData,
  sortedRankings,
  orderBy,
  order,
  onSort,
  page,
  onPageChange,
  selectedPipeline,
  onSelectTheme,
  onOpenSources,
}) {
  if (isLoading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="300px">
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Paper elevation={1}>
      <Box
        sx={{
          p: 1.5,
          borderBottom: 1,
          borderColor: 'divider',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
        }}
      >
        <Box sx={{ fontSize: '14px', fontWeight: 600 }}>Theme Rankings</Box>
        {rankingsData && (
          <Box display="flex" gap={1} alignItems="center">
            <Chip
              label={selectedPipeline === 'technical' ? 'Technical' : 'Fundamental'}
              size="small"
              color={selectedPipeline === 'technical' ? 'primary' : 'secondary'}
            />
            <Chip label={`${rankingsData.total_themes} themes | ${rankingsData.date || 'Today'}`} size="small" />
          </Box>
        )}
      </Box>
      <TableContainer sx={{ maxHeight: 'calc(100vh - 400px)' }}>
        <Table stickyHeader size="small">
          <TableHead>
            <TableRow>
              <TableCell>
                <TableSortLabel active={orderBy === 'rank'} direction={orderBy === 'rank' ? order : 'asc'} onClick={() => onSort('rank')}>
                  #
                </TableSortLabel>
              </TableCell>
              <TableCell>Theme</TableCell>
              <TableCell>Status</TableCell>
              <TableCell align="center">
                <TableSortLabel
                  active={orderBy === 'momentum_score'}
                  direction={orderBy === 'momentum_score' ? order : 'asc'}
                  onClick={() => onSort('momentum_score')}
                >
                  Mom
                </TableSortLabel>
              </TableCell>
              <TableCell align="right">
                <TableSortLabel
                  active={orderBy === 'mention_velocity'}
                  direction={orderBy === 'mention_velocity' ? order : 'asc'}
                  onClick={() => onSort('mention_velocity')}
                >
                  Vel
                </TableSortLabel>
              </TableCell>
              <TableCell align="right">
                <TableSortLabel
                  active={orderBy === 'mentions_7d'}
                  direction={orderBy === 'mentions_7d' ? order : 'asc'}
                  onClick={() => onSort('mentions_7d')}
                >
                  7D
                </TableSortLabel>
              </TableCell>
              <TableCell align="right">
                <TableSortLabel
                  active={orderBy === 'basket_rs_vs_spy'}
                  direction={orderBy === 'basket_rs_vs_spy' ? order : 'asc'}
                  onClick={() => onSort('basket_rs_vs_spy')}
                >
                  RS
                </TableSortLabel>
              </TableCell>
              <TableCell align="right">
                <TableSortLabel
                  active={orderBy === 'basket_return_1w'}
                  direction={orderBy === 'basket_return_1w' ? order : 'asc'}
                  onClick={() => onSort('basket_return_1w')}
                >
                  1W
                </TableSortLabel>
              </TableCell>
              <TableCell align="right">
                <TableSortLabel
                  active={orderBy === 'pct_above_50ma'}
                  direction={orderBy === 'pct_above_50ma' ? order : 'asc'}
                  onClick={() => onSort('pct_above_50ma')}
                >
                  50MA
                </TableSortLabel>
              </TableCell>
              <TableCell align="right">#</TableCell>
              <TableCell>Tickers</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {sortedRankings.map((row) => (
              <TableRow
                key={row.theme}
                hover
                onClick={() => onOpenSources({ id: row.theme_cluster_id, name: row.theme })}
                sx={{ cursor: 'pointer' }}
              >
                <TableCell>
                  <Box
                    component="span"
                    sx={{
                      backgroundColor:
                        row.rank <= 5 ? 'success.main' : row.rank <= 10 ? 'warning.main' : 'error.main',
                      color: 'white',
                      padding: '1px 4px',
                      borderRadius: '2px',
                      fontSize: '10px',
                      fontWeight: 600,
                      fontFamily: 'monospace',
                    }}
                  >
                    {row.rank}
                  </Box>
                </TableCell>
                <TableCell sx={{ maxWidth: 180, overflow: 'hidden' }}>
                  <Box display="flex" alignItems="center" gap={0.5} sx={{ minWidth: 0 }}>
                    <Box
                      sx={{
                        fontWeight: 500,
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        whiteSpace: 'nowrap',
                        flex: 1,
                      }}
                      title={row.theme}
                    >
                      {row.theme}
                    </Box>
                    <Tooltip title="View details">
                      <IconButton
                        size="small"
                        onClick={(event) => {
                          event.stopPropagation();
                          onSelectTheme({ id: row.theme_cluster_id, name: row.theme });
                        }}
                        sx={{ p: 0.25 }}
                        aria-label={`View details for ${row.theme}`}
                      >
                        <TimelineIcon sx={{ fontSize: 14 }} />
                      </IconButton>
                    </Tooltip>
                  </Box>
                  {row.first_seen && (
                    <Box sx={{ fontSize: '9px', color: 'text.secondary' }}>
                      Since {new Date(row.first_seen).toLocaleDateString()}
                    </Box>
                  )}
                </TableCell>
                <TableCell>
                  <Box
                    component="span"
                    sx={{
                      fontSize: '9px',
                      padding: '1px 4px',
                      borderRadius: '2px',
                      border: '1px solid',
                      borderColor: THEME_STATUS_COLORS[row.status]
                        ? `${THEME_STATUS_COLORS[row.status]}.main`
                        : 'grey.400',
                      color: THEME_STATUS_COLORS[row.status]
                        ? `${THEME_STATUS_COLORS[row.status]}.main`
                        : 'text.secondary',
                    }}
                  >
                    {row.status}
                  </Box>
                </TableCell>
                <TableCell align="center">
                  <MomentumBar score={row.momentum_score} />
                </TableCell>
                <TableCell align="right">
                  <VelocityIndicator velocity={row.mention_velocity} />
                </TableCell>
                <TableCell align="right" sx={{ fontFamily: 'monospace' }}>
                  {row.mentions_7d}
                </TableCell>
                <TableCell
                  align="right"
                  sx={{
                    fontFamily: 'monospace',
                    fontWeight: 600,
                    color:
                      row.basket_rs_vs_spy >= 60
                        ? 'success.main'
                        : row.basket_rs_vs_spy <= 40
                          ? 'error.main'
                          : 'text.primary',
                  }}
                >
                  {row.basket_rs_vs_spy?.toFixed(0)}
                </TableCell>
                <TableCell
                  align="right"
                  sx={{
                    fontFamily: 'monospace',
                    color:
                      row.basket_return_1w > 0
                        ? 'success.main'
                        : row.basket_return_1w < 0
                          ? 'error.main'
                          : 'text.primary',
                  }}
                >
                  {row.basket_return_1w > 0 ? '+' : ''}
                  {row.basket_return_1w?.toFixed(1)}%
                </TableCell>
                <TableCell
                  align="right"
                  sx={{
                    fontFamily: 'monospace',
                    color:
                      row.pct_above_50ma >= 70
                        ? 'success.main'
                        : row.pct_above_50ma <= 30
                          ? 'error.main'
                          : 'text.primary',
                  }}
                >
                  {row.pct_above_50ma?.toFixed(0)}%
                </TableCell>
                <TableCell align="right" sx={{ fontFamily: 'monospace' }}>
                  {row.num_constituents}
                </TableCell>
                <TableCell>
                  <Box display="flex" gap={0.25} flexWrap="nowrap">
                    {row.top_tickers?.slice(0, 8).map((ticker) => (
                      <Box
                        key={ticker}
                        component="span"
                        sx={{
                          fontSize: '9px',
                          padding: '1px 3px',
                          backgroundColor: '#1976d2',
                          color: 'white',
                          borderRadius: '2px',
                        }}
                      >
                        {ticker}
                      </Box>
                    ))}
                    {row.top_tickers?.length > 8 && (
                      <Box component="span" sx={{ fontSize: '9px', color: 'text.secondary' }}>
                        +{row.top_tickers.length - 8}
                      </Box>
                    )}
                  </Box>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
      <TablePagination
        component="div"
        count={rankingsData?.total_themes || 0}
        page={page}
        onPageChange={(event, newPage) => onPageChange(newPage)}
        rowsPerPage={PAGE_SIZE}
        rowsPerPageOptions={[PAGE_SIZE]}
        labelDisplayedRows={({ from, to, count }) => `${from}-${to} of ${count !== -1 ? count : `more than ${to}`}`}
      />
    </Paper>
  );
}
