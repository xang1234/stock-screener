import {
  Box,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TableSortLabel,
} from '@mui/material';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import TrendingDownIcon from '@mui/icons-material/TrendingDown';
import { GROUP_RS_FIELDS, formatGroupRs } from './groupRankingFields';

const RankChangeCell = ({ value }) => {
  if (value === null || value === undefined) {
    return <Box sx={{ color: 'text.secondary', fontFamily: 'monospace' }}>-</Box>;
  }
  const color = value > 0 ? 'success.main' : value < 0 ? 'error.main' : 'text.secondary';
  const prefix = value > 0 ? '+' : '';
  return (
    <Box display="flex" alignItems="center" justifyContent="flex-end" sx={{ fontFamily: 'monospace' }}>
      {value > 0 && <TrendingUpIcon sx={{ fontSize: 12, mr: 0.25, color }} />}
      {value < 0 && <TrendingDownIcon sx={{ fontSize: 12, mr: 0.25, color }} />}
      <Box component="span" sx={{ color, fontWeight: value !== 0 ? 600 : 400, fontSize: '11px' }}>
        {prefix}{value}
      </Box>
    </Box>
  );
};

const HistoricalRankCell = ({ currentRank, rankChange }) => {
  if (rankChange == null || currentRank == null) {
    return <Box sx={{ color: 'text.secondary', fontFamily: 'monospace', fontSize: '11px' }}>-</Box>;
  }
  return (
    <Box sx={{ fontFamily: 'monospace', fontSize: '11px', textAlign: 'right' }}>
      {currentRank + rankChange}
    </Box>
  );
};

export default function LiveGroupRankingsTable({
  rankings,
  order,
  orderBy,
  onSort,
  onSelectGroup,
  showHistoricalRanks,
}) {
  return (
            <TableContainer sx={{ maxHeight: 'calc(100vh - 200px)', flexGrow: 1 }}>
              <Table stickyHeader size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>
                      <TableSortLabel
                        active={orderBy === 'rank'}
                        direction={orderBy === 'rank' ? order : 'asc'}
                        onClick={() => onSort('rank')}
                      >
                        Rank
                      </TableSortLabel>
                    </TableCell>
                    <TableCell>Industry Group</TableCell>
                    {GROUP_RS_FIELDS.map(({ field, label }) => (
                      <TableCell key={field} align="right">
                        <TableSortLabel
                          active={orderBy === field}
                          direction={orderBy === field ? order : 'asc'}
                          onClick={() => onSort(field)}
                        >
                          {label}
                        </TableSortLabel>
                      </TableCell>
                    ))}
                    <TableCell align="right">
                      <TableSortLabel
                        active={orderBy === 'median_rs_rating'}
                        direction={orderBy === 'median_rs_rating' ? order : 'asc'}
                        onClick={() => onSort('median_rs_rating')}
                      >
                        Med RS
                      </TableSortLabel>
                    </TableCell>
                    <TableCell align="right">
                      <TableSortLabel
                        active={orderBy === 'weighted_avg_rs_rating'}
                        direction={orderBy === 'weighted_avg_rs_rating' ? order : 'asc'}
                        onClick={() => onSort('weighted_avg_rs_rating')}
                      >
                        Wtd RS
                      </TableSortLabel>
                    </TableCell>
                    <TableCell align="right">
                      <TableSortLabel
                        active={orderBy === 'rs_std_dev'}
                        direction={orderBy === 'rs_std_dev' ? order : 'asc'}
                        onClick={() => onSort('rs_std_dev')}
                      >
                        Disp
                      </TableSortLabel>
                    </TableCell>
                    <TableCell align="right">
                      <TableSortLabel
                        active={orderBy === 'num_stocks'}
                        direction={orderBy === 'num_stocks' ? order : 'asc'}
                        onClick={() => onSort('num_stocks')}
                      >
                        #
                      </TableSortLabel>
                    </TableCell>
                    <TableCell align="right">
                      <TableSortLabel
                        active={orderBy === 'pct_rs_above_80'}
                        direction={orderBy === 'pct_rs_above_80' ? order : 'asc'}
                        onClick={() => onSort('pct_rs_above_80')}
                      >
                        80+%
                      </TableSortLabel>
                    </TableCell>
                    <TableCell align="right">Top</TableCell>
                    <TableCell align="right">
                      <TableSortLabel
                        active={orderBy === 'rank_change_1w'}
                        direction={orderBy === 'rank_change_1w' ? order : 'asc'}
                        onClick={() => onSort('rank_change_1w')}
                      >
                        1W
                      </TableSortLabel>
                    </TableCell>
                    <TableCell align="right">
                      <TableSortLabel
                        active={orderBy === 'rank_change_1m'}
                        direction={orderBy === 'rank_change_1m' ? order : 'asc'}
                        onClick={() => onSort('rank_change_1m')}
                      >
                        1M Δ
                      </TableSortLabel>
                    </TableCell>
                    <TableCell align="right">
                      <TableSortLabel
                        active={orderBy === 'rank_change_3m'}
                        direction={orderBy === 'rank_change_3m' ? order : 'asc'}
                        onClick={() => onSort('rank_change_3m')}
                      >
                        3M Δ
                      </TableSortLabel>
                    </TableCell>
                    <TableCell align="right">
                      <TableSortLabel
                        active={orderBy === 'rank_change_6m'}
                        direction={orderBy === 'rank_change_6m' ? order : 'asc'}
                        onClick={() => onSort('rank_change_6m')}
                      >
                        6M
                      </TableSortLabel>
                    </TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {rankings.map((row) => (
                    <TableRow
                      key={row.industry_group}
                      hover
                      onClick={() => onSelectGroup(row.industry_group)}
                      sx={{ cursor: 'pointer' }}
                    >
                      <TableCell>
                        <Box
                          component="span"
                          sx={{
                            backgroundColor: row.rank <= 20 ? 'success.main' : row.rank >= 177 ? 'error.main' : 'warning.main',
                            color: row.rank <= 20 || row.rank >= 177 ? 'white' : 'warning.contrastText',
                            padding: '1px 5px',
                            borderRadius: '2px',
                            fontSize: '10px',
                            fontWeight: 600,
                            fontFamily: 'monospace',
                          }}
                        >
                          {row.rank}
                        </Box>
                      </TableCell>
                      <TableCell sx={{ maxWidth: 180, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                        {row.industry_group}
                      </TableCell>
                      {GROUP_RS_FIELDS.map(({ field }) => (
                        <TableCell
                          key={field}
                          align="right"
                          sx={{
                            fontFamily: 'monospace',
                            ...(field === 'avg_rs_rating' && {
                              fontWeight: 600,
                              color: row[field] >= 70
                                ? 'success.main'
                                : row[field] <= 30
                                  ? 'error.main'
                                  : 'text.primary',
                            }),
                          }}
                        >
                          {formatGroupRs(row[field])}
                        </TableCell>
                      ))}
                      <TableCell align="right" sx={{ fontFamily: 'monospace' }}>
                        {row.median_rs_rating != null ? row.median_rs_rating.toFixed(1) : '-'}
                      </TableCell>
                      <TableCell align="right" sx={{ fontFamily: 'monospace' }}>
                        {row.weighted_avg_rs_rating != null ? row.weighted_avg_rs_rating.toFixed(1) : '-'}
                      </TableCell>
                      <TableCell align="right" sx={{ fontFamily: 'monospace' }}>
                        {row.rs_std_dev != null ? row.rs_std_dev.toFixed(1) : '-'}
                      </TableCell>
                      <TableCell align="right" sx={{ fontFamily: 'monospace' }}>{row.num_stocks}</TableCell>
                      <TableCell align="right" sx={{ fontFamily: 'monospace' }}>
                        {row.pct_rs_above_80 != null
                          ? `${row.pct_rs_above_80.toFixed(1)}%`
                          : row.num_stocks
                            ? `${(((row.num_stocks_rs_above_80 ?? 0) / row.num_stocks) * 100).toFixed(1)}%`
                            : '-'}
                      </TableCell>
                      <TableCell align="right" sx={{ color: 'text.secondary', fontWeight: 500 }}>
                        {row.top_symbol || '-'}
                      </TableCell>
                      <TableCell align="right">
                        {showHistoricalRanks ? (
                          <HistoricalRankCell currentRank={row.rank} rankChange={row.rank_change_1w} />
                        ) : (
                          <RankChangeCell value={row.rank_change_1w} />
                        )}
                      </TableCell>
                      <TableCell align="right">
                        {showHistoricalRanks ? (
                          <HistoricalRankCell currentRank={row.rank} rankChange={row.rank_change_1m} />
                        ) : (
                          <RankChangeCell value={row.rank_change_1m} />
                        )}
                      </TableCell>
                      <TableCell align="right">
                        {showHistoricalRanks ? (
                          <HistoricalRankCell currentRank={row.rank} rankChange={row.rank_change_3m} />
                        ) : (
                          <RankChangeCell value={row.rank_change_3m} />
                        )}
                      </TableCell>
                      <TableCell align="right">
                        {showHistoricalRanks ? (
                          <HistoricalRankCell currentRank={row.rank} rankChange={row.rank_change_6m} />
                        ) : (
                          <RankChangeCell value={row.rank_change_6m} />
                        )}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
  );
}
