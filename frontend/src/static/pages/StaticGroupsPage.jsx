import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  Alert,
  Box,
  CircularProgress,
  Grid,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Typography,
} from '@mui/material';
import { useStaticManifest, fetchStaticJson } from '../dataClient';
import StaticGroupDetailModal from '../StaticGroupDetailModal';
import RankChangeCell from '../../components/shared/RankChangeCell';

function MoversCard({ title, rows }) {
  return (
    <Paper elevation={0} sx={{ p: 1.5, height: '100%', border: '1px solid', borderColor: 'divider' }}>
      <Typography variant="subtitle1" sx={{ fontWeight: 600, fontSize: '13px', textTransform: 'uppercase', letterSpacing: '0.5px', mb: 0.5 }}>
        {title}
      </Typography>
      <TableContainer>
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>Group</TableCell>
              <TableCell align="right">Rank</TableCell>
              <TableCell align="right">Change</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {(rows || []).slice(0, 5).map((row) => (
              <TableRow key={`${title}-${row.industry_group}`}>
                <TableCell>{row.industry_group}</TableCell>
                <TableCell align="right">{row.rank}</TableCell>
                <TableCell align="right">
                  <RankChangeCell value={row.rank_change_1w} />
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
    </Paper>
  );
}

function StaticGroupsPage() {
  const manifestQuery = useStaticManifest();
  const groupsQuery = useQuery({
    queryKey: ['staticGroups', manifestQuery.data?.pages?.groups?.path],
    queryFn: () => fetchStaticJson(manifestQuery.data.pages.groups.path),
    enabled: Boolean(manifestQuery.data?.pages?.groups?.path),
    staleTime: Infinity,
  });
  const [selectedGroup, setSelectedGroup] = useState(null);

  if (manifestQuery.isLoading || groupsQuery.isLoading) {
    return (
      <Box display="flex" justifyContent="center" py={8}>
        <CircularProgress />
      </Box>
    );
  }

  if (manifestQuery.isError || groupsQuery.isError) {
    return <Alert severity="error">Failed to load group rankings.</Alert>;
  }

  if (!groupsQuery.data?.available) {
    return <Alert severity="info">{groupsQuery.data?.message || 'No group rankings are available.'}</Alert>;
  }

  const payload = groupsQuery.data.payload || {};
  const rankings = payload.rankings?.rankings || [];
  const movers = payload.movers || {};
  const moversPeriod = payload.movers_period || movers.period || '3m';
  const groupDetails = payload.group_details || {};

  return (
    <Box>
      <Typography variant="h5" sx={{ fontWeight: 700, letterSpacing: '-0.5px', mb: 0.5 }}>
        Industry Group Rankings
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 2, fontSize: '12px' }}>
        Latest ranking date: {payload.rankings?.date || '-'}.
      </Typography>

      <Grid container spacing={1.5} sx={{ mb: 2 }}>
        <Grid item xs={12} md={6}>
          <MoversCard title={`Top Gainers (${moversPeriod.toUpperCase()})`} rows={movers.gainers} />
        </Grid>
        <Grid item xs={12} md={6}>
          <MoversCard title={`Top Losers (${moversPeriod.toUpperCase()})`} rows={movers.losers} />
        </Grid>
      </Grid>

      <Paper elevation={0} sx={{ p: 1.5, border: '1px solid', borderColor: 'divider' }}>
        <Typography variant="subtitle1" sx={{ fontWeight: 600, fontSize: '13px', textTransform: 'uppercase', letterSpacing: '0.5px', mb: 0.5 }}>
          Current Rankings
        </Typography>
        <TableContainer>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell align="center">Rank</TableCell>
                <TableCell>Group</TableCell>
                <TableCell align="center">Avg RS</TableCell>
                <TableCell align="center">Stocks</TableCell>
                <TableCell align="right">1W</TableCell>
                <TableCell align="right">1M</TableCell>
                <TableCell align="right">3M</TableCell>
                <TableCell align="right">6M</TableCell>
                <TableCell>Top Stock</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {rankings.map((row) => (
                <TableRow
                  key={row.industry_group}
                  hover
                  onClick={() => setSelectedGroup(row.industry_group)}
                  tabIndex={0}
                  onKeyDown={(event) => {
                    if (event.key === 'Enter' || event.key === ' ') {
                      event.preventDefault();
                      setSelectedGroup(row.industry_group);
                    }
                  }}
                  sx={{ cursor: 'pointer' }}
                >
                  <TableCell align="center" sx={{ fontFamily: 'monospace', fontWeight: 600 }}>{row.rank}</TableCell>
                  <TableCell>{row.industry_group}</TableCell>
                  <TableCell align="center" sx={{ fontFamily: 'monospace' }}>{row.avg_rs_rating?.toFixed?.(1) ?? '-'}</TableCell>
                  <TableCell align="center" sx={{ fontFamily: 'monospace' }}>{row.num_stocks}</TableCell>
                  <TableCell align="right"><RankChangeCell value={row.rank_change_1w} /></TableCell>
                  <TableCell align="right"><RankChangeCell value={row.rank_change_1m} /></TableCell>
                  <TableCell align="right"><RankChangeCell value={row.rank_change_3m} /></TableCell>
                  <TableCell align="right"><RankChangeCell value={row.rank_change_6m} /></TableCell>
                  <TableCell sx={{ fontWeight: 500, color: 'text.secondary', fontSize: '12px' }}>
                    {row.top_symbol || '-'}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      </Paper>

      <StaticGroupDetailModal
        group={selectedGroup}
        detail={selectedGroup ? groupDetails[selectedGroup] : null}
        open={!!selectedGroup}
        onClose={() => setSelectedGroup(null)}
      />
    </Box>
  );
}

export default StaticGroupsPage;
