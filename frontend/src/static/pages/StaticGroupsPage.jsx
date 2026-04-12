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
    <Paper sx={{ p: 2, height: '100%' }}>
      <Typography variant="h6" gutterBottom>
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
      <Typography variant="h4" gutterBottom>
        Industry Group Rankings
      </Typography>
      <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
        Latest ranking date: {payload.rankings?.date || '-'}.
      </Typography>

      <Grid container spacing={2} sx={{ mb: 3 }}>
        <Grid item xs={12} md={6}>
          <MoversCard title={`Top Gainers (${moversPeriod.toUpperCase()})`} rows={movers.gainers} />
        </Grid>
        <Grid item xs={12} md={6}>
          <MoversCard title={`Top Losers (${moversPeriod.toUpperCase()})`} rows={movers.losers} />
        </Grid>
      </Grid>

      <Paper sx={{ p: 2 }}>
        <Typography variant="h6" gutterBottom>
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
