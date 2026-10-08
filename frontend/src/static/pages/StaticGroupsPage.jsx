import { lazy, Suspense, useEffect, useMemo, useState } from 'react';
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
  TableSortLabel,
  Typography,
} from '@mui/material';
import {
  useStaticManifest,
  fetchStaticJson,
  resolveStaticMarketEntry,
  staticQueryOptions,
  useStaticGeneration,
  useStaticGroupsRRG,
  useStaticGroupMatrix,
} from '../dataClient';
import { useStaticChartIndex } from '../chartClient';
import StaticGroupDetailModal from '../StaticGroupDetailModal';
import RRGChart from '../../components/Charts/RRGChart';
import GroupViewTabs from '../../features/groups/GroupViewTabs';
import { DEFAULT_MATRIX_PREFERENCES } from '../../features/groups/matrix/groupMatrixModel';
import { useRRGScopeSelection } from '../../components/Charts/useRRGScopeSelection';
import RankChangeCell from '../../components/shared/RankChangeCell';
import TickerCell from '../../components/common/TickerCell';
import { useStaticMarket } from '../StaticMarketContext';
import {
  STATIC_GROUP_RANKING_COLUMNS,
  formatGroupRs,
} from '../../features/groups/groupRankingFields';
import { sortGroupRankings } from '../../features/groups/groupRankingSort';
import {
  groupRsCellSx,
  groupRsTone,
} from '../../features/groups/groupRsVisualEncoding';

const GroupMatrixPanel = lazy(() => import('../../features/groups/matrix/GroupMatrixPanel'));

function SortableTableCell({ field, label, align = 'left', orderBy, order, onSort }) {
  return (
    <TableCell align={align}>
      <TableSortLabel
        active={orderBy === field}
        direction={orderBy === field ? order : 'asc'}
        onClick={() => onSort(field)}
      >
        {label}
      </TableSortLabel>
    </TableCell>
  );
}

function StaticRankingCell({ row, column }) {
  const align = column.staticAlign || column.align;

  if (column.kind === 'rank') {
    return (
      <TableCell align={align} sx={{ fontFamily: 'monospace', fontWeight: 600 }}>
        {row.rank}
      </TableCell>
    );
  }

  if (column.kind === 'group') {
    return <TableCell>{row.industry_group}</TableCell>;
  }

  if (column.kind === 'rs') {
    const value = row[column.field];
    return (
      <TableCell
        align={align}
        data-rs-tone={groupRsTone(value)}
        sx={{ fontFamily: 'monospace', ...groupRsCellSx(value) }}
      >
        {formatGroupRs(value)}
      </TableCell>
    );
  }

  if (column.kind === 'integer') {
    return (
      <TableCell align={align} sx={{ fontFamily: 'monospace' }}>
        {row[column.field]}
      </TableCell>
    );
  }

  if (column.kind === 'rankChange') {
    return (
      <TableCell align={align}>
        <RankChangeCell value={row[column.field]} />
      </TableCell>
    );
  }

  return (
    <TableCell align={align} sx={{ fontSize: '12px' }}>
      <TickerCell
        symbol={row.top_symbol}
        companyName={row.top_symbol_name}
      />
    </TableCell>
  );
}

function MoversCard({ title, rows }) {
  return (
    <Paper elevation={0} sx={{ p: 1.5, height: '100%', border: '1px solid', borderColor: 'divider' }}>
      <Typography variant="subtitle1" sx={{ fontWeight: 600, fontSize: '13px', textTransform: 'uppercase', letterSpacing: '0.5px', mb: 0.5 }}>
        {title}
      </Typography>
      <TableContainer>
        <Table size="small" sx={{ minWidth: 360 }}>
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

function GroupsTableView({ movers, moversPeriod, rankings, onSelectGroup }) {
  const [orderBy, setOrderBy] = useState('rank');
  const [order, setOrder] = useState('asc');
  const sortedRankings = useMemo(
    () => sortGroupRankings(rankings, orderBy, order),
    [rankings, orderBy, order],
  );

  const handleSort = (field) => {
    const isAsc = orderBy === field && order === 'asc';
    setOrder(isAsc ? 'desc' : 'asc');
    setOrderBy(field);
  };

  return (
    <>
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
          <Table size="small" sx={{ minWidth: 1040 }}>
            <TableHead>
              <TableRow>
                {STATIC_GROUP_RANKING_COLUMNS.map((column) => (
                  <SortableTableCell
                    key={column.field}
                    field={column.field}
                    label={column.staticLabel || column.label}
                    align={column.staticAlign || column.align}
                    orderBy={orderBy}
                    order={order}
                    onSort={handleSort}
                  />
                ))}
              </TableRow>
            </TableHead>
            <TableBody>
              {sortedRankings.map((row) => (
                <TableRow
                  key={row.industry_group}
                  hover
                  onClick={() => onSelectGroup(row.industry_group)}
                  tabIndex={0}
                  onKeyDown={(event) => {
                    if (event.key === 'Enter' || event.key === ' ') {
                      event.preventDefault();
                      onSelectGroup(row.industry_group);
                    }
                  }}
                  sx={{ cursor: 'pointer' }}
                >
                  {STATIC_GROUP_RANKING_COLUMNS.map((column) => (
                    <StaticRankingCell
                      key={column.field}
                      row={row}
                      column={column}
                    />
                  ))}
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      </Paper>
    </>
  );
}

function StaticGroupsPage() {
  const manifestQuery = useStaticManifest();
  const { selectedMarket } = useStaticMarket();
  const marketEntry = useMemo(
    () => resolveStaticMarketEntry(manifestQuery.data, selectedMarket),
    [manifestQuery.data, selectedMarket],
  );
  const { generation, dataRoot } = useStaticGeneration();
  const groupsQuery = useQuery(staticQueryOptions({
    key: ['staticGroups', marketEntry.pages?.groups?.path],
    generation,
    queryFn: () => fetchStaticJson(marketEntry.pages.groups.path, dataRoot),
    enabled: Boolean(marketEntry.pages?.groups?.path),
    gcTime: undefined,
  }));
  const chartIndexQuery = useStaticChartIndex(marketEntry.assets?.charts?.path);
  const rrgQuery = useStaticGroupsRRG(marketEntry);
  const rrgAvailable = Boolean(marketEntry.assets?.groups_rrg?.path);
  const [selectedGroup, setSelectedGroup] = useState(null);
  const [view, setView] = useState('table');
  const [matrixPreferences, setMatrixPreferences] = useState(DEFAULT_MATRIX_PREFERENCES);
  const matrixAvailable = Boolean(marketEntry.assets?.groups_matrix?.path);
  const matrixQuery = useStaticGroupMatrix(marketEntry, view === 'matrix');
  useEffect(() => {
    setSelectedGroup(null);
    if (!matrixAvailable && view === 'matrix') setView('table');
  }, [selectedMarket, matrixAvailable, view]);
  const [rrgScope, setRrgScope] = useState('groups'); // 'groups' | 'sectors'
  const { availableScopes: availableRrgScopes } = useRRGScopeSelection({
    view,
    scope: rrgScope,
    setView,
    setScope: setRrgScope,
    rrgAvailable,
    bundle: rrgQuery.data,
  });

  if (manifestQuery.isLoading) {
    return (
      <Box display="flex" justifyContent="center" py={8}>
        <CircularProgress />
      </Box>
    );
  }

  if (manifestQuery.isError) {
    return <Alert severity="error">Failed to load group rankings.</Alert>;
  }

  const payload = groupsQuery.data?.payload || {};
  const rankings = payload.rankings?.rankings || [];
  const movers = payload.movers || {};
  const moversPeriod = payload.movers_period || movers.period || '1w';
  const groupDetails = payload.group_details || {};

  return (
    <Box>
      <Typography variant="h5" sx={{ fontWeight: 700, letterSpacing: '-0.5px', mb: 0.5 }}>
        {marketEntry.display_name} Group Rankings
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 2, fontSize: '12px' }}>
        Latest ranking date: {payload.rankings?.date || '-'}.
      </Typography>

      <GroupViewTabs
          view={view}
          onView={setView}
          scope={rrgScope}
          onScope={setRrgScope}
          rrgAvailable={rrgAvailable}
          matrixAvailable={matrixAvailable}
          availableScopes={availableRrgScopes}
          sx={{ mb: 2 }}
        />

      <Box role="tabpanel" id={`group-panel-${view}`} aria-labelledby={`group-tab-${view}`}>
      {view === 'matrix' ? (
        <Suspense fallback={<CircularProgress />}>
          <GroupMatrixPanel key={selectedMarket} data={matrixQuery.data} isLoading={matrixQuery.isLoading}
            error={matrixQuery.error} onRetry={() => matrixQuery.refetch()}
            preferences={matrixPreferences} onPreferencesChange={setMatrixPreferences} />
        </Suspense>
      ) : view === 'rrg' ? (
        <RRGChart
          data={rrgQuery.data?.payload?.[rrgScope]}
          isLoading={rrgQuery.isLoading}
          error={rrgQuery.isError ? rrgQuery.error : null}
          onSelectGroup={(name) => rrgScope === 'groups' && setSelectedGroup(name)}
        />
      ) : groupsQuery.isLoading ? <CircularProgress />
        : groupsQuery.isError ? <Alert severity="error">Failed to load group rankings.</Alert>
        : !groupsQuery.data?.available ? <Alert severity="info">{groupsQuery.data?.message || 'No group rankings are available.'}</Alert>
        : (
        <GroupsTableView
          movers={movers}
          moversPeriod={moversPeriod}
          rankings={rankings}
          onSelectGroup={setSelectedGroup}
        />
      )}

      </Box>

      <StaticGroupDetailModal
        group={selectedGroup}
        detail={selectedGroup ? groupDetails[selectedGroup] : null}
        chartIndex={chartIndexQuery.data}
        open={!!selectedGroup}
        onClose={() => setSelectedGroup(null)}
      />
    </Box>
  );
}

export default StaticGroupsPage;
