import { useMemo, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  Alert,
  Box,
  CircularProgress,
  Paper,
  Tab,
  Tabs,
  Typography,
} from '@mui/material';

import BreadthContextStrip from '../../components/Breadth/BreadthContextStrip';
import BreadthContributorDialog from '../../components/Breadth/BreadthContributorDialog';
import BreadthHistoryTable from '../../components/Breadth/BreadthHistoryTable';
import { useBreadthContributors } from '../../components/Breadth/useBreadthContributors';
import BreadthChart from '../../components/Charts/BreadthChart';
import BreadthGroupAttribution from '../components/BreadthGroupAttribution';
import {
  fetchStaticBreadthContributorIndex,
  fetchStaticBreadthContributors,
  fetchStaticJson,
  resolveStaticMarketEntry,
  staticQueryOptions,
  useStaticGeneration,
  useStaticManifest,
  withGeneration,
} from '../dataClient';
import { useStaticMarket } from '../StaticMarketContext';

const RANGE_DAYS = { '1M': 31, '3M': 90 };

function StaticBreadthPage() {
  const manifestQuery = useStaticManifest();
  const { selectedMarket } = useStaticMarket();
  const marketEntry = useMemo(
    () => resolveStaticMarketEntry(manifestQuery.data, selectedMarket),
    [manifestQuery.data, selectedMarket],
  );
  const { generation, dataRoot } = useStaticGeneration();
  const breadthQuery = useQuery(staticQueryOptions({
    key: ['staticBreadth', marketEntry.pages?.breadth?.path],
    generation,
    queryFn: () => fetchStaticJson(marketEntry.pages.breadth.path, dataRoot),
    enabled: Boolean(marketEntry.pages?.breadth?.path),
  }));
  const contributorIndexPath = marketEntry.assets?.breadth_contributors?.index_path;
  const contributorDrilldown = useBreadthContributors({
    market: marketEntry.market,
    indexQueryKey: contributorIndexPath
      ? withGeneration(['staticBreadthContributors', 'index', marketEntry.market, contributorIndexPath], generation)
      : null,
    loadIndex: contributorIndexPath
      ? () => fetchStaticBreadthContributorIndex(contributorIndexPath, dataRoot)
      : null,
    loadDate: contributorIndexPath
      ? (date) => fetchStaticBreadthContributors(contributorIndexPath, date, dataRoot)
      : null,
    indexStaleTime: Infinity,
    documentStaleTime: Infinity,
    documentKeySuffix: withGeneration([], generation),
  });
  const [timeRange, setTimeRange] = useState('1M');
  const [selectedTab, setSelectedTab] = useState(0);

  const payload = breadthQuery.data?.payload || {};
  const groupAttribution = payload.group_attribution || null;
  const attributionAvailable = Boolean(groupAttribution?.available);
  const displayName = marketEntry.display_name;
  const filteredChartData = useMemo(() => {
    const allData = payload.chart_data || payload.history_90d || [];
    return allData.slice(-(RANGE_DAYS[timeRange] || 31));
  }, [payload.chart_data, payload.history_90d, timeRange]);
  const filteredBenchmarkData = useMemo(() => {
    const allData = payload.benchmark_overlay ?? payload.spy_overlay ?? [];
    return allData.slice(-(RANGE_DAYS[timeRange] || 31));
  }, [payload.benchmark_overlay, payload.spy_overlay, timeRange]);
  const benchmarkLabel = payload.benchmark_symbol
    || (marketEntry.market === 'US' ? 'SPY' : 'Benchmark');
  if (manifestQuery.isLoading || breadthQuery.isLoading) {
    return (
      <Box display="flex" justifyContent="center" py={8}>
        <CircularProgress />
      </Box>
    );
  }
  if (manifestQuery.isError || breadthQuery.isError) {
    return <Alert severity="error">Failed to load breadth data.</Alert>;
  }
  if (breadthQuery.data?.available === false) {
    return (
      <Alert severity="info">
        {breadthQuery.data?.message || 'No breadth snapshot is available.'}
      </Alert>
    );
  }

  const current = payload.current || {};
  const history = payload.history_90d || [];
  return (
    <Box>
      <Typography variant="h5" sx={{ fontWeight: 700, letterSpacing: '-0.5px', mb: 0.5 }}>
        {displayName} Breadth
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 2, fontSize: 12 }}>
        Breadth snapshot published {breadthQuery.data.published_at || breadthQuery.data.generated_at}.
      </Typography>

      <Tabs
        value={selectedTab}
        onChange={(_event, value) => setSelectedTab(value)}
        sx={{ mb: 2, borderBottom: 1, borderColor: 'divider', minHeight: 36 }}
      >
        <Tab label="Overview" sx={{ minHeight: 36, fontSize: 12 }} />
        <Tab
          label="By Group"
          sx={{ minHeight: 36, fontSize: 12 }}
          disabled={!attributionAvailable && groupAttribution == null}
        />
      </Tabs>

      {selectedTab === 0 && (
        <>
          <Paper variant="outlined" sx={{ p: 1.5, mb: 2 }}>
            <BreadthContextStrip row={current} />
          </Paper>
          <Box sx={{ height: { xs: 390, md: 460 }, mb: 2 }}>
            <BreadthChart
              breadthData={filteredChartData}
              spyData={filteredBenchmarkData}
              benchmarkLabel={benchmarkLabel}
              isLoading={false}
              error={null}
              timeRange={timeRange}
              onTimeRangeChange={setTimeRange}
              availableRanges={['1M', '3M']}
              fillContainer
            />
          </Box>
          {history.length > 0 && (
            <Paper variant="outlined">
              <Box sx={{ p: 1.5, borderBottom: 1, borderColor: 'divider' }}>
                <Typography sx={{ fontWeight: 700, fontSize: 13 }}>
                  Recent Sessions
                </Typography>
              </Box>
              <BreadthHistoryTable
                rows={history}
                maxRows={20}
                contributorDates={contributorDrilldown.availableDates}
                onContributorCellClick={contributorDrilldown.open}
              />
            </Paper>
          )}
        </>
      )}

      {selectedTab === 1 && <BreadthGroupAttribution attribution={groupAttribution} />}
      <BreadthContributorDialog
        open={Boolean(contributorDrilldown.selected)}
        metric={contributorDrilldown.selected?.metric}
        row={contributorDrilldown.selected?.row}
        view={contributorDrilldown.viewState.view}
        isLoading={contributorDrilldown.dialogQuery.isLoading}
        error={contributorDrilldown.dialogQuery.error}
        inconsistent={contributorDrilldown.viewState.inconsistent}
        onRetry={() => contributorDrilldown.dialogQuery.refetch()}
        onClose={contributorDrilldown.close}
      />
    </Box>
  );
}

export default StaticBreadthPage;
