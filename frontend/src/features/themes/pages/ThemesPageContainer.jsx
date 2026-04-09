import { useEffect, useMemo, useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { Alert, Box, Button, CircularProgress, Container } from '@mui/material';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import {
  dismissAlert,
  getAlerts,
  getCandidateThemeQueue,
  getEmergingThemes,
  getFailedItemsCount,
  getL1Categories,
  getL1Rankings,
  getMergeSuggestions,
  getPipelineObservability,
  getThemeRankings,
  getThemesBootstrap,
  runPipelineAsync,
} from '../../../api/themes';
import ThemeSourcesModal from '../../../components/Themes/ThemeSourcesModal';
import ThemeReviewDialog from '../../../components/Themes/ThemeReviewDialog';
import ThemeSettingsDialog from '../../../components/Themes/ThemeSettingsDialog';
import ArticleBrowserModal from '../../../components/Themes/ArticleBrowserModal';
import ModelSettingsModal from '../../../components/Themes/ModelSettingsModal';
import ThemeTaxonomyTable from '../../../components/Themes/ThemeTaxonomyTable';
import { usePipeline } from '../../../contexts/usePipeline';
import { useRuntime } from '../../../contexts/RuntimeContext';
import { DEFAULT_SOURCE_TYPES } from '../constants';
import ThemeDetailModal from '../components/ThemeDetailModal';
import ThemeInsightsCards from '../components/ThemeInsightsCards';
import ThemesFiltersPanel from '../components/ThemesFiltersPanel';
import ThemesPageHeader from '../components/ThemesPageHeader';
import ThemesRankingsTable from '../components/ThemesRankingsTable';

const PAGE_SIZE = 50;
const VALID_THEME_VIEWS = new Set(['grouped', 'flat']);

function ThemesPage() {
  const [selectedTab, setSelectedTab] = useState('all');
  const [selectedTheme, setSelectedTheme] = useState(null);
  const [orderBy, setOrderBy] = useState('rank');
  const [order, setOrder] = useState('asc');
  const [sourcesTheme, setSourcesTheme] = useState(null);
  const [selectedSourceTypes, setSelectedSourceTypes] = useState(DEFAULT_SOURCE_TYPES);
  const [reviewDialogOpen, setReviewDialogOpen] = useState(false);
  const [reviewDialogTab, setReviewDialogTab] = useState(0);
  const [settingsDialogOpen, setSettingsDialogOpen] = useState(false);
  const [settingsDialogTab, setSettingsDialogTab] = useState(0);
  const [articleBrowserOpen, setArticleBrowserOpen] = useState(false);
  const [modelSettingsOpen, setModelSettingsOpen] = useState(false);
  const [selectedPipeline, setSelectedPipeline] = useState('technical');
  const [page, setPage] = useState(0);
  const [themeView, setThemeView] = useState(() => {
    const storedThemeView = localStorage.getItem('themeView');
    return VALID_THEME_VIEWS.has(storedThemeView) ? storedThemeView : 'grouped';
  });
  const [categoryFilter, setCategoryFilter] = useState(null);
  const [bootstrapSettledVariants, setBootstrapSettledVariants] = useState({});
  const [dismissingAlertId, setDismissingAlertId] = useState(null);

  const { runtimeReady, uiSnapshots } = useRuntime();
  const { isPipelineRunning, startPipeline } = usePipeline();
  const queryClient = useQueryClient();

  const snapshotEnabled = runtimeReady && Boolean(uiSnapshots?.themes);
  const bootstrapVariantKey = `${selectedPipeline}:${themeView}`;
  const bootstrapSettled = Boolean(bootstrapSettledVariants[bootstrapVariantKey]);
  const liveQueriesEnabled = runtimeReady && (!snapshotEnabled || bootstrapSettled);

  const handleSourceTypeToggle = (sourceType) => {
    setSelectedSourceTypes((previous) => {
      if (previous.includes(sourceType)) {
        if (previous.length === 1) {
          return previous;
        }
        return previous.filter((source) => source !== sourceType);
      }
      return [...previous, sourceType];
    });
  };

  const handleTabChange = (_, value) => {
    setSelectedTab(value);
    setPage(0);
  };

  const handleSourceTypeToggleWithReset = (sourceType) => {
    handleSourceTypeToggle(sourceType);
    setPage(0);
  };

  const handlePipelineChangeWithReset = (_, newPipeline) => {
    if (newPipeline !== null) {
      setSelectedPipeline(newPipeline);
      setPage(0);
      setCategoryFilter(null);
    }
  };

  const {
    data: rankingsData,
    isLoading: isLoadingRankings,
    error: errorRankings,
    refetch: refetchRankings,
  } = useQuery({
    queryKey: ['themeRankings', selectedTab, selectedSourceTypes, selectedPipeline, page],
    queryFn: () =>
      getThemeRankings(
        PAGE_SIZE,
        selectedTab === 'all' ? null : selectedTab,
        selectedSourceTypes,
        selectedPipeline,
        page * PAGE_SIZE
      ),
    enabled: liveQueriesEnabled && themeView === 'flat',
    refetchInterval: 60000,
    staleTime: 60_000,
  });

  const { data: groupedRankingsData, refetch: refetchGroupedRankings } = useQuery({
    queryKey: ['l1Rankings', selectedPipeline, null, 0, 'momentum_score', 'desc'],
    queryFn: () =>
      getL1Rankings({
        pipeline: selectedPipeline,
        category: null,
        limit: PAGE_SIZE,
        offset: 0,
        sortBy: 'momentum_score',
        sortOrder: 'desc',
      }),
    enabled: liveQueriesEnabled && themeView === 'grouped',
    staleTime: 60_000,
  });

  const { data: emerging, isLoading: isLoadingEmerging } = useQuery({
    queryKey: ['emergingThemes', selectedPipeline],
    queryFn: () => getEmergingThemes(1.5, 3, selectedPipeline),
    enabled: liveQueriesEnabled,
    staleTime: 60_000,
  });

  const { data: alerts, isLoading: isLoadingAlerts } = useQuery({
    queryKey: ['themeAlerts'],
    queryFn: () => getAlerts(false, 50),
    enabled: liveQueriesEnabled,
    staleTime: 60_000,
  });

  const themesBootstrapQuery = useQuery({
    queryKey: ['themesBootstrap', selectedPipeline, themeView],
    queryFn: () => getThemesBootstrap({ pipeline: selectedPipeline, themeView }),
    enabled: snapshotEnabled && !bootstrapSettled,
    retry: false,
    staleTime: 60_000,
  });
  const isBootstrappingVariant = snapshotEnabled && !bootstrapSettled && themesBootstrapQuery.isPending;

  useEffect(() => {
    if (!snapshotEnabled) {
      return;
    }
    if (themesBootstrapQuery.isError) {
      setBootstrapSettledVariants((previous) => ({ ...previous, [bootstrapVariantKey]: true }));
      return;
    }
    if (!themesBootstrapQuery.isSuccess) {
      return;
    }
    if (themesBootstrapQuery.data?.is_stale) {
      setBootstrapSettledVariants((previous) => ({ ...previous, [bootstrapVariantKey]: true }));
      return;
    }

    const payload = themesBootstrapQuery.data?.payload ?? {};
    queryClient.setQueryData(['emergingThemes', selectedPipeline], payload.emerging ?? null);
    queryClient.setQueryData(['themeAlerts'], payload.alerts ?? null);
    queryClient.setQueryData(['mergeSuggestions', 'pending'], { total: payload.pending_merge_count ?? 0, suggestions: [] });
    queryClient.setQueryData(['candidateThemeQueue', selectedPipeline, 'summary'], payload.candidate_queue_summary ?? null);
    queryClient.setQueryData(['failedItemsCount', selectedPipeline], payload.failed_items_count ?? null);
    queryClient.setQueryData(['pipelineObservability', selectedPipeline], payload.observability ?? null);

    if (payload.l1_categories) {
      queryClient.setQueryData(['l1Categories', selectedPipeline], payload.l1_categories);
    }
    if (payload.l1_rankings) {
      queryClient.setQueryData(['l1Rankings', selectedPipeline, null, 0, 'momentum_score', 'desc'], payload.l1_rankings);
    }
    if (payload.rankings) {
      queryClient.setQueryData(['themeRankings', 'all', DEFAULT_SOURCE_TYPES, selectedPipeline, 0], payload.rankings);
    }

    setBootstrapSettledVariants((previous) => ({ ...previous, [bootstrapVariantKey]: true }));
  }, [
    bootstrapVariantKey,
    queryClient,
    selectedPipeline,
    snapshotEnabled,
    themesBootstrapQuery.data,
    themesBootstrapQuery.isError,
    themesBootstrapQuery.isSuccess,
  ]);

  const dismissMutation = useMutation({
    mutationFn: dismissAlert,
    onMutate: async (alertId) => {
      setDismissingAlertId(alertId);
      await queryClient.cancelQueries({ queryKey: ['themeAlerts'] });
      const previousAlerts = queryClient.getQueryData(['themeAlerts']);
      queryClient.setQueryData(['themeAlerts'], (old) => {
        if (!old) {
          return old;
        }
        return {
          ...old,
          total: old.total - 1,
          unread: old.alerts.find((alert) => alert.id === alertId && !alert.is_read) ? old.unread - 1 : old.unread,
          alerts: old.alerts.filter((alert) => alert.id !== alertId),
        };
      });
      return { previousAlerts };
    },
    onError: (_, __, context) => {
      if (context?.previousAlerts) {
        queryClient.setQueryData(['themeAlerts'], context.previousAlerts);
      }
    },
    onSettled: () => {
      setDismissingAlertId(null);
      queryClient.invalidateQueries({ queryKey: ['themeAlerts'] });
    },
  });

  const handleDismissAlert = (alertId) => {
    dismissMutation.mutate(alertId);
  };

  const { data: pendingMerges } = useQuery({
    queryKey: ['mergeSuggestions', 'pending'],
    queryFn: () => getMergeSuggestions('pending', 100),
    enabled: liveQueriesEnabled,
    staleTime: 60_000,
  });

  const { data: candidateQueueSummary } = useQuery({
    queryKey: ['candidateThemeQueue', selectedPipeline, 'summary'],
    queryFn: () => getCandidateThemeQueue({ limit: 1, offset: 0, pipeline: selectedPipeline }),
    enabled: liveQueriesEnabled,
    staleTime: 60_000,
  });

  const { data: failedCount } = useQuery({
    queryKey: ['failedItemsCount', selectedPipeline],
    queryFn: () => getFailedItemsCount(selectedPipeline),
    enabled: liveQueriesEnabled,
    staleTime: 60_000,
  });

  const { data: l1Categories } = useQuery({
    queryKey: ['l1Categories', selectedPipeline],
    queryFn: () => getL1Categories(selectedPipeline),
    enabled: liveQueriesEnabled && themeView === 'grouped',
    staleTime: 60_000,
  });

  const { data: observability, isLoading: isLoadingObservability } = useQuery({
    queryKey: ['pipelineObservability', selectedPipeline],
    queryFn: () => getPipelineObservability(selectedPipeline, 30),
    enabled: liveQueriesEnabled,
    refetchInterval: 60000,
    staleTime: 60_000,
  });

  const handleViewChange = (_, newView) => {
    if (newView !== null && VALID_THEME_VIEWS.has(newView)) {
      setThemeView(newView);
      localStorage.setItem('themeView', newView);
      setPage(0);
    }
  };

  const handleRunPipeline = async (lookbackDays = null) => {
    try {
      const result = await runPipelineAsync(selectedPipeline, lookbackDays);
      startPipeline(result.run_id);
    } catch (error) {
      console.error('Pipeline error:', error);
    }
  };

  const handleSort = (property) => {
    const isAsc = orderBy === property && order === 'asc';
    setOrder(isAsc ? 'desc' : 'asc');
    setOrderBy(property);
  };

  const sortedRankings = useMemo(() => {
    if (!rankingsData?.rankings) {
      return [];
    }
    return [...rankingsData.rankings].sort((a, b) => {
      let left = a[orderBy];
      let right = b[orderBy];

      if (left === null || left === undefined) {
        left = order === 'asc' ? Infinity : -Infinity;
      }
      if (right === null || right === undefined) {
        right = order === 'asc' ? Infinity : -Infinity;
      }
      if (order === 'asc') {
        return left < right ? -1 : left > right ? 1 : 0;
      }
      return left > right ? -1 : left < right ? 1 : 0;
    });
  }, [rankingsData?.rankings, orderBy, order]);

  const topTrendingThemes = useMemo(() => {
    if (themeView === 'grouped') {
      return (groupedRankingsData?.rankings ?? []).slice(0, 5).map((theme) => ({
        id: theme.id,
        name: theme.display_name,
        rank: theme.rank,
        momentum_score: theme.momentum_score,
      }));
    }
    return (rankingsData?.rankings ?? []).slice(0, 5).map((theme) => ({
      id: theme.theme_cluster_id,
      name: theme.theme,
      rank: theme.rank,
      momentum_score: theme.momentum_score,
    }));
  }, [groupedRankingsData?.rankings, rankingsData?.rankings, themeView]);

  const pendingReviewCount =
    (pendingMerges?.total ?? pendingMerges?.suggestions?.length ?? pendingMerges?.length ?? 0) +
    (candidateQueueSummary?.total || 0);

  const handleRefresh = () => {
    if (themeView === 'grouped') {
      refetchGroupedRankings();
      return;
    }
    refetchRankings();
  };

  if (!runtimeReady) {
    return (
      <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
        <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
          <CircularProgress />
        </Box>
      </Container>
    );
  }

  if (themeView === 'flat' && errorRankings) {
    return (
      <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
        <Alert severity="error">Error loading themes: {errorRankings.message}</Alert>
        <Box mt={2}>
          <Button
            variant="contained"
            startIcon={<PlayArrowIcon />}
            onClick={() => handleRunPipeline()}
            disabled={isPipelineRunning}
          >
            {isPipelineRunning ? 'Running Pipeline...' : 'Run Discovery Pipeline'}
          </Button>
        </Box>
      </Container>
    );
  }

  return (
    <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
      <ThemesPageHeader
        selectedPipeline={selectedPipeline}
        onPipelineChange={handlePipelineChangeWithReset}
        pendingReviewCount={pendingReviewCount}
        onOpenReview={() => {
          setReviewDialogTab(0);
          setReviewDialogOpen(true);
        }}
        onOpenArticles={() => setArticleBrowserOpen(true)}
        onOpenSettings={() => {
          setSettingsDialogTab(0);
          setSettingsDialogOpen(true);
        }}
        onOpenModelSettings={() => setModelSettingsOpen(true)}
        failedCount={failedCount?.failed_count || 0}
        isPipelineRunning={isPipelineRunning}
        onRunPipeline={handleRunPipeline}
        onRefresh={handleRefresh}
      />

      <ThemeInsightsCards
        emerging={emerging}
        isLoadingEmerging={isLoadingEmerging}
        topTrendingThemes={topTrendingThemes}
        onSelectTheme={setSelectedTheme}
        observability={observability}
        isLoadingObservability={isLoadingObservability}
        alerts={alerts}
        isLoadingAlerts={isLoadingAlerts}
        onDismissAlert={handleDismissAlert}
        dismissingAlertId={dismissingAlertId}
      />

      <ThemesFiltersPanel
        themeView={themeView}
        onViewChange={handleViewChange}
        l1Categories={l1Categories}
        categoryFilter={categoryFilter}
        onCategoryFilterChange={setCategoryFilter}
        selectedTab={selectedTab}
        onTabChange={handleTabChange}
        selectedSourceTypes={selectedSourceTypes}
        onSourceTypeToggle={handleSourceTypeToggleWithReset}
      />

      {themeView === 'grouped' ? (
        isBootstrappingVariant ? (
          <Box display="flex" justifyContent="center" alignItems="center" minHeight="300px">
            <CircularProgress />
          </Box>
        ) : (
          <ThemeTaxonomyTable
            pipeline={selectedPipeline}
            categoryFilter={categoryFilter}
            onThemeClick={(theme) => setSelectedTheme(theme)}
          />
        )
      ) : (
        <ThemesRankingsTable
          isLoading={isBootstrappingVariant || isLoadingRankings}
          rankingsData={rankingsData}
          sortedRankings={sortedRankings}
          orderBy={orderBy}
          order={order}
          onSort={handleSort}
          page={page}
          onPageChange={setPage}
          selectedPipeline={selectedPipeline}
          onSelectTheme={setSelectedTheme}
          onOpenSources={setSourcesTheme}
        />
      )}

      {selectedTheme && (
        <ThemeDetailModal
          themeId={selectedTheme.id}
          themeName={selectedTheme.name}
          selectedPipeline={selectedPipeline}
          open={Boolean(selectedTheme)}
          onClose={() => setSelectedTheme(null)}
        />
      )}

      {sourcesTheme && (
        <ThemeSourcesModal
          open={Boolean(sourcesTheme)}
          onClose={() => setSourcesTheme(null)}
          themeId={sourcesTheme.id}
          themeName={sourcesTheme.name}
        />
      )}

      <ThemeReviewDialog
        open={reviewDialogOpen}
        onClose={() => setReviewDialogOpen(false)}
        pipeline={selectedPipeline}
        initialTab={reviewDialogTab}
        mergeCount={pendingMerges?.total ?? pendingMerges?.suggestions?.length ?? pendingMerges?.length ?? 0}
        candidateCount={candidateQueueSummary?.total || 0}
      />

      <ThemeSettingsDialog
        open={settingsDialogOpen}
        onClose={() => setSettingsDialogOpen(false)}
        pipeline={selectedPipeline}
        initialTab={settingsDialogTab}
      />

      <ArticleBrowserModal
        open={articleBrowserOpen}
        onClose={() => setArticleBrowserOpen(false)}
        pipeline={selectedPipeline}
      />

      <ModelSettingsModal open={modelSettingsOpen} onClose={() => setModelSettingsOpen(false)} />
    </Container>
  );
}

export default ThemesPage;
