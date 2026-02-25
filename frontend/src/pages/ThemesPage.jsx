import { useState, useMemo } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
  Container,
  Typography,
  Box,
  Alert,
  CircularProgress,
  Paper,
  Grid,
  Card,
  CardContent,
  Chip,
  Tab,
  Tabs,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TableSortLabel,
  TablePagination,
  Dialog,
  DialogTitle,
  DialogContent,
  IconButton,
  Button,
  LinearProgress,
  Badge,
  ToggleButton,
  ToggleButtonGroup,
  Tooltip,
} from '@mui/material';
import TrendingFlatIcon from '@mui/icons-material/TrendingFlat';
import AccountBalanceIcon from '@mui/icons-material/AccountBalance';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import TrendingDownIcon from '@mui/icons-material/TrendingDown';
import WhatshotIcon from '@mui/icons-material/Whatshot';
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome';
import BubbleChartIcon from '@mui/icons-material/BubbleChart';
import NotificationsIcon from '@mui/icons-material/Notifications';
import CloseIcon from '@mui/icons-material/Close';
import RefreshIcon from '@mui/icons-material/Refresh';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import TimelineIcon from '@mui/icons-material/Timeline';
import NewReleasesIcon from '@mui/icons-material/NewReleases';
import ShowChartIcon from '@mui/icons-material/ShowChart';
import SettingsIcon from '@mui/icons-material/Settings';
import CompareArrowsIcon from '@mui/icons-material/CompareArrows';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  ResponsiveContainer,
  AreaChart,
  Area,
} from 'recharts';
import {
  getThemeRankings,
  getEmergingThemes,
  getThemeDetail,
  getThemeHistory,
  getAlerts,
  dismissAlert,
  runPipelineAsync,
  getCandidateThemeQueue,
  getMergeSuggestions,
  getThemeRelationshipGraph,
  getFailedItemsCount,
  getPipelineObservability,
} from '../api/themes';
import ArticleIcon from '@mui/icons-material/Article';
import SmartToyIcon from '@mui/icons-material/SmartToy';
import ManageSourcesModal from '../components/Themes/ManageSourcesModal';
import ThemeSourcesModal from '../components/Themes/ThemeSourcesModal';
import ThemeMergeReviewModal from '../components/Themes/ThemeMergeReviewModal';
import ThemeCandidateReviewModal from '../components/Themes/ThemeCandidateReviewModal';
import ThemePolicySettingsModal from '../components/Themes/ThemePolicySettingsModal';
import ArticleBrowserModal from '../components/Themes/ArticleBrowserModal';
import ModelSettingsModal from '../components/Themes/ModelSettingsModal';
import { usePipeline } from '../contexts/PipelineContext';

// Status badge colors
const statusColors = {
  trending: 'success',
  emerging: 'warning',
  active: 'info',
  fading: 'error',
  dormant: 'default',
};

// Source types for filtering
const SOURCE_TYPES = [
  { value: 'substack', label: 'Substack' },
  { value: 'twitter', label: 'Twitter' },
  { value: 'news', label: 'News' },
  { value: 'reddit', label: 'Reddit' },
];

// Helper component for momentum score bar
const MomentumBar = ({ score }) => {
  const color = score >= 70 ? 'success' : score >= 50 ? 'warning' : 'error';
  return (
    <Box sx={{ display: 'flex', alignItems: 'center', minWidth: 80 }}>
      <Box sx={{ width: '100%', mr: 0.5 }}>
        <LinearProgress
          variant="determinate"
          value={Math.min(score, 100)}
          color={color}
          sx={{ height: 6, borderRadius: 3 }}
        />
      </Box>
      <Box sx={{ minWidth: 28, fontSize: '11px', fontWeight: 600, fontFamily: 'monospace' }}>
        {score?.toFixed(0)}
      </Box>
    </Box>
  );
};

// Velocity indicator
const VelocityIndicator = ({ velocity }) => {
  if (!velocity) return <Box sx={{ color: 'text.secondary', fontFamily: 'monospace', fontSize: '11px' }}>-</Box>;

  const isAccelerating = velocity > 1;
  const color = velocity >= 2 ? 'success.main' : velocity >= 1.5 ? 'warning.main' : velocity >= 1 ? 'info.main' : 'text.secondary';

  return (
    <Box display="flex" alignItems="center" sx={{ fontFamily: 'monospace' }}>
      {isAccelerating && <TrendingUpIcon sx={{ fontSize: 12, mr: 0.25, color }} />}
      <Box component="span" sx={{ color, fontWeight: velocity >= 1.5 ? 600 : 400, fontSize: '11px' }}>
        {velocity.toFixed(1)}x
      </Box>
    </Box>
  );
};

// Emerging Themes Card
const EmergingThemesCard = ({ themes, isLoading }) => {
  if (isLoading) {
    return (
      <Card variant="outlined">
        <CardContent sx={{ p: 1.5, '&:last-child': { pb: 1.5 } }}>
          <Box display="flex" justifyContent="center" p={1}>
            <CircularProgress size={20} />
          </Box>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card variant="outlined" sx={{ height: '100%' }}>
      <CardContent sx={{ p: 1.5, '&:last-child': { pb: 1.5 } }}>
        <Box display="flex" alignItems="center" mb={1}>
          <NewReleasesIcon sx={{ color: 'warning.main', mr: 0.5, fontSize: 18 }} />
          <Box sx={{ fontSize: '12px', fontWeight: 600 }}>
            Emerging Themes
          </Box>
          {themes?.count > 0 && (
            <Chip label={themes.count} size="small" color="warning" sx={{ ml: 0.5 }} />
          )}
        </Box>

        {themes?.themes?.length > 0 ? (
          <Box>
            {themes.themes.slice(0, 5).map((theme, index) => (
              <Box
                key={theme.theme}
                sx={{
                  py: 0.5,
                  borderBottom: index < 4 ? 1 : 0,
                  borderColor: 'divider',
                  display: 'flex',
                  alignItems: 'center',
                }}
              >
                <AutoAwesomeIcon sx={{ fontSize: 14, color: 'warning.main', mr: 0.5 }} />
                <Box sx={{ flex: 1, minWidth: 0 }}>
                  <Box sx={{ fontSize: '11px', fontWeight: 500 }}>
                    {theme.theme}
                  </Box>
                  <Box display="flex" gap={0.5} mt={0.25}>
                    <Chip
                      label={`${theme.velocity.toFixed(1)}x`}
                      size="small"
                      variant="outlined"
                      color="warning"
                      sx={{ height: 16, fontSize: '9px' }}
                    />
                    <Chip
                      label={`${theme.mentions_7d} ment.`}
                      size="small"
                      variant="outlined"
                      sx={{ height: 16, fontSize: '9px' }}
                    />
                  </Box>
                </Box>
              </Box>
            ))}
          </Box>
        ) : (
          <Box sx={{ fontSize: '11px', color: 'text.secondary', textAlign: 'center', py: 1 }}>
            No emerging themes detected
          </Box>
        )}
      </CardContent>
    </Card>
  );
};

// Alerts Card
const AlertsCard = ({ alerts, isLoading, onDismiss, dismissingId }) => {
  const [hoveredId, setHoveredId] = useState(null);

  if (isLoading) {
    return (
      <Card variant="outlined">
        <CardContent sx={{ p: 1.5, '&:last-child': { pb: 1.5 } }}>
          <Box display="flex" justifyContent="center" p={1}>
            <CircularProgress size={20} />
          </Box>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card variant="outlined" sx={{ height: '100%' }}>
      <CardContent sx={{ p: 1.5, '&:last-child': { pb: 1.5 } }}>
        <Box display="flex" alignItems="center" mb={1}>
          <Badge badgeContent={alerts?.unread || 0} color="error">
            <NotificationsIcon sx={{ color: 'primary.main', fontSize: 18 }} />
          </Badge>
          <Box sx={{ fontSize: '12px', fontWeight: 600, ml: 1 }}>
            Alerts
          </Box>
          {alerts?.total > 0 && (
            <Box sx={{ fontSize: '10px', color: 'text.secondary', ml: 0.5 }}>
              ({alerts.total})
            </Box>
          )}
        </Box>

        {alerts?.alerts?.length > 0 ? (
          <Box sx={{ maxHeight: 200, overflowY: 'auto' }}>
            {alerts.alerts.map((alert) => (
              <Box
                key={alert.id}
                onMouseEnter={() => setHoveredId(alert.id)}
                onMouseLeave={() => setHoveredId(null)}
                sx={{
                  py: 0.5,
                  opacity: alert.is_read ? 0.6 : 1,
                  borderLeft: 2,
                  borderColor: alert.severity === 'warning' ? 'warning.main' : 'info.main',
                  pl: 1,
                  mb: 0.5,
                  display: 'flex',
                  alignItems: 'flex-start',
                  justifyContent: 'space-between',
                  position: 'relative',
                }}
              >
                <Box sx={{ flex: 1, minWidth: 0 }}>
                  <Box sx={{ fontSize: '11px', fontWeight: alert.is_read ? 400 : 600 }}>
                    {alert.title}
                  </Box>
                  <Box sx={{ fontSize: '10px', color: 'text.secondary' }}>
                    {new Date(alert.triggered_at).toLocaleDateString()}
                  </Box>
                </Box>
                {(hoveredId === alert.id || dismissingId === alert.id) && (
                  <IconButton
                    size="small"
                    onClick={(e) => {
                      e.stopPropagation();
                      onDismiss?.(alert.id);
                    }}
                    disabled={dismissingId === alert.id}
                    sx={{
                      p: 0.25,
                      ml: 0.5,
                      opacity: dismissingId === alert.id ? 0.5 : 1,
                    }}
                  >
                    {dismissingId === alert.id ? (
                      <CircularProgress size={12} />
                    ) : (
                      <CloseIcon sx={{ fontSize: 14 }} />
                    )}
                  </IconButton>
                )}
              </Box>
            ))}
          </Box>
        ) : (
          <Box sx={{ fontSize: '11px', color: 'text.secondary', textAlign: 'center', py: 1 }}>
            No alerts
          </Box>
        )}
      </CardContent>
    </Card>
  );
};

const PipelineObservabilityCard = ({ observability, isLoading }) => {
  if (isLoading) {
    return (
      <Card variant="outlined">
        <CardContent sx={{ p: 1.5, '&:last-child': { pb: 1.5 } }}>
          <Box display="flex" justifyContent="center" p={1}>
            <CircularProgress size={20} />
          </Box>
        </CardContent>
      </Card>
    );
  }

  const metrics = observability?.metrics || {};
  const alerts = observability?.alerts || [];
  const statChip = (label, value, color = 'default') => (
    <Chip
      key={label}
      label={`${label}: ${value}`}
      size="small"
      color={color}
      variant="outlined"
      sx={{ height: 18, fontSize: '10px' }}
    />
  );

  return (
    <Card variant="outlined" sx={{ height: '100%' }}>
      <CardContent sx={{ p: 1.5, '&:last-child': { pb: 1.5 } }}>
        <Box display="flex" alignItems="center" mb={1}>
          <TimelineIcon sx={{ color: 'primary.main', fontSize: 18 }} />
          <Box sx={{ fontSize: '12px', fontWeight: 600, ml: 1 }}>
            Pipeline Health
          </Box>
          {alerts.length > 0 && (
            <Chip
              label={`${alerts.length} alerts`}
              size="small"
              color="warning"
              sx={{ ml: 1, height: 18, fontSize: '10px' }}
            />
          )}
        </Box>

        <Box display="flex" gap={0.5} flexWrap="wrap" mb={1}>
          {statChip('Parse', `${((metrics.parse_failure_rate || 0) * 100).toFixed(1)}%`, metrics.parse_failure_rate > 0.25 ? 'warning' : 'default')}
          {statChip('No-mention', `${((metrics.processed_without_mentions_ratio || 0) * 100).toFixed(1)}%`, metrics.processed_without_mentions_ratio > 0.2 ? 'warning' : 'default')}
          {statChip('New clusters', `${((metrics.new_cluster_rate || 0) * 100).toFixed(1)}%`, metrics.new_cluster_rate > 0.45 ? 'warning' : 'default')}
          {statChip('Merge proxy', `${((metrics.merge_precision_proxy || 0) * 100).toFixed(1)}%`, metrics.merge_precision_proxy < 0.55 ? 'warning' : 'default')}
        </Box>

        {alerts.length > 0 ? (
          <Box sx={{ maxHeight: 120, overflowY: 'auto' }}>
            {alerts.slice(0, 3).map((item) => (
              <Box
                key={item.key}
                sx={{
                  mb: 0.75,
                  px: 1,
                  py: 0.5,
                  borderLeft: 2,
                  borderColor: item.severity === 'warning' ? 'warning.main' : 'info.main',
                  backgroundColor: 'background.paper',
                }}
              >
                <Box sx={{ fontSize: '10px', fontWeight: 600 }}>{item.title}</Box>
                <Box sx={{ fontSize: '10px', color: 'text.secondary' }}>{item.description}</Box>
                <Box sx={{ fontSize: '10px', mt: 0.25 }}>
                  <a href={item.runbook_url} target="_blank" rel="noreferrer">Runbook</a>
                </Box>
              </Box>
            ))}
          </Box>
        ) : (
          <Box sx={{ fontSize: '11px', color: 'text.secondary', textAlign: 'center', py: 1 }}>
            No policy breaches
          </Box>
        )}
      </CardContent>
    </Card>
  );
};

// Theme Detail Modal
const ThemeDetailModal = ({ themeId, themeName, open, onClose, selectedPipeline }) => {
  const { data: detail, isLoading: isLoadingDetail } = useQuery({
    queryKey: ['themeDetail', themeId],
    queryFn: () => getThemeDetail(themeId),
    enabled: !!themeId && open,
  });

  const { data: history, isLoading: isLoadingHistory } = useQuery({
    queryKey: ['themeHistory', themeId],
    queryFn: () => getThemeHistory(themeId, 30),
    enabled: !!themeId && open,
  });

  const { data: relationshipGraph, isLoading: isLoadingGraph } = useQuery({
    queryKey: ['themeRelationshipGraph', themeId, selectedPipeline],
    queryFn: () => getThemeRelationshipGraph(themeId, { pipeline: selectedPipeline, limit: 120 }),
    enabled: !!themeId && open,
  });

  const isLoading = isLoadingDetail || isLoadingHistory || isLoadingGraph;

  return (
    <Dialog open={open} onClose={onClose} maxWidth="lg" fullWidth>
      <DialogTitle>
        <Box display="flex" justifyContent="space-between" alignItems="center">
          <Box display="flex" alignItems="center">
            <BubbleChartIcon sx={{ mr: 1, color: 'primary.main' }} />
            <Typography variant="h6">{themeName}</Typography>
            {detail?.theme?.is_validated && (
              <Chip label="Validated" size="small" color="success" sx={{ ml: 1 }} />
            )}
          </Box>
          <IconButton onClick={onClose} size="small">
            <CloseIcon />
          </IconButton>
        </Box>
      </DialogTitle>
      <DialogContent>
        {isLoading ? (
          <Box display="flex" justifyContent="center" p={4}>
            <CircularProgress />
          </Box>
        ) : detail ? (
          <Box>
            {/* Metrics Grid */}
            {detail.metrics && (
              <Grid container spacing={2} mb={3}>
                <Grid item xs={6} md={2}>
                  <Box textAlign="center" p={1} bgcolor="grey.50" borderRadius={1}>
                    <Typography variant="h5" fontWeight="bold">
                      #{detail.metrics.rank || '-'}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      Rank
                    </Typography>
                  </Box>
                </Grid>
                <Grid item xs={6} md={2}>
                  <Box textAlign="center" p={1} bgcolor="grey.50" borderRadius={1}>
                    <Typography variant="h5" fontWeight="bold" color={
                      detail.metrics.momentum_score >= 70 ? 'success.main' :
                      detail.metrics.momentum_score >= 50 ? 'warning.main' : 'error.main'
                    }>
                      {detail.metrics.momentum_score?.toFixed(0) || '-'}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      Momentum
                    </Typography>
                  </Box>
                </Grid>
                <Grid item xs={6} md={2}>
                  <Box textAlign="center" p={1} bgcolor="grey.50" borderRadius={1}>
                    <Typography variant="h5" fontWeight="bold">
                      {detail.metrics.mention_velocity?.toFixed(1) || '-'}x
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      Velocity
                    </Typography>
                  </Box>
                </Grid>
                <Grid item xs={6} md={2}>
                  <Box textAlign="center" p={1} bgcolor="grey.50" borderRadius={1}>
                    <Typography variant="h5" fontWeight="bold">
                      {detail.metrics.basket_rs_vs_spy?.toFixed(0) || '-'}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      RS vs SPY
                    </Typography>
                  </Box>
                </Grid>
                <Grid item xs={6} md={2}>
                  <Box textAlign="center" p={1} bgcolor="grey.50" borderRadius={1}>
                    <Typography variant="h5" fontWeight="bold">
                      {detail.metrics.pct_above_50ma?.toFixed(0) || '-'}%
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      Above 50MA
                    </Typography>
                  </Box>
                </Grid>
                <Grid item xs={6} md={2}>
                  <Box textAlign="center" p={1} bgcolor="grey.50" borderRadius={1}>
                    <Typography variant="h5" fontWeight="bold">
                      {detail.metrics.num_constituents || 0}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      Stocks
                    </Typography>
                  </Box>
                </Grid>
              </Grid>
            )}

            {/* Chart */}
            {history?.history?.length > 0 && (
              <Box mb={3}>
                <Typography variant="subtitle2" gutterBottom>
                  Momentum Score History
                </Typography>
                <Box sx={{ height: 200 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={history.history}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis
                        dataKey="date"
                        tick={{ fontSize: 10 }}
                        tickFormatter={(val) => val.slice(5)}
                      />
                      <YAxis domain={[0, 100]} tick={{ fontSize: 10 }} />
                      <RechartsTooltip />
                      <Area
                        type="monotone"
                        dataKey="momentum_score"
                        stroke="#1976d2"
                        fill="#1976d2"
                        fillOpacity={0.2}
                        name="Momentum"
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                </Box>
              </Box>
            )}

            {/* Constituents Table */}
            {detail.constituents?.length > 0 && (
              <Box>
                <Box sx={{ fontSize: '12px', fontWeight: 600, mb: 0.5 }}>
                  Theme Constituents ({detail.constituents.length})
                </Box>
                <TableContainer sx={{ maxHeight: 250 }}>
                  <Table size="small" stickyHeader>
                    <TableHead>
                      <TableRow>
                        <TableCell>Sym</TableCell>
                        <TableCell align="right">Ment.</TableCell>
                        <TableCell align="right">Conf</TableCell>
                        <TableCell align="right">Corr</TableCell>
                        <TableCell>Src</TableCell>
                        <TableCell>Last</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {detail.constituents.map((stock) => (
                        <TableRow key={stock.symbol} hover>
                          <TableCell sx={{ fontWeight: 600 }}>
                            {stock.symbol}
                          </TableCell>
                          <TableCell align="right" sx={{ fontFamily: 'monospace' }}>{stock.mention_count}</TableCell>
                          <TableCell align="right" sx={{
                            fontFamily: 'monospace',
                            color: stock.confidence >= 0.8 ? 'success.main' : stock.confidence >= 0.5 ? 'warning.main' : 'text.secondary'
                          }}>
                            {(stock.confidence * 100).toFixed(0)}%
                          </TableCell>
                          <TableCell align="right" sx={{ fontFamily: 'monospace' }}>
                            {stock.correlation_to_theme?.toFixed(2) || '-'}
                          </TableCell>
                          <TableCell>
                            <Box
                              component="span"
                              sx={{
                                fontSize: '9px',
                                padding: '1px 3px',
                                backgroundColor: 'grey.100',
                                borderRadius: '2px',
                              }}
                            >
                              {stock.source || 'unk'}
                            </Box>
                          </TableCell>
                          <TableCell sx={{ fontSize: '10px', color: 'text.secondary', fontFamily: 'monospace' }}>
                            {stock.last_mentioned_at
                              ? new Date(stock.last_mentioned_at).toLocaleDateString()
                              : '-'}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              </Box>
            )}

            {detail.relationships?.length > 0 && (
              <Box mt={3}>
                <Box sx={{ fontSize: '12px', fontWeight: 600, mb: 0.5 }}>
                  Theme Relationships ({detail.relationships.length})
                </Box>
                <TableContainer sx={{ maxHeight: 220 }}>
                  <Table size="small" stickyHeader>
                    <TableHead>
                      <TableRow>
                        <TableCell>Direction</TableCell>
                        <TableCell>Type</TableCell>
                        <TableCell>Theme</TableCell>
                        <TableCell align="right">Conf</TableCell>
                        <TableCell>Source</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {detail.relationships.map((rel) => (
                        <TableRow key={rel.relation_id} hover>
                          <TableCell>
                            <Chip
                              size="small"
                              label={rel.direction}
                              color={rel.direction === 'outgoing' ? 'primary' : 'default'}
                              variant="outlined"
                            />
                          </TableCell>
                          <TableCell>
                            <Tooltip
                              title={
                                rel.relationship_type === 'subset'
                                  ? 'Subset: one theme is mostly contained by another. Review before merging.'
                                  : rel.relationship_type === 'related'
                                    ? 'Related: topical overlap exists, but themes can still be distinct.'
                                    : 'Distinct: themes should remain separate unless new evidence appears.'
                              }
                            >
                              <Chip
                                size="small"
                                label={rel.relationship_type}
                                color={
                                  rel.relationship_type === 'subset'
                                    ? 'warning'
                                    : rel.relationship_type === 'related'
                                      ? 'info'
                                      : 'default'
                                }
                              />
                            </Tooltip>
                          </TableCell>
                          <TableCell sx={{ fontWeight: 600 }}>
                            {rel.peer_theme_display_name || rel.peer_theme_name || '-'}
                          </TableCell>
                          <TableCell align="right" sx={{ fontFamily: 'monospace' }}>
                            {(Math.max(0, Math.min(1, rel.confidence || 0)) * 100).toFixed(0)}%
                          </TableCell>
                          <TableCell sx={{ fontSize: '10px', color: 'text.secondary', fontFamily: 'monospace' }}>
                            {rel.provenance || '-'}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              </Box>
            )}

            {relationshipGraph?.edges?.length > 0 && (
              <Box mt={3}>
                <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 0.5 }}>
                  <Box sx={{ fontSize: '12px', fontWeight: 600 }}>
                    Relationship Graph Context
                  </Box>
                  <Box sx={{ display: 'flex', gap: 0.5 }}>
                    <Chip size="small" label={`${relationshipGraph.total_nodes} nodes`} variant="outlined" />
                    <Chip size="small" label={`${relationshipGraph.total_edges} edges`} variant="outlined" />
                  </Box>
                </Box>
                <TableContainer sx={{ maxHeight: 220 }}>
                  <Table size="small" stickyHeader>
                    <TableHead>
                      <TableRow>
                        <TableCell>Source</TableCell>
                        <TableCell>Relation</TableCell>
                        <TableCell>Target</TableCell>
                        <TableCell align="right">Conf</TableCell>
                        <TableCell>Provenance</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {relationshipGraph.edges.slice(0, 100).map((edge) => (
                        <TableRow key={edge.relation_id} hover>
                          <TableCell sx={{ fontWeight: 600 }}>{edge.source_theme_name || edge.source_theme_id}</TableCell>
                          <TableCell>
                            <Chip
                              size="small"
                              label={edge.relationship_type}
                              color={
                                edge.relationship_type === 'subset'
                                  ? 'warning'
                                  : edge.relationship_type === 'related'
                                    ? 'info'
                                    : 'default'
                              }
                            />
                          </TableCell>
                          <TableCell sx={{ fontWeight: 600 }}>{edge.target_theme_name || edge.target_theme_id}</TableCell>
                          <TableCell align="right" sx={{ fontFamily: 'monospace' }}>
                            {(Math.max(0, Math.min(1, edge.confidence || 0)) * 100).toFixed(0)}%
                          </TableCell>
                          <TableCell sx={{ fontSize: '10px', color: 'text.secondary', fontFamily: 'monospace' }}>
                            {edge.provenance || '-'}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              </Box>
            )}
          </Box>
        ) : (
          <Typography color="text.secondary">No data available</Typography>
        )}
      </DialogContent>
    </Dialog>
  );
};

// Pagination constants
const PAGE_SIZE = 50;

// Main Page Component
function ThemesPage() {
  const [selectedTab, setSelectedTab] = useState('all');
  const [selectedTheme, setSelectedTheme] = useState(null);
  const [orderBy, setOrderBy] = useState('rank');
  const [order, setOrder] = useState('asc');
  const [sourcesModalOpen, setSourcesModalOpen] = useState(false);
  const [sourcesTheme, setSourcesTheme] = useState(null);
  const [selectedSourceTypes, setSelectedSourceTypes] = useState(
    SOURCE_TYPES.map(s => s.value)  // All selected by default
  );
  const [mergeModalOpen, setMergeModalOpen] = useState(false);
  const [candidateModalOpen, setCandidateModalOpen] = useState(false);
  const [articleBrowserOpen, setArticleBrowserOpen] = useState(false);
  const [modelSettingsOpen, setModelSettingsOpen] = useState(false);
  const [policySettingsOpen, setPolicySettingsOpen] = useState(false);
  const [selectedPipeline, setSelectedPipeline] = useState('technical');
  const [page, setPage] = useState(0);

  // Use global pipeline context
  const { isPipelineRunning, startPipeline } = usePipeline();

  // Handle pipeline toggle - defined above with handlePipelineChangeWithReset

  // Toggle source type selection
  const handleSourceTypeToggle = (sourceType) => {
    setSelectedSourceTypes(prev => {
      if (prev.includes(sourceType)) {
        // Don't allow deselecting all - keep at least one
        if (prev.length === 1) return prev;
        return prev.filter(s => s !== sourceType);
      }
      return [...prev, sourceType];
    });
  };

  // Reset page when filters change
  const handleTabChange = (e, v) => {
    setSelectedTab(v);
    setPage(0);
  };

  const handleSourceTypeToggleWithReset = (sourceType) => {
    handleSourceTypeToggle(sourceType);
    setPage(0);
  };

  const handlePipelineChangeWithReset = (event, newPipeline) => {
    if (newPipeline !== null) {
      setSelectedPipeline(newPipeline);
      setPage(0);
    }
  };

  // Fetch theme rankings
  const {
    data: rankingsData,
    isLoading: isLoadingRankings,
    error: errorRankings,
    refetch: refetchRankings,
  } = useQuery({
    queryKey: ['themeRankings', selectedTab, selectedSourceTypes, selectedPipeline, page],
    queryFn: () => getThemeRankings(PAGE_SIZE, selectedTab === 'all' ? null : selectedTab, selectedSourceTypes, selectedPipeline, page * PAGE_SIZE),
    refetchInterval: 60000,
  });

  // Fetch emerging themes
  const { data: emerging, isLoading: isLoadingEmerging } = useQuery({
    queryKey: ['emergingThemes', selectedPipeline],
    queryFn: () => getEmergingThemes(1.5, 3, selectedPipeline),
  });

  // Fetch alerts
  const { data: alerts, isLoading: isLoadingAlerts } = useQuery({
    queryKey: ['themeAlerts'],
    queryFn: () => getAlerts(false, 50),
  });

  // Query client for cache invalidation
  const queryClient = useQueryClient();

  // Dismiss alert state and mutation
  const [dismissingAlertId, setDismissingAlertId] = useState(null);

  const dismissMutation = useMutation({
    mutationFn: dismissAlert,
    onMutate: async (alertId) => {
      setDismissingAlertId(alertId);
      // Cancel outgoing refetches
      await queryClient.cancelQueries({ queryKey: ['themeAlerts'] });
      // Snapshot previous value
      const previousAlerts = queryClient.getQueryData(['themeAlerts']);
      // Optimistically update
      queryClient.setQueryData(['themeAlerts'], (old) => {
        if (!old) return old;
        return {
          ...old,
          total: old.total - 1,
          unread: old.alerts.find(a => a.id === alertId && !a.is_read) ? old.unread - 1 : old.unread,
          alerts: old.alerts.filter(a => a.id !== alertId),
        };
      });
      return { previousAlerts };
    },
    onError: (err, alertId, context) => {
      // Rollback on error
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

  // Fetch pending merge suggestions count
  const { data: pendingMerges } = useQuery({
    queryKey: ['mergeSuggestions', 'pending'],
    queryFn: () => getMergeSuggestions('pending', 100),
  });

  const { data: candidateQueueSummary } = useQuery({
    queryKey: ['candidateThemeQueue', selectedPipeline, 'summary'],
    queryFn: () => getCandidateThemeQueue({ limit: 1, offset: 0, pipeline: selectedPipeline }),
  });

  // Fetch failed items count for retry badge
  const { data: failedCount } = useQuery({
    queryKey: ['failedItemsCount', selectedPipeline],
    queryFn: () => getFailedItemsCount(selectedPipeline),
  });

  const { data: observability, isLoading: isLoadingObservability } = useQuery({
    queryKey: ['pipelineObservability', selectedPipeline],
    queryFn: () => getPipelineObservability(selectedPipeline, 30),
    refetchInterval: 60000,
  });

  // Start async pipeline using global context
  const handleRunPipeline = async () => {
    try {
      // Run pipeline for selected pipeline only (not both)
      const result = await runPipelineAsync(selectedPipeline);
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

  // Sort rankings - memoized to prevent re-sorting on unrelated state changes (modal opens, etc.)
  const sortedRankings = useMemo(() => {
    if (!rankingsData?.rankings) return [];
    return [...rankingsData.rankings].sort((a, b) => {
      let aVal = a[orderBy];
      let bVal = b[orderBy];

      if (aVal === null || aVal === undefined) aVal = order === 'asc' ? Infinity : -Infinity;
      if (bVal === null || bVal === undefined) bVal = order === 'asc' ? Infinity : -Infinity;

      if (order === 'asc') {
        return aVal < bVal ? -1 : aVal > bVal ? 1 : 0;
      }
      return aVal > bVal ? -1 : aVal < bVal ? 1 : 0;
    });
  }, [rankingsData?.rankings, orderBy, order]);

  if (errorRankings) {
    return (
      <Container maxWidth="xl" sx={{ mt: 4, mb: 4 }}>
        <Alert severity="error">
          Error loading themes: {errorRankings.message}
        </Alert>
        <Box mt={2}>
          <Button
            variant="contained"
            startIcon={<PlayArrowIcon />}
            onClick={handleRunPipeline}
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
      {/* Header */}
      <Box sx={{ mb: 4, display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <Box>
          <Box display="flex" alignItems="center" gap={2}>
            <Typography variant="h4">
              <WhatshotIcon sx={{ mr: 1, verticalAlign: 'middle', color: 'error.main' }} />
              Theme Discovery
            </Typography>
            {/* Pipeline Toggle */}
            <ToggleButtonGroup
              value={selectedPipeline}
              exclusive
              onChange={handlePipelineChangeWithReset}
              size="small"
              sx={{ height: 32 }}
            >
              <ToggleButton value="technical" sx={{ px: 2 }}>
                <TrendingFlatIcon sx={{ mr: 0.5, fontSize: 18 }} />
                Technical
              </ToggleButton>
              <ToggleButton value="fundamental" sx={{ px: 2 }}>
                <AccountBalanceIcon sx={{ mr: 0.5, fontSize: 18 }} />
                Fundamental
              </ToggleButton>
            </ToggleButtonGroup>
          </Box>
          <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
            {selectedPipeline === 'technical'
              ? 'Price action, momentum, RS, chart patterns, breakouts'
              : 'Earnings, valuation, macro themes, analyst coverage'
            }
          </Typography>
        </Box>
        <Box display="flex" gap={1}>
          <Badge
            badgeContent={pendingMerges?.suggestions?.length || pendingMerges?.length || 0}
            color="warning"
            max={99}
          >
            <Button
              size="small"
              variant="outlined"
              startIcon={<CompareArrowsIcon />}
              onClick={() => setMergeModalOpen(true)}
              sx={{ fontSize: '0.65rem', py: 0.15, px: 0.5, '& .MuiSvgIcon-root': { fontSize: '0.85rem' } }}
            >
              Review Merges
            </Button>
          </Badge>
          <Badge
            badgeContent={candidateQueueSummary?.total || 0}
            color="info"
            max={99}
          >
            <Button
              size="small"
              variant="outlined"
              startIcon={<AutoAwesomeIcon />}
              onClick={() => setCandidateModalOpen(true)}
              sx={{ fontSize: '0.65rem', py: 0.15, px: 0.5, '& .MuiSvgIcon-root': { fontSize: '0.85rem' } }}
            >
              Review Candidates
            </Button>
          </Badge>
          <Button
            size="small"
            variant="outlined"
            startIcon={<ArticleIcon />}
            onClick={() => setArticleBrowserOpen(true)}
            sx={{ fontSize: '0.65rem', py: 0.15, px: 0.5, '& .MuiSvgIcon-root': { fontSize: '0.85rem' } }}
          >
            Browse Articles
          </Button>
          <Button
            size="small"
            variant="outlined"
            startIcon={<SettingsIcon />}
            onClick={() => setPolicySettingsOpen(true)}
            sx={{ fontSize: '0.65rem', py: 0.15, px: 0.5, '& .MuiSvgIcon-root': { fontSize: '0.85rem' } }}
          >
            Policy Controls
          </Button>
          <Button
            size="small"
            variant="outlined"
            startIcon={<SettingsIcon />}
            onClick={() => setSourcesModalOpen(true)}
            sx={{ fontSize: '0.65rem', py: 0.15, px: 0.5, '& .MuiSvgIcon-root': { fontSize: '0.85rem' } }}
          >
            Manage Sources
          </Button>
          <Button
            size="small"
            variant="outlined"
            startIcon={<SmartToyIcon />}
            onClick={() => setModelSettingsOpen(true)}
            sx={{ fontSize: '0.65rem', py: 0.15, px: 0.5, '& .MuiSvgIcon-root': { fontSize: '0.85rem' } }}
          >
            LLM Settings
          </Button>
          <Tooltip title={failedCount?.failed_count > 0 ? `${failedCount.failed_count} items pending retry` : ''}>
            <Badge
              badgeContent={failedCount?.failed_count || 0}
              color="error"
              max={999}
              invisible={!failedCount?.failed_count}
            >
              <Button
                size="small"
                variant="outlined"
                startIcon={isPipelineRunning ? <CircularProgress size={12} /> : <PlayArrowIcon />}
                onClick={handleRunPipeline}
                disabled={isPipelineRunning}
                sx={{ fontSize: '0.65rem', py: 0.15, px: 0.5, '& .MuiSvgIcon-root': { fontSize: '0.85rem' } }}
              >
                {isPipelineRunning ? 'Running...' : 'Run Pipeline'}
              </Button>
            </Badge>
          </Tooltip>
          <Button
            size="small"
            variant="outlined"
            startIcon={<RefreshIcon />}
            onClick={() => refetchRankings()}
            sx={{ fontSize: '0.65rem', py: 0.15, px: 0.5, '& .MuiSvgIcon-root': { fontSize: '0.85rem' } }}
          >
            Refresh
          </Button>
        </Box>
      </Box>

      {/* Summary Cards */}
      <Grid container spacing={2} mb={3}>
        <Grid item xs={12} md={3}>
          <EmergingThemesCard themes={emerging} isLoading={isLoadingEmerging} />
        </Grid>
        <Grid item xs={12} md={3}>
          <Card variant="outlined" sx={{ height: '100%' }}>
            <CardContent>
              <Box display="flex" alignItems="center" mb={2}>
                <ShowChartIcon sx={{ color: 'success.main', mr: 1 }} />
                <Typography variant="subtitle1" fontWeight="bold">
                  Top Trending
                </Typography>
              </Box>
              {rankingsData?.rankings?.slice(0, 5).map((theme, index) => (
                <Box
                  key={theme.theme}
                  display="flex"
                  justifyContent="space-between"
                  alignItems="center"
                  py={0.75}
                  borderBottom={index < 4 ? 1 : 0}
                  borderColor="divider"
                  sx={{ cursor: 'pointer' }}
                  onClick={() => setSelectedTheme({ id: theme.theme_cluster_id, name: theme.theme })}
                >
                  <Box display="flex" alignItems="center">
                    <Typography variant="body2" color="text.secondary" sx={{ mr: 1, minWidth: 20 }}>
                      #{theme.rank}
                    </Typography>
                    <Typography variant="body2" fontWeight="medium" sx={{ maxWidth: 200 }} noWrap>
                      {theme.theme}
                    </Typography>
                  </Box>
                  <MomentumBar score={theme.momentum_score} />
                </Box>
              ))}
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={3}>
          <PipelineObservabilityCard
            observability={observability}
            isLoading={isLoadingObservability}
          />
        </Grid>
        <Grid item xs={12} md={3}>
          <AlertsCard
            alerts={alerts}
            isLoading={isLoadingAlerts}
            onDismiss={handleDismissAlert}
            dismissingId={dismissingAlertId}
          />
        </Grid>
      </Grid>

      {/* Status Filter Tabs */}
      <Paper sx={{ mb: 3 }}>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs value={selectedTab} onChange={handleTabChange}>
            <Tab label="All Themes" value="all" />
            <Tab
              label={
                <Box display="flex" alignItems="center">
                  Trending
                  <Chip label="Hot" size="small" color="success" sx={{ ml: 0.5, height: 18 }} />
                </Box>
              }
              value="trending"
            />
            <Tab label="Emerging" value="emerging" />
            <Tab label="Active" value="active" />
            <Tab label="Fading" value="fading" />
          </Tabs>
        </Box>

        {/* Source Type Filter */}
        <Box sx={{ px: 2, py: 1.5, display: 'flex', alignItems: 'center', gap: 1, borderTop: 1, borderColor: 'divider' }}>
          <Typography variant="body2" color="text.secondary" sx={{ mr: 1 }}>
            Sources:
          </Typography>
          {SOURCE_TYPES.map((source) => (
            <Chip
              key={source.value}
              label={source.label}
              size="small"
              variant={selectedSourceTypes.includes(source.value) ? 'filled' : 'outlined'}
              color={selectedSourceTypes.includes(source.value) ? 'primary' : 'default'}
              onClick={() => handleSourceTypeToggleWithReset(source.value)}
              sx={{ cursor: 'pointer' }}
            />
          ))}
        </Box>
      </Paper>

      {/* Rankings Table */}
      {isLoadingRankings ? (
        <Box display="flex" justifyContent="center" alignItems="center" minHeight="300px">
          <CircularProgress />
        </Box>
      ) : (
        <Paper elevation={1}>
          <Box sx={{ p: 1.5, borderBottom: 1, borderColor: 'divider', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <Box sx={{ fontSize: '14px', fontWeight: 600 }}>
              Theme Rankings
            </Box>
            {rankingsData && (
              <Box display="flex" gap={1} alignItems="center">
                <Chip
                  label={selectedPipeline === 'technical' ? 'Technical' : 'Fundamental'}
                  size="small"
                  color={selectedPipeline === 'technical' ? 'primary' : 'secondary'}
                />
                <Chip
                  label={`${rankingsData.total_themes} themes | ${rankingsData.date || 'Today'}`}
                  size="small"
                />
              </Box>
            )}
          </Box>
          <TableContainer sx={{ maxHeight: 'calc(100vh - 400px)' }}>
            <Table stickyHeader size="small">
              <TableHead>
                <TableRow>
                  <TableCell>
                    <TableSortLabel
                      active={orderBy === 'rank'}
                      direction={orderBy === 'rank' ? order : 'asc'}
                      onClick={() => handleSort('rank')}
                    >
                      #
                    </TableSortLabel>
                  </TableCell>
                  <TableCell>Theme</TableCell>
                  <TableCell>Status</TableCell>
                  <TableCell align="center">
                    <TableSortLabel
                      active={orderBy === 'momentum_score'}
                      direction={orderBy === 'momentum_score' ? order : 'asc'}
                      onClick={() => handleSort('momentum_score')}
                    >
                      Mom
                    </TableSortLabel>
                  </TableCell>
                  <TableCell align="right">
                    <TableSortLabel
                      active={orderBy === 'mention_velocity'}
                      direction={orderBy === 'mention_velocity' ? order : 'asc'}
                      onClick={() => handleSort('mention_velocity')}
                    >
                      Vel
                    </TableSortLabel>
                  </TableCell>
                  <TableCell align="right">
                    <TableSortLabel
                      active={orderBy === 'mentions_7d'}
                      direction={orderBy === 'mentions_7d' ? order : 'asc'}
                      onClick={() => handleSort('mentions_7d')}
                    >
                      7D
                    </TableSortLabel>
                  </TableCell>
                  <TableCell align="right">
                    <TableSortLabel
                      active={orderBy === 'basket_rs_vs_spy'}
                      direction={orderBy === 'basket_rs_vs_spy' ? order : 'asc'}
                      onClick={() => handleSort('basket_rs_vs_spy')}
                    >
                      RS
                    </TableSortLabel>
                  </TableCell>
                  <TableCell align="right">
                    <TableSortLabel
                      active={orderBy === 'basket_return_1w'}
                      direction={orderBy === 'basket_return_1w' ? order : 'asc'}
                      onClick={() => handleSort('basket_return_1w')}
                    >
                      1W
                    </TableSortLabel>
                  </TableCell>
                  <TableCell align="right">
                    <TableSortLabel
                      active={orderBy === 'pct_above_50ma'}
                      direction={orderBy === 'pct_above_50ma' ? order : 'asc'}
                      onClick={() => handleSort('pct_above_50ma')}
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
                    onClick={() => setSourcesTheme({ id: row.theme_cluster_id, name: row.theme })}
                    sx={{ cursor: 'pointer' }}
                  >
                    <TableCell>
                      <Box
                        component="span"
                        sx={{
                          backgroundColor: row.rank <= 5 ? 'success.main' : row.rank <= 10 ? 'warning.main' : 'error.main',
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
                            onClick={(e) => {
                              e.stopPropagation();
                              setSelectedTheme({ id: row.theme_cluster_id, name: row.theme });
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
                          borderColor: statusColors[row.status] ? `${statusColors[row.status]}.main` : 'grey.400',
                          color: statusColors[row.status] ? `${statusColors[row.status]}.main` : 'text.secondary',
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
                    <TableCell align="right" sx={{ fontFamily: 'monospace' }}>{row.mentions_7d}</TableCell>
                    <TableCell align="right" sx={{
                      fontFamily: 'monospace',
                      fontWeight: 600,
                      color: row.basket_rs_vs_spy >= 60 ? 'success.main' : row.basket_rs_vs_spy <= 40 ? 'error.main' : 'text.primary'
                    }}>
                      {row.basket_rs_vs_spy?.toFixed(0)}
                    </TableCell>
                    <TableCell align="right" sx={{
                      fontFamily: 'monospace',
                      color: row.basket_return_1w > 0 ? 'success.main' : row.basket_return_1w < 0 ? 'error.main' : 'text.primary'
                    }}>
                      {row.basket_return_1w > 0 ? '+' : ''}{row.basket_return_1w?.toFixed(1)}%
                    </TableCell>
                    <TableCell align="right" sx={{
                      fontFamily: 'monospace',
                      color: row.pct_above_50ma >= 70 ? 'success.main' : row.pct_above_50ma <= 30 ? 'error.main' : 'text.primary'
                    }}>
                      {row.pct_above_50ma?.toFixed(0)}%
                    </TableCell>
                    <TableCell align="right" sx={{ fontFamily: 'monospace' }}>{row.num_constituents}</TableCell>
                    <TableCell>
                      <Box display="flex" gap={0.25} flexWrap="nowrap">
                        {row.top_tickers?.slice(0, 8).map((ticker) => (
                          <Box
                            key={ticker}
                            component="span"
                            sx={{ fontSize: '9px', padding: '1px 3px', backgroundColor: '#1976d2', color: 'white', borderRadius: '2px' }}
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
            onPageChange={(e, newPage) => setPage(newPage)}
            rowsPerPage={PAGE_SIZE}
            rowsPerPageOptions={[PAGE_SIZE]}
            labelDisplayedRows={({ from, to, count }) =>
              `${from}-${to} of ${count !== -1 ? count : `more than ${to}`}`
            }
          />
        </Paper>
      )}

      {/* Theme Detail Modal */}
      {selectedTheme && (
        <ThemeDetailModal
          themeId={selectedTheme.id}
          themeName={selectedTheme.name}
          selectedPipeline={selectedPipeline}
          open={!!selectedTheme}
          onClose={() => setSelectedTheme(null)}
        />
      )}

      {/* Manage Sources Modal */}
      <ManageSourcesModal
        open={sourcesModalOpen}
        onClose={() => setSourcesModalOpen(false)}
      />

      {/* Theme Sources Modal */}
      {sourcesTheme && (
        <ThemeSourcesModal
          open={!!sourcesTheme}
          onClose={() => setSourcesTheme(null)}
          themeId={sourcesTheme.id}
          themeName={sourcesTheme.name}
        />
      )}

      {/* Theme Merge Review Modal */}
      <ThemeMergeReviewModal
        open={mergeModalOpen}
        onClose={() => setMergeModalOpen(false)}
      />

      <ThemeCandidateReviewModal
        open={candidateModalOpen}
        onClose={() => setCandidateModalOpen(false)}
        pipeline={selectedPipeline}
      />

      {/* Article Browser Modal */}
      <ArticleBrowserModal
        open={articleBrowserOpen}
        onClose={() => setArticleBrowserOpen(false)}
      />

      {/* Model Settings Modal */}
      <ModelSettingsModal
        open={modelSettingsOpen}
        onClose={() => setModelSettingsOpen(false)}
      />

      <ThemePolicySettingsModal
        open={policySettingsOpen}
        onClose={() => setPolicySettingsOpen(false)}
        pipeline={selectedPipeline}
      />
    </Container>
  );
}

export default ThemesPage;
