import { useQuery } from '@tanstack/react-query';
import {
  Box,
  Chip,
  CircularProgress,
  Dialog,
  DialogContent,
  DialogTitle,
  IconButton,
  Link,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Tooltip,
  Typography,
} from '@mui/material';
import BubbleChartIcon from '@mui/icons-material/BubbleChart';
import CloseIcon from '@mui/icons-material/Close';
import {
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  ResponsiveContainer,
  AreaChart,
  Area,
} from 'recharts';
import {
  getThemeDetail,
  getThemeHistory,
  getThemeRelationshipGraph,
  getThemeMentions,
} from '../../../api/themes';
import TranslatedText from '../../../components/common/TranslatedText';

function getSafeExternalUrl(url) {
  if (!url) {
    return null;
  }
  try {
    const parsedUrl = new URL(url, window.location.origin);
    return parsedUrl.protocol === 'http:' || parsedUrl.protocol === 'https:' ? parsedUrl.href : null;
  } catch {
    return null;
  }
}

export default function ThemeDetailModal({ themeId, themeName, open, onClose, selectedPipeline }) {
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

  const { data: mentions, isLoading: isLoadingMentions } = useQuery({
    queryKey: ['themeMentions', themeId],
    queryFn: () => getThemeMentions(themeId, 50),
    enabled: !!themeId && open,
  });

  const isLoading = isLoadingDetail || isLoadingHistory || isLoadingGraph || isLoadingMentions;

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
            {detail.metrics && (
              <Box
                sx={{
                  mb: 3,
                  display: 'grid',
                  gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))',
                  gap: 2,
                }}
              >
                <Box textAlign="center" p={1} bgcolor="action.hover" borderRadius={1}>
                  <Typography variant="h5" fontWeight="bold">
                    #{detail.metrics.rank || '-'}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    Rank
                  </Typography>
                </Box>
                <Box textAlign="center" p={1} bgcolor="action.hover" borderRadius={1}>
                  <Typography
                    variant="h5"
                    fontWeight="bold"
                    color={
                      detail.metrics.momentum_score >= 70
                        ? 'success.main'
                        : detail.metrics.momentum_score >= 50
                          ? 'warning.main'
                          : 'error.main'
                    }
                  >
                    {detail.metrics.momentum_score?.toFixed(0) || '-'}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    Momentum
                  </Typography>
                </Box>
                <Box textAlign="center" p={1} bgcolor="action.hover" borderRadius={1}>
                  <Typography variant="h5" fontWeight="bold">
                    {detail.metrics.mention_velocity?.toFixed(1) || '-'}x
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    Velocity
                  </Typography>
                </Box>
                <Box textAlign="center" p={1} bgcolor="action.hover" borderRadius={1}>
                  <Typography variant="h5" fontWeight="bold">
                    {detail.metrics.basket_rs_vs_spy?.toFixed(0) || '-'}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    RS vs SPY
                  </Typography>
                </Box>
                <Box textAlign="center" p={1} bgcolor="action.hover" borderRadius={1}>
                  <Typography variant="h5" fontWeight="bold">
                    {detail.metrics.pct_above_50ma?.toFixed(0) || '-'}%
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    Above 50MA
                  </Typography>
                </Box>
                <Box textAlign="center" p={1} bgcolor="action.hover" borderRadius={1}>
                  <Typography variant="h5" fontWeight="bold">
                    {detail.metrics.num_constituents || 0}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    Stocks
                  </Typography>
                </Box>
              </Box>
            )}

            {history?.history?.length > 0 && (
              <Box mb={3}>
                <Typography variant="subtitle2" gutterBottom>
                  Momentum Score History
                </Typography>
                <Box sx={{ height: 200 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={history.history}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="date" tick={{ fontSize: 10 }} tickFormatter={(value) => value.slice(5)} />
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
                          <TableCell sx={{ fontWeight: 600 }}>{stock.symbol}</TableCell>
                          <TableCell align="right" sx={{ fontFamily: 'monospace' }}>
                            {stock.mention_count}
                          </TableCell>
                          <TableCell
                            align="right"
                            sx={{
                              fontFamily: 'monospace',
                              color:
                                stock.confidence >= 0.8
                                  ? 'success.main'
                                  : stock.confidence >= 0.5
                                    ? 'warning.main'
                                    : 'text.secondary',
                            }}
                          >
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
                                backgroundColor: 'action.selected',
                                borderRadius: '2px',
                              }}
                            >
                              {stock.source || 'unk'}
                            </Box>
                          </TableCell>
                          <TableCell sx={{ fontSize: '10px', color: 'text.secondary', fontFamily: 'monospace' }}>
                            {stock.last_mentioned_at ? new Date(stock.last_mentioned_at).toLocaleDateString() : '-'}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              </Box>
            )}

            <Box mt={3}>
              <Box sx={{ fontSize: '12px', fontWeight: 600, mb: 0.5 }}>
                Source Articles ({mentions?.total_count || 0})
              </Box>
              {mentions?.mentions?.length > 0 ? (
                <TableContainer sx={{ maxHeight: 300 }}>
                  <Table size="small" stickyHeader>
                    <TableHead>
                      <TableRow>
                        <TableCell>Title</TableCell>
                        <TableCell>Excerpt</TableCell>
                        <TableCell>Source</TableCell>
                        <TableCell>Sentiment</TableCell>
                        <TableCell>Date</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {mentions.mentions.map((mention) => {
                        const safeContentUrl = getSafeExternalUrl(mention.content_url);
                        return (
                          <TableRow key={mention.mention_id} hover>
                            <TableCell sx={{ maxWidth: 200 }}>
                              {safeContentUrl ? (
                              <Link
                                href={safeContentUrl}
                                target="_blank"
                                rel="noopener noreferrer"
                                underline="hover"
                                sx={{
                                  fontSize: '11px',
                                  fontWeight: 500,
                                  display: '-webkit-box',
                                  WebkitLineClamp: 2,
                                  WebkitBoxOrient: 'vertical',
                                  overflow: 'hidden',
                                }}
                              >
                                {mention.content_title || 'Untitled'}
                              </Link>
                            ) : (
                              <Box
                                sx={{
                                  fontSize: '11px',
                                  fontWeight: 500,
                                  display: '-webkit-box',
                                  WebkitLineClamp: 2,
                                  WebkitBoxOrient: 'vertical',
                                  overflow: 'hidden',
                                }}
                              >
                                {mention.content_title || 'Untitled'}
                              </Box>
                            )}
                            </TableCell>
                            <TableCell sx={{ maxWidth: 280 }}>
                              <TranslatedText
                                originalText={mention.excerpt}
                                translatedText={mention.translated_excerpt}
                                sourceLanguage={mention.source_language}
                                translationMetadata={mention.translation_metadata}
                                typographySx={{
                                  fontSize: '10px',
                                  color: 'text.secondary',
                                  display: '-webkit-box',
                                  WebkitLineClamp: 2,
                                  WebkitBoxOrient: 'vertical',
                                  overflow: 'hidden',
                                }}
                              />
                            </TableCell>
                            <TableCell>
                              <Box display="flex" alignItems="center" gap={0.5}>
                                {mention.source_name && (
                                  <Box sx={{ fontSize: '10px', fontWeight: 500 }}>{mention.source_name}</Box>
                                )}
                                <Box
                                  component="span"
                                  sx={{
                                    fontSize: '9px',
                                    padding: '1px 3px',
                                    backgroundColor: 'action.selected',
                                    borderRadius: '2px',
                                  }}
                                >
                                  {mention.source_type}
                                </Box>
                              </Box>
                            </TableCell>
                            <TableCell>
                              <Chip
                                size="small"
                                label={mention.sentiment || 'neutral'}
                                color={
                                  mention.sentiment === 'bullish'
                                    ? 'success'
                                    : mention.sentiment === 'bearish'
                                      ? 'error'
                                      : 'default'
                                }
                                sx={{ height: 18, fontSize: '9px' }}
                              />
                            </TableCell>
                            <TableCell
                              sx={{
                                fontSize: '10px',
                                color: 'text.secondary',
                                fontFamily: 'monospace',
                                whiteSpace: 'nowrap',
                              }}
                            >
                              {mention.published_at ? new Date(mention.published_at).toLocaleDateString() : '-'}
                            </TableCell>
                          </TableRow>
                        );
                      })}
                    </TableBody>
                  </Table>
                </TableContainer>
              ) : (
                <Box sx={{ fontSize: '11px', color: 'text.secondary', textAlign: 'center', py: 2 }}>
                  No articles found
                </Box>
              )}
            </Box>

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
                      {detail.relationships.map((relationship) => (
                        <TableRow key={relationship.relation_id} hover>
                          <TableCell>
                            <Chip
                              size="small"
                              label={relationship.direction}
                              color={relationship.direction === 'outgoing' ? 'primary' : 'default'}
                              variant="outlined"
                            />
                          </TableCell>
                          <TableCell>
                            <Tooltip
                              title={
                                relationship.relationship_type === 'subset'
                                  ? 'Subset: one theme is mostly contained by another. Review before merging.'
                                  : relationship.relationship_type === 'related'
                                    ? 'Related: topical overlap exists, but themes can still be distinct.'
                                    : 'Distinct: themes should remain separate unless new evidence appears.'
                              }
                            >
                              <Chip
                                size="small"
                                label={relationship.relationship_type}
                                color={
                                  relationship.relationship_type === 'subset'
                                    ? 'warning'
                                    : relationship.relationship_type === 'related'
                                      ? 'info'
                                      : 'default'
                                }
                              />
                            </Tooltip>
                          </TableCell>
                          <TableCell sx={{ fontWeight: 600 }}>
                            {relationship.peer_theme_display_name || relationship.peer_theme_name || '-'}
                          </TableCell>
                          <TableCell align="right" sx={{ fontFamily: 'monospace' }}>
                            {(Math.max(0, Math.min(1, relationship.confidence || 0)) * 100).toFixed(0)}%
                          </TableCell>
                          <TableCell sx={{ fontSize: '10px', color: 'text.secondary', fontFamily: 'monospace' }}>
                            {relationship.provenance || '-'}
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
                  <Box sx={{ fontSize: '12px', fontWeight: 600 }}>Relationship Graph Context</Box>
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
}
