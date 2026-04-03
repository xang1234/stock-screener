import { useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Link as RouterLink, useParams } from 'react-router-dom';
import {
  Alert,
  Box,
  Card,
  CardContent,
  Chip,
  CircularProgress,
  Divider,
  Grid,
  Link,
  List,
  ListItem,
  ListItemText,
  Stack,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Typography,
} from '@mui/material';

import { getStockDecisionDashboard } from '../../api/stocks';
import CandlestickChart from '../Charts/CandlestickChart';
import AddToWatchlistMenu from '../common/AddToWatchlistMenu';
import {
  formatLargeNumber,
  formatPercent,
  formatRatio,
  getScoreColor,
} from '../../utils/formatUtils';

function MetricChip({ label, value, color = 'default' }) {
  return (
    <Chip
      label={`${label}: ${value ?? '-'}`}
      size="small"
      color={color}
      variant={color === 'default' ? 'outlined' : 'filled'}
    />
  );
}

function FactorList({ title, items, emptyText }) {
  return (
    <Box>
      <Typography variant="subtitle2" sx={{ mb: 1, fontWeight: 700 }}>
        {title}
      </Typography>
      {items?.length ? (
        <List dense sx={{ py: 0 }}>
          {items.map((item) => (
            <ListItem key={`${item.screener_name}:${item.criterion_name}`} sx={{ px: 0 }}>
              <ListItemText
                primary={`${item.criterion_name} (${item.screener_name})`}
                secondary={`${item.score.toFixed(1)} / ${item.max_score.toFixed(1)}`}
              />
            </ListItem>
          ))}
        </List>
      ) : (
        <Typography variant="body2" color="text.secondary">
          {emptyText}
        </Typography>
      )}
    </Box>
  );
}

function StockDetails() {
  const { symbol } = useParams();

  const { data, isLoading, error } = useQuery({
    queryKey: ['stockDecisionDashboard', symbol],
    queryFn: () => getStockDecisionDashboard(symbol),
    enabled: Boolean(symbol),
    staleTime: 60_000,
  });

  const headerChips = useMemo(() => {
    if (!data) return [];
    const chart = data.chart?.chart_data || {};
    return [
      { label: 'Stage', value: chart.stage },
      { label: 'RS', value: chart.rs_rating != null ? chart.rs_rating.toFixed(0) : null },
      { label: 'EPS', value: chart.eps_rating },
      { label: 'VCP', value: chart.vcp_detected ? 'Yes' : 'No' },
      { label: 'ADR', value: chart.adr_percent != null ? formatPercent(chart.adr_percent) : null },
    ].filter((chip) => chip.value != null && chip.value !== '-');
  }, [data]);

  if (isLoading) {
    return (
      <Box display="flex" justifyContent="center" py={6}>
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Alert severity="error">
        Failed to load stock workspace: {error.message}
      </Alert>
    );
  }

  if (!data) {
    return (
      <Alert severity="warning">
        No stock workspace data is available.
      </Alert>
    );
  }

  const info = data.info || {};
  const fundamentals = data.fundamentals || {};
  const technicals = data.technicals || {};
  const chart = data.chart?.chart_data || {};
  const price = info.current_price ?? technicals.current_price ?? chart.current_price;
  const decision = data.decision_summary;

  return (
    <Box>
      <Stack
        direction={{ xs: 'column', md: 'row' }}
        justifyContent="space-between"
        alignItems={{ xs: 'flex-start', md: 'center' }}
        spacing={2}
        sx={{ mb: 2 }}
      >
        <Box>
          <Typography variant="h4" sx={{ fontWeight: 800, letterSpacing: '-0.02em' }}>
            {data.symbol}
          </Typography>
          <Typography variant="body1" color="text.secondary">
            {info.name || 'Unknown company'}
          </Typography>
          <Stack direction="row" spacing={1} sx={{ mt: 1, flexWrap: 'wrap' }}>
            {info.sector && <Chip size="small" label={info.sector} />}
            {info.industry && <Chip size="small" variant="outlined" label={info.industry} />}
            {price != null && <Chip size="small" color="primary" label={`$${Number(price).toFixed(2)}`} />}
          </Stack>
        </Box>

        <Stack direction="row" spacing={1}>
          <AddToWatchlistMenu
            symbols={data.symbol}
            trigger={<Chip color="primary" clickable label="Add to Watchlist" />}
          />
          {decision.rating && (
            <Chip
              label={decision.rating}
              sx={{
                bgcolor: getScoreColor(decision.composite_score) || 'action.selected',
                color: '#fff',
                fontWeight: 700,
              }}
            />
          )}
        </Stack>
      </Stack>

      {data.degraded_reasons?.length > 0 && (
        <Alert severity="info" sx={{ mb: 2 }}>
          Workspace is partially degraded: {data.degraded_reasons.join(', ')}.
        </Alert>
      )}

      <Grid container spacing={2}>
        <Grid item xs={12} lg={8}>
          <Card sx={{ overflow: 'hidden' }}>
            <Box sx={{ px: 2, py: 1.5, borderBottom: 1, borderColor: 'divider' }}>
              <Typography variant="subtitle1" sx={{ fontWeight: 700, mb: 1 }}>
                Chart
              </Typography>
              <Stack direction="row" spacing={1} sx={{ flexWrap: 'wrap' }}>
                {headerChips.map((chip) => (
                  <MetricChip key={chip.label} label={chip.label} value={chip.value} />
                ))}
              </Stack>
            </Box>
            <Box sx={{ p: 1 }}>
              <CandlestickChart
                symbol={data.symbol}
                priceData={data.chart?.price_history || []}
                height={520}
              />
            </Box>
          </Card>
        </Grid>

        <Grid item xs={12} lg={4}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Typography variant="subtitle1" sx={{ fontWeight: 700, mb: 2 }}>
                Decision Summary
              </Typography>
              <Stack spacing={1.25}>
                <MetricChip
                  label="Composite"
                  value={decision.composite_score != null ? decision.composite_score.toFixed(1) : '-'}
                  color="primary"
                />
                <MetricChip label="Screeners" value={`${decision.screeners_passed}/${decision.screeners_total}`} />
                <MetricChip label="Method" value={decision.composite_method || '-'} />
                <MetricChip label="Feature Date" value={decision.freshness?.feature_as_of_date || '-'} />
                <MetricChip label="Breadth Date" value={decision.freshness?.breadth_date || '-'} />
                <MetricChip label="Price History" value={decision.freshness?.has_price_history ? 'Ready' : 'Missing'} />
              </Stack>
              <Divider sx={{ my: 2 }} />
              <FactorList
                title="Top Strengths"
                items={decision.top_strengths}
                emptyText="No positive criteria available."
              />
              <Divider sx={{ my: 2 }} />
              <FactorList
                title="Top Weaknesses"
                items={decision.top_weaknesses}
                emptyText="No failing criteria were recorded."
              />
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} lg={6}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Typography variant="subtitle1" sx={{ fontWeight: 700, mb: 2 }}>
                Screener Breakdown
              </Typography>
              <Stack spacing={2}>
                {data.screener_explanations?.length ? data.screener_explanations.map((screener) => (
                  <Box key={screener.screener_name}>
                    <Stack direction="row" justifyContent="space-between" alignItems="center">
                      <Typography variant="subtitle2" sx={{ textTransform: 'capitalize', fontWeight: 700 }}>
                        {screener.screener_name.replaceAll('_', ' ')}
                      </Typography>
                      <Chip
                        size="small"
                        label={`${screener.score.toFixed(1)} · ${screener.rating}`}
                        color={screener.passes ? 'success' : 'default'}
                      />
                    </Stack>
                    <List dense sx={{ py: 0.5 }}>
                      {screener.criteria.slice(0, 5).map((criterion) => (
                        <ListItem key={criterion.name} sx={{ px: 0 }}>
                          <ListItemText
                            primary={criterion.name}
                            secondary={`${criterion.score.toFixed(1)} / ${criterion.max_score.toFixed(1)}`}
                          />
                          <Chip
                            size="small"
                            label={criterion.passed ? 'Pass' : 'Weak'}
                            color={criterion.passed ? 'success' : 'warning'}
                            variant={criterion.passed ? 'filled' : 'outlined'}
                          />
                        </ListItem>
                      ))}
                    </List>
                  </Box>
                )) : (
                  <Typography variant="body2" color="text.secondary">
                    Screener explanations are unavailable for this symbol.
                  </Typography>
                )}
              </Stack>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} lg={6}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Typography variant="subtitle1" sx={{ fontWeight: 700, mb: 2 }}>
                Industry Peers
              </Typography>
              {data.peers?.length ? (
                <TableContainer>
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell>Symbol</TableCell>
                        <TableCell>Company</TableCell>
                        <TableCell align="right">Comp</TableCell>
                        <TableCell align="right">RS</TableCell>
                        <TableCell align="right">Stage</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {data.peers.slice(0, 8).map((peer) => (
                        <TableRow key={peer.symbol}>
                          <TableCell>
                            <Link component={RouterLink} to={`/stock/${peer.symbol}`} underline="hover">
                              {peer.symbol}
                            </Link>
                          </TableCell>
                          <TableCell>{peer.company_name || '-'}</TableCell>
                          <TableCell align="right">{peer.composite_score?.toFixed(1) || '-'}</TableCell>
                          <TableCell align="right">{peer.rs_rating?.toFixed(0) || '-'}</TableCell>
                          <TableCell align="right">{peer.stage ?? '-'}</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              ) : (
                <Typography variant="body2" color="text.secondary">
                  No peer data is available from the latest published feature run.
                </Typography>
              )}
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} lg={6}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Typography variant="subtitle1" sx={{ fontWeight: 700, mb: 2 }}>
                Related Themes
              </Typography>
              {data.themes?.length ? (
                <Stack spacing={1.5}>
                  {data.themes.map((theme) => (
                    <Box key={theme.theme_id}>
                      <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 0.5, flexWrap: 'wrap' }}>
                        <Chip size="small" color="secondary" label={theme.display_name} />
                        {theme.status && <Chip size="small" variant="outlined" label={theme.status} />}
                        {theme.lifecycle_state && <Chip size="small" variant="outlined" label={theme.lifecycle_state} />}
                      </Stack>
                      <Typography variant="body2" color="text.secondary">
                        Momentum {formatRatio(theme.momentum_score)} · Velocity {formatRatio(theme.mention_velocity)}x · 1M basket {formatPercent(theme.basket_return_1m)}
                      </Typography>
                    </Box>
                  ))}
                </Stack>
              ) : (
                <Typography variant="body2" color="text.secondary">
                  This symbol is not currently linked to an active theme cluster.
                </Typography>
              )}
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} lg={6}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Typography variant="subtitle1" sx={{ fontWeight: 700, mb: 2 }}>
                Market Regime and Risk Context
              </Typography>
              <Stack direction="row" spacing={1} sx={{ mb: 2 }}>
                <Chip
                  color={data.regime.label === 'offense' ? 'success' : data.regime.label === 'defense' ? 'warning' : 'default'}
                  label={data.regime.label}
                />
                <MetricChip label="5D Ratio" value={formatRatio(data.regime.ratio_5day)} />
                <MetricChip label="10D Ratio" value={formatRatio(data.regime.ratio_10day)} />
              </Stack>
              <Typography variant="body2" sx={{ mb: 2 }}>
                {data.regime.summary}
              </Typography>
              <Divider sx={{ my: 2 }} />
              <Stack spacing={1}>
                <Typography variant="body2" color="text.secondary">
                  Market cap: {formatLargeNumber(fundamentals.market_cap ?? info.market_cap, '$')}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  EPS growth (quarterly): {formatPercent(fundamentals.eps_growth_quarterly ?? chart.eps_growth_qq)}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Revenue growth: {formatPercent(fundamentals.revenue_growth ?? chart.sales_growth_qq)}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  52W range: {technicals.low_52w != null && technicals.high_52w != null
                    ? `$${Number(technicals.low_52w).toFixed(2)} - $${Number(technicals.high_52w).toFixed(2)}`
                    : '-'}
                </Typography>
              </Stack>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
}

export default StockDetails;
