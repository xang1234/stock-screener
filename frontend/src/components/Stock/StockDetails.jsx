import { useMemo, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Link as RouterLink, useNavigate, useParams } from 'react-router-dom';
import {
  Accordion,
  AccordionDetails,
  AccordionSummary,
  Alert,
  Box,
  Chip,
  CircularProgress,
  Divider,
  Link,
  Stack,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  ToggleButton,
  ToggleButtonGroup,
  Typography,
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import CancelIcon from '@mui/icons-material/Cancel';

import { getStockDecisionDashboard, getStockValidation } from '../../api/stocks';
import CandlestickChart from '../Charts/CandlestickChart';
import StockMetricsSidebar from '../Scan/StockMetricsSidebar';
import AddToWatchlistMenu from '../common/AddToWatchlistMenu';
import PeerComparisonModal from '../Scan/PeerComparisonModal';
import {
  ValidationDegradedAlert,
  ValidationFailureClustersTable,
  ValidationRecentEventsTable,
  ValidationSummaryCards,
} from '../Validation/ValidationPanels';
import { getGroupRankColor, getStageColor } from '../../utils/colorUtils';
import {
  formatLargeNumber,
  formatPercent,
  formatRatio,
  getScoreColor,
} from '../../utils/formatUtils';
import { useStrategyProfile } from '../../contexts/StrategyProfileContext';

const CriteriaList = ({ title, items, emptyText }) => (
  <Box sx={{ flex: 1 }}>
    <SectionHeader>{title}</SectionHeader>
    {items?.length ? (
      <Box sx={{ display: 'grid', gridTemplateColumns: '1fr', gap: 0.25 }}>
        {items.map((item) => (
          <Box key={`${item.screener_name}:${item.criterion_name}`} sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <Typography variant="caption" color="text.secondary" sx={{ fontSize: '0.75rem' }}>
              {item.criterion_name}
            </Typography>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
              <Typography variant="body2" sx={{ fontSize: '0.75rem' }}>
                {item.score.toFixed(1)}/{item.max_score.toFixed(1)}
              </Typography>
              <BoolIndicator value={item.passed} />
            </Box>
          </Box>
        ))}
      </Box>
    ) : (
      <Typography variant="caption" color="text.secondary">{emptyText}</Typography>
    )}
  </Box>
);

const MetricRow = ({ label, value, color }) => (
  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
    <Typography variant="caption" color="text.secondary" sx={{ fontSize: '0.75rem' }}>
      {label}
    </Typography>
    <Typography variant="body2" fontWeight="medium" sx={{ color: color || 'text.primary', fontSize: '0.8rem' }}>
      {value}
    </Typography>
  </Box>
);

const SectionHeader = ({ children }) => (
  <Typography
    variant="caption"
    color="text.secondary"
    sx={{ fontWeight: 'bold', letterSpacing: 0.5, fontSize: '0.65rem', mb: 0.5, display: 'block' }}
  >
    {children}
  </Typography>
);

const BoolIndicator = ({ value }) => {
  if (value) return <CheckCircleIcon sx={{ fontSize: 14, color: '#4caf50' }} />;
  return <CancelIcon sx={{ fontSize: 14, color: '#9e9e9e' }} />;
};

function MetricBox({ label, value, bgcolor }) {
  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
      <Box sx={{ borderRadius: 1, px: 1.5, py: 0.5, textAlign: 'center', minWidth: 36, bgcolor }}>
        <Typography variant="body2" noWrap sx={{ fontSize: '0.8rem', color: 'white', fontWeight: 'bold' }}>
          {value}
        </Typography>
      </Box>
      <Typography variant="caption" color="text.secondary" sx={{ fontSize: '0.65rem', mt: 0.25 }}>
        {label}
      </Typography>
    </Box>
  );
}

function InfoBox({ label, value }) {
  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
      <Box sx={{
        border: 1,
        borderColor: 'divider',
        borderRadius: 1,
        px: 1.5,
        py: 0.5,
        minWidth: 80,
        maxWidth: 180,
        textAlign: 'center',
        bgcolor: 'background.paper',
      }}>
        <Typography variant="body2" noWrap sx={{ fontSize: '0.8rem' }}>
          {value || '-'}
        </Typography>
      </Box>
      <Typography variant="caption" color="text.secondary" sx={{ fontSize: '0.65rem', mt: 0.25 }}>
        {label}
      </Typography>
    </Box>
  );
}

function StockDetails() {
  const { ticker: symbol } = useParams();
  const navigate = useNavigate();
  const { activeProfile, activeProfileDetail } = useStrategyProfile();
  const [peerModalOpen, setPeerModalOpen] = useState(false);
  const [validationLookbackDays, setValidationLookbackDays] = useState(365);

  const { data, isLoading, error } = useQuery({
    queryKey: ['stockDecisionDashboard', symbol, activeProfile],
    queryFn: () => getStockDecisionDashboard(symbol, activeProfile),
    enabled: Boolean(symbol),
    staleTime: 60_000,
  });
  const {
    data: validationData,
    isLoading: isValidationLoading,
    error: validationError,
  } = useQuery({
    queryKey: ['stockValidation', symbol, validationLookbackDays],
    queryFn: () => getStockValidation(symbol, validationLookbackDays),
    enabled: Boolean(symbol),
    staleTime: 60_000,
  });

  const sidebarFundamentals = useMemo(
    () => data ? { ...(data.fundamentals || {}), symbol: data.symbol } : {},
    [data]
  );

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
  const decision = data.decision_summary || {
    composite_score: null,
    rating: null,
    screeners_passed: 0,
    screeners_total: 0,
    composite_method: null,
    top_strengths: [],
    top_weaknesses: [],
    freshness: data.freshness || {},
  };
  const freshness = decision.freshness || data.freshness || {};
  const regime = data.regime || null;
  const eventRisk = data.event_risk || { notes: [] };
  const regimeActions = data.regime_actions || {
    stance: regime?.label || 'unavailable',
    sizing_guidance: 'probe',
    avoid_new_entries: false,
    preferred_setups: [],
    caution_flags: [],
    summary: 'Action guidance is unavailable.',
  };
  const strengthsTitle = activeProfileDetail?.stock_action?.strengths_title || 'Top Strengths';
  const weaknessesTitle = activeProfileDetail?.stock_action?.weaknesses_title || 'Top Weaknesses';

  const adrValue = chart.adr_percent;
  const epsRating = chart.eps_rating;
  const groupRank = chart.ibd_group_rank;

  return (
    <Box>
      {/* ─── Header ─── */}
      <Box
        sx={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          px: 2,
          py: 1.5,
          borderBottom: 1,
          borderColor: 'divider',
          bgcolor: 'background.default',
          borderRadius: '4px 4px 0 0',
          flexWrap: 'wrap',
          gap: 1,
        }}
      >
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <Box>
            <Typography variant="h5" fontWeight="bold">
              {data.symbol}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              {info.name || 'Unknown company'}
            </Typography>
          </Box>

          {groupRank != null && (
            <MetricBox label="Grp Rnk" value={groupRank} bgcolor={getGroupRankColor(groupRank)} />
          )}
          {adrValue != null && (
            <MetricBox
              label="ADR"
              value={`${Number(adrValue).toFixed(1)}%`}
              bgcolor={Number(adrValue) >= 4 ? 'success.main' : Number(adrValue) >= 2 ? 'warning.main' : 'error.main'}
            />
          )}
          {epsRating != null && (
            <MetricBox
              label="EPS Rtg"
              value={epsRating}
              bgcolor={epsRating >= 80 ? 'success.main' : epsRating >= 50 ? 'warning.main' : 'error.main'}
            />
          )}
          {(chart.ibd_industry_group || info.sector || info.industry) && (
            <Box sx={{ display: 'flex', gap: 1.5, ml: 1 }}>
              {chart.ibd_industry_group && <InfoBox label="IBD" value={chart.ibd_industry_group} />}
              {(chart.gics_sector || info.sector) && <InfoBox label="Sector" value={chart.gics_sector || info.sector} />}
              {(chart.gics_industry || info.industry) && <InfoBox label="Industry" value={chart.gics_industry || info.industry} />}
            </Box>
          )}
        </Box>

        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          {price != null && (
            <Chip size="small" color="primary" label={`$${Number(price).toFixed(2)}`} />
          )}
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
        </Box>
      </Box>

      {data.degraded_reasons?.length > 0 && (
        <Alert severity="info" sx={{ borderRadius: 0 }}>
          Workspace is partially degraded: {data.degraded_reasons.join(', ')}.
        </Alert>
      )}

      {/* ─── Split Panel: Sidebar + Chart ─── */}
      <Box sx={{ display: 'flex', borderBottom: 1, borderColor: 'divider', height: 520 }}>
        <StockMetricsSidebar
          stockData={(data.chart?.chart_data?.source !== 'unavailable' && data.chart?.chart_data) || null}
          fundamentals={sidebarFundamentals}
          onViewPeers={() => setPeerModalOpen(true)}
        />
        <Box sx={{ flex: 1, overflow: 'hidden', bgcolor: 'background.paper' }}>
          <CandlestickChart
            key={data.symbol}
            symbol={data.symbol}
            priceData={data.chart?.price_history || []}
            height={520}
          />
        </Box>
      </Box>

      {/* ─── Accordion Sections ─── */}
      <Box sx={{ mt: 1 }}>

        {/* Decision Summary */}
        <Accordion>
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Typography variant="subtitle1" sx={{ fontWeight: 700 }}>
              Decision Summary
            </Typography>
            {decision.composite_score != null && (
              <Chip
                size="small"
                label={`${decision.composite_score.toFixed(1)} · ${decision.screeners_passed}/${decision.screeners_total}`}
                sx={{ ml: 2, bgcolor: getScoreColor(decision.composite_score) || 'action.selected', color: '#fff' }}
              />
            )}
          </AccordionSummary>
          <AccordionDetails>
            <SectionHeader>OVERVIEW</SectionHeader>
            <Box sx={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 1.5, mb: 2 }}>
              <MetricRow label="Composite" value={decision.composite_score != null ? decision.composite_score.toFixed(1) : '-'} color="primary.main" />
              <MetricRow label="Screeners" value={`${decision.screeners_passed}/${decision.screeners_total}`} />
              <MetricRow label="Method" value={decision.composite_method || '-'} />
              <MetricRow label="Feature Date" value={freshness.feature_as_of_date || '-'} />
              <MetricRow label="Breadth Date" value={freshness.breadth_date || '-'} />
              <MetricRow label="Price History" value={freshness.has_price_history ? 'Ready' : 'Missing'} />
            </Box>
            <Divider sx={{ my: 1.5 }} />
            <Box sx={{ display: 'flex', gap: 4 }}>
              <CriteriaList title={strengthsTitle.toUpperCase()} items={decision.top_strengths} emptyText="No positive criteria." />
              <CriteriaList title={weaknessesTitle.toUpperCase()} items={decision.top_weaknesses} emptyText="No failing criteria." />
            </Box>
          </AccordionDetails>
        </Accordion>

        {/* Screener Breakdown */}
        <Accordion>
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Typography variant="subtitle1" sx={{ fontWeight: 700 }}>
              Screener Breakdown
            </Typography>
          </AccordionSummary>
          <AccordionDetails>
            {data.screener_explanations?.length ? (
              <Stack spacing={1.5}>
                {data.screener_explanations.map((screener, idx) => (
                  <Box key={screener.screener_name}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 0.5 }}>
                      <Typography variant="caption" sx={{ textTransform: 'uppercase', fontWeight: 'bold', letterSpacing: 0.5, fontSize: '0.65rem' }}>
                        {screener.screener_name.replaceAll('_', ' ')}
                      </Typography>
                      <Chip
                        size="small"
                        label={`${screener.score.toFixed(1)} · ${screener.rating}`}
                        color={screener.passes ? 'success' : 'default'}
                        sx={{ height: 20, fontSize: '0.7rem' }}
                      />
                    </Box>
                    <Box sx={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', columnGap: 1.5, rowGap: 0.25 }}>
                      {screener.criteria.map((criterion) => (
                        <Box key={criterion.name} sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                          <Typography variant="caption" color="text.secondary" sx={{ fontSize: '0.7rem' }}>
                            {criterion.name}
                          </Typography>
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
                            <Typography variant="caption" sx={{ fontSize: '0.7rem' }}>
                              {criterion.score.toFixed(1)}/{criterion.max_score.toFixed(1)}
                            </Typography>
                            <BoolIndicator value={criterion.passed} />
                          </Box>
                        </Box>
                      ))}
                    </Box>
                    {idx < data.screener_explanations.length - 1 && (
                      <Divider sx={{ mt: 1 }} />
                    )}
                  </Box>
                ))}
              </Stack>
            ) : (
              <Typography variant="body2" color="text.secondary">
                Screener explanations are unavailable for this symbol.
              </Typography>
            )}
          </AccordionDetails>
        </Accordion>

        {/* Industry Peers */}
        <Accordion>
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Typography variant="subtitle1" sx={{ fontWeight: 700 }}>
              Industry Peers
            </Typography>
            {data.peers?.length > 0 && (
              <Chip size="small" variant="outlined" label={`${data.peers.length}`} sx={{ ml: 2 }} />
            )}
          </AccordionSummary>
          <AccordionDetails>
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
                    {data.peers.slice(0, 12).map((peer) => (
                      <TableRow key={peer.symbol}>
                        <TableCell>
                          <Link component={RouterLink} to={`/stocks/${encodeURIComponent(peer.symbol)}`} underline="hover">
                            {peer.symbol}
                          </Link>
                        </TableCell>
                        <TableCell>{peer.company_name || '-'}</TableCell>
                        <TableCell align="right">{peer.composite_score?.toFixed(1) || '-'}</TableCell>
                        <TableCell align="right">{peer.rs_rating?.toFixed(0) || '-'}</TableCell>
                        <TableCell align="right">
                          {peer.stage != null ? (
                            <Chip
                              label={`S${peer.stage}`}
                              size="small"
                              sx={{
                                bgcolor: getStageColor(peer.stage),
                                color: 'white',
                                fontSize: '0.65rem',
                                height: 18,
                                '& .MuiChip-label': { px: 0.75 },
                              }}
                            />
                          ) : '-'}
                        </TableCell>
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
          </AccordionDetails>
        </Accordion>

        {/* Related Themes */}
        <Accordion>
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Typography variant="subtitle1" sx={{ fontWeight: 700 }}>
              Related Themes
            </Typography>
          </AccordionSummary>
          <AccordionDetails>
            {data.themes?.length ? (
              <Box sx={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 1.5 }}>
                {data.themes.map((theme) => (
                  <Box key={theme.theme_id} sx={{ border: 1, borderColor: 'divider', borderRadius: 1, p: 1 }}>
                    <Stack direction="row" spacing={0.5} alignItems="center" sx={{ mb: 0.5 }}>
                      <Chip size="small" color="secondary" label={theme.display_name} sx={{ fontSize: '0.7rem', height: 20 }} />
                      {theme.status && <Chip size="small" variant="outlined" label={theme.status} sx={{ fontSize: '0.65rem', height: 18 }} />}
                    </Stack>
                    <Box sx={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 0.25 }}>
                      <MetricRow label="Momentum" value={formatRatio(theme.momentum_score)} />
                      <MetricRow label="Velocity" value={`${formatRatio(theme.mention_velocity)}x`} />
                      <MetricRow label="1M Basket" value={formatPercent(theme.basket_return_1m)} />
                      {theme.lifecycle_state && <MetricRow label="Lifecycle" value={theme.lifecycle_state} />}
                    </Box>
                  </Box>
                ))}
              </Box>
            ) : (
              <Typography variant="body2" color="text.secondary">
                This symbol is not currently linked to an active theme cluster.
              </Typography>
            )}
          </AccordionDetails>
        </Accordion>

        {/* Market Regime & Risk */}
        <Accordion>
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Typography variant="subtitle1" sx={{ fontWeight: 700 }}>
              Market Regime & Risk Context
            </Typography>
            {regime && (
              <Chip
                size="small"
                label={regime.label}
                color={regime.label === 'offense' ? 'success' : regime.label === 'defense' ? 'warning' : 'default'}
                sx={{ ml: 2, height: 20, fontSize: '0.7rem' }}
              />
            )}
          </AccordionSummary>
          <AccordionDetails>
            <Box sx={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', columnGap: 1.5, rowGap: 0.5 }}>
              {regime ? (
                <>
                  <MetricRow label="Stance" value={regime.label} />
                  <MetricRow label="5D Ratio" value={formatRatio(regime.ratio_5day)} />
                  <MetricRow label="10D Ratio" value={formatRatio(regime.ratio_10day)} />
                </>
              ) : (
                <Typography variant="body2" color="text.secondary" sx={{ gridColumn: '1 / -1' }}>
                  Market regime context is unavailable for this symbol.
                </Typography>
              )}
              <MetricRow label="Market Cap" value={formatLargeNumber(fundamentals.market_cap ?? info.market_cap, '$')} />
              <MetricRow label="EPS Q/Q" value={formatPercent(fundamentals.eps_growth_quarterly ?? chart.eps_growth_qq)} />
              <MetricRow label="Revenue Growth" value={formatPercent(fundamentals.revenue_growth ?? chart.sales_growth_qq)} />
              <MetricRow
                label="52W Range"
                value={technicals.low_52w != null && technicals.high_52w != null
                  ? `$${Number(technicals.low_52w).toFixed(0)}-$${Number(technicals.high_52w).toFixed(0)}`
                  : '-'}
              />
            </Box>
            {regime?.summary && (
              <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 1 }}>
                {regime.summary}
              </Typography>
            )}
          </AccordionDetails>
        </Accordion>

        <Accordion>
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Typography variant="subtitle1" sx={{ fontWeight: 700 }}>
              Event Risk & Action Plan
            </Typography>
            <Chip
              size="small"
              sx={{ ml: 2, height: 20, fontSize: '0.7rem' }}
              color={eventRisk.earnings_window_risk === 'imminent' ? 'error' : eventRisk.earnings_window_risk === 'caution' ? 'warning' : 'default'}
              label={eventRisk.earnings_window_risk || 'safe'}
            />
          </AccordionSummary>
          <AccordionDetails>
            <Stack spacing={1.5}>
              <Box sx={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', columnGap: 1.5, rowGap: 0.5 }}>
                <MetricRow label="Next Earnings" value={eventRisk.next_earnings_date || '-'} />
                <MetricRow label="Days Until" value={eventRisk.days_until_earnings ?? '-'} />
                <MetricRow label="Window Risk" value={eventRisk.earnings_window_risk || '-'} />
                <MetricRow label="Beat Count (4)" value={eventRisk.beat_count_last_4 ?? 0} />
                <MetricRow label="Miss Count (4)" value={eventRisk.miss_count_last_4 ?? 0} />
                <MetricRow label="Recent Earnings" value={eventRisk.recent_earnings_count ?? 0} />
                <MetricRow label="Avg Gap" value={formatPercent(eventRisk.avg_post_earnings_gap_pct)} />
                <MetricRow label="Avg 5S Return" value={formatPercent(eventRisk.avg_post_earnings_5s_return_pct)} />
                <MetricRow label="Inst Ownership" value={formatPercent(eventRisk.institutional_ownership_current)} />
                <MetricRow label="Inst Δ 90D" value={formatPercent(eventRisk.institutional_ownership_delta_90d)} />
                <MetricRow label="Sizing" value={regimeActions.sizing_guidance} />
                <MetricRow label="Avoid New Entries" value={regimeActions.avoid_new_entries ? 'Yes' : 'No'} />
              </Box>

              <Box>
                <SectionHeader>ACTION PLAN</SectionHeader>
                <Typography variant="body2">{regimeActions.summary}</Typography>
              </Box>

              <Box>
                <SectionHeader>PREFERRED SETUPS</SectionHeader>
                {regimeActions.preferred_setups?.length ? (
                  <Stack direction="row" spacing={0.75} flexWrap="wrap" useFlexGap>
                    {regimeActions.preferred_setups.map((setup, index) => (
                      <Chip key={`${setup}-${index}`} label={setup} size="small" variant="outlined" />
                    ))}
                  </Stack>
                ) : (
                  <Typography variant="caption" color="text.secondary">
                    No setup guidance is available.
                  </Typography>
                )}
              </Box>

              <Box>
                <SectionHeader>CAUTION FLAGS</SectionHeader>
                {regimeActions.caution_flags?.length ? (
                  <Stack spacing={0.5}>
                    {regimeActions.caution_flags.map((flag, index) => (
                      <Typography key={`${flag}-${index}`} variant="caption" color="text.secondary">
                        {flag}
                      </Typography>
                    ))}
                  </Stack>
                ) : (
                  <Typography variant="caption" color="text.secondary">
                    No active caution flags.
                  </Typography>
                )}
              </Box>

              <Box>
                <SectionHeader>EVENT NOTES</SectionHeader>
                {eventRisk.notes?.length ? (
                  <Stack spacing={0.5}>
                    {eventRisk.notes.map((note, index) => (
                      <Typography key={`${note}-${index}`} variant="caption" color="text.secondary">
                        {note}
                      </Typography>
                    ))}
                  </Stack>
                ) : (
                  <Typography variant="caption" color="text.secondary">
                    No additional event-risk notes.
                  </Typography>
                )}
              </Box>
            </Stack>
          </AccordionDetails>
        </Accordion>

        <Accordion>
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Typography variant="subtitle1" sx={{ fontWeight: 700 }}>
              Historical Validation
            </Typography>
          </AccordionSummary>
          <AccordionDetails>
            <Stack spacing={1.5}>
              <Box>
                <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.5 }}>
                  Lookback
                </Typography>
                <ToggleButtonGroup
                  exclusive
                  size="small"
                  value={validationLookbackDays}
                  onChange={(_, value) => value && setValidationLookbackDays(value)}
                >
                  <ToggleButton value={90}>90D</ToggleButton>
                  <ToggleButton value={180}>180D</ToggleButton>
                  <ToggleButton value={365}>365D</ToggleButton>
                </ToggleButtonGroup>
              </Box>

              {isValidationLoading && (
                <Box display="flex" justifyContent="center" py={2}>
                  <CircularProgress size={24} />
                </Box>
              )}

              {validationError && (
                <Alert severity="error">
                  Failed to load historical validation: {validationError.message}
                </Alert>
              )}

              {validationData && (
                <Stack spacing={2}>
                  <ValidationDegradedAlert degradedReasons={validationData.degraded_reasons} />
                  {(validationData.source_breakdown || []).map((section) => (
                    <Box key={section.source_kind} sx={{ border: 1, borderColor: 'divider', borderRadius: 1, p: 1.5 }}>
                      <Typography variant="subtitle2" sx={{ fontWeight: 700, mb: 1 }}>
                        {section.source_kind === 'scan_pick' ? 'Scan Picks' : 'Theme Alerts'}
                      </Typography>
                      <Stack spacing={1.5}>
                        <ValidationDegradedAlert degradedReasons={section.degraded_reasons} />
                        <ValidationSummaryCards horizons={section.horizons} />
                      </Stack>
                    </Box>
                  ))}

                  <Box>
                    <Typography variant="subtitle2" sx={{ fontWeight: 700, mb: 1 }}>
                      Recent Events
                    </Typography>
                    <ValidationRecentEventsTable events={validationData.recent_events} showSourceKind />
                  </Box>

                  <Box>
                    <Typography variant="subtitle2" sx={{ fontWeight: 700, mb: 1 }}>
                      Failure Clusters
                    </Typography>
                    <ValidationFailureClustersTable clusters={validationData.failure_clusters} />
                  </Box>
                </Stack>
              )}
            </Stack>
          </AccordionDetails>
        </Accordion>
      </Box>

      <PeerComparisonModal
        open={peerModalOpen}
        onClose={() => setPeerModalOpen(false)}
        symbol={data.symbol}
        onOpenChart={(sym) => {
          setPeerModalOpen(false);
          navigate(`/stocks/${encodeURIComponent(sym)}`);
        }}
      />
    </Box>
  );
}

export default StockDetails;
