import { Alert, Box, Chip, CircularProgress, Drawer, IconButton, Typography } from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import CancelIcon from '@mui/icons-material/Cancel';
import WarningAmberIcon from '@mui/icons-material/WarningAmber';
import { formatPatternName, getScoreColor } from '../../utils/formatUtils';

// Human-readable names for common check/flag strings
const CHECK_NAME_MAP = {
  setup_score_ok: 'Setup score meets threshold',
  quality_floor_ok: 'Quality score above floor',
  readiness_floor_ok: 'Readiness score above floor',
  in_early_zone: 'Within early-stage zone',
  atr14_within_limit: 'ATR14 within volatility limit',
  volume_sufficient: 'Volume above minimum',
  rs_leadership_ok: 'RS leadership confirmed',
  stage_ok: 'Stage is acceptable',
  setup_score_below_threshold: 'Setup score below threshold',
  quality_below_threshold: 'Quality score below threshold',
  readiness_below_threshold: 'Readiness score below threshold',
  outside_early_zone: 'Outside early-stage zone',
  atr14_pct_exceeds_limit: 'ATR14 exceeds volatility limit',
  volume_below_minimum: 'Volume below minimum',
  rs_leadership_insufficient: 'RS leadership insufficient',
  stage_not_ok: 'Stage not acceptable',
  ma_alignment_ok: 'MA alignment meets minimum',
  ma_alignment_insufficient: 'MA alignment below minimum',
  rs_rating_ok: 'RS rating meets minimum',
  rs_rating_insufficient: 'RS rating below minimum',
  no_primary_pattern: 'No primary pattern detected',
  insufficient_data: 'Insufficient data',
  primary_pattern_selected: 'Primary pattern selected',
  primary_pattern_fallback_selected: 'Primary selected via fallback',
  detector_pipeline_executed: 'Detector pipeline executed',
  cross_detector_calibration_applied: 'Cross-detector calibration applied',
  too_extended: 'Entry is too extended past pivot',
  breaks_50d_support: 'Price below 50-day moving average',
  low_liquidity: 'Average daily volume below liquidity threshold',
  earnings_soon: 'Earnings announcement within risk window',
};

// Operational flags with hard severity (structural break — renders red).
// All other operational flags are soft (caution — renders amber).
const HARD_FLAGS = new Set(['breaks_50d_support']);

const getFlagCode = (flag) => {
  if (!flag) return '';
  if (typeof flag === 'string') return String(flag).split(':')[0];
  if (typeof flag === 'object' && !Array.isArray(flag) && flag.code) return String(flag.code);
  return '';
};

const getFlagMessage = (flag) => {
  if (!flag) return '';
  if (typeof flag === 'string') return String(flag);
  if (typeof flag === 'object' && !Array.isArray(flag)) {
    if (flag.message) return String(flag.message);
    if (flag.code) return String(flag.code);
  }
  return '';
};

/**
 * Format a check/flag string to human-readable text
 */
const formatCheckName = (name) => {
  if (!name) return '-';
  if (CHECK_NAME_MAP[name]) return CHECK_NAME_MAP[name];
  // Fallback: snake_case to Title Case
  return String(name)
    .replace(/:/g, ' \u2013 ')
    .split('_')
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(' ');
};

/**
 * Section header
 */
const DrawerSection = ({ title, children }) => (
  <Box sx={{ mb: 2 }}>
    <Typography
      variant="caption"
      color="text.secondary"
      sx={{ fontWeight: 'bold', letterSpacing: 0.5, fontSize: '0.7rem', mb: 0.75, display: 'block' }}
    >
      {title}
    </Typography>
    {children}
  </Box>
);

/**
 * Score grid item
 */
const ScoreItem = ({ label, value }) => (
  <Box sx={{ textAlign: 'center' }}>
    <Typography variant="h6" fontWeight="bold" sx={{ color: getScoreColor(value), lineHeight: 1.2 }}>
      {value != null ? value.toFixed(1) : '-'}
    </Typography>
    <Typography variant="caption" color="text.secondary" sx={{ fontSize: '0.65rem' }}>
      {label}
    </Typography>
  </Box>
);

/**
 * Check list item with icon
 */
const CheckItem = ({ icon, text, color }) => (
  <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.75, py: 0.25 }}>
    {icon}
    <Typography variant="body2" sx={{ fontSize: '0.78rem', color: color || 'text.primary' }}>
      {text}
    </Typography>
  </Box>
);

/**
 * SetupEngineDrawer — slide-out panel showing the full explain payload
 * for a stock's Setup Engine evaluation.
 *
 * @param {Object} props
 * @param {boolean} props.open - Whether drawer is open
 * @param {Function} props.onClose - Close handler
 * @param {Object} props.stockData - Stock result from scan (needs se_* fields)
 */
function SetupEngineDrawer({ open, onClose, stockData, isLoading = false }) {
  if (!stockData) return null;

  const explain =
    stockData.se_explain && typeof stockData.se_explain === 'object' && !Array.isArray(stockData.se_explain)
      ? stockData.se_explain
      : null;
  const candidates = Array.isArray(stockData.se_candidates) ? stockData.se_candidates : null;

  const hasInsufficientData = explain?.invalidation_flags?.some(
    (f) => {
      const code = getFlagCode(f);
      const text = getFlagMessage(f);
      return code === 'insufficient_data' || text.startsWith('data_policy:insufficient') || (code === 'data_policy' && text.includes('insufficient'));
    }
  );
  const hasDegradedData = !hasInsufficientData && explain?.invalidation_flags?.some(
    (f) => {
      const text = getFlagMessage(f);
      const code = getFlagCode(f);
      return text.startsWith('data_policy:degraded') || (code === 'data_policy' && text.includes('degraded'));
    }
  );

  return (
    <Drawer
      anchor="right"
      open={open}
      onClose={onClose}
      sx={{ zIndex: 1400 }}
      slotProps={{
        backdrop: { sx: { backgroundColor: 'transparent' } },
      }}
      PaperProps={{
        sx: { width: 420, bgcolor: 'background.paper' },
      }}
    >
      {/* Header */}
      <Box
        sx={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          px: 2,
          py: 1.5,
          borderBottom: 1,
          borderColor: 'divider',
        }}
      >
        <Typography variant="subtitle1" fontWeight="bold">
          Setup Engine Details{stockData.symbol ? ` — ${stockData.symbol}` : ''}
        </Typography>
        <IconButton onClick={onClose} size="small">
          <CloseIcon fontSize="small" />
        </IconButton>
      </Box>

      {/* Content */}
      <Box sx={{ p: 2, overflow: 'auto', flex: 1 }}>
        {isLoading ? (
          <Box sx={{ mt: 2, display: 'flex', justifyContent: 'center' }}>
            <CircularProgress size={24} />
          </Box>
        ) : !explain ? (
          <Typography variant="body2" color="text.secondary" sx={{ mt: 2, textAlign: 'center' }}>
            Setup details not available for this stock.
          </Typography>
        ) : (
          <>
            {/* Data quality banners */}
            {hasInsufficientData && (
              <Alert severity="warning" sx={{ mb: 2 }}>
                Insufficient historical data — scores and pattern detection are unavailable for this stock.
              </Alert>
            )}
            {hasDegradedData && (
              <Alert severity="info" sx={{ mb: 2 }}>
                Some data sources are degraded. Scores may be less reliable.
              </Alert>
            )}

            {/* Section 1: Setup Summary */}
            <DrawerSection title="SETUP SUMMARY">
              {/* Pattern + Confidence */}
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1.5 }}>
                <Typography variant="body1" fontWeight="bold">
                  {formatPatternName(stockData.se_pattern_primary)}
                </Typography>
                {stockData.se_pattern_confidence != null && (
                  <Chip
                    label={`${stockData.se_pattern_confidence.toFixed(0)}%`}
                    size="small"
                    sx={{
                      fontSize: '0.7rem',
                      height: 22,
                      bgcolor: getScoreColor(stockData.se_pattern_confidence),
                      color: 'white',
                    }}
                  />
                )}
              </Box>

              {/* Scores Grid */}
              {(stockData.se_setup_score != null ||
                stockData.se_quality_score != null ||
                stockData.se_readiness_score != null) && (
                <Box sx={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 1, mb: 1.5 }}>
                  <ScoreItem label="Setup" value={stockData.se_setup_score} />
                  <ScoreItem label="Quality" value={stockData.se_quality_score} />
                  <ScoreItem label="Readiness" value={stockData.se_readiness_score} />
                </Box>
              )}

              {/* Ready chip */}
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                {stockData.se_setup_ready != null && (
                  <Chip
                    label={stockData.se_setup_ready ? 'READY' : 'NOT READY'}
                    size="small"
                    sx={{
                      fontSize: '0.7rem',
                      height: 22,
                      bgcolor: stockData.se_setup_ready ? '#4caf50' : '#9e9e9e',
                      color: 'white',
                      fontWeight: 'bold',
                    }}
                  />
                )}
                {stockData.se_timeframe && (
                  <Chip
                    label={stockData.se_timeframe}
                    size="small"
                    variant="outlined"
                    sx={{ fontSize: '0.7rem', height: 22 }}
                  />
                )}
              </Box>

              {/* Pivot Info */}
              <Box sx={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 0.5, mt: 1 }}>
                {stockData.se_pivot_price != null && (
                  <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                    <Typography variant="caption" color="text.secondary">Pivot</Typography>
                    <Typography variant="body2" fontWeight="medium" sx={{ fontSize: '0.8rem' }}>
                      ${stockData.se_pivot_price.toFixed(2)}
                    </Typography>
                  </Box>
                )}
                {stockData.se_pivot_type && (
                  <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                    <Typography variant="caption" color="text.secondary">Type</Typography>
                    <Typography variant="body2" sx={{ fontSize: '0.8rem' }}>
                      {formatCheckName(stockData.se_pivot_type)}
                    </Typography>
                  </Box>
                )}
                {stockData.se_pivot_date && (
                  <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                    <Typography variant="caption" color="text.secondary">Pivot Date</Typography>
                    <Typography variant="body2" sx={{ fontSize: '0.8rem' }}>
                      {stockData.se_pivot_date}
                    </Typography>
                  </Box>
                )}
                {stockData.se_distance_to_pivot_pct != null && (
                  <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                    <Typography variant="caption" color="text.secondary">Dist to Pivot</Typography>
                    <Typography
                      variant="body2"
                      fontWeight="medium"
                      sx={{
                        fontSize: '0.8rem',
                        color: stockData.se_distance_to_pivot_pct <= 0 ? '#4caf50' : '#ff9800',
                      }}
                    >
                      {stockData.se_distance_to_pivot_pct >= 0 ? '+' : ''}
                      {stockData.se_distance_to_pivot_pct.toFixed(1)}%
                    </Typography>
                  </Box>
                )}
                {stockData.se_atr14_pct != null && (
                  <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                    <Typography variant="caption" color="text.secondary">ATR14</Typography>
                    <Typography variant="body2" sx={{ fontSize: '0.8rem' }}>
                      {stockData.se_atr14_pct.toFixed(2)}%
                    </Typography>
                  </Box>
                )}
              </Box>
            </DrawerSection>

            {/* Section 2: Passed Checks */}
            {explain.passed_checks?.length > 0 && (
              <DrawerSection title="PASSED CHECKS">
                {explain.passed_checks.map((check) => (
                  <CheckItem
                    key={check}
                    icon={<CheckCircleIcon sx={{ fontSize: 16, color: '#4caf50' }} />}
                    text={formatCheckName(check)}
                  />
                ))}
              </DrawerSection>
            )}

            {/* Section 3: Failed Checks */}
            {explain.failed_checks?.length > 0 && (
              <DrawerSection title="FAILED CHECKS">
                {explain.failed_checks.map((check) => (
                  <CheckItem
                    key={check}
                    icon={<CancelIcon sx={{ fontSize: 16, color: '#f44336' }} />}
                    text={formatCheckName(check)}
                    color="#f44336"
                  />
                ))}
              </DrawerSection>
            )}

            {/* Section 4: Key Levels */}
            {explain.key_levels && typeof explain.key_levels === 'object' && !Array.isArray(explain.key_levels) && Object.keys(explain.key_levels).length > 0 && (
              <DrawerSection title="KEY LEVELS">
                {Object.entries(explain.key_levels).map(([name, price]) => (
                  <Box key={name} sx={{ display: 'flex', justifyContent: 'space-between', py: 0.25 }}>
                    <Typography variant="body2" sx={{ fontSize: '0.78rem' }}>
                      {formatCheckName(name)}
                    </Typography>
                    <Typography variant="body2" fontWeight="medium" sx={{ fontSize: '0.78rem' }}>
                      {price != null && Number.isFinite(Number(price)) ? `$${Number(price).toFixed(2)}` : '-'}
                    </Typography>
                  </Box>
                ))}
              </DrawerSection>
            )}

            {/* Section 5: Invalidation Flags */}
            {explain.invalidation_flags?.length > 0 && (
              <DrawerSection title="INVALIDATION FLAGS">
                {explain.invalidation_flags.map((flag, idx) => {
                  const flagBase = getFlagCode(flag);
                  const flagText = getFlagMessage(flag);
                  const isHard = HARD_FLAGS.has(flagBase);
                  const flagColor = isHard ? '#f44336' : '#ff9800';
                  return (
                    <CheckItem
                      key={`${flagBase || 'flag'}-${idx}`}
                      icon={<WarningAmberIcon sx={{ fontSize: 16, color: flagColor }} />}
                      text={formatCheckName(flagText)}
                      color={flagColor}
                    />
                  );
                })}
              </DrawerSection>
            )}

            {/* Section 6: Pattern Candidates (optional) */}
            {candidates?.length > 0 && (
              <DrawerSection title="PATTERN CANDIDATES">
                {candidates.map((c, idx) => (
                  <Box
                    key={`${c.pattern || 'candidate'}-${idx}`}
                    sx={{
                      border: 1,
                      borderColor: 'divider',
                      borderRadius: 1,
                      p: 1.5,
                      mb: 1,
                    }}
                  >
                    {/* Candidate header */}
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.75 }}>
                      <Typography variant="body2" fontWeight="bold">
                        {formatPatternName(c.pattern)}
                      </Typography>
                      {c.confidence_pct != null && (
                        <Chip
                          label={`${Number(c.confidence_pct).toFixed(0)}%`}
                          size="small"
                          sx={{
                            fontSize: '0.65rem',
                            height: 18,
                            bgcolor: getScoreColor(Number(c.confidence_pct)),
                            color: 'white',
                          }}
                        />
                      )}
                      {c.timeframe && (
                        <Chip
                          label={c.timeframe}
                          size="small"
                          variant="outlined"
                          sx={{ fontSize: '0.65rem', height: 18 }}
                        />
                      )}
                    </Box>

                    {/* Candidate scores */}
                    <Box sx={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 0.5, mb: 0.75 }}>
                      {c.setup_score != null && (
                        <Box sx={{ textAlign: 'center' }}>
                          <Typography variant="body2" fontWeight="medium" sx={{ color: getScoreColor(c.setup_score), fontSize: '0.8rem' }}>
                            {c.setup_score.toFixed(1)}
                          </Typography>
                          <Typography variant="caption" color="text.secondary" sx={{ fontSize: '0.6rem' }}>Setup</Typography>
                        </Box>
                      )}
                      {c.quality_score != null && (
                        <Box sx={{ textAlign: 'center' }}>
                          <Typography variant="body2" fontWeight="medium" sx={{ color: getScoreColor(c.quality_score), fontSize: '0.8rem' }}>
                            {c.quality_score.toFixed(1)}
                          </Typography>
                          <Typography variant="caption" color="text.secondary" sx={{ fontSize: '0.6rem' }}>Quality</Typography>
                        </Box>
                      )}
                      {c.readiness_score != null && (
                        <Box sx={{ textAlign: 'center' }}>
                          <Typography variant="body2" fontWeight="medium" sx={{ color: getScoreColor(c.readiness_score), fontSize: '0.8rem' }}>
                            {c.readiness_score.toFixed(1)}
                          </Typography>
                          <Typography variant="caption" color="text.secondary" sx={{ fontSize: '0.6rem' }}>Readiness</Typography>
                        </Box>
                      )}
                    </Box>

                    {/* Candidate pivot */}
                    {c.pivot_price != null && (
                      <Typography variant="caption" color="text.secondary" sx={{ fontSize: '0.7rem' }}>
                        Pivot: ${c.pivot_price.toFixed(2)}
                        {c.pivot_type ? ` (${formatCheckName(c.pivot_type)})` : ''}
                        {c.pivot_date ? ` \u2014 ${c.pivot_date}` : ''}
                      </Typography>
                    )}

                    {/* Candidate checks */}
                    {c.checks && typeof c.checks === 'object' && !Array.isArray(c.checks) && Object.keys(c.checks).length > 0 && (
                      <Box sx={{ mt: 0.5 }}>
                        {Object.entries(c.checks).map(([name, passed]) => (
                          <Box key={name} sx={{ display: 'flex', alignItems: 'center', gap: 0.5, py: 0.1 }}>
                            {passed ? (
                              <CheckCircleIcon sx={{ fontSize: 12, color: '#4caf50' }} />
                            ) : (
                              <CancelIcon sx={{ fontSize: 12, color: '#f44336' }} />
                            )}
                            <Typography variant="caption" sx={{ fontSize: '0.68rem' }}>
                              {formatCheckName(name)}
                            </Typography>
                          </Box>
                        ))}
                      </Box>
                    )}

                    {/* Candidate notes */}
                    {c.notes?.length > 0 && (
                      <Box sx={{ mt: 0.5 }}>
                        {c.notes.map((note, i) => (
                          <Typography key={i} variant="caption" color="text.secondary" sx={{ fontSize: '0.68rem', display: 'block' }}>
                            {note}
                          </Typography>
                        ))}
                      </Box>
                    )}
                  </Box>
                ))}
              </DrawerSection>
            )}
          </>
        )}
      </Box>
    </Drawer>
  );
}

export default SetupEngineDrawer;
