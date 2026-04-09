/**
 * Technical Indicators Display Component
 *
 * Shows detailed technical analysis metrics including RS rating, stage, MA analysis, and VCP
 */
import {
  Card,
  CardContent,
  Typography,
  Box,
  Grid,
  Chip,
  LinearProgress,
  Divider,
} from '@mui/material';
import {
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  TrendingFlat as TrendingFlatIcon,
  CheckCircle as CheckIcon,
  Cancel as CancelIcon,
} from '@mui/icons-material';

/**
 * RS Rating Display
 */
export const RSRatingCard = ({ rsRating, relativePerformance }) => {
  const getRatingColor = (rating) => {
    if (rating >= 80) return 'success';
    if (rating >= 70) return 'info';
    if (rating >= 60) return 'warning';
    return 'error';
  };

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Relative Strength Rating
        </Typography>

        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center', my: 3 }}>
          <Box sx={{ position: 'relative', display: 'inline-flex' }}>
            <Box
              sx={{
                width: 120,
                height: 120,
                borderRadius: '50%',
                border: `8px solid`,
                borderColor: `${getRatingColor(rsRating)}.main`,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                bgcolor: 'background.paper',
              }}
            >
              <Typography variant="h3" fontWeight="bold" color={`${getRatingColor(rsRating)}.main`}>
                {rsRating}
              </Typography>
            </Box>
          </Box>
        </Box>

        <Box sx={{ mb: 2 }}>
          <Typography variant="body2" color="text.secondary" gutterBottom>
            Performance vs SPY
          </Typography>
          <LinearProgress
            variant="determinate"
            value={Math.min(Math.max((relativePerformance + 50) * 2, 0), 100)}
            color={relativePerformance > 0 ? 'success' : 'error'}
            sx={{ height: 8, borderRadius: 1 }}
          />
          <Typography variant="body2" align="right" sx={{ mt: 0.5 }}>
            {relativePerformance > 0 ? '+' : ''}{relativePerformance.toFixed(1)}%
          </Typography>
        </Box>

        <Divider sx={{ my: 2 }} />

        <Box>
          <Typography variant="caption" color="text.secondary">
            Rating Scale
          </Typography>
          <Grid container spacing={1} sx={{ mt: 0.5 }}>
            <Grid item xs={3}>
              <Chip label="80+" color="success" size="small" sx={{ width: '100%' }} />
            </Grid>
            <Grid item xs={3}>
              <Chip label="70-79" color="info" size="small" sx={{ width: '100%' }} />
            </Grid>
            <Grid item xs={3}>
              <Chip label="60-69" color="warning" size="small" sx={{ width: '100%' }} />
            </Grid>
            <Grid item xs={3}>
              <Chip label="<60" color="error" size="small" sx={{ width: '100%' }} />
            </Grid>
          </Grid>
        </Box>
      </CardContent>
    </Card>
  );
};

/**
 * Stage Analysis Display
 */
export const StageAnalysisCard = ({ stage, stageName, description, confidence }) => {
  const getStageInfo = (stageNum) => {
    switch (stageNum) {
      case 1:
        return { color: 'info', icon: <TrendingFlatIcon />, label: 'Basing' };
      case 2:
        return { color: 'success', icon: <TrendingUpIcon />, label: 'Advancing' };
      case 3:
        return { color: 'warning', icon: <TrendingFlatIcon />, label: 'Topping' };
      case 4:
        return { color: 'error', icon: <TrendingDownIcon />, label: 'Declining' };
      default:
        return { color: 'default', icon: <TrendingFlatIcon />, label: 'Unknown' };
    }
  };

  const stageInfo = getStageInfo(stage);

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Weinstein Stage Analysis
        </Typography>

        <Box sx={{ textAlign: 'center', my: 3 }}>
          <Chip
            icon={stageInfo.icon}
            label={`Stage ${stage}: ${stageName}`}
            color={stageInfo.color}
            sx={{
              fontSize: '1.1rem',
              py: 3,
              px: 2,
              '& .MuiChip-label': { px: 2 },
            }}
          />
        </Box>

        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
          {description}
        </Typography>

        <Divider sx={{ my: 2 }} />

        <Box>
          <Typography variant="body2" color="text.secondary" gutterBottom>
            Confidence Score
          </Typography>
          <LinearProgress
            variant="determinate"
            value={confidence}
            color={confidence >= 70 ? 'success' : confidence >= 50 ? 'warning' : 'error'}
            sx={{ height: 8, borderRadius: 1 }}
          />
          <Typography variant="body2" align="right" sx={{ mt: 0.5 }}>
            {confidence}%
          </Typography>
        </Box>

        <Box sx={{ mt: 2 }}>
          <Typography variant="caption" color="text.secondary">
            Stage Definitions
          </Typography>
          <Box sx={{ mt: 1 }}>
            <Typography variant="caption" display="block">
              <strong>Stage 1:</strong> Basing (sideways movement)
            </Typography>
            <Typography variant="caption" display="block">
              <strong>Stage 2:</strong> Advancing (uptrend) ✓ IDEAL
            </Typography>
            <Typography variant="caption" display="block">
              <strong>Stage 3:</strong> Topping (distribution)
            </Typography>
            <Typography variant="caption" display="block">
              <strong>Stage 4:</strong> Declining (downtrend)
            </Typography>
          </Box>
        </Box>
      </CardContent>
    </Card>
  );
};

/**
 * MA Alignment Display
 */
export const MAAlignmentCard = ({ maAnalysis }) => {
  const { alignment, ma_200_trend, separation } = maAnalysis;

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Moving Average Alignment
        </Typography>

        <Box sx={{ mb: 3 }}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
            <Typography variant="body2">
              Alignment Status
            </Typography>
            {alignment.perfect_alignment ? (
              <Chip icon={<CheckIcon />} label="Perfect" color="success" size="small" />
            ) : (
              <Chip icon={<CancelIcon />} label={alignment.status} color="warning" size="small" />
            )}
          </Box>
          <LinearProgress
            variant="determinate"
            value={alignment.alignment_score}
            color={alignment.perfect_alignment ? 'success' : 'warning'}
            sx={{ height: 8, borderRadius: 1 }}
          />
          <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5, display: 'block' }}>
            {alignment.conditions_met} of {alignment.total_conditions} conditions met
          </Typography>
        </Box>

        <Divider sx={{ my: 2 }} />

        <Grid container spacing={2}>
          <Grid item xs={12}>
            <Typography variant="body2" fontWeight="bold" gutterBottom>
              Alignment Criteria
            </Typography>
          </Grid>

          <Grid item xs={12}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <Typography variant="body2">Price &gt; 50-day MA</Typography>
              {alignment.details.price_above_50 ? (
                <CheckIcon color="success" fontSize="small" />
              ) : (
                <CancelIcon color="error" fontSize="small" />
              )}
            </Box>
          </Grid>

          <Grid item xs={12}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <Typography variant="body2">50-day &gt; 150-day MA</Typography>
              {alignment.details.ma_50_above_150 ? (
                <CheckIcon color="success" fontSize="small" />
              ) : (
                <CancelIcon color="error" fontSize="small" />
              )}
            </Box>
          </Grid>

          <Grid item xs={12}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <Typography variant="body2">150-day &gt; 200-day MA</Typography>
              {alignment.details.ma_150_above_200 ? (
                <CheckIcon color="success" fontSize="small" />
              ) : (
                <CancelIcon color="error" fontSize="small" />
              )}
            </Box>
          </Grid>

          <Grid item xs={12}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <Typography variant="body2">200-day MA trending up</Typography>
              {ma_200_trend.trending_up ? (
                <CheckIcon color="success" fontSize="small" />
              ) : (
                <CancelIcon color="error" fontSize="small" />
              )}
            </Box>
          </Grid>
        </Grid>

        <Divider sx={{ my: 2 }} />

        <Box>
          <Typography variant="body2" color="text.secondary" gutterBottom>
            Price Separation
          </Typography>
          <Grid container spacing={1}>
            <Grid item xs={6}>
              <Typography variant="caption" display="block">
                From 50-day MA
              </Typography>
              <Typography variant="body2" fontWeight="bold">
                {separation.separation_from_50 > 0 ? '+' : ''}
                {separation.separation_from_50}%
              </Typography>
            </Grid>
            <Grid item xs={6}>
              <Typography variant="caption" display="block">
                From 200-day MA
              </Typography>
              <Typography variant="body2" fontWeight="bold">
                {separation.separation_from_200 > 0 ? '+' : ''}
                {separation.separation_from_200}%
              </Typography>
            </Grid>
          </Grid>
          {separation.overextended && (
            <Typography variant="caption" color="warning.main" sx={{ mt: 1, display: 'block' }}>
              ⚠️ Price may be overextended
            </Typography>
          )}
        </Box>
      </CardContent>
    </Card>
  );
};

/**
 * VCP Pattern Display
 */
export const VCPPatternCard = ({ vcpData }) => {
  const {
    vcp_detected,
    vcp_score,
    num_bases,
    contracting_depth,
    contracting_volume,
    tight_near_highs,
    bases_depth,
    distance_from_high_pct,
  } = vcpData;

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          VCP Pattern Detection
        </Typography>

        <Box sx={{ textAlign: 'center', my: 3 }}>
          {vcp_detected ? (
            <Chip
              icon={<CheckIcon />}
              label="VCP DETECTED"
              color="success"
              sx={{
                fontSize: '1rem',
                py: 2.5,
                px: 2,
              }}
            />
          ) : (
            <Chip
              label="NO VCP"
              color="default"
              sx={{
                fontSize: '1rem',
                py: 2.5,
                px: 2,
              }}
            />
          )}
        </Box>

        <Box sx={{ mb: 2 }}>
          <Typography variant="body2" color="text.secondary" gutterBottom>
            VCP Score
          </Typography>
          <LinearProgress
            variant="determinate"
            value={vcp_score}
            color={vcp_score >= 70 ? 'success' : vcp_score >= 50 ? 'warning' : 'error'}
            sx={{ height: 10, borderRadius: 1 }}
          />
          <Typography variant="body2" align="right" sx={{ mt: 0.5 }}>
            {vcp_score}
          </Typography>
        </Box>

        <Divider sx={{ my: 2 }} />

        <Grid container spacing={2}>
          <Grid item xs={12}>
            <Typography variant="body2" fontWeight="bold" gutterBottom>
              Pattern Characteristics
            </Typography>
          </Grid>

          <Grid item xs={6}>
            <Box>
              <Typography variant="caption" color="text.secondary">
                Bases Found
              </Typography>
              <Typography variant="body1" fontWeight="bold">
                {num_bases}
              </Typography>
            </Box>
          </Grid>

          <Grid item xs={6}>
            <Box>
              <Typography variant="caption" color="text.secondary">
                From High
              </Typography>
              <Typography variant="body1" fontWeight="bold">
                {distance_from_high_pct}%
              </Typography>
            </Box>
          </Grid>

          <Grid item xs={12}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <Typography variant="body2">Contracting Depth</Typography>
              {contracting_depth ? (
                <CheckIcon color="success" fontSize="small" />
              ) : (
                <CancelIcon color="error" fontSize="small" />
              )}
            </Box>
          </Grid>

          <Grid item xs={12}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <Typography variant="body2">Contracting Volume</Typography>
              {contracting_volume ? (
                <CheckIcon color="success" fontSize="small" />
              ) : (
                <CancelIcon color="error" fontSize="small" />
              )}
            </Box>
          </Grid>

          <Grid item xs={12}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <Typography variant="body2">Tight Near Highs</Typography>
              {tight_near_highs ? (
                <CheckIcon color="success" fontSize="small" />
              ) : (
                <CancelIcon color="error" fontSize="small" />
              )}
            </Box>
          </Grid>
        </Grid>

        {bases_depth && bases_depth.length > 0 && (
          <>
            <Divider sx={{ my: 2 }} />
            <Box>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                Base Depths (should be contracting)
              </Typography>
              <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                {bases_depth.map((depth, index) => (
                  <Chip
                    key={index}
                    label={`${depth}%`}
                    size="small"
                    variant="outlined"
                  />
                ))}
              </Box>
            </Box>
          </>
        )}
      </CardContent>
    </Card>
  );
};

export default {
  RSRatingCard,
  StageAnalysisCard,
  MAAlignmentCard,
  VCPPatternCard,
};
