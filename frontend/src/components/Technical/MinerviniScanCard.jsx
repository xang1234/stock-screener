/**
 * Minervini Scan Results Card
 *
 * Displays comprehensive Minervini template analysis results
 */
import {
  Card,
  CardContent,
  Typography,
  Box,
  Chip,
  LinearProgress,
  Grid,
  Divider,
  CircularProgress,
  Alert,
} from '@mui/material';
import {
  CheckCircle as CheckIcon,
  Cancel as CancelIcon,
  TrendingUp as TrendingUpIcon,
} from '@mui/icons-material';
import { useQuery } from '@tanstack/react-query';
import { getMinerviniScan } from '../../api/stocks';

/**
 * Score bar component
 */
const ScoreBar = ({ label, points, maxPoints, passes }) => {
  const percentage = (points / maxPoints) * 100;
  const color = passes ? 'success' : percentage > 50 ? 'warning' : 'error';

  return (
    <Box sx={{ mb: 2 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
        <Typography variant="body2">{label}</Typography>
        <Typography variant="body2" fontWeight="bold">
          {points.toFixed(1)} / {maxPoints}
        </Typography>
      </Box>
      <LinearProgress
        variant="determinate"
        value={percentage}
        color={color}
        sx={{ height: 8, borderRadius: 1 }}
      />
    </Box>
  );
};

/**
 * Stage indicator component
 */
const StageIndicator = ({ stage, stageName }) => {
  const getStageColor = (stageNum) => {
    switch (stageNum) {
      case 2:
        return 'success';
      case 1:
        return 'info';
      case 3:
        return 'warning';
      case 4:
        return 'error';
      default:
        return 'default';
    }
  };

  return (
    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
      <Chip
        label={`Stage ${stage}`}
        color={getStageColor(stage)}
        size="small"
      />
      <Typography variant="body2" color="text.secondary">
        {stageName}
      </Typography>
    </Box>
  );
};

/**
 * MinerviniScanCard Component
 */
const MinerviniScanCard = ({ symbol, includeVCP = true }) => {
  const { data, isLoading, error } = useQuery({
    queryKey: ['minerviniScan', symbol, includeVCP],
    queryFn: () => getMinerviniScan(symbol, includeVCP),
    enabled: !!symbol,
  });

  if (!symbol) {
    return (
      <Card>
        <CardContent>
          <Typography color="text.secondary">
            Enter a stock symbol to view Minervini analysis
          </Typography>
        </CardContent>
      </Card>
    );
  }

  if (isLoading) {
    return (
      <Card>
        <CardContent sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
          <CircularProgress />
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card>
        <CardContent>
          <Alert severity="error">
            Error loading Minervini scan: {error.message}
          </Alert>
        </CardContent>
      </Card>
    );
  }

  const {
    passes_template,
    minervini_score,
    score_breakdown,
    rs_rating,
    stage,
    stage_name,
    ma_alignment,
    vcp_detected,
    current_price,
    above_52w_low_pct,
    from_52w_high_pct,
  } = data;

  return (
    <Card>
      <CardContent>
        {/* Header */}
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
          <Typography variant="h6">
            Minervini Template Analysis
          </Typography>
          {passes_template ? (
            <Chip
              icon={<CheckIcon />}
              label="PASSES TEMPLATE"
              color="success"
              sx={{ fontWeight: 'bold' }}
            />
          ) : (
            <Chip
              icon={<CancelIcon />}
              label="DOES NOT PASS"
              color="error"
              sx={{ fontWeight: 'bold' }}
            />
          )}
        </Box>

        {/* Overall Score */}
        <Box sx={{ mb: 3 }}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
            <Typography variant="body1" fontWeight="bold">
              Overall Score
            </Typography>
            <Typography variant="h5" fontWeight="bold" color={minervini_score >= 70 ? 'success.main' : 'error.main'}>
              {minervini_score.toFixed(1)}
            </Typography>
          </Box>
          <LinearProgress
            variant="determinate"
            value={minervini_score}
            color={minervini_score >= 70 ? 'success' : 'error'}
            sx={{ height: 12, borderRadius: 1 }}
          />
          <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5, display: 'block' }}>
            Passing score: 70+
          </Typography>
        </Box>

        <Divider sx={{ my: 2 }} />

        {/* Score Breakdown */}
        <Typography variant="subtitle2" gutterBottom fontWeight="bold">
          Score Breakdown
        </Typography>

        {/* RS Rating */}
        {score_breakdown.rs_rating && (
          <ScoreBar
            label={`RS Rating (${score_breakdown.rs_rating.value})`}
            points={score_breakdown.rs_rating.points}
            maxPoints={score_breakdown.rs_rating.max_points}
            passes={score_breakdown.rs_rating.passes}
          />
        )}

        {/* Stage */}
        {score_breakdown.stage && (
          <ScoreBar
            label={`Stage ${score_breakdown.stage.value} (${stage_name})`}
            points={score_breakdown.stage.points}
            maxPoints={score_breakdown.stage.max_points}
            passes={score_breakdown.stage.passes}
          />
        )}

        {/* MA Alignment */}
        {score_breakdown.ma_alignment && (
          <ScoreBar
            label={`MA Alignment (${score_breakdown.ma_alignment.value.toFixed(0)}%)`}
            points={score_breakdown.ma_alignment.points}
            maxPoints={score_breakdown.ma_alignment.max_points}
            passes={score_breakdown.ma_alignment.passes}
          />
        )}

        {/* 52-Week Position */}
        {score_breakdown.position_52w && (
          <ScoreBar
            label={`52-Week Position (${above_52w_low_pct}% above low, ${from_52w_high_pct}% from high)`}
            points={score_breakdown.position_52w.points}
            maxPoints={score_breakdown.position_52w.max_points}
            passes={score_breakdown.position_52w.passes}
          />
        )}

        {/* VCP */}
        {score_breakdown.vcp && (
          <ScoreBar
            label={`VCP Pattern (${score_breakdown.vcp.value.toFixed(0)}%)`}
            points={score_breakdown.vcp.points}
            maxPoints={score_breakdown.vcp.max_points}
            passes={score_breakdown.vcp.passes}
          />
        )}

        <Divider sx={{ my: 2 }} />

        {/* Key Metrics */}
        <Typography variant="subtitle2" gutterBottom fontWeight="bold">
          Key Metrics
        </Typography>

        <Grid container spacing={2}>
          <Grid item xs={6}>
            <Box>
              <Typography variant="caption" color="text.secondary">
                Current Price
              </Typography>
              <Typography variant="body1" fontWeight="bold">
                ${current_price?.toFixed(2)}
              </Typography>
            </Box>
          </Grid>

          <Grid item xs={6}>
            <Box>
              <Typography variant="caption" color="text.secondary">
                RS Rating
              </Typography>
              <Typography variant="body1" fontWeight="bold">
                {rs_rating}
                {rs_rating >= 80 && (
                  <TrendingUpIcon
                    sx={{ ml: 0.5, fontSize: 16, color: 'success.main', verticalAlign: 'middle' }}
                  />
                )}
              </Typography>
            </Box>
          </Grid>

          <Grid item xs={6}>
            <Box>
              <Typography variant="caption" color="text.secondary">
                Stage
              </Typography>
              <Box sx={{ mt: 0.5 }}>
                <StageIndicator stage={stage} stageName={stage_name} />
              </Box>
            </Box>
          </Grid>

          <Grid item xs={6}>
            <Box>
              <Typography variant="caption" color="text.secondary">
                MA Alignment
              </Typography>
              <Typography variant="body1" fontWeight="bold">
                {ma_alignment ? (
                  <Chip icon={<CheckIcon />} label="Aligned" color="success" size="small" />
                ) : (
                  <Chip icon={<CancelIcon />} label="Not Aligned" color="error" size="small" />
                )}
              </Typography>
            </Box>
          </Grid>

          {vcp_detected !== null && (
            <Grid item xs={12}>
              <Box>
                <Typography variant="caption" color="text.secondary">
                  VCP Pattern
                </Typography>
                <Typography variant="body1" fontWeight="bold">
                  {vcp_detected ? (
                    <Chip icon={<CheckIcon />} label="VCP Detected" color="success" size="small" />
                  ) : (
                    <Chip label="No VCP" color="default" size="small" />
                  )}
                </Typography>
              </Box>
            </Grid>
          )}
        </Grid>
      </CardContent>
    </Card>
  );
};

export default MinerviniScanCard;
