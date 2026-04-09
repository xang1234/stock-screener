import {
  Badge,
  Box,
  Button,
  CircularProgress,
  IconButton,
  ToggleButton,
  ToggleButtonGroup,
  Tooltip,
  Typography,
} from '@mui/material';
import AccountBalanceIcon from '@mui/icons-material/AccountBalance';
import ArticleIcon from '@mui/icons-material/Article';
import CompareArrowsIcon from '@mui/icons-material/CompareArrows';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import RefreshIcon from '@mui/icons-material/Refresh';
import Replay30Icon from '@mui/icons-material/Replay30';
import SettingsIcon from '@mui/icons-material/Settings';
import SmartToyIcon from '@mui/icons-material/SmartToy';
import TrendingFlatIcon from '@mui/icons-material/TrendingFlat';
import WhatshotIcon from '@mui/icons-material/Whatshot';

export default function ThemesPageHeader({
  selectedPipeline,
  onPipelineChange,
  pendingReviewCount,
  onOpenReview,
  onOpenArticles,
  onOpenSettings,
  onOpenModelSettings,
  failedCount,
  isPipelineRunning,
  onRunPipeline,
  onRefresh,
}) {
  return (
    <Box sx={{ mb: 4, display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
      <Box>
        <Box display="flex" alignItems="center" gap={2}>
          <Typography variant="h4">
            <WhatshotIcon sx={{ mr: 1, verticalAlign: 'middle', color: 'error.main' }} />
            Theme Discovery
          </Typography>
          <ToggleButtonGroup
            value={selectedPipeline}
            exclusive
            onChange={onPipelineChange}
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
            : 'Earnings, valuation, macro themes, analyst coverage'}
        </Typography>
      </Box>
      <Box display="flex" gap={1} alignItems="center">
        <Badge badgeContent={pendingReviewCount} color="warning" max={99}>
          <Button
            size="small"
            variant="outlined"
            startIcon={<CompareArrowsIcon />}
            onClick={onOpenReview}
            sx={{ fontSize: '0.65rem', py: 0.15, px: 0.5, '& .MuiSvgIcon-root': { fontSize: '0.85rem' } }}
          >
            Review
          </Button>
        </Badge>
        <Button
          size="small"
          variant="outlined"
          startIcon={<ArticleIcon />}
          onClick={onOpenArticles}
          sx={{ fontSize: '0.65rem', py: 0.15, px: 0.5, '& .MuiSvgIcon-root': { fontSize: '0.85rem' } }}
        >
          Articles
        </Button>
        <Button
          size="small"
          variant="outlined"
          startIcon={<SettingsIcon />}
          onClick={onOpenSettings}
          sx={{ fontSize: '0.65rem', py: 0.15, px: 0.5, '& .MuiSvgIcon-root': { fontSize: '0.85rem' } }}
        >
          Settings
        </Button>
        <Button
          size="small"
          variant="outlined"
          startIcon={<SmartToyIcon />}
          onClick={onOpenModelSettings}
          sx={{ fontSize: '0.65rem', py: 0.15, px: 0.5, '& .MuiSvgIcon-root': { fontSize: '0.85rem' } }}
        >
          LLM
        </Button>
        <Box sx={{ borderLeft: 1, borderColor: 'divider', height: 24, mx: 0.5 }} />
        <Tooltip
          title={
            isPipelineRunning
              ? 'Pipeline running...'
              : failedCount > 0
                ? `Run Pipeline (${failedCount} pending retry)`
                : 'Run Pipeline'
          }
        >
          <Badge badgeContent={failedCount} color="error" max={999} invisible={!failedCount}>
            <span>
              <IconButton size="small" onClick={() => onRunPipeline()} disabled={isPipelineRunning} color="primary">
                {isPipelineRunning ? <CircularProgress size={18} /> : <PlayArrowIcon fontSize="small" />}
              </IconButton>
            </span>
          </Badge>
        </Tooltip>
        <Tooltip title="Backfill 30d">
          <span>
            <IconButton size="small" onClick={() => onRunPipeline(30)} disabled={isPipelineRunning}>
              <Replay30Icon fontSize="small" />
            </IconButton>
          </span>
        </Tooltip>
        <Tooltip title="Refresh">
          <IconButton size="small" onClick={onRefresh}>
            <RefreshIcon fontSize="small" />
          </IconButton>
        </Tooltip>
      </Box>
    </Box>
  );
}
