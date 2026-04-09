import {
  Badge,
  Box,
  Card,
  CardContent,
  Chip,
  CircularProgress,
  Grid,
  IconButton,
  Typography,
} from '@mui/material';
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome';
import CloseIcon from '@mui/icons-material/Close';
import NewReleasesIcon from '@mui/icons-material/NewReleases';
import NotificationsIcon from '@mui/icons-material/Notifications';
import ShowChartIcon from '@mui/icons-material/ShowChart';
import TimelineIcon from '@mui/icons-material/Timeline';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import { LinearProgress } from '@mui/material';

function normalizeVelocity(velocity) {
  const numericVelocity = Number(velocity);
  if (!Number.isFinite(numericVelocity) || numericVelocity <= 0) {
    return null;
  }
  return numericVelocity;
}

function formatVelocityLabel(velocity) {
  const numericVelocity = normalizeVelocity(velocity);
  return numericVelocity ? `${numericVelocity.toFixed(1)}x` : '-';
}

export function MomentumBar({ score }) {
  const numericScore = Number.isFinite(score) ? Math.max(0, Math.min(score, 100)) : 0;
  const color = numericScore >= 70 ? 'success' : numericScore >= 50 ? 'warning' : 'error';
  return (
    <Box sx={{ display: 'flex', alignItems: 'center', minWidth: 80 }}>
      <Box sx={{ width: '100%', mr: 0.5 }}>
        <LinearProgress
          variant="determinate"
          value={numericScore}
          color={color}
          sx={{ height: 6, borderRadius: 3 }}
        />
      </Box>
      <Box sx={{ minWidth: 28, fontSize: '11px', fontWeight: 600, fontFamily: 'monospace' }}>
        {numericScore.toFixed(0)}
      </Box>
    </Box>
  );
}

export function VelocityIndicator({ velocity }) {
  const numericVelocity = normalizeVelocity(velocity);
  if (!numericVelocity) {
    return (
      <Box sx={{ color: 'text.secondary', fontFamily: 'monospace', fontSize: '11px' }}>
        -
      </Box>
    );
  }

  const isAccelerating = numericVelocity > 1;
  const color =
    numericVelocity >= 2
      ? 'success.main'
      : numericVelocity >= 1.5
        ? 'warning.main'
        : numericVelocity >= 1
          ? 'info.main'
          : 'text.secondary';

  return (
    <Box display="flex" alignItems="center" sx={{ fontFamily: 'monospace' }}>
      {isAccelerating && <TrendingUpIcon sx={{ fontSize: 12, mr: 0.25, color }} />}
      <Box component="span" sx={{ color, fontWeight: numericVelocity >= 1.5 ? 600 : 400, fontSize: '11px' }}>
        {numericVelocity.toFixed(1)}x
      </Box>
    </Box>
  );
}

function EmergingThemesCard({ themes, isLoading }) {
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
          <Box sx={{ fontSize: '12px', fontWeight: 600 }}>Emerging Themes</Box>
          {themes?.count > 0 && <Chip label={themes.count} size="small" color="warning" sx={{ ml: 0.5 }} />}
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
                  <Box sx={{ fontSize: '11px', fontWeight: 500 }}>{theme.theme}</Box>
                  <Box display="flex" gap={0.5} mt={0.25}>
                    <Chip
                      label={formatVelocityLabel(theme.velocity)}
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
}

function TopTrendingThemesCard({ topTrendingThemes = [], onSelectTheme }) {
  return (
    <Card variant="outlined" sx={{ height: '100%' }}>
      <CardContent>
        <Box display="flex" alignItems="center" mb={2}>
          <ShowChartIcon sx={{ color: 'success.main', mr: 1 }} />
          <Typography variant="subtitle1" fontWeight="bold">
            Top Trending
          </Typography>
        </Box>
        {topTrendingThemes.map((theme, index) => (
          <Box
            key={theme.id}
            display="flex"
            justifyContent="space-between"
            alignItems="center"
            py={0.75}
            borderBottom={index < 4 ? 1 : 0}
            borderColor="divider"
            sx={{ cursor: 'pointer' }}
            role="button"
            tabIndex={0}
            onClick={() => onSelectTheme?.({ id: theme.id, name: theme.name })}
            onKeyDown={(event) => {
              if (event.key === 'Enter' || event.key === ' ') {
                event.preventDefault();
                onSelectTheme?.({ id: theme.id, name: theme.name });
              }
            }}
          >
            <Box display="flex" alignItems="center">
              <Typography variant="body2" color="text.secondary" sx={{ mr: 1, minWidth: 20 }}>
                #{theme.rank}
              </Typography>
              <Typography variant="body2" fontWeight="medium" sx={{ maxWidth: 200 }} noWrap>
                {theme.name}
              </Typography>
            </Box>
            <MomentumBar score={theme.momentum_score} />
          </Box>
        ))}
      </CardContent>
    </Card>
  );
}

function AlertsCard({ alerts, isLoading, onDismiss, dismissingId }) {
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
          <Box sx={{ fontSize: '12px', fontWeight: 600, ml: 1 }}>Alerts</Box>
          {alerts?.total > 0 && (
            <Box sx={{ fontSize: '10px', color: 'text.secondary', ml: 0.5 }}>({alerts.total})</Box>
          )}
        </Box>

        {alerts?.alerts?.length > 0 ? (
          <Box sx={{ maxHeight: 200, overflowY: 'auto' }}>
            {alerts.alerts.map((alert) => (
              <Box
                key={alert.id}
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
                  '&:hover .dismiss-action, &:focus-within .dismiss-action': {
                    opacity: 1,
                  },
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
                <IconButton
                  className="dismiss-action"
                  size="small"
                  onClick={(event) => {
                    event.stopPropagation();
                    onDismiss?.(alert.id);
                  }}
                  disabled={dismissingId === alert.id}
                  sx={{
                    p: 0.25,
                    ml: 0.5,
                    opacity: dismissingId === alert.id ? 0.5 : 0.35,
                    transition: 'opacity 120ms ease',
                  }}
                  aria-label={`Dismiss alert ${alert.title}`}
                >
                  {dismissingId === alert.id ? <CircularProgress size={12} /> : <CloseIcon sx={{ fontSize: 14 }} />}
                </IconButton>
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
}

function PipelineObservabilityCard({ observability, isLoading }) {
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
          <Box sx={{ fontSize: '12px', fontWeight: 600, ml: 1 }}>Pipeline Health</Box>
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
          {statChip(
            'Parse',
            `${((metrics.parse_failure_rate || 0) * 100).toFixed(1)}%`,
            metrics.parse_failure_rate > 0.25 ? 'warning' : 'default'
          )}
          {statChip(
            'No-mention',
            `${((metrics.processed_without_mentions_ratio || 0) * 100).toFixed(1)}%`,
            metrics.processed_without_mentions_ratio > 0.2 ? 'warning' : 'default'
          )}
          {statChip(
            'New clusters',
            `${((metrics.new_cluster_rate || 0) * 100).toFixed(1)}%`,
            metrics.new_cluster_rate > 0.45 ? 'warning' : 'default'
          )}
          {statChip(
            'Merge proxy',
            `${((metrics.merge_precision_proxy || 0) * 100).toFixed(1)}%`,
            metrics.merge_precision_proxy < 0.55 ? 'warning' : 'default'
          )}
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
                  <a href={item.runbook_url} target="_blank" rel="noreferrer">
                    Runbook
                  </a>
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
}

export default function ThemeInsightsCards({
  emerging,
  isLoadingEmerging,
  topTrendingThemes,
  onSelectTheme,
  observability,
  isLoadingObservability,
  alerts,
  isLoadingAlerts,
  onDismissAlert,
  dismissingAlertId,
}) {
  return (
    <Grid container spacing={2} mb={3}>
      <Grid item xs={12} md={3}>
        <EmergingThemesCard themes={emerging} isLoading={isLoadingEmerging} />
      </Grid>
      <Grid item xs={12} md={3}>
        <TopTrendingThemesCard topTrendingThemes={topTrendingThemes} onSelectTheme={onSelectTheme} />
      </Grid>
      <Grid item xs={12} md={3}>
        <PipelineObservabilityCard observability={observability} isLoading={isLoadingObservability} />
      </Grid>
      <Grid item xs={12} md={3}>
        <AlertsCard
          alerts={alerts}
          isLoading={isLoadingAlerts}
          onDismiss={onDismissAlert}
          dismissingId={dismissingAlertId}
        />
      </Grid>
    </Grid>
  );
}
