import { useMemo, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { Link as RouterLink } from 'react-router-dom';
import {
  Alert,
  Box,
  Button,
  Chip,
  CircularProgress,
  Link,
  Paper,
  Stack,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Typography,
} from '@mui/material';
import DownloadIcon from '@mui/icons-material/Download';
import ContentCopyIcon from '@mui/icons-material/ContentCopy';

import { getDailyDigest, getDailyDigestMarkdown } from '../api/digest';
import { ValidationDegradedAlert, ValidationSummaryCards } from '../components/Validation/ValidationPanels';
import { useStrategyProfile } from '../contexts/StrategyProfileContext';

function formatDate(value) {
  if (!value) return '-';
  if (typeof value === 'string' && /^\d{4}-\d{2}-\d{2}$/.test(value)) {
    const [year, month, day] = value.split('-').map(Number);
    return new Date(year, month - 1, day).toLocaleDateString();
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString();
}

function formatPercent(value) {
  if (value == null) return '-';
  return `${Number(value).toFixed(1)}%`;
}

function severityToMui(severity) {
  if (severity === 'critical') return 'error';
  return severity || 'info';
}

function SectionCard({ title, children }) {
  return (
    <Paper sx={{ p: 2 }}>
      <Stack spacing={1.5}>
        <Typography variant="h6" sx={{ fontWeight: 700 }}>
          {title}
        </Typography>
        {children}
      </Stack>
    </Paper>
  );
}

function DigestValidationBlock({ title, snapshot }) {
  return (
    <Stack spacing={1}>
      <Typography variant="subtitle2" sx={{ fontWeight: 700 }}>
        {title}
      </Typography>
      <ValidationDegradedAlert degradedReasons={snapshot?.degraded_reasons} />
      <ValidationSummaryCards horizons={snapshot?.horizons || []} />
    </Stack>
  );
}

function Metric({ label, value }) {
  return (
    <Paper variant="outlined" sx={{ p: 1.25 }}>
      <Typography variant="caption" color="text.secondary">
        {label}
      </Typography>
      <Typography variant="body2" sx={{ fontWeight: 700 }}>
        {value}
      </Typography>
    </Paper>
  );
}

function ThemeTable({ themes }) {
  return (
    <TableContainer>
      <Table size="small">
        <TableHead>
          <TableRow>
            <TableCell>Theme</TableCell>
            <TableCell align="right">Momentum</TableCell>
            <TableCell align="right">Velocity</TableCell>
            <TableCell align="right">1M Basket</TableCell>
            <TableCell>Status</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {themes.map((theme) => (
            <TableRow key={theme.theme_id}>
              <TableCell>{theme.display_name}</TableCell>
              <TableCell align="right">{theme.momentum_score?.toFixed?.(1) ?? '-'}</TableCell>
              <TableCell align="right">{theme.mention_velocity?.toFixed?.(2) ?? '-'}</TableCell>
              <TableCell align="right">{formatPercent(theme.basket_return_1m)}</TableCell>
              <TableCell>{theme.status || '-'}</TableCell>
            </TableRow>
          ))}
          {themes.length === 0 && (
            <TableRow>
              <TableCell colSpan={5} align="center">
                <Typography variant="body2" color="text.secondary">
                  No ranked themes are available.
                </Typography>
              </TableCell>
            </TableRow>
          )}
        </TableBody>
      </Table>
    </TableContainer>
  );
}

function SymbolList({ symbols }) {
  if (!symbols?.length) {
    return <Typography variant="body2" color="text.secondary">-</Typography>;
  }

  return (
    <Stack direction="row" spacing={0.75} flexWrap="wrap">
      {symbols.map((symbol) => (
        <Link
          key={symbol}
          component={RouterLink}
          to={`/stocks/${encodeURIComponent(symbol)}`}
          underline="hover"
          sx={{ fontSize: '0.8rem' }}
        >
          {symbol}
        </Link>
      ))}
    </Stack>
  );
}

function DigestPage() {
  const { activeProfile, activeProfileDetail } = useStrategyProfile();
  const queryClient = useQueryClient();
  const [exportError, setExportError] = useState('');
  const [exportMessage, setExportMessage] = useState('');
  const [isExporting, setIsExporting] = useState(false);

  const cachedDigest =
    queryClient.getQueryData(['dailyDigest', activeProfile]) ||
    queryClient.getQueryData(['dailyDigest']);

  const { data, isLoading, error } = useQuery({
    queryKey: ['dailyDigest', activeProfile],
    queryFn: () => getDailyDigest(undefined, activeProfile),
    staleTime: 60_000,
    placeholderData: (previousData) => previousData || cachedDigest,
  });

  const digestData = data || cachedDigest;

  const freshnessSummary = useMemo(() => {
    if (!digestData?.freshness) {
      return [];
    }
    return [
      ['Feature run', formatDate(digestData.freshness.latest_feature_as_of_date)],
      ['Breadth', formatDate(digestData.freshness.latest_breadth_date)],
      ['Theme metrics', formatDate(digestData.freshness.latest_theme_metrics_date)],
      ['Theme alerts', formatDate(digestData.freshness.latest_theme_alert_at)],
    ];
  }, [digestData]);

  const handleMarkdownExport = async (mode) => {
    setExportError('');
      setExportMessage('');
      setIsExporting(true);
      try {
      const markdown = await getDailyDigestMarkdown(digestData?.as_of_date, activeProfile);
      if (mode === 'copy') {
        if (!navigator?.clipboard?.writeText) {
          throw new Error('Clipboard access is unavailable in this environment.');
        }
        await navigator.clipboard.writeText(markdown);
        setExportMessage('Markdown copied to clipboard.');
      } else {
        const blob = new Blob([markdown], { type: 'text/markdown;charset=utf-8' });
        const url = window.URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `daily_digest_${digestData?.as_of_date || 'latest'}.md`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        window.URL.revokeObjectURL(url);
        setExportMessage('Markdown download started.');
      }
    } catch (requestError) {
      setExportError(requestError.message || 'Failed to export markdown.');
    } finally {
      setIsExporting(false);
    }
  };

  if (isLoading && !digestData) {
    return (
      <Box display="flex" justifyContent="center" py={6}>
        <CircularProgress />
      </Box>
    );
  }

  if (error && !digestData) {
    return <Alert severity="error">Failed to load daily digest: {error.message}</Alert>;
  }

  if (!digestData) {
    return <Alert severity="warning">No digest data is available.</Alert>;
  }

  const orderedSections = activeProfileDetail?.digest?.section_order || [
    'market',
    'leaders',
    'themes',
    'validation',
    'watchlists',
    'risks',
  ];

  const sections = {
    market: (
      <SectionCard title="Market Stance" key="market">
        <Typography variant="body2">{digestData.market.summary}</Typography>
        <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr 1fr', md: 'repeat(5, 1fr)' }, gap: 1 }}>
          <Metric label="Up 4%+" value={digestData.market.breadth_metrics?.up_4pct ?? '-'} />
          <Metric label="Down 4%+" value={digestData.market.breadth_metrics?.down_4pct ?? '-'} />
          <Metric label="5D Ratio" value={digestData.market.breadth_metrics?.ratio_5day?.toFixed?.(2) ?? '-'} />
          <Metric label="10D Ratio" value={digestData.market.breadth_metrics?.ratio_10day?.toFixed?.(2) ?? '-'} />
          <Metric label="Universe" value={digestData.market.breadth_metrics?.total_stocks_scanned ?? '-'} />
        </Box>
      </SectionCard>
    ),
    leaders: (
      <SectionCard title="Leaders" key="leaders">
        <TableContainer>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>Symbol</TableCell>
                <TableCell>Name</TableCell>
                <TableCell align="right">Score</TableCell>
                <TableCell>Rating</TableCell>
                <TableCell>Industry Group</TableCell>
                <TableCell>Reason</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {digestData.leaders.map((leader) => (
                <TableRow key={leader.symbol}>
                  <TableCell>
                    <Link component={RouterLink} to={`/stocks/${encodeURIComponent(leader.symbol)}`} underline="hover">
                      {leader.symbol}
                    </Link>
                  </TableCell>
                  <TableCell>{leader.name || '-'}</TableCell>
                  <TableCell align="right">{leader.composite_score?.toFixed?.(1) ?? '-'}</TableCell>
                  <TableCell>{leader.rating || '-'}</TableCell>
                  <TableCell>{leader.industry_group || '-'}</TableCell>
                  <TableCell>{leader.reason_summary}</TableCell>
                </TableRow>
              ))}
              {digestData.leaders.length === 0 && (
                <TableRow>
                  <TableCell colSpan={6} align="center">
                    <Typography variant="body2" color="text.secondary">
                      No leader candidates are available.
                    </Typography>
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </TableContainer>
      </SectionCard>
    ),
    themes: (
      <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', xl: '1fr 1fr' }, gap: 2 }} key="themes">
        <SectionCard title="Themes">
          <Typography variant="subtitle2" sx={{ fontWeight: 700 }}>
            Leaders
          </Typography>
          <ThemeTable themes={digestData.themes.leaders} />
          <Typography variant="subtitle2" sx={{ fontWeight: 700, pt: 1 }}>
            Laggards
          </Typography>
          <ThemeTable themes={digestData.themes.laggards} />
        </SectionCard>

        <SectionCard title="Recent Theme Alerts">
          <Stack spacing={1}>
            {digestData.themes.recent_alerts.map((alert) => (
              <Paper key={alert.alert_id} variant="outlined" sx={{ p: 1.25 }}>
                <Stack spacing={0.5}>
                  <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap">
                    <Chip size="small" label={alert.alert_type.replace('_', ' ')} />
                    <Chip size="small" variant="outlined" label={alert.severity || 'info'} />
                    <Typography variant="caption" color="text.secondary">
                      {formatDate(alert.triggered_at)}
                    </Typography>
                  </Stack>
                  <Typography variant="subtitle2" sx={{ fontWeight: 700 }}>
                    {alert.title}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    {alert.theme || 'Unknown theme'}
                  </Typography>
                  <Stack direction="row" spacing={1} flexWrap="wrap">
                    {alert.related_tickers.map((symbol) => (
                      <Chip key={`${alert.alert_id}:${symbol}`} size="small" label={symbol} />
                    ))}
                  </Stack>
                </Stack>
              </Paper>
            ))}
            {digestData.themes.recent_alerts.length === 0 && (
              <Typography variant="body2" color="text.secondary">
                No recent theme alerts are available.
              </Typography>
            )}
          </Stack>
        </SectionCard>
      </Box>
    ),
    validation: (
      <SectionCard title="Validation Snapshot" key="validation">
        <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', xl: '1fr 1fr' }, gap: 2 }}>
          <DigestValidationBlock title="Scan Picks" snapshot={digestData.validation.scan_pick} />
          <DigestValidationBlock title="Theme Alerts" snapshot={digestData.validation.theme_alert} />
        </Box>
      </SectionCard>
    ),
    watchlists: (
      <SectionCard title="Watchlist Highlights" key="watchlists">
        <TableContainer>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>Watchlist</TableCell>
                <TableCell>Leader Overlap</TableCell>
                <TableCell>Theme Alert Overlap</TableCell>
                <TableCell>Notes</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {digestData.watchlists.map((watchlist) => (
                <TableRow key={watchlist.watchlist_id}>
                  <TableCell>{watchlist.watchlist_name}</TableCell>
                  <TableCell><SymbolList symbols={watchlist.matched_symbols} /></TableCell>
                  <TableCell><SymbolList symbols={watchlist.alert_symbols} /></TableCell>
                  <TableCell>{watchlist.notes}</TableCell>
                </TableRow>
              ))}
              {digestData.watchlists.length === 0 && (
                <TableRow>
                  <TableCell colSpan={4} align="center">
                    <Typography variant="body2" color="text.secondary">
                      No watchlist highlights are available.
                    </Typography>
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </TableContainer>
      </SectionCard>
    ),
    risks: (
      <SectionCard title="Risk Notes" key="risks">
        <Stack spacing={1}>
          {digestData.risks.map((risk, index) => (
            <Alert key={`${risk.kind}:${index}`} severity={severityToMui(risk.severity)}>
              <strong>{risk.kind.replace(/_/g, ' ')}</strong>: {risk.message}
            </Alert>
          ))}
          {digestData.risks.length === 0 && (
            <Typography variant="body2" color="text.secondary">
              No material risk notes are available.
            </Typography>
          )}
        </Stack>
      </SectionCard>
    ),
  };

  return (
    <Stack spacing={2}>
      <Paper sx={{ p: 2 }}>
        <Stack spacing={1.5}>
          <Stack
            direction={{ xs: 'column', md: 'row' }}
            justifyContent="space-between"
            alignItems={{ xs: 'flex-start', md: 'center' }}
            spacing={1.5}
          >
            <Box>
              <Typography variant="h5" sx={{ fontWeight: 700 }}>
                Daily Digest
              </Typography>
              <Typography variant="body2" color="text.secondary">
                As of {formatDate(digestData.as_of_date)}. Shared daily summary built from breadth, leaders, themes, validation, and watchlists.
              </Typography>
            </Box>
            <Stack direction={{ xs: 'column', sm: 'row' }} spacing={1}>
              <Chip label={`Profile: ${activeProfileDetail?.label || activeProfile}`} size="small" variant="outlined" />
              <Button
                variant="outlined"
                size="small"
                startIcon={<ContentCopyIcon fontSize="small" />}
                onClick={() => handleMarkdownExport('copy')}
                disabled={isExporting}
              >
                Copy Markdown
              </Button>
              <Button
                variant="contained"
                size="small"
                startIcon={<DownloadIcon fontSize="small" />}
                onClick={() => handleMarkdownExport('download')}
                disabled={isExporting}
              >
                Download Markdown
              </Button>
            </Stack>
          </Stack>

          <Stack direction={{ xs: 'column', md: 'row' }} spacing={1} alignItems={{ xs: 'flex-start', md: 'center' }}>
            <Chip
              color={digestData.market.stance === 'offense' ? 'success' : digestData.market.stance === 'defense' ? 'warning' : 'default'}
              label={`Stance: ${digestData.market.stance}`}
              size="small"
            />
            <Typography variant="caption" color="text.secondary">
              Validation window: {digestData.validation.lookback_days} days
            </Typography>
          </Stack>

          {exportMessage && <Alert severity="success">{exportMessage}</Alert>}
          {exportError && <Alert severity="error">{exportError}</Alert>}
          {digestData.degraded_reasons?.length > 0 && (
            <Alert severity="info">
              Digest sections are partially degraded: {digestData.degraded_reasons.join(', ')}.
            </Alert>
          )}
        </Stack>
      </Paper>

      {orderedSections.map((sectionKey) => sections[sectionKey]).filter(Boolean)}

      <SectionCard title="Freshness">
        <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: 'repeat(4, 1fr)' }, gap: 1 }}>
          {freshnessSummary.map(([label, value]) => (
            <Metric key={label} label={label} value={value} />
          ))}
        </Box>
      </SectionCard>
    </Stack>
  );
}

export default DigestPage;
