import { useMemo, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  Alert,
  Box,
  CircularProgress,
  Grid,
  Paper,
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
import { useStaticManifest, fetchStaticJson } from '../dataClient';

function SummaryMetric({ label, value }) {
  return (
    <Paper sx={{ p: 2, height: '100%' }}>
      <Typography variant="caption" color="text.secondary">
        {label}
      </Typography>
      <Typography variant="h6" sx={{ mt: 0.5 }}>
        {value}
      </Typography>
    </Paper>
  );
}

function StaticThemesPage() {
  const manifestQuery = useStaticManifest();
  const themesIndexQuery = useQuery({
    queryKey: ['staticThemesIndex', manifestQuery.data?.pages?.themes?.path],
    queryFn: () => fetchStaticJson(manifestQuery.data.pages.themes.path),
    enabled: Boolean(manifestQuery.data?.pages?.themes?.path),
    staleTime: Infinity,
  });
  const [pipeline, setPipeline] = useState('technical');
  const [themeView, setThemeView] = useState('grouped');
  const variantKey = `${pipeline}:${themeView}`;
  const variants = themesIndexQuery.data?.variants ?? null;
  const variantMeta = variants?.[variantKey];
  const fallbackVariantKey = useMemo(() => {
    if (variantMeta?.available) {
      return variantKey;
    }

    const variantEntries = variants ? Object.entries(variants) : [];
    const samePipelineVariant = variantEntries.find(
      ([key, meta]) => meta?.available && key.startsWith(`${pipeline}:`)
    );
    if (samePipelineVariant) {
      return samePipelineVariant[0];
    }

    const firstAvailableVariant = variantEntries.find(([_key, meta]) => meta?.available);
    return firstAvailableVariant?.[0] || null;
  }, [pipeline, variantKey, variantMeta?.available, variants]);
  const activeVariantKey = variantMeta?.available ? variantKey : fallbackVariantKey;
  const activeVariantMeta = activeVariantKey ? variants?.[activeVariantKey] : null;
  const activeThemeView = activeVariantKey?.split(':')[1] || themeView;
  const isFallbackActive = Boolean(activeVariantKey && activeVariantKey !== variantKey);
  const variantQuery = useQuery({
    queryKey: ['staticThemesVariant', activeVariantMeta?.path],
    queryFn: () => fetchStaticJson(activeVariantMeta.path),
    enabled: Boolean(activeVariantMeta?.available && activeVariantMeta?.path),
    staleTime: Infinity,
  });

  const rankings = useMemo(() => {
    const payload = variantQuery.data?.payload || {};
    if (activeThemeView === 'grouped') {
      return payload.l1_rankings?.rankings || [];
    }
    return payload.rankings?.rankings || [];
  }, [activeThemeView, variantQuery.data]);

  if (manifestQuery.isLoading || themesIndexQuery.isLoading) {
    return (
      <Box display="flex" justifyContent="center" py={8}>
        <CircularProgress />
      </Box>
    );
  }

  if (manifestQuery.isError || themesIndexQuery.isError) {
    return <Alert severity="error">Failed to load themes data.</Alert>;
  }

  if (!themesIndexQuery.data?.available) {
    return <Alert severity="info">Themes are unavailable in this static build.</Alert>;
  }

  if (!activeVariantMeta?.available) {
    return <Alert severity="warning">No exported theme views are available in this static build.</Alert>;
  }

  if (variantQuery.isLoading) {
    return (
      <Box display="flex" justifyContent="center" py={8}>
        <CircularProgress />
      </Box>
    );
  }

  if (variantQuery.isError) {
    return <Alert severity="error">Failed to load the selected theme view.</Alert>;
  }

  const payload = variantQuery.data?.payload || {};
  const emerging = payload.emerging?.themes || [];
  const fallbackLabel = activeVariantKey
    ? activeVariantKey
        .split(':')
        .map((value) => value.charAt(0).toUpperCase() + value.slice(1))
        .join(' / ')
    : null;

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Themes
      </Typography>
      <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
        Published {variantQuery.data?.published_at || variantQuery.data?.generated_at}. Theme operations are read-only in the static site.
      </Typography>

      <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap', mb: 3 }}>
        <ToggleButtonGroup
          value={pipeline}
          exclusive
          onChange={(_event, value) => value && setPipeline(value)}
          size="small"
        >
          <ToggleButton value="technical">Technical</ToggleButton>
          <ToggleButton value="fundamental">Fundamental</ToggleButton>
        </ToggleButtonGroup>
        <ToggleButtonGroup
          value={themeView}
          exclusive
          onChange={(_event, value) => value && setThemeView(value)}
          size="small"
        >
          <ToggleButton value="grouped">Grouped</ToggleButton>
          <ToggleButton value="flat">Flat</ToggleButton>
        </ToggleButtonGroup>
      </Box>

      {isFallbackActive && fallbackLabel && (
        <Alert severity="warning" sx={{ mb: 3 }}>
          The selected theme view is unavailable in this export. Showing {fallbackLabel} instead.
        </Alert>
      )}

      <Grid container spacing={2} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={4}>
          <SummaryMetric label="Emerging Themes" value={payload.emerging?.count ?? 0} />
        </Grid>
        <Grid item xs={12} sm={4}>
          <SummaryMetric label="Pending Merge Suggestions" value={payload.pending_merge_count ?? 0} />
        </Grid>
        <Grid item xs={12} sm={4}>
          <SummaryMetric label="Retryable Failed Items" value={payload.failed_items_count?.failed_count ?? 0} />
        </Grid>
      </Grid>

      {emerging.length > 0 && (
        <Paper sx={{ p: 2, mb: 3 }}>
          <Typography variant="h6" gutterBottom>
            Emerging Themes
          </Typography>
          <TableContainer>
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>Theme</TableCell>
                  <TableCell align="right">Mentions 7D</TableCell>
                  <TableCell align="right">Velocity</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {emerging.slice(0, 10).map((row) => (
                  <TableRow key={row.theme}>
                    <TableCell>{row.theme}</TableCell>
                    <TableCell align="right">{row.mentions_7d}</TableCell>
                    <TableCell align="right">{row.velocity?.toFixed?.(2) ?? '-'}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </Paper>
      )}

      <Paper sx={{ p: 2 }}>
        <Typography variant="h6" gutterBottom>
          {activeThemeView === 'grouped' ? 'Grouped Rankings' : 'Flat Rankings'}
        </Typography>
        <TableContainer>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell align="right">Rank</TableCell>
                <TableCell>Theme</TableCell>
                <TableCell align="right">Momentum</TableCell>
                <TableCell align="right">Mentions 7D</TableCell>
                <TableCell align="right">Constituents</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {rankings.slice(0, 50).map((row) => (
                <TableRow key={row.id ?? row.theme_cluster_id}>
                  <TableCell align="right">{row.rank ?? '-'}</TableCell>
                  <TableCell>{row.display_name || row.theme}</TableCell>
                  <TableCell align="right">{row.momentum_score?.toFixed?.(2) ?? '-'}</TableCell>
                  <TableCell align="right">{row.mentions_7d ?? '-'}</TableCell>
                  <TableCell align="right">{row.num_constituents ?? '-'}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      </Paper>
    </Box>
  );
}

export default StaticThemesPage;
