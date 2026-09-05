import { useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Alert, Box, Button, CircularProgress, Typography } from '@mui/material';
import { Link as RouterLink, useNavigate, useParams } from 'react-router-dom';

import OptionsSymbolDetailView from '../../features/options/OptionsSymbolDetailView';
import { useStaticMarket } from '../StaticMarketContext';
import { resolveStaticMarketEntry, useStaticManifest } from '../dataClient';
import { getStaticOptionsManifest, staticOptionsSymbolQueryOptions } from '../optionsClient';

export default function StaticOptionsSymbolPage() {
  const navigate = useNavigate();
  const { symbol: routeSymbol = '' } = useParams();
  const symbol = routeSymbol.trim().toUpperCase();
  const rootManifest = useStaticManifest();
  const { selectedMarket } = useStaticMarket();
  const marketEntry = resolveStaticMarketEntry(rootManifest.data, selectedMarket);
  const market = marketEntry.market || selectedMarket;
  const optionsPath = marketEntry.pages?.options?.path;
  const manifestQuery = useQuery({
    queryKey: ['options-analytics', 'manifest', 'static', market, optionsPath],
    queryFn: () => getStaticOptionsManifest(marketEntry),
    enabled: market === 'US' && Boolean(optionsPath),
    staleTime: Infinity,
    gcTime: Infinity,
  });
  const detailSetup = useMemo(() => {
    if (!manifestQuery.data) return { options: null, setupError: null };
    try {
      return { options: staticOptionsSymbolQueryOptions(manifestQuery.data, symbol), setupError: null };
    } catch (error) {
      return { options: null, setupError: error };
    }
  }, [manifestQuery.data, symbol]);
  const detailQuery = useQuery({
    ...(detailSetup.options || {
      queryKey: ['options-analytics', 'symbol', 'static', 'unavailable', symbol],
      queryFn: async () => null,
    }),
    enabled: Boolean(detailSetup.options),
  });

  if (manifestQuery.isLoading || (detailSetup.options && detailQuery.isLoading)) {
    return <Box sx={{ display: 'flex', justifyContent: 'center', p: 6 }}><CircularProgress /></Box>;
  }
  if (manifestQuery.isError || detailSetup.setupError || detailQuery.isError || !detailQuery.data) {
    return (
      <Alert severity="info">
        <Typography variant="h6" component="h1">{symbol || 'This symbol'} is not in the published options cohort</Typography>
        <Typography variant="body2" sx={{ mb: 1 }}>Static detail files exist only for today’s advertised symbols.</Typography>
        <Button component={RouterLink} to="/options" color="inherit">Back to Command Center</Button>
      </Alert>
    );
  }

  return <OptionsSymbolDetailView data={detailQuery.data} onBack={() => navigate('/options')} />;
}
