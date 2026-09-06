import { useQuery } from '@tanstack/react-query';
import { Alert, Box, CircularProgress } from '@mui/material';
import { useNavigate } from 'react-router-dom';

import OptionsCommandCenterView from '../../features/options/OptionsCommandCenterView';
import { useStaticMarket } from '../StaticMarketContext';
import { resolveStaticMarketEntry, useStaticManifest } from '../dataClient';
import {
  getStaticOptionsManifest,
  staticOptionsCommandCenterQueryOptions,
} from '../optionsClient';

export default function StaticOptionsPage() {
  const navigate = useNavigate();
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
  const commandOptions = manifestQuery.data
    ? staticOptionsCommandCenterQueryOptions(manifestQuery.data)
    : { queryKey: ['options-analytics', 'command-center', 'static', 'pending'], queryFn: async () => null };
  const commandQuery = useQuery({ ...commandOptions, enabled: Boolean(manifestQuery.data) });

  if (manifestQuery.isLoading || commandQuery.isLoading) {
    return <Box sx={{ display: 'flex', justifyContent: 'center', p: 6 }}><CircularProgress /></Box>;
  }
  if (manifestQuery.isError || commandQuery.isError || !commandQuery.data) {
    return <Alert severity="info">Options analytics are not available in this static snapshot.</Alert>;
  }

  return (
    <OptionsCommandCenterView
      data={commandQuery.data}
      onOpenSymbol={(symbol) => navigate(`/options/${encodeURIComponent(symbol)}`)}
    />
  );
}
