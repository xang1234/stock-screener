import { useQuery } from '@tanstack/react-query';
import { Alert, Box, CircularProgress } from '@mui/material';
import { useNavigate } from 'react-router-dom';

import OptionsCommandCenterView from '../../features/options/OptionsCommandCenterView';
import { useStaticMarket } from '../StaticMarketContext';
import {
  resolveStaticMarketEntry,
  staticQueryOptions,
  useStaticGeneration,
  useStaticManifest,
} from '../dataClient';
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
  const generation = useStaticGeneration();
  const manifestQuery = useQuery(staticQueryOptions({
    key: ['options-analytics', 'manifest', 'static', market, optionsPath],
    generation: generation.generation,
    queryFn: () => getStaticOptionsManifest(marketEntry, generation.dataRoot),
    enabled: market === 'US' && Boolean(optionsPath),
  }));
  const commandOptions = manifestQuery.data
    ? staticOptionsCommandCenterQueryOptions(manifestQuery.data, generation)
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
