import { useQuery } from '@tanstack/react-query';
import { Alert, Box, Button, CircularProgress, Typography } from '@mui/material';
import { Link as RouterLink, useNavigate, useParams } from 'react-router-dom';

import { getOptionsSymbolDetail } from '../api/optionsAnalytics';
import OptionsSymbolDetailView from '../features/options/OptionsSymbolDetailView';
import { optionsSymbolQueryKey } from '../features/options/optionsContract';

export default function OptionsSymbolPage() {
  const navigate = useNavigate();
  const { symbol: routeSymbol = '' } = useParams();
  const symbol = routeSymbol.trim().toUpperCase();
  const detailQuery = useQuery({
    queryKey: optionsSymbolQueryKey({ mode: 'live', runId: 'published', symbol }),
    queryFn: () => getOptionsSymbolDetail(symbol),
    enabled: Boolean(symbol),
  });

  if (detailQuery.isLoading) {
    return <Box sx={{ display: 'flex', justifyContent: 'center', p: 6 }}><CircularProgress /></Box>;
  }
  if (detailQuery.isError || !detailQuery.data) {
    return (
      <Alert severity="info">
        <Typography variant="h6" component="h1">{symbol || 'This symbol'} is not in the published options cohort</Typography>
        <Typography variant="body2" sx={{ mb: 1 }}>It may have dropped out of today’s current Candidates and Leaders.</Typography>
        <Button component={RouterLink} to="/options" color="inherit">Back to Command Center</Button>
      </Alert>
    );
  }

  return <OptionsSymbolDetailView data={detailQuery.data} onBack={() => navigate('/options')} />;
}
