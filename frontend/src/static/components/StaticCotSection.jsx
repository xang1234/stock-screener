import { useState } from 'react';
import { Alert, Box } from '@mui/material';

import CotPositioningView from '../../features/cot/CotPositioningView';
import { sliceCotHistory } from '../../features/cot/cotContract';
import { useStaticCotHistory, useStaticCotIndex } from '../cotClient';

const StaticCotSection = ({ manifest }) => {
  const advertised = Boolean(manifest?.assets?.cot?.path);
  const [selectedSlug, setSelectedSlug] = useState('sp-500');
  const [range, setRange] = useState('1y');
  const indexQuery = useStaticCotIndex(manifest);
  const index = indexQuery.data || null;
  const historyQuery = useStaticCotHistory(index, selectedSlug, '5y', !indexQuery.isPlaceholderData);
  const history = historyQuery.data ? sliceCotHistory(historyQuery.data, range) : null;

  if (!advertised) return null;
  if (indexQuery.isError) {
    return <Alert data-testid="static-cot-section" severity="warning">Static COT data is unavailable.</Alert>;
  }

  return (
    <Box data-testid="static-cot-section" sx={{ mb: 2 }}>
      <CotPositioningView
        catalog={index?.catalog || null}
        history={history}
        selectedSlug={selectedSlug}
        range={range}
        onSelectInstrument={setSelectedSlug}
        onRangeChange={setRange}
        isLoading={indexQuery.isLoading || historyQuery.isLoading}
        error={historyQuery.error}
      />
    </Box>
  );
};

export default StaticCotSection;
