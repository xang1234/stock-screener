import { useCallback, useEffect, useMemo, useState } from 'react';
import { keepPreviousData, useQuery } from '@tanstack/react-query';
import { Chip } from '@mui/material';

import { getSocialEvidence, getSocialSymbols } from '../../api/socialSignals';
import { useIsMobileShell } from '../../components/Layout/shellNav';
import { isSocialSignalMarket, SOCIAL_SIGNAL_MARKETS_COPY } from './socialSignalMarkets';
import { formatSocialScore, socialStateLabel } from './socialSignalPresentation';

const WINDOW = '7d';
const CHIP_COLOR = { actionable: 'success', watch: 'info', risk_off: 'warning', context: 'default' };

const chipLabel = (item) => (item.queue_score == null
  ? socialStateLabel(item.state)
  : `${socialStateLabel(item.state)} ${formatSocialScore(item.queue_score)}`);

export default function useScanSocialColumn(rows, enabled) {
  const isMobile = useIsMobileShell();
  // Phone cards have no Social cell, so they make no lookup.
  const lookupEnabled = enabled && !isMobile;
  const symbols = useMemo(() => (lookupEnabled ? [...new Set((rows || [])
    .filter((row) => isSocialSignalMarket(row.market))
    .map((row) => row.symbol))].sort() : []), [rows, lookupEnabled]);
  const lookup = useQuery({
    queryKey: ['scanSocialSymbols', WINDOW, symbols],
    queryFn: () => getSocialSymbols(symbols, WINDOW),
    enabled: symbols.length > 0,
    staleTime: 60_000,
    placeholderData: keepPreviousData,
  });
  const [selected, setSelected] = useState(null);
  useEffect(() => { if (!lookupEnabled) setSelected(null); }, [lookupEnabled]);
  const evidence = useQuery({
    queryKey: ['socialSignals', 'evidence', selected?.candidate_key, WINDOW],
    queryFn: () => getSocialEvidence(selected.candidate_key, WINDOW),
    enabled: Boolean(selected),
  });
  const close = useCallback(() => setSelected(null), []);
  const renderCell = useMemo(() => {
    if (!symbols.length || (!lookup.data && lookup.isError) || lookup.data?.available === false) return null;
    const thisPage = lookup.data && !lookup.isPlaceholderData
      ? new Map(lookup.data.items.map((item) => [item.canonical_symbol, item]))
      : null;
    return function renderSocialCell(row) {
      if (!isSocialSignalMarket(row.market)) return <span title={SOCIAL_SIGNAL_MARKETS_COPY}>n/a</span>;
      if (!thisPage) return '…';
      const item = thisPage.get(row.symbol);
      if (!item) return <span title="Not mentioned in the latest 7-day Social run">-</span>;
      return (
        <Chip
          size="small" label={chipLabel(item)} color={CHIP_COLOR[item.state]}
          variant={item.state === 'context' ? 'outlined' : 'filled'}
          title={`${item.mention_count} mentions in the published 7-day window. Open the evidence.`}
          onClick={(event) => { event.stopPropagation(); setSelected(item); }}
        />
      );
    };
  }, [symbols.length, lookup.isError, lookup.data, lookup.isPlaceholderData]);
  return { renderCell, drawer: { open: Boolean(selected), selected, query: evidence, onClose: close } };
}
