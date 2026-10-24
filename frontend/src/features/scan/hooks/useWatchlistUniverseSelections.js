import { useEffect, useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';

import { getWatchlistSymbols, getWatchlists } from '../../../api/userWatchlists';
import { WATCHLIST_MARKET } from '../constants';
import { withWatchlistUniverse } from '../universeSelection';

// Adds the saved watchlists to the universe picker as the Watchlist market and
// loads the selected watchlist's symbols. A selection whose watchlist was
// deleted elsewhere is cleared, so Scan never targets a missing watchlist.
export function useWatchlistUniverseSelections({
  universeSelections,
  universeMarket,
  universeScope,
  setUniverseScope,
}) {
  const watchlistMarketSelected = universeMarket === WATCHLIST_MARKET;
  const selectedWatchlistId = watchlistMarketSelected && universeScope ? Number(universeScope) : null;
  const watchlistsQuery = useQuery({
    queryKey: ['userWatchlists'],
    queryFn: getWatchlists,
    enabled: watchlistMarketSelected,
  });
  const symbolsQuery = useQuery({
    queryKey: ['userWatchlistSymbols', selectedWatchlistId],
    queryFn: () => getWatchlistSymbols(selectedWatchlistId),
    enabled: selectedWatchlistId !== null,
    // The list can change in another tab or through MCP, so refetch on mount
    // and when this tab becomes visible again. An edit made while this tab
    // stays visible is not seen until then.
    staleTime: 0,
    refetchOnWindowFocus: true,
  });
  const watchlists = watchlistsQuery.data?.watchlists;
  const selectedRemoved = selectedWatchlistId !== null
    && watchlists !== undefined
    && !watchlists.some((watchlist) => watchlist.id === selectedWatchlistId);

  useEffect(() => {
    if (selectedRemoved) {
      setUniverseScope(null);
    }
  }, [selectedRemoved, setUniverseScope]);

  return useMemo(
    () => withWatchlistUniverse(universeSelections, {
      watchlists,
      listFailed: watchlistsQuery.isError,
      selected: selectedWatchlistId === null ? null : {
        id: String(selectedWatchlistId),
        symbols: symbolsQuery.data?.symbols,
        failed: symbolsQuery.isError,
      },
    }),
    [
      selectedWatchlistId,
      symbolsQuery.data,
      symbolsQuery.isError,
      universeSelections,
      watchlists,
      watchlistsQuery.isError,
    ]
  );
}
