// Item changes alter both the watchlist table payload and the symbol list the
// Scan page scans, so every item mutation refreshes both.
export const invalidateWatchlistItems = (queryClient, watchlistId) => Promise.all([
  queryClient.invalidateQueries({ queryKey: ['userWatchlistData', watchlistId] }),
  queryClient.invalidateQueries({ queryKey: ['userWatchlistSymbols', watchlistId] }),
]);
