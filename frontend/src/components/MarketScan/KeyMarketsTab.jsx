/**
 * Key Markets tab displaying TradingView charts with keyboard navigation.
 */
import { useState, useEffect, useCallback } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Box, Typography, IconButton, CircularProgress } from '@mui/material';
import SettingsIcon from '@mui/icons-material/Settings';
import TradingViewChart from './TradingViewChart';
import SymbolNavigator from './SymbolNavigator';
import WatchlistManager from './WatchlistManager';
import { getWatchlist } from '../../api/marketScan';

function KeyMarketsTab() {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [managerOpen, setManagerOpen] = useState(false);

  // Fetch watchlist from backend
  const { data: watchlist, isLoading, refetch } = useQuery({
    queryKey: ['marketScan', 'key_markets'],
    queryFn: () => getWatchlist('key_markets'),
  });

  const symbols = watchlist?.symbols || [];
  const currentSymbol = symbols[currentIndex];

  // Navigation functions
  const goNext = useCallback(() => {
    if (symbols.length > 0) {
      setCurrentIndex((prev) => (prev + 1) % symbols.length);
    }
  }, [symbols.length]);

  const goPrevious = useCallback(() => {
    if (symbols.length > 0) {
      setCurrentIndex((prev) => (prev - 1 + symbols.length) % symbols.length);
    }
  }, [symbols.length]);

  // Keyboard navigation
  useEffect(() => {
    const handleKeyDown = (e) => {
      // Ignore if typing in input
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

      if (e.code === 'Space') {
        e.preventDefault();
        if (e.shiftKey) {
          goPrevious();
        } else {
          goNext();
        }
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [goNext, goPrevious]);

  // Reset index if it's out of bounds after list changes
  useEffect(() => {
    if (currentIndex >= symbols.length && symbols.length > 0) {
      setCurrentIndex(0);
    }
  }, [symbols.length, currentIndex]);

  if (isLoading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" height="100%">
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* Compact header with navigation */}
      <Box
        sx={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          mb: 0.5,
          pb: 0.5,
          borderBottom: 1,
          borderColor: 'divider',
        }}
      >
        <Box display="flex" alignItems="center" gap={1}>
          <Typography variant="h6" fontWeight={700} color="primary.main">
            {currentSymbol?.symbol || 'No symbols'}
          </Typography>
          {currentSymbol?.display_name && (
            <Typography variant="body2" color="text.secondary">
              {currentSymbol.display_name}
            </Typography>
          )}
          <Typography variant="caption" color="text.disabled" sx={{ ml: 1 }}>
            Space / Shift+Space
          </Typography>
        </Box>

        <Box display="flex" alignItems="center" gap={1}>
          <SymbolNavigator
            currentIndex={currentIndex}
            total={symbols.length}
            onPrevious={goPrevious}
            onNext={goNext}
            onSelectIndex={setCurrentIndex}
            symbols={symbols}
          />
          <IconButton
            onClick={() => setManagerOpen(true)}
            title="Manage watchlist"
            size="small"
          >
            <SettingsIcon fontSize="small" />
          </IconButton>
        </Box>
      </Box>

      {/* TradingView Chart */}
      <Box sx={{ flex: 1, minHeight: 0 }}>
        {currentSymbol ? (
          <TradingViewChart
            key={currentSymbol.symbol}
            symbol={currentSymbol.symbol}
            interval="D"
            range="4M"
            hideSidebar={true}
          />
        ) : (
          <Box
            display="flex"
            justifyContent="center"
            alignItems="center"
            height="100%"
          >
            <Typography color="text.secondary">
              No symbols in watchlist. Click the settings icon to add some.
            </Typography>
          </Box>
        )}
      </Box>

      {/* Watchlist Manager Modal */}
      <WatchlistManager
        open={managerOpen}
        onClose={() => setManagerOpen(false)}
        listName="key_markets"
        onUpdate={refetch}
      />
    </Box>
  );
}

export default KeyMarketsTab;
