/**
 * Market Scan page with vertical side tabs for different scan views.
 */
import { lazy, Suspense, useEffect, useMemo, useState } from 'react';
import { Box, CircularProgress, Tabs, Tab, Paper } from '@mui/material';
import DailyMarketSnapshotTab from '../components/MarketScan/DailyMarketSnapshotTab';
import { useRuntime } from '../contexts/RuntimeContext';

const KeyMarketsTab = lazy(() => import('../components/MarketScan/KeyMarketsTab'));
const ThemesTab = lazy(() => import('../components/MarketScan/ThemesTab'));
const WatchlistsTab = lazy(() => import('../components/MarketScan/WatchlistsTab'));
const StockbeeMmTab = lazy(() => import('../components/MarketScan/StockbeeMmTab'));

function LazyTabFallback() {
  return (
    <Box display="flex" justifyContent="center" alignItems="center" height="100%">
      <CircularProgress size={24} />
    </Box>
  );
}

function renderLazyTab(TabComponent) {
  return (
    <Suspense fallback={<LazyTabFallback />}>
      <TabComponent />
    </Suspense>
  );
}

function MarketScanPage() {
  const { features } = useRuntime();
  const [selectedTab, setSelectedTab] = useState(0);
  // Daily Snapshot first: it renders from one cached payload, while Key
  // Markets mounts the TradingView widget (~180 external requests) — only
  // pay that cost when the user opens that tab.
  const subTabs = useMemo(() => ([
    { id: 'daily_snapshot', label: 'Daily Snapshot', render: () => <DailyMarketSnapshotTab /> },
    { id: 'key_markets', label: 'Key Markets', render: () => renderLazyTab(KeyMarketsTab) },
    ...(features.themes
      ? [{ id: 'themes', label: 'Themes', render: () => renderLazyTab(ThemesTab) }]
      : []),
    { id: 'watchlists', label: 'Watchlists', render: () => renderLazyTab(WatchlistsTab) },
    { id: 'stockbee_mm', label: 'Stockbee MM', render: () => renderLazyTab(StockbeeMmTab) },
  ]), [features.themes]);

  useEffect(() => {
    if (selectedTab >= subTabs.length) {
      setSelectedTab(0);
    }
  }, [selectedTab, subTabs.length]);

  return (
    <Box sx={{ display: 'flex', height: 'calc(100vh - 70px)' }}>
      {/* Left sidebar with vertical tabs */}
      <Paper
        elevation={1}
        sx={{
          width: 120,
          flexShrink: 0,
          borderRight: 1,
          borderColor: 'divider',
        }}
      >
        <Tabs
          orientation="vertical"
          value={selectedTab}
          onChange={(e, v) => setSelectedTab(v)}
          sx={{
            '& .MuiTab-root': {
              alignItems: 'flex-start',
              textAlign: 'left',
              minHeight: 36,
              px: 1.5,
              fontSize: '12px',
            },
          }}
        >
          {subTabs.map((tab) => (
            <Tab
              key={tab.id}
              label={tab.label}
            />
          ))}
        </Tabs>
      </Paper>

      {/* Main content area */}
      <Box sx={{ flex: 1, overflow: 'hidden', p: 1 }}>
        {subTabs[selectedTab]?.render()}
      </Box>
    </Box>
  );
}

export default MarketScanPage;
