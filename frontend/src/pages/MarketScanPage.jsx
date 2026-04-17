/**
 * Market Scan page with vertical side tabs for different scan views.
 */
import { useEffect, useMemo, useState } from 'react';
import { Box, Tabs, Tab, Paper } from '@mui/material';
import KeyMarketsTab from '../components/MarketScan/KeyMarketsTab';
import DailyMarketSnapshotTab from '../components/MarketScan/DailyMarketSnapshotTab';
import DigestTab from '../components/MarketScan/DigestTab';
import ThemesTab from '../components/MarketScan/ThemesTab';
import WatchlistsTab from '../components/MarketScan/WatchlistsTab';
import StockbeeMmTab from '../components/MarketScan/StockbeeMmTab';
import { useRuntime } from '../contexts/RuntimeContext';

function MarketScanPage() {
  const { features } = useRuntime();
  const [selectedTab, setSelectedTab] = useState(0);
  const subTabs = useMemo(() => ([
    { id: 'key_markets', label: 'Key Markets', render: () => <KeyMarketsTab /> },
    { id: 'daily_snapshot', label: 'Daily Snapshot', render: () => <DailyMarketSnapshotTab /> },
    { id: 'digest', label: 'Digest', render: () => <DigestTab /> },
    ...(features.themes
      ? [{ id: 'themes', label: 'Themes', render: () => <ThemesTab /> }]
      : []),
    { id: 'watchlists', label: 'Watchlists', render: () => <WatchlistsTab /> },
    { id: 'stockbee_mm', label: 'Stockbee MM', render: () => <StockbeeMmTab /> },
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
