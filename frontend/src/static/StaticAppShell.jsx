import { HashRouter as Router, Navigate, Route, Routes } from 'react-router-dom';
import StaticLayout from './StaticLayout';
import StaticHomePage from './pages/StaticHomePage';
import StaticScanPage from './pages/StaticScanPage';
import StaticBreadthPage from './pages/StaticBreadthPage';
import StaticGroupsPage from './pages/StaticGroupsPage';
import { StaticMarketProvider } from './StaticMarketContext';
import { getStaticSupportedMarkets, useStaticManifest } from './dataClient';

function StaticAppContent() {
  const manifestQuery = useStaticManifest();
  const supportedMarkets = getStaticSupportedMarkets(manifestQuery.data);
  const defaultMarket = manifestQuery.data?.default_market || supportedMarkets[0] || 'US';

  return (
    <StaticMarketProvider
      supportedMarkets={supportedMarkets}
      defaultMarket={defaultMarket}
    >
      <StaticLayout>
        <Routes>
          <Route path="/" element={<StaticHomePage />} />
          <Route path="/scan" element={<StaticScanPage />} />
          <Route path="/breadth" element={<StaticBreadthPage />} />
          <Route path="/groups" element={<StaticGroupsPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </StaticLayout>
    </StaticMarketProvider>
  );
}

function StaticAppShell() {
  return (
    <Router>
      <StaticAppContent />
    </Router>
  );
}

export default StaticAppShell;
