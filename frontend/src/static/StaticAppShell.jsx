import { lazy, Suspense } from 'react';
import { HashRouter as Router, Navigate, Route, Routes } from 'react-router-dom';
import StaticLayout from './StaticLayout';
import { PageSkeleton } from '../components/common/LoadingSkeletons';
import StaticHomePage from './pages/StaticHomePage';
import StaticScanPage from './pages/StaticScanPage';
import StaticBreadthPage from './pages/StaticBreadthPage';
import StaticGroupsPage from './pages/StaticGroupsPage';
import { StaticMarketProvider } from './StaticMarketContext';
import {
  getStaticSupportedMarkets,
  getStaticUnavailableMarkets,
  useStaticManifest,
} from './dataClient';

const StaticOptionsPage = lazy(() => import('./pages/StaticOptionsPage'));
const StaticOptionsSymbolPage = lazy(() => import('./pages/StaticOptionsSymbolPage'));
const StaticScoreboardPage = lazy(() => import('./pages/StaticScoreboardPage'));

function StaticAppContent() {
  const manifestQuery = useStaticManifest();
  const supportedMarkets = getStaticSupportedMarkets(manifestQuery.data);
  const defaultMarket = manifestQuery.data?.default_market || supportedMarkets[0] || 'US';

  return (
    <StaticMarketProvider
      supportedMarkets={supportedMarkets}
      unavailableMarkets={getStaticUnavailableMarkets(manifestQuery.data)}
      defaultMarket={defaultMarket}
    >
      <StaticLayout>
        {/* Lazy pages (Options) load inside the layout, keeping the app bar. */}
        <Suspense fallback={<PageSkeleton />}>
          <Routes>
            <Route path="/" element={<StaticHomePage />} />
            <Route path="/scan" element={<StaticScanPage />} />
            <Route path="/breadth" element={<StaticBreadthPage />} />
            <Route path="/groups" element={<StaticGroupsPage />} />
            <Route path="/options" element={<StaticOptionsPage />} />
            <Route path="/options/:symbol" element={<StaticOptionsSymbolPage />} />
            <Route path="/scoreboard" element={<StaticScoreboardPage />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </Suspense>
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
