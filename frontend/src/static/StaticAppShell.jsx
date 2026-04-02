import { HashRouter as Router, Navigate, Route, Routes } from 'react-router-dom';
import StaticLayout from './StaticLayout';
import StaticHomePage from './pages/StaticHomePage';
import StaticScanPage from './pages/StaticScanPage';
import StaticBreadthPage from './pages/StaticBreadthPage';
import StaticGroupsPage from './pages/StaticGroupsPage';

function StaticAppShell() {
  return (
    <Router>
      <StaticLayout>
        <Routes>
          <Route path="/" element={<StaticHomePage />} />
          <Route path="/scan" element={<StaticScanPage />} />
          <Route path="/breadth" element={<StaticBreadthPage />} />
          <Route path="/groups" element={<StaticGroupsPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </StaticLayout>
    </Router>
  );
}

export default StaticAppShell;
