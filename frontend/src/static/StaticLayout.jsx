import { useEffect, useState } from 'react';
import {
  Alert,
  AppBar,
  Box,
  Button,
  Chip,
  Container,
  FormControl,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  MenuItem,
  Select,
  Toolbar,
} from '@mui/material';
import { useLocation } from 'react-router-dom';
import GitHubIcon from '@mui/icons-material/GitHub';
import CandlestickChartIcon from '@mui/icons-material/CandlestickChart';
import LockOutlinedIcon from '@mui/icons-material/LockOutlined';
import ScoreboardIcon from '@mui/icons-material/Scoreboard';
import {
  BrandMark,
  ColorModeButton,
  ColorModeListItem,
  DesktopNav,
  MobileBottomNav,
} from '../components/Layout/AppNav';
import { CORE_NAV_ITEMS, mobileNavOffset, safeGutters, useIsMobileShell } from '../components/Layout/shellNav';
import { useStaticMarket } from './StaticMarketContext';
import {
  getStaticSupportedMarkets,
  getStaticUnavailableMarkets,
  resolveStaticMarketEntry,
  useStaticManifest,
} from './dataClient';
import { useStaticGenerationLifecycle } from './useStaticGenerationLifecycle';
import { REPO_URL, SITE_NAME, SITE_TITLE } from './siteMeta';
import { marketFlag } from '../utils/marketFlags';
import { isStaticOptionsAvailable } from '../features/options/optionsAvailability';

const OPTIONS_NAV_ITEM = { path: '/options', label: 'Options', matchPrefix: true, icon: <CandlestickChartIcon /> };
const SCOREBOARD_NAV_ITEM = { path: '/scoreboard', label: 'Scoreboard', icon: <ScoreboardIcon /> };

// Late once the publish says so, or once its deadline passes with nothing new
// published: a stalled pipeline must not leave a market "current" (#503).
const publicationStatus = (publication, now) => {
  if (!publication) return null;
  const session = publication.session_date;
  if (publication.state === 'unknown' || !session) return { label: 'Status unknown', color: 'default' };
  const deadline = publication.late_after ? Date.parse(publication.late_after) : NaN;
  if (publication.state === 'stale') return { label: `Late · ${session}`, color: 'warning' };
  // No deadline means nothing could turn this late: unknown, never Current forever.
  if (!Number.isFinite(deadline)) return { label: 'Status unknown', color: 'default' };
  if (now > deadline) return { label: `Late · ${session}`, color: 'warning' };
  return { label: `Current · ${session}`, color: 'success' };
};

const useMinuteClock = () => {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const timer = setInterval(() => setNow(Date.now()), 60_000);
    return () => clearInterval(timer);
  }, []);
  return now;
};

function StaticLayout({ children }) {
  const isMobile = useIsMobileShell();
  const manifestQuery = useStaticManifest();
  const { expired } = useStaticGenerationLifecycle();
  const supportedMarkets = getStaticSupportedMarkets(manifestQuery.data);
  const { selectedMarket, setSelectedMarket } = useStaticMarket();
  const marketEntry = resolveStaticMarketEntry(manifestQuery.data, selectedMarket);
  const selectedUnavailable = getStaticUnavailableMarkets(manifestQuery.data).includes(selectedMarket);
  const now = useMinuteClock();
  const status = selectedUnavailable ? null : publicationStatus(marketEntry.publication, now);
  const allNavItems = [...CORE_NAV_ITEMS, OPTIONS_NAV_ITEM, SCOREBOARD_NAV_ITEM];
  const navItems = [
    ...CORE_NAV_ITEMS,
    ...(isStaticOptionsAvailable(marketEntry) ? [OPTIONS_NAV_ITEM] : []),
    ...(marketEntry.assets?.scoreboard?.path ? [SCOREBOARD_NAV_ITEM] : []),
  ];
  const { pathname } = useLocation();
  // Title from every route: /options and /scoreboard still render (an unavailable notice) when their nav items are hidden.
  const pageLabel = allNavItems.find((item) => (
    item.matchPrefix ? pathname.startsWith(item.path) : pathname === item.path
  ))?.label;
  const titleMarket = selectedUnavailable ? selectedMarket : marketEntry.market;
  useEffect(() => {
    // The landing page is the only URL search engines see (hash routes), so it keeps the descriptive title.
    document.title = pathname === '/'
      ? SITE_TITLE
      : [pageLabel, titleMarket, SITE_NAME].filter(Boolean).join(' · ');
  }, [pathname, pageLabel, titleMarket]);

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
      <AppBar position="static">
        <Toolbar
          variant="dense"
          sx={{ ...safeGutters, minHeight: 52, gap: { xs: 1, md: 1.5 }, overflowX: 'auto', '& > *': { flexShrink: 0 } }}
        >
          <BrandMark label={isMobile ? null : SITE_NAME} />
          {isMobile ? (
            <LockOutlinedIcon role="img" aria-label="Read-only" titleAccess="Read-only" sx={{ fontSize: 18, color: 'text.secondary' }} />
          ) : (
            <Chip label="Read-only" size="small" variant="outlined" />
          )}
          <FormControl size="small" sx={{ minWidth: { xs: 0, md: 140 } }}>
            <Select
              value={selectedUnavailable ? selectedMarket : marketEntry.market}
              onChange={(event) => setSelectedMarket(event.target.value)}
              displayEmpty
              // Phones show flag + code ("🇭🇰 HK") so the status chip still fits at 360px.
              renderValue={isMobile ? (market) => `${marketFlag(market)} ${market}`.trim() : undefined}
              sx={{ height: 32, fontSize: '13px', fontWeight: 600, borderRadius: 2 }}
              inputProps={{ 'aria-label': 'Static market selector' }}
            >
              {supportedMarkets.map((market) => {
                const label = manifestQuery.data?.markets?.[market]?.display_name || market;
                const flag = marketFlag(market);
                return (
                  <MenuItem key={market} value={market}>
                    {flag ? `${flag}  ${label}` : label}
                  </MenuItem>
                );
              })}
              {getStaticUnavailableMarkets(manifestQuery.data).map((market) => {
                const flag = marketFlag(market);
                const label = `${market} — unavailable`;
                return (
                  <MenuItem key={market} value={market} disabled>
                    {flag ? `${flag}  ${label}` : label}
                  </MenuItem>
                );
              })}
            </Select>
          </FormControl>
          {status && (
            <Chip
              label={status.label}
              color={status.color}
              variant="outlined"
              size="small"
              role="status"
              aria-label={`${marketEntry.market} data status: ${status.label}`}
            />
          )}
          {!isMobile && (
            <>
              <Box sx={{ flexGrow: 1 }} />
              <DesktopNav items={navItems} />
              <Button
                href={REPO_URL}
                target="_blank"
                rel="noopener noreferrer"
                size="small"
                color="inherit"
                variant="outlined"
                startIcon={<GitHubIcon />}
                aria-label="Star on GitHub (opens in a new tab)"
                // Icon-only until lg: the bar already scrolls sideways around 900-1200px.
                sx={{ textTransform: 'none', borderRadius: 2, minWidth: 0, '& .MuiButton-startIcon': { mr: { md: 0, lg: 1 } } }}
              >
                <Box component="span" sx={{ display: { md: 'none', lg: 'inline' } }}>Star on GitHub</Box>
              </Button>
              <ColorModeButton />
            </>
          )}
        </Toolbar>
      </AppBar>

      <Container maxWidth="xl" sx={{ ...safeGutters, mt: 1.5, mb: 1.5, flex: 1, pb: { xs: mobileNavOffset(8), md: 0 } }}>
        {expired && (
          <Alert
            severity="info"
            sx={{ mb: 1.5 }}
            action={(
              <Button color="inherit" size="small" onClick={() => window.location.reload()}>
                Reload
              </Button>
            )}
          >
            Some data could not be loaded, most likely because newer data has been published. Reload to get the latest.
          </Alert>
        )}
        {selectedUnavailable ? (
          <Alert
            severity="warning"
            action={(
              <Button color="inherit" size="small" onClick={() => manifestQuery.refetch()}>
                Retry
              </Button>
            )}
          >
            {selectedMarket} data is not available in this publish. Retry, or choose another market above.
          </Alert>
        ) : children}
      </Container>

      {isMobile && (
        <MobileBottomNav
          items={navItems}
          extras={(close) => (
            <>
              <ListItem>
                <ListItemIcon><LockOutlinedIcon /></ListItemIcon>
                <ListItemText primary="Read-only" secondary="A published daily snapshot; nothing here can be changed." />
              </ListItem>
              <ListItemButton
                component="a"
                href={REPO_URL}
                target="_blank"
                rel="noopener noreferrer"
                aria-label="Star on GitHub (opens in a new tab)"
                onClick={close}
              >
                <ListItemIcon><GitHubIcon /></ListItemIcon>
                <ListItemText primary="Star on GitHub" secondary="Open source. Self-host it for live scans." />
              </ListItemButton>
              <ColorModeListItem onDone={close} />
            </>
          )}
        />
      )}
    </Box>
  );
}

export default StaticLayout;
