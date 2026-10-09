import { useContext } from 'react';
import {
  Alert,
  AppBar,
  Box,
  Button,
  Chip,
  Container,
  FormControl,
  MenuItem,
  Select,
  Tab,
  Tabs,
  Toolbar,
  Typography,
  IconButton,
  useTheme,
} from '@mui/material';
import { Link as RouterLink, useLocation } from 'react-router-dom';
import ShowChartIcon from '@mui/icons-material/ShowChart';
import Brightness4Icon from '@mui/icons-material/Brightness4';
import Brightness7Icon from '@mui/icons-material/Brightness7';
import { ColorModeContext } from '../contexts/ColorModeContext';
import { useStaticMarket } from './StaticMarketContext';
import {
  getStaticSupportedMarkets,
  getStaticUnavailableMarkets,
  resolveStaticMarketEntry,
  useStaticManifest,
} from './dataClient';
import { useStaticGenerationLifecycle } from './useStaticGenerationLifecycle';
import { marketFlag } from '../utils/marketFlags';
import { isStaticOptionsAvailable } from '../features/options/optionsAvailability';

const NAV_ITEMS = [
  { path: '/', label: 'Daily' },
  { path: '/scan', label: 'Scan' },
  { path: '/breadth', label: 'Breadth' },
  { path: '/groups', label: 'Groups' },
];

function StaticLayout({ children }) {
  const location = useLocation();
  const theme = useTheme();
  const colorMode = useContext(ColorModeContext);
  const manifestQuery = useStaticManifest();
  const { expired } = useStaticGenerationLifecycle();
  const supportedMarkets = getStaticSupportedMarkets(manifestQuery.data);
  const { selectedMarket, setSelectedMarket } = useStaticMarket();
  const marketEntry = resolveStaticMarketEntry(manifestQuery.data, selectedMarket);
  const selectedUnavailable = getStaticUnavailableMarkets(manifestQuery.data).includes(selectedMarket);
  const navItems = isStaticOptionsAvailable(marketEntry)
    ? [...NAV_ITEMS, { path: '/options', label: 'Options', matchPrefix: true }]
    : NAV_ITEMS;

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
      <AppBar position="static" sx={{ minHeight: 48 }}>
        <Toolbar variant="dense" sx={{ minHeight: 48, flexWrap: 'wrap', rowGap: 0.5 }}>
          <ShowChartIcon sx={{ mr: 1, fontSize: 20 }} />
          <Typography variant="subtitle1" component="div" sx={{ fontWeight: 600 }}>
            STOCK SCANNER DAILY
          </Typography>
          <Chip
            label="Read-only"
            size="small"
            color="info"
            sx={{ ml: 1.5, height: 22, fontSize: '11px' }}
          />
          <Box sx={{ ml: 1.5, minWidth: 140 }}>
            <FormControl size="small" fullWidth>
              <Select
                value={selectedUnavailable ? selectedMarket : marketEntry.market}
                onChange={(event) => setSelectedMarket(event.target.value)}
                displayEmpty
                sx={{
                  color: 'inherit',
                  backgroundColor: 'rgba(255,255,255,0.12)',
                  height: 30,
                  '& .MuiOutlinedInput-notchedOutline': {
                    borderColor: 'rgba(255,255,255,0.35)',
                  },
                  '& .MuiSvgIcon-root': {
                    color: 'inherit',
                  },
                }}
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
          </Box>
          <Box sx={{ flexGrow: 1 }} />
          <Tabs
            value={navItems.find((item) => (
              item.matchPrefix ? location.pathname.startsWith(item.path) : item.path === location.pathname
            ))?.path || false}
            variant="scrollable"
            scrollButtons="auto"
            allowScrollButtonsMobile
            textColor="inherit"
            TabIndicatorProps={{ sx: { bgcolor: 'common.white' } }}
            sx={{ minHeight: 40, '& .MuiTab-root': { minHeight: 40 } }}
          >
            {navItems.map((item) => (
              <Tab
                key={item.path}
                component={RouterLink}
                to={item.path}
                value={item.path}
                label={item.label}
                sx={{ fontSize: '12px', textTransform: 'none', px: 1.5, py: 0.5 }}
              />
            ))}
          </Tabs>
          <IconButton
            sx={{ ml: 0.5 }}
            onClick={colorMode.toggleColorMode}
            color="inherit"
            title={theme.palette.mode === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
            size="small"
          >
            {theme.palette.mode === 'dark' ? <Brightness7Icon fontSize="small" /> : <Brightness4Icon fontSize="small" />}
          </IconButton>
        </Toolbar>
      </AppBar>

      <Container maxWidth="xl" sx={{ mt: 1.5, mb: 1.5, flex: 1 }}>
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
    </Box>
  );
}

export default StaticLayout;
