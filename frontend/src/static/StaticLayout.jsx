import { useContext } from 'react';
import {
  AppBar,
  Box,
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
import { getStaticSupportedMarkets, resolveStaticMarketEntry, useStaticManifest } from './dataClient';
import { marketFlag } from '../utils/marketFlags';

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
  const supportedMarkets = getStaticSupportedMarkets(manifestQuery.data);
  const { selectedMarket, setSelectedMarket } = useStaticMarket();
  const marketEntry = resolveStaticMarketEntry(manifestQuery.data, selectedMarket);

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
                value={marketEntry.market}
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
              </Select>
            </FormControl>
          </Box>
          <Box sx={{ flexGrow: 1 }} />
          <Tabs
            value={NAV_ITEMS.some((item) => item.path === location.pathname) ? location.pathname : false}
            variant="scrollable"
            scrollButtons="auto"
            allowScrollButtonsMobile
            textColor="inherit"
            TabIndicatorProps={{ sx: { bgcolor: 'common.white' } }}
            sx={{ minHeight: 40, '& .MuiTab-root': { minHeight: 40 } }}
          >
            {NAV_ITEMS.map((item) => (
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
        {children}
      </Container>
    </Box>
  );
}

export default StaticLayout;
