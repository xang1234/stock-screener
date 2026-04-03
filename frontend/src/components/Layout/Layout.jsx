import { useContext, useState } from 'react';
import {
  AppBar,
  Box,
  Button,
  Container,
  InputBase,
  Toolbar,
  Typography,
  IconButton,
  useTheme,
} from '@mui/material';
import { Link as RouterLink, useLocation, useNavigate } from 'react-router-dom';
import ShowChartIcon from '@mui/icons-material/ShowChart';
import Brightness4Icon from '@mui/icons-material/Brightness4';
import Brightness7Icon from '@mui/icons-material/Brightness7';
import SettingsIcon from '@mui/icons-material/Settings';
import { ColorModeContext } from '../../contexts/ColorModeContext';
import PipelineProgressCard from '../PipelineProgressCard';
import TaskSettingsModal from '../Settings/TaskSettingsModal';
import CacheStatus from '../Scan/CacheStatus';
import DesktopBootstrapBanner from '../App/DesktopBootstrapBanner';
import { useRuntime } from '../../contexts/RuntimeContext';

function TickerSearch() {
  const navigate = useNavigate();
  const [tickerInput, setTickerInput] = useState('');

  const handleTickerSubmit = (event) => {
    if (event.key !== 'Enter') return;
    const ticker = tickerInput.trim().toUpperCase();
    if (!ticker) return;
    navigate(`/stocks/${encodeURIComponent(ticker)}`);
    setTickerInput('');
    event.target.blur();
  };

  return (
    <InputBase
      value={tickerInput}
      onChange={(e) => setTickerInput(e.target.value)}
      onKeyDown={handleTickerSubmit}
      placeholder="TICKER ↵"
      size="small"
      inputProps={{ 'aria-label': 'Go to ticker' }}
      sx={{
        px: 1.5,
        py: 0.25,
        width: 300,
        fontSize: '15px',
        backgroundColor: 'rgba(255,255,255,0.15)',
        borderRadius: 1,
        color: 'inherit',
        '& input::placeholder': {
          color: 'rgba(255,255,255,0.7)',
          opacity: 1,
          fontStyle: 'italic',
          letterSpacing: '1px',
        },
      }}
    />
  );
}

function Layout({ children }) {
  const location = useLocation();
  const theme = useTheme();
  const colorMode = useContext(ColorModeContext);
  const { auth, desktopMode, features, isLoggingOut, logout } = useRuntime();
  const [settingsOpen, setSettingsOpen] = useState(false);

  const navItems = [
    { path: '/', label: 'Routine' },
    { path: '/scan', label: 'Bulk Scanner' },
    { path: '/breadth', label: 'Market Breadth' },
    { path: '/groups', label: 'Group Rankings' },
    ...(features.themes ? [{ path: '/themes', label: 'Themes' }] : []),
    ...(features.chatbot ? [{ path: '/chatbot', label: 'Chatbot' }] : []),
  ];

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
      <AppBar position="static" sx={{ minHeight: 48 }}>
        <Toolbar variant="dense" sx={{ minHeight: 48 }}>
          <ShowChartIcon sx={{ mr: 1, fontSize: 20 }} />
          <Typography variant="subtitle1" component="div" sx={{ fontWeight: 600 }}>
            STOCK SCANNER
          </Typography>
          {!desktopMode && (
            <Box sx={{ ml: 2 }}>
              <CacheStatus />
            </Box>
          )}
          <Box sx={{ flexGrow: 1 }} />
          <TickerSearch />
          <Box sx={{ flexGrow: 1 }} />
          {navItems.map((item) => {
            const isActive = location.pathname === item.path;
            return (
              <Button
                key={item.path}
                color="inherit"
                component={RouterLink}
                to={item.path}
                size="small"
                sx={{
                  backgroundColor: isActive ? 'rgba(255, 255, 255, 0.15)' : 'transparent',
                  borderBottom: isActive ? '2px solid white' : '2px solid transparent',
                  borderRadius: 0,
                  fontWeight: isActive ? 600 : 400,
                  fontSize: '12px',
                  px: 1.5,
                  py: 0.5,
                  '&:hover': {
                    backgroundColor: 'rgba(255, 255, 255, 0.25)',
                  },
                }}
              >
                {item.label}
              </Button>
            );
          })}
          {features.tasks && (
            <IconButton
              sx={{ ml: 1 }}
              onClick={() => setSettingsOpen(true)}
              color="inherit"
              title="Scheduled Tasks"
              size="small"
            >
              <SettingsIcon fontSize="small" />
            </IconButton>
          )}
          {auth?.required && auth?.authenticated && (
            <Button
              color="inherit"
              size="small"
              onClick={() => logout()}
              disabled={isLoggingOut}
              sx={{ ml: 1, fontSize: '12px' }}
            >
              Sign out
            </Button>
          )}
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

      <Container maxWidth="xl" sx={{ mt: 1, mb: 1, flex: 1 }}>
        <DesktopBootstrapBanner />
        {children}
      </Container>

      {features.themes && <PipelineProgressCard />}

      {features.tasks && <TaskSettingsModal open={settingsOpen} onClose={() => setSettingsOpen(false)} />}
    </Box>
  );
}

export default Layout;
