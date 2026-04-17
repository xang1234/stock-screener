import { useContext, useEffect, useState } from 'react';
import {
  AppBar,
  Box,
  Button,
  Container,
  Drawer,
  Fab,
  InputBase,
  ListItemText,
  Menu,
  MenuItem,
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
import PersonIcon from '@mui/icons-material/Person';
import MonitorHeartIcon from '@mui/icons-material/MonitorHeart';
import SmartToyIcon from '@mui/icons-material/SmartToy';
import { ColorModeContext } from '../../contexts/ColorModeContext';
import { AssistantChat } from '../AssistantChat';
import PipelineProgressCard from '../PipelineProgressCard';
import TaskSettingsModal from '../Settings/TaskSettingsModal';
import CacheStatus from '../Scan/CacheStatus';
import { AssistantChatProvider } from '../../contexts/AssistantChatContext';
import { useRuntime } from '../../contexts/RuntimeContext';
import { useStrategyProfile } from '../../contexts/StrategyProfileContext';

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
      placeholder="TICKER"
      size="small"
      inputProps={{ 'aria-label': 'Go to ticker' }}
      sx={{
        px: 1.5,
        py: 0.25,
        width: 150,
        fontSize: '15px',
        backgroundColor: 'rgba(255,255,255,0.15)',
        borderRadius: 1,
        color: 'inherit',
        '& input::placeholder': {
          color: 'rgba(255,255,255,0.7)',
          opacity: 1,
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
  const { auth, features, isLoggingOut, logout } = useRuntime();
  const { activeProfile, activeProfileDetail, profiles, setActiveProfile } = useStrategyProfile();
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [assistantOpen, setAssistantOpen] = useState(false);
  const [cacheStatusEnabled, setCacheStatusEnabled] = useState(false);

  useEffect(() => {
    setCacheStatusEnabled(false);
    const timer = window.setTimeout(() => {
      setCacheStatusEnabled(true);
    }, 2500);
    return () => window.clearTimeout(timer);
  }, [location.pathname]);

  const assistantAvailable = features.chatbot && (!auth?.required || auth?.authenticated);

  const navItems = [
    { path: '/', label: 'Daily' },
    { path: '/scan', label: 'Scan' },
    { path: '/breadth', label: 'Breadth' },
    { path: '/groups', label: 'Groups' },
    { path: '/digest', label: 'Digest' },
    { path: '/validation', label: 'Backtest' },
    ...(features.themes ? [{ path: '/themes', label: 'Themes' }] : []),
    ...(features.chatbot ? [{ path: '/chatbot', label: 'Assistant' }] : []),
  ];

  const [profileMenuAnchor, setProfileMenuAnchor] = useState(null);
  const profileMenuOpen = Boolean(profileMenuAnchor);
  const profileOptions = profiles.length
    ? profiles
    : [{ profile: activeProfile, label: activeProfileDetail?.label || activeProfile }];
  const activeProfileLabel = profileOptions.find((p) => p.profile === activeProfile)?.label
    || activeProfileDetail?.label
    || activeProfile;

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
      <AppBar position="static" sx={{ minHeight: 48 }}>
        <Toolbar variant="dense" sx={{ minHeight: 48 }}>
          <ShowChartIcon sx={{ mr: 1, fontSize: 20 }} />
          <Typography variant="subtitle1" component="div" sx={{ fontWeight: 600 }}>
            STOCK SCANNER
          </Typography>
          <Box sx={{ ml: 2 }}>
            <CacheStatus enabled={cacheStatusEnabled} />
          </Box>
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
          <IconButton
            sx={{ ml: 1 }}
            onClick={(event) => setProfileMenuAnchor(event.currentTarget)}
            color="inherit"
            title={`Profile: ${activeProfileLabel}`}
            aria-label="Select strategy profile"
            aria-haspopup="menu"
            aria-expanded={profileMenuOpen}
            size="small"
          >
            <PersonIcon fontSize="small" />
          </IconButton>
          <Menu
            anchorEl={profileMenuAnchor}
            open={profileMenuOpen}
            onClose={() => setProfileMenuAnchor(null)}
            anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
            transformOrigin={{ vertical: 'top', horizontal: 'right' }}
          >
            {profileOptions.map((profile) => (
              <MenuItem
                key={profile.profile}
                selected={profile.profile === activeProfile}
                onClick={() => {
                  setActiveProfile(profile.profile);
                  setProfileMenuAnchor(null);
                }}
              >
                <ListItemText primary={profile.label} />
              </MenuItem>
            ))}
          </Menu>
          {features.tasks && (
            <IconButton
              sx={{ ml: 0.5 }}
              onClick={() => setSettingsOpen(true)}
              color="inherit"
              title="Scheduled Tasks"
              size="small"
            >
              <SettingsIcon fontSize="small" />
            </IconButton>
          )}
          <IconButton
            sx={{ ml: 0.5 }}
            component={RouterLink}
            to="/operations"
            color="inherit"
            aria-label="Operations / telemetry"
            title="Operations / telemetry"
            size="small"
          >
            <MonitorHeartIcon fontSize="small" />
          </IconButton>
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
        {children}
      </Container>

      {features.themes && <PipelineProgressCard />}

      {features.tasks && <TaskSettingsModal open={settingsOpen} onClose={() => setSettingsOpen(false)} />}

      {assistantAvailable && location.pathname !== '/chatbot' && (
        <>
          <Fab
            color="secondary"
            aria-label="Open assistant"
            onClick={() => setAssistantOpen(true)}
            sx={{
              position: 'fixed',
              right: 24,
              bottom: 24,
              zIndex: theme.zIndex.drawer - 1,
            }}
          >
            <SmartToyIcon />
          </Fab>

          <Drawer
            anchor="right"
            open={assistantOpen}
            onClose={() => setAssistantOpen(false)}
            PaperProps={{
              sx: {
                width: { xs: '100%', sm: 440 },
              },
            }}
          >
            {assistantOpen && (
              <AssistantChatProvider>
                <AssistantChat mode="drawer" onClose={() => setAssistantOpen(false)} />
              </AssistantChatProvider>
            )}
          </Drawer>
        </>
      )}
    </Box>
  );
}

export default Layout;
