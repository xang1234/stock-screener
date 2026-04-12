import { useContext } from 'react';
import {
  AppBar,
  Box,
  Button,
  Chip,
  Container,
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

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
      <AppBar position="static" sx={{ minHeight: 40 }}>
        <Toolbar variant="dense" sx={{ minHeight: 40 }}>
          <ShowChartIcon sx={{ mr: 0.75, fontSize: 16 }} />
          <Typography variant="subtitle2" component="div" sx={{ fontWeight: 600, letterSpacing: '0.05em', fontSize: '11px' }}>
            STOCK SCANNER DAILY
          </Typography>
          <Chip
            label="Read-only"
            size="small"
            color="info"
            sx={{ ml: 1, height: 18, fontSize: '9px' }}
          />
          <Box sx={{ flexGrow: 1 }} />
          {NAV_ITEMS.map((item) => {
            const isActive = location.pathname === item.path;
            return (
              <Button
                key={item.path}
                color="inherit"
                component={RouterLink}
                to={item.path}
                size="small"
                sx={{
                  backgroundColor: isActive ? 'rgba(255, 255, 255, 0.08)' : 'transparent',
                  borderBottom: isActive ? '2px solid #5C6BC0' : '2px solid transparent',
                  borderRadius: 0,
                  fontWeight: isActive ? 600 : 400,
                  fontSize: '11px',
                  px: 1.25,
                  py: 0.25,
                  minWidth: 0,
                  letterSpacing: '0.03em',
                  '&:hover': {
                    backgroundColor: 'rgba(255, 255, 255, 0.05)',
                  },
                }}
              >
                {item.label}
              </Button>
            );
          })}
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
