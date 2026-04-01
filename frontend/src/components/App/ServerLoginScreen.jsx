import { useState } from 'react';
import {
  Alert,
  Box,
  Button,
  CircularProgress,
  Paper,
  Stack,
  TextField,
  Typography,
} from '@mui/material';

function ServerLoginScreen({
  auth,
  isLoggingIn = false,
  loginError = null,
  onLogin,
}) {
  const [password, setPassword] = useState('');

  const handleSubmit = async (event) => {
    event.preventDefault();
    if (!password.trim() || !onLogin) {
      return;
    }
    try {
      await onLogin(password);
      setPassword('');
    } catch {
      // Leave the entered password in place so the user can correct and retry.
    }
  };

  return (
    <Box
      sx={{
        minHeight: '100vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        px: 2,
        background:
          'radial-gradient(circle at top, rgba(25,118,210,0.18), transparent 42%), linear-gradient(180deg, #0d1726 0%, #10151c 100%)',
      }}
    >
      <Paper
        elevation={8}
        sx={{
          width: '100%',
          maxWidth: 460,
          p: 4,
          borderRadius: 3,
          backgroundColor: 'rgba(15, 23, 42, 0.94)',
          color: '#f8fafc',
        }}
      >
        <Stack spacing={2.5} component="form" onSubmit={handleSubmit}>
          <Box>
            <Typography variant="overline" sx={{ letterSpacing: 2, color: 'rgba(148, 163, 184, 0.9)' }}>
              Server Access
            </Typography>
            <Typography variant="h5" sx={{ fontWeight: 700, mt: 0.5 }}>
              Sign in to Stock Scanner
            </Typography>
            <Typography variant="body2" sx={{ color: 'rgba(226, 232, 240, 0.85)', mt: 1 }}>
              Server mode now requires an authenticated browser session before protected API routes are available.
            </Typography>
          </Box>

          {!auth?.configured && (
            <Alert severity="warning">
              {auth?.message || 'Server authentication is required but not configured.'}
            </Alert>
          )}

          {loginError && (
            <Alert severity="error">
              {loginError}
            </Alert>
          )}

          <TextField
            label="Server password"
            type="password"
            value={password}
            onChange={(event) => setPassword(event.target.value)}
            disabled={isLoggingIn || !auth?.configured}
            autoComplete="current-password"
            fullWidth
            InputLabelProps={{ shrink: true }}
            sx={{
              '& .MuiInputBase-root': {
                color: '#f8fafc',
              },
              '& .MuiOutlinedInput-notchedOutline': {
                borderColor: 'rgba(148, 163, 184, 0.35)',
              },
            }}
          />

          <Button
            type="submit"
            variant="contained"
            disabled={isLoggingIn || !auth?.configured || !password.trim()}
            sx={{ minHeight: 44, fontWeight: 700 }}
          >
            {isLoggingIn ? <CircularProgress size={20} color="inherit" /> : 'Sign in'}
          </Button>
        </Stack>
      </Paper>
    </Box>
  );
}

export default ServerLoginScreen;
