import { Component } from 'react';
import { Box, Button, Typography } from '@mui/material';

// Chrome, Firefox and Safari word a failed dynamic import differently; the
// last is Vite's preload helper failing on a chunk's CSS.
const CHUNK_LOAD_ERROR = /Failed to fetch dynamically imported module|error loading dynamically imported module|Importing a module script failed|Unable to preload CSS/i;
const RELOADED_KEY = 'app-chunk-reload';

// ponytail: one auto-reload per tab session; a second deploy in the same session
// shows the fallback instead. Store a timestamp if that becomes common.
function reloadOnce() {
  try {
    if (sessionStorage.getItem(RELOADED_KEY)) return;
    sessionStorage.setItem(RELOADED_KEY, '1');
  } catch {
    return; // Without the guard a reload could loop, so leave it to the button.
  }
  window.location.reload();
}

// After a deploy, an open tab's lazy pages point at chunks that no longer
// exist. Reloading fetches the new index.html and its new chunk names.
export default class AppErrorBoundary extends Component {
  state = { error: null };

  static getDerivedStateFromError(error) {
    return { error };
  }

  componentDidCatch(error) {
    if (CHUNK_LOAD_ERROR.test(error?.message ?? '')) reloadOnce();
  }

  render() {
    if (!this.state.error) return this.props.children;
    return (
      <Box role="alert" sx={{ p: 4, textAlign: 'center' }}>
        <Typography variant="h6" gutterBottom>Something went wrong</Typography>
        <Button variant="contained" onClick={() => window.location.reload()}>Reload</Button>
      </Box>
    );
  }
}
