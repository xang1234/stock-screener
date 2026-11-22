import { Component } from 'react';
import { Box, Button, Typography } from '@mui/material';

// Chrome, Firefox and Safari word a failed dynamic import differently; the
// last is Vite's preload helper failing on a chunk's CSS.
const CHUNK_LOAD_ERROR = /Failed to fetch dynamically imported module|error loading dynamically imported module|Importing a module script failed|Unable to preload CSS/i;
const RELOADED_KEY = 'app-chunk-reload';

// A reload that still hits a missing chunk within this window is a loop, not a
// new deploy, so the fallback shows instead.
const RELOAD_LOOP_WINDOW_MS = 30000;

function reloadOnce() {
  try {
    const last = Number(sessionStorage.getItem(RELOADED_KEY));
    if (last && Date.now() - last < RELOAD_LOOP_WINDOW_MS) return;
    sessionStorage.setItem(RELOADED_KEY, String(Date.now()));
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

  // The router sits inside this boundary, so Back is the only navigation left
  // once it shows; clear the error so the previous page renders again.
  componentDidMount() {
    window.addEventListener('popstate', this.reset);
  }

  componentWillUnmount() {
    window.removeEventListener('popstate', this.reset);
  }

  reset = () => {
    if (this.state.error) this.setState({ error: null });
  };

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
