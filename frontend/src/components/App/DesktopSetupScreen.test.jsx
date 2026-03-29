import { createTheme, ThemeProvider } from '@mui/material';
import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import DesktopSetupScreen from './DesktopSetupScreen';

const startSetup = vi.fn();

const runtimeState = {
  isStartingSetup: false,
  setup: {
    status: 'idle',
    message: 'Desktop setup has not started',
    percent: 0,
    steps: [],
  },
  setupFailed: false,
  setupOptions: [
    {
      id: 'quick_start',
      label: 'Quick Start',
      description: 'Install starter data, open immediately, and continue updates in the background.',
      recommended: true,
    },
    {
      id: 'download_latest',
      label: 'Download Latest Before Opening',
      description: 'Install starter data, then wait for the first core local refresh to complete.',
      recommended: false,
    },
  ],
  setupRunning: false,
  startSetup,
};

vi.mock('../../contexts/RuntimeContext', () => ({
  useRuntime: () => runtimeState,
}));

describe('DesktopSetupScreen', () => {
  it('renders both setup options and delegates button clicks', () => {
    render(
      <ThemeProvider theme={createTheme()}>
        <DesktopSetupScreen />
      </ThemeProvider>
    );

    expect(screen.getAllByText(/Quick Start/i).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/Download Latest Before Opening/i).length).toBeGreaterThan(0);

    fireEvent.click(screen.getAllByRole('button', { name: /Quick Start/i })[0]);
    expect(startSetup).toHaveBeenCalledWith('quick_start', false);
  });
});
