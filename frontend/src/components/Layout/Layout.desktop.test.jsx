import { createTheme, ThemeProvider } from '@mui/material';
import { render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { describe, expect, it, vi } from 'vitest';

import Layout from './Layout';
import { ColorModeContext } from '../../contexts/ColorModeContext';

const runtimeState = {
  desktopMode: true,
  features: {
    themes: false,
    chatbot: false,
    tasks: false,
  },
};

vi.mock('../../contexts/RuntimeContext', () => ({
  useRuntime: () => runtimeState,
}));

vi.mock('../PipelineProgressCard', () => ({
  default: () => <div data-testid="pipeline-card" />,
}));

vi.mock('../Settings/TaskSettingsModal', () => ({
  default: () => <div data-testid="task-settings-modal" />,
}));

vi.mock('../Scan/CacheStatus', () => ({
  default: () => <div data-testid="cache-status" />,
}));

vi.mock('../App/DesktopBootstrapBanner', () => ({
  default: () => <div data-testid="bootstrap-banner" />,
}));

vi.mock('./SymbolSearchDialog', () => ({
  default: () => <div data-testid="symbol-search-dialog" />,
}));

function renderLayout() {
  return render(
    <ThemeProvider theme={createTheme()}>
      <ColorModeContext.Provider value={{ toggleColorMode: vi.fn(), mode: 'dark' }}>
        <MemoryRouter initialEntries={['/']}>
          <Layout>
            <div>content</div>
          </Layout>
        </MemoryRouter>
      </ColorModeContext.Provider>
    </ThemeProvider>
  );
}

describe('Layout desktop capability gating', () => {
  it('hides themes, chatbot, task controls, and cache status in desktop core mode', () => {
    renderLayout();

    expect(screen.queryByText('Themes')).not.toBeInTheDocument();
    expect(screen.queryByText('Chatbot')).not.toBeInTheDocument();
    expect(screen.queryByTitle('Scheduled Tasks')).not.toBeInTheDocument();
    expect(screen.queryByTestId('cache-status')).not.toBeInTheDocument();
    expect(screen.getByTestId('bootstrap-banner')).toBeInTheDocument();
  });
});
