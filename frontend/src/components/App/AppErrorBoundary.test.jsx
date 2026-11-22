import { lazy, Suspense } from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import AppErrorBoundary from './AppErrorBoundary';

function Crash() {
  throw new Error('boom');
}

// A lazy page whose chunk was deleted by a deploy.
const MissingChunkPage = () => {
  const Page = lazy(() => Promise.reject(
    new TypeError('Failed to fetch dynamically imported module: https://app/assets/Page-old.js'),
  ));
  return <Suspense fallback={null}><Page /></Suspense>;
};

describe('AppErrorBoundary', () => {
  let reload;

  beforeEach(() => {
    sessionStorage.clear();
    reload = vi.fn();
    vi.stubGlobal('location', { ...window.location, reload });
    vi.spyOn(console, 'error').mockImplementation(() => {});
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it('shows a reload fallback on a render error', () => {
    render(<AppErrorBoundary><Crash /></AppErrorBoundary>);

    expect(screen.getByText('Something went wrong')).toBeInTheDocument();
    expect(reload).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole('button', { name: 'Reload' }));
    expect(reload).toHaveBeenCalledTimes(1);
  });

  it('reloads once on a missing chunk, then shows the fallback', async () => {
    const first = render(<AppErrorBoundary><MissingChunkPage /></AppErrorBoundary>);
    await screen.findByText('Something went wrong');
    expect(reload).toHaveBeenCalledTimes(1);

    // The reloaded tab hits the same missing chunk.
    first.unmount();
    render(<AppErrorBoundary><MissingChunkPage /></AppErrorBoundary>);
    await screen.findByText('Something went wrong');
    expect(reload).toHaveBeenCalledTimes(1);
  });

  it('does not auto-reload when sessionStorage is unavailable', async () => {
    vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => {
      throw new Error('SecurityError');
    });
    render(<AppErrorBoundary><MissingChunkPage /></AppErrorBoundary>);
    await screen.findByText('Something went wrong');
    expect(reload).not.toHaveBeenCalled();
  });
});
