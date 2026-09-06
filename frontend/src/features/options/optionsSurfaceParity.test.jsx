import { cleanup } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import { renderWithProviders } from '../../test/renderWithProviders';
import OptionsCommandCenterView from './OptionsCommandCenterView';
import { normalizeOptionsCommandCenter } from './optionsContract';
import { commandCenterFixture } from './__fixtures__/optionsResponses';

describe('options surface parity', () => {
  it('renders equivalent live and static contracts through the same presentation', () => {
    const live = normalizeOptionsCommandCenter(structuredClone(commandCenterFixture));
    const staticPayload = normalizeOptionsCommandCenter(structuredClone(commandCenterFixture), {
      expectedRunId: commandCenterFixture.run_id,
    });
    const liveView = renderWithProviders(
      <OptionsCommandCenterView data={live} onOpenSymbol={() => {}} />,
    );
    const liveText = liveView.container.textContent;
    cleanup();
    const staticView = renderWithProviders(
      <OptionsCommandCenterView data={staticPayload} onOpenSymbol={() => {}} />,
    );

    expect(staticView.container.textContent).toBe(liveText);
    expect(staticView.container.textContent).not.toMatch(/refresh|inflow|bullish|bearish/i);
  });
});
