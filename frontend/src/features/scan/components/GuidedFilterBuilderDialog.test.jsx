import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { vi } from 'vitest';

import { renderWithProviders } from '../../../test/renderWithProviders';
import { createEmptyExpression } from '../filterExpressionModel';
import { EXPRESSION_LIMITS } from '../scanFilterFields';
import GuidedFilterBuilderDialog from './GuidedFilterBuilderDialog';

describe('GuidedFilterBuilderDialog', () => {
  it('keeps draft changes private until a valid expression is applied', async () => {
    const user = userEvent.setup();
    const onApply = vi.fn();
    const onClose = vi.fn();
    renderWithProviders(
      <GuidedFilterBuilderDialog
        open
        expression={createEmptyExpression()}
        onClose={onClose}
        onApply={onApply}
      />,
    );

    await user.click(screen.getByRole('button', { name: /add named setup/i }));
    await user.click(screen.getByRole('button', { name: /apply logic/i }));
    expect(onApply).not.toHaveBeenCalled();
    expect(screen.getByText(/composite score needs a minimum or maximum/i)).toBeInTheDocument();

    await user.type(screen.getByLabelText('Minimum'), '80');
    await user.click(screen.getByRole('button', { name: /apply logic/i }));

    expect(onApply).toHaveBeenCalledTimes(1);
    expect(onApply.mock.calls[0][0].groups[0]).toMatchObject({
      name: 'Setup 1',
      match: 'all',
      enabled: true,
      conditions: [
        { kind: 'range', field: 'composite_score', min: 80, max: null },
      ],
    });
    expect(onClose).not.toHaveBeenCalled();
  });

  it('cancels without applying the draft', async () => {
    const user = userEvent.setup();
    const onApply = vi.fn();
    const onClose = vi.fn();
    renderWithProviders(
      <GuidedFilterBuilderDialog
        open
        expression={createEmptyExpression()}
        onClose={onClose}
        onApply={onApply}
      />,
    );

    await user.click(screen.getByRole('button', { name: /cancel/i }));
    expect(onClose).toHaveBeenCalledTimes(1);
    expect(onApply).not.toHaveBeenCalled();
  });

  it('keeps a contradictory range in the dialog and explains the error', async () => {
    const user = userEvent.setup();
    const onApply = vi.fn();
    renderWithProviders(
      <GuidedFilterBuilderDialog
        open
        expression={createEmptyExpression()}
        onClose={vi.fn()}
        onApply={onApply}
      />,
    );

    await user.click(screen.getByRole('button', { name: /add named setup/i }));
    await user.type(screen.getByLabelText('Minimum'), '100');
    await user.type(screen.getByLabelText('Maximum'), '10');
    await user.click(screen.getByRole('button', { name: /apply logic/i }));

    expect(onApply).not.toHaveBeenCalled();
    expect(screen.getByText(/composite score minimum cannot exceed maximum/i)).toBeInTheDocument();
  });

  it('keeps an oversized categorical rule in the dialog and explains the limit', async () => {
    const user = userEvent.setup();
    const onApply = vi.fn();
    const expression = createEmptyExpression();
    expression.groups = [{
      id: 'industry',
      name: 'Industry setup',
      match: 'all',
      enabled: true,
      conditions: [{
        kind: 'categorical',
        field: 'ibd_industry_group',
        values: Array.from(
          { length: EXPRESSION_LIMITS.maxCategoricalValues + 1 },
          (_, index) => `Industry ${index + 1}`,
        ),
        mode: 'include',
      }],
    }];
    renderWithProviders(
      <GuidedFilterBuilderDialog
        open
        expression={expression}
        onClose={vi.fn()}
        onApply={onApply}
      />,
    );

    await user.click(screen.getByRole('button', { name: /apply logic/i }));

    expect(onApply).not.toHaveBeenCalled();
    expect(screen.getByText(
      `IBD industry allows at most ${EXPRESSION_LIMITS.maxCategoricalValues} values.`,
    )).toBeInTheDocument();
  });
});
