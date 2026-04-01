import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';

import ServerLoginScreen from './ServerLoginScreen';
import { renderWithProviders } from '../../test/renderWithProviders';

describe('ServerLoginScreen', () => {
  it('keeps the entered password when login fails', async () => {
    const user = userEvent.setup();
    const onLogin = vi.fn().mockRejectedValue(new Error('Invalid password'));

    renderWithProviders(
      <ServerLoginScreen
        auth={{ configured: true }}
        onLogin={onLogin}
      />
    );

    const passwordInput = screen.getByLabelText('Server password');
    await user.type(passwordInput, 'wrong-pass');
    await user.click(screen.getByRole('button', { name: 'Sign in' }));

    expect(onLogin).toHaveBeenCalledWith('wrong-pass');
    expect(passwordInput).toHaveValue('wrong-pass');
  });
});
