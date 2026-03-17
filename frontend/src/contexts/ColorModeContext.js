import { createContext } from 'react';

export const ColorModeContext = createContext({
  toggleColorMode: () => {},
  mode: 'dark',
});
