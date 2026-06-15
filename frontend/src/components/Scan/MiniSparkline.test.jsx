import { render } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import { MiniPriceSparkline } from './MiniSparkline';

describe('MiniPriceSparkline', () => {
  it('rerenders when interior data points change', () => {
    const { container, rerender } = render(
      <MiniPriceSparkline
        data={[1, 2, 3, 1]}
        trend={1}
        width={120}
        height={28}
      />
    );
    const initialPath = container.querySelector('path[stroke]')?.getAttribute('d');

    rerender(
      <MiniPriceSparkline
        data={[1, 2.5, 3, 1]}
        trend={1}
        width={120}
        height={28}
      />
    );

    expect(container.querySelector('path[stroke]')?.getAttribute('d')).not.toEqual(initialPath);
  });
});
