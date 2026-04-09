import { fireEvent, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { renderWithProviders } from '../../test/renderWithProviders';
import AssistantMessageBubble from './AssistantMessageBubble';

describe('AssistantMessageBubble', () => {
  it('renders inline citations and expandable source details', () => {
    renderWithProviders(
      <AssistantMessageBubble
        message={{
          role: 'assistant',
          content: 'Internal breadth improved [1].',
          created_at: '2026-04-09T00:00:00Z',
          source_references: [
            {
              reference_number: 1,
              type: 'internal',
              title: 'Breadth snapshot',
              url: '/breadth',
              section: 'As of 2026-04-09',
              snippet: 'Breadth ratio remains above 1.5.',
            },
          ],
        }}
      />,
    );

    expect(screen.getByRole('link', { name: '[1]' })).toHaveAttribute('href', '/breadth');

    fireEvent.click(screen.getByText('Sources (1)'));

    expect(screen.getByText('Breadth snapshot')).toBeInTheDocument();
    expect(screen.getByText(/As of 2026-04-09/)).toBeInTheDocument();
    expect(screen.getByText(/Breadth ratio remains above 1.5/)).toBeInTheDocument();
  });

  it('ignores unnumbered references when wiring inline numeric citations', () => {
    renderWithProviders(
      <AssistantMessageBubble
        message={{
          role: 'assistant',
          content: 'Use [1] for the internal snapshot.',
          created_at: '2026-04-09T00:00:00Z',
          source_references: [
            {
              type: 'web',
              title: 'Reuters',
              url: 'https://example.com/reuters',
            },
            {
              reference_number: 1,
              type: 'internal',
              title: 'Feature run snapshot',
              url: '/stocks/NVDA',
            },
          ],
        }}
      />,
    );

    expect(screen.getByRole('link', { name: '[1]' })).toHaveAttribute('href', '/stocks/NVDA');
  });
});
