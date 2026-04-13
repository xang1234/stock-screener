import { describe, expect, it } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/react';

import TranslatedText from './TranslatedText';

describe('TranslatedText', () => {
  it('renders plain text with no chip when source language is missing', () => {
    render(<TranslatedText originalText="Hello world" />);

    expect(screen.getByText('Hello world')).toBeInTheDocument();
    expect(screen.queryByRole('button')).toBeNull();
  });

  it('renders plain text with no chip for English sources', () => {
    render(
      <TranslatedText
        originalText="NVIDIA dominates accelerator shipments."
        sourceLanguage="en"
      />
    );

    expect(screen.getByText(/NVIDIA dominates/)).toBeInTheDocument();
    expect(screen.queryByText('EN')).toBeNull();
  });

  it('suppresses chip for BCP-47 English region tags (en-US, EN-GB)', () => {
    // External translation providers may emit region-suffixed tags.
    const { rerender } = render(
      <TranslatedText originalText="Hello" sourceLanguage="en-US" />
    );
    expect(screen.queryByText('EN-US')).toBeNull();

    rerender(<TranslatedText originalText="Hello" sourceLanguage="EN-GB" />);
    expect(screen.queryByText('EN-GB')).toBeNull();
  });

  it('shows translated text and language chip when source is non-English', () => {
    render(
      <TranslatedText
        originalText="半導体大手は増産計画を発表した。"
        translatedText="Major semiconductor firms announced expansion plans."
        sourceLanguage="ja"
      />
    );

    expect(screen.getByText(/Major semiconductor firms/)).toBeInTheDocument();
    expect(screen.getByText('JA')).toBeInTheDocument();
  });

  it('shows a confidence badge when translation_metadata.confidence is present', () => {
    render(
      <TranslatedText
        originalText="半導体大手は増産計画を発表した。"
        translatedText="Major semiconductor firms announced expansion plans."
        sourceLanguage="ja"
        translationMetadata={{ provider: 'deepl', confidence: 0.92 }}
      />
    );

    expect(screen.getByText('92%')).toBeInTheDocument();
  });

  it('reveals the original text when the expand toggle is clicked', () => {
    render(
      <TranslatedText
        originalText="半導体大手は増産計画を発表した。"
        translatedText="Major semiconductor firms announced expansion plans."
        sourceLanguage="ja"
      />
    );

    // Original text is not in the accessible DOM before expansion (MUI
    // Collapse with unmountOnExit keeps it out of the tree).
    expect(screen.queryByText('半導体大手は増産計画を発表した。')).toBeNull();

    const toggle = screen.getByRole('button', { name: /show original/i });
    fireEvent.click(toggle);

    expect(screen.getByText('半導体大手は増産計画を発表した。')).toBeInTheDocument();
  });

  it('does not show the expand toggle when translated text equals the original', () => {
    render(
      <TranslatedText
        originalText="Hello world"
        translatedText="Hello world"
        sourceLanguage="ja"
      />
    );

    // No-op translation (likely a translation-skipped path) should not
    // offer a misleading "Show original" toggle.
    expect(screen.queryByRole('button', { name: /show original/i })).toBeNull();
  });

  it('renders a dash placeholder when no text is available', () => {
    render(<TranslatedText originalText={null} translatedText={null} />);

    expect(screen.getByText('-')).toBeInTheDocument();
  });
});
