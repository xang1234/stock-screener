import { useState } from 'react';
import { Box, Chip, Collapse, IconButton, Tooltip, Typography } from '@mui/material';
import ExpandLessIcon from '@mui/icons-material/ExpandLess';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';

function hasTranslation(translatedText, originalText) {
  if (!translatedText || !originalText) return false;
  return translatedText.trim() !== originalText.trim();
}

function displayLanguageCode(sourceLanguage) {
  if (!sourceLanguage) return null;
  const trimmed = sourceLanguage.trim();
  if (!trimmed) return null;
  if (trimmed.toLowerCase() === 'en') return null;
  return trimmed.toUpperCase();
}

function formatConfidence(confidence) {
  if (confidence === null || confidence === undefined) return null;
  const pct = Math.round(confidence * 100);
  return `${pct}%`;
}

function confidenceChipColor(confidence) {
  if (confidence === null || confidence === undefined) return 'default';
  if (confidence >= 0.9) return 'success';
  if (confidence >= 0.7) return 'default';
  return 'warning';
}

/**
 * Render text that may be a translation of a non-English source, with a
 * language chip, confidence badge, and an expandable "Show original"
 * disclosure. English sources (source_language absent or 'en') render
 * as plain text so the common case stays visually quiet.
 */
export default function TranslatedText({
  originalText,
  translatedText,
  sourceLanguage,
  translationMetadata,
  variant = 'body2',
  typographySx,
  boxSx,
}) {
  const [expanded, setExpanded] = useState(false);
  const languageCode = displayLanguageCode(sourceLanguage);
  const hasTranslatedVariant = hasTranslation(translatedText, originalText);
  const isTranslated = Boolean(languageCode && hasTranslatedVariant);

  const primaryText = isTranslated ? translatedText : originalText;
  const fallback = primaryText || originalText || '';

  if (!languageCode) {
    return (
      <Typography variant={variant} sx={typographySx} title={originalText || undefined}>
        {fallback || '-'}
      </Typography>
    );
  }

  const confidenceLabel = formatConfidence(translationMetadata?.confidence);
  const providerLabel = translationMetadata?.provider;
  const targetLanguage = translationMetadata?.target_language;

  const chipTooltip = [
    `Source language: ${languageCode}`,
    targetLanguage ? `Translated to: ${targetLanguage.toUpperCase()}` : null,
    providerLabel ? `Provider: ${providerLabel}` : null,
  ]
    .filter(Boolean)
    .join(' · ');

  return (
    <Box sx={boxSx}>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, flexWrap: 'wrap', mb: 0.5 }}>
        <Tooltip title={chipTooltip} arrow>
          <Chip
            label={languageCode}
            size="small"
            variant="outlined"
            sx={{ height: 18, fontSize: '0.65rem', fontWeight: 600 }}
          />
        </Tooltip>
        {confidenceLabel && (
          <Tooltip title="Translation confidence" arrow>
            <Chip
              label={confidenceLabel}
              size="small"
              color={confidenceChipColor(translationMetadata?.confidence)}
              variant="outlined"
              sx={{ height: 18, fontSize: '0.65rem' }}
            />
          </Tooltip>
        )}
        {isTranslated && originalText && (
          <Tooltip title={expanded ? 'Hide original' : 'Show original'} arrow>
            <IconButton
              size="small"
              onClick={(event) => {
                // Guard against callers rendering TranslatedText inside a <Link>
                // where the default/bubbled click would trigger navigation.
                event.preventDefault();
                event.stopPropagation();
                setExpanded((prev) => !prev);
              }}
              sx={{ p: 0.25 }}
              aria-label={expanded ? 'Hide original text' : 'Show original text'}
            >
              {expanded ? (
                <ExpandLessIcon sx={{ fontSize: 14 }} />
              ) : (
                <ExpandMoreIcon sx={{ fontSize: 14 }} />
              )}
            </IconButton>
          </Tooltip>
        )}
      </Box>
      <Typography variant={variant} sx={typographySx}>
        {fallback || '-'}
      </Typography>
      {isTranslated && originalText && (
        <Collapse in={expanded} unmountOnExit>
          <Box
            sx={{
              mt: 0.75,
              pt: 0.75,
              borderTop: 1,
              borderColor: 'divider',
              fontStyle: 'italic',
            }}
          >
            <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.25 }}>
              Original ({languageCode})
            </Typography>
            <Typography variant={variant} color="text.secondary" sx={typographySx}>
              {originalText}
            </Typography>
          </Box>
        </Collapse>
      )}
    </Box>
  );
}
