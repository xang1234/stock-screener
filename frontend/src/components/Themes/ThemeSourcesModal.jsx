import { useQuery } from '@tanstack/react-query';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Typography,
  Box,
  CircularProgress,
  Chip,
  IconButton,
  Tooltip,
  Link,
} from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import OpenInNewIcon from '@mui/icons-material/OpenInNew';
import ArticleIcon from '@mui/icons-material/Article';
import { getThemeMentions } from '../../api/themes';
import TranslatedText from '../common/TranslatedText';

// Sentiment color mapping
const getSentimentColor = (sentiment) => {
  switch (sentiment) {
    case 'bullish':
      return 'success';
    case 'bearish':
      return 'error';
    default:
      return 'default';
  }
};

// Confidence color
const getConfidenceColor = (confidence) => {
  if (confidence >= 0.8) return 'success.main';
  if (confidence >= 0.5) return 'warning.main';
  return 'text.secondary';
};

// Source type label
const sourceTypeLabels = {
  substack: 'Substack',
  twitter: 'Twitter',
  news: 'News',
  reddit: 'Reddit',
};

const attachmentStatusLabels = {
  pending: 'Attachments processing',
  processing: 'Attachments processing',
  complete: 'Attachments ready',
  partial: 'Attachments partially prepared',
  failed: 'Attachments unavailable',
};

function getSafeExternalUrl(url) {
  if (!url) return null;
  try {
    const parsedUrl = new URL(url, window.location.origin);
    return parsedUrl.protocol === 'http:' || parsedUrl.protocol === 'https:'
      ? parsedUrl.href
      : null;
  } catch {
    return null;
  }
}

export function AttachmentEvidence({ status, attachments }) {
  if (!status || status === 'none') return null;

  return (
    <Box sx={{ mt: 0.75 }}>
      <Chip
        label={attachmentStatusLabels[status] || 'Attachment status unavailable'}
        size="small"
        variant="outlined"
        color={status === 'failed' ? 'warning' : 'default'}
        sx={{ height: 20, fontSize: '0.68rem', mb: attachments?.length ? 0.5 : 0 }}
      />
      {attachments?.map((attachment) => {
        const safeUrl = getSafeExternalUrl(attachment.url);
        const attachmentLabel = attachment.kind === 'image' ? 'Image attachment' : 'Linked article';
        return (
          <Box key={`${attachment.kind}-${attachment.url}`} sx={{ fontSize: '0.72rem', lineHeight: 1.45 }}>
            {safeUrl ? (
              <Link href={safeUrl} target="_blank" rel="noopener noreferrer">
                {attachmentLabel}
              </Link>
            ) : (
              <Typography component="span" variant="caption">{attachmentLabel}</Typography>
            )}
            {attachment.status && attachment.status !== 'complete' && (
              <Typography component="span" variant="caption" color="text.secondary">
                {' '}({attachment.status})
              </Typography>
            )}
            {attachment.error_code && (
              <Typography variant="caption" color="text.secondary" display="block">
                Attachment unavailable: {attachment.error_code}
              </Typography>
            )}
            {attachment.warnings?.map((warning) => (
              <Typography key={warning} variant="caption" color="text.secondary" display="block">
                Attachment note: {warning}
              </Typography>
            ))}
          </Box>
        );
      })}
    </Box>
  );
}

function ThemeSourcesModal({ open, onClose, themeId, themeName }) {
  const { data: mentionsData, isLoading, error } = useQuery({
    queryKey: ['themeMentions', themeId],
    queryFn: () => getThemeMentions(themeId),
    enabled: open && !!themeId,
  });

  const formatDate = (dateString) => {
    if (!dateString) return '-';
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
    });
  };

  return (
    <Dialog
      open={open}
      onClose={onClose}
      maxWidth="xl"
      fullWidth
      PaperProps={{
        sx: { minHeight: '60vh', maxHeight: '90vh' },
      }}
    >
      <DialogTitle>
        <Box display="flex" justifyContent="space-between" alignItems="center">
          <Box display="flex" alignItems="center">
            <ArticleIcon sx={{ mr: 1, color: 'primary.main' }} />
            <Typography variant="h6">News Sources: {themeName}</Typography>
          </Box>
          <IconButton onClick={onClose} size="small">
            <CloseIcon />
          </IconButton>
        </Box>
      </DialogTitle>

      <DialogContent dividers>
        {isLoading && (
          <Box display="flex" justifyContent="center" p={5}>
            <CircularProgress />
          </Box>
        )}

        {error && (
          <Typography color="error">
            Error loading sources: {error.message}
          </Typography>
        )}

        {mentionsData && mentionsData.mentions?.length > 0 && (
          <>
            <Box mb={2}>
              <Typography variant="body2" color="text.secondary">
                Found {mentionsData.total_count} news items mentioning this
                theme
              </Typography>
            </Box>

            <TableContainer component={Paper} variant="outlined">
              <Table size="small" stickyHeader>
                <TableHead>
                  <TableRow>
                    <TableCell sx={{ minWidth: 250 }}>Title</TableCell>
                    <TableCell sx={{ minWidth: 300 }}>Excerpt</TableCell>
                    <TableCell align="center" sx={{ minWidth: 100 }}>
                      Source
                    </TableCell>
                    <TableCell sx={{ minWidth: 100 }}>Author</TableCell>
                    <TableCell align="center" sx={{ minWidth: 90 }}>
                      Sentiment
                    </TableCell>
                    <TableCell align="center" sx={{ minWidth: 90 }}>
                      Confidence
                    </TableCell>
                    <TableCell sx={{ minWidth: 100 }}>Date</TableCell>
                    <TableCell sx={{ minWidth: 120 }}>Tickers</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {mentionsData.mentions.map((mention) => (
                    <TableRow key={mention.mention_id} hover>
                      <TableCell>
                        <Box display="flex" alignItems="flex-start">
                          <Typography
                            variant="body2"
                            fontWeight="medium"
                            sx={{
                              maxWidth: 230,
                              overflow: 'hidden',
                              textOverflow: 'ellipsis',
                              whiteSpace: 'nowrap',
                            }}
                          >
                            {mention.content_title || 'Untitled'}
                          </Typography>
                          {mention.content_url && (
                            <Tooltip title="Open original">
                              <IconButton
                                size="small"
                                component={Link}
                                href={mention.content_url}
                                target="_blank"
                                rel="noopener noreferrer"
                                sx={{ ml: 0.5, p: 0.25 }}
                              >
                                <OpenInNewIcon sx={{ fontSize: 16 }} />
                              </IconButton>
                            </Tooltip>
                          )}
                        </Box>
                      </TableCell>

                      <TableCell>
                        {mention.claim_support?.theme === 'inferred' && (
                          <Typography variant="caption" color="text.secondary" display="block">
                            Theme inferred from source context
                          </Typography>
                        )}
                        {mention.development && (
                          <Typography variant="body2" sx={{ mb: 0.75 }}>
                            <strong>Development: </strong>{mention.development}
                          </Typography>
                        )}
                        <AttachmentEvidence
                          status={mention.attachment_status}
                          attachments={mention.attachments}
                        />
                        <TranslatedText
                          originalText={mention.excerpt}
                          translatedText={mention.translated_excerpt}
                          sourceLanguage={mention.source_language}
                          translationMetadata={mention.translation_metadata}
                          typographySx={{
                            display: '-webkit-box',
                            WebkitLineClamp: 3,
                            WebkitBoxOrient: 'vertical',
                            overflow: 'hidden',
                            maxWidth: 280,
                            color: 'text.secondary',
                          }}
                        />
                      </TableCell>

                      <TableCell align="center">
                        <Chip
                          label={
                            sourceTypeLabels[mention.source_type] ||
                            mention.source_type
                          }
                          size="small"
                          variant="outlined"
                        />
                        {mention.source_name && (
                          <Typography
                            variant="caption"
                            display="block"
                            color="text.secondary"
                          >
                            {mention.source_name}
                          </Typography>
                        )}
                      </TableCell>

                      <TableCell>
                        <Typography variant="body2">
                          {mention.author || '-'}
                        </Typography>
                      </TableCell>

                      <TableCell align="center">
                        {mention.sentiment ? (
                          <Chip
                            label={mention.sentiment}
                            size="small"
                            color={getSentimentColor(mention.sentiment)}
                            sx={{ textTransform: 'capitalize' }}
                          />
                        ) : (
                          '-'
                        )}
                      </TableCell>

                      <TableCell align="center">
                        <Typography
                          variant="body2"
                          fontWeight="medium"
                          color={getConfidenceColor(mention.confidence)}
                        >
                          {mention.confidence
                            ? `${(mention.confidence * 100).toFixed(0)}%`
                            : '-'}
                        </Typography>
                      </TableCell>

                      <TableCell>
                        <Typography variant="caption" color="text.secondary">
                          {formatDate(mention.published_at)}
                        </Typography>
                      </TableCell>

                      <TableCell>
                        <Box display="flex" gap={0.5} flexWrap="wrap">
                          {mention.tickers?.slice(0, 3).map((ticker) => (
                            <Chip
                              key={ticker}
                              label={ticker}
                              size="small"
                              variant="outlined"
                              sx={{ fontSize: '0.7rem', height: 20 }}
                            />
                          ))}
                          {mention.tickers?.length > 3 && (
                            <Chip
                              label={`+${mention.tickers.length - 3}`}
                              size="small"
                              sx={{ fontSize: '0.7rem', height: 20 }}
                            />
                          )}
                        </Box>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </>
        )}

        {mentionsData && mentionsData.mentions?.length === 0 && (
          <Typography color="text.secondary" textAlign="center" sx={{ p: 3 }}>
            No news sources found for this theme.
          </Typography>
        )}
      </DialogContent>

      <DialogActions>
        <Button onClick={onClose}>Close</Button>
      </DialogActions>
    </Dialog>
  );
}

export default ThemeSourcesModal;
