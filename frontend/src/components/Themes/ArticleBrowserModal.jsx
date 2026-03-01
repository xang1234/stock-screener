import { useState, useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  Box,
  Button,
  IconButton,
  TextField,
  InputAdornment,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TableSortLabel,
  TablePagination,
  Chip,
  CircularProgress,
  Typography,
  Link,
  Tooltip,
} from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import DownloadIcon from '@mui/icons-material/Download';
import SearchIcon from '@mui/icons-material/Search';
import OpenInNewIcon from '@mui/icons-material/OpenInNew';
import ArticleIcon from '@mui/icons-material/Article';
import { getContentItems, exportContentItems } from '../../api/themes';

const SOURCE_TYPE_OPTIONS = [
  { value: '', label: 'All Sources' },
  { value: 'substack', label: 'Substack' },
  { value: 'twitter', label: 'Twitter' },
  { value: 'news', label: 'News' },
  { value: 'reddit', label: 'Reddit' },
];

const SENTIMENT_OPTIONS = [
  { value: '', label: 'All Sentiments' },
  { value: 'bullish', label: 'Bullish' },
  { value: 'bearish', label: 'Bearish' },
  { value: 'neutral', label: 'Neutral' },
];

const ROWS_PER_PAGE_OPTIONS = [10, 25, 50, 100];

const sentimentColors = {
  bullish: 'success',
  bearish: 'error',
  neutral: 'default',
};

const processingStatusColors = {
  pending: 'default',
  in_progress: 'info',
  processed: 'success',
  failed_retryable: 'warning',
  failed_terminal: 'error',
};

const sourceTypeColors = {
  substack: '#FF6719',
  twitter: '#1DA1F2',
  news: '#6B7280',
  reddit: '#FF4500',
};

function ArticleBrowserModal({ open, onClose, pipeline }) {
  // State for filters and pagination
  const [search, setSearch] = useState('');
  const [debouncedSearch, setDebouncedSearch] = useState('');
  const [sourceType, setSourceType] = useState('');
  const [sentiment, setSentiment] = useState('');
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(25);
  const [orderBy, setOrderBy] = useState('published_at');
  const [order, setOrder] = useState('desc');
  const [exporting, setExporting] = useState(false);

  // Debounce search
  useMemo(() => {
    const timer = setTimeout(() => {
      setDebouncedSearch(search);
      setPage(0); // Reset to first page on search
    }, 300);
    return () => clearTimeout(timer);
  }, [search]);

  // Build query params
  const queryParams = useMemo(() => ({
    search: debouncedSearch || undefined,
    source_type: sourceType || undefined,
    sentiment: sentiment || undefined,
    pipeline: pipeline || undefined,
    limit: rowsPerPage,
    offset: page * rowsPerPage,
    sort_by: orderBy,
    sort_order: order,
  }), [debouncedSearch, sourceType, sentiment, pipeline, rowsPerPage, page, orderBy, order]);

  // Fetch content items
  const { data, isLoading, error } = useQuery({
    queryKey: ['contentItems', pipeline, queryParams],
    queryFn: () => getContentItems(queryParams),
    enabled: open,
    keepPreviousData: true,
  });

  const handleSort = (property) => {
    const isAsc = orderBy === property && order === 'asc';
    setOrder(isAsc ? 'desc' : 'asc');
    setOrderBy(property);
    setPage(0);
  };

  const handleChangePage = (event, newPage) => {
    setPage(newPage);
  };

  const handleChangeRowsPerPage = (event) => {
    setRowsPerPage(parseInt(event.target.value, 10));
    setPage(0);
  };

  const handleExport = async () => {
    try {
      setExporting(true);
      const exportParams = {
        search: debouncedSearch || undefined,
        source_type: sourceType || undefined,
        sentiment: sentiment || undefined,
        pipeline: pipeline || undefined,
        sort_by: orderBy,
        sort_order: order,
      };
      const blob = await exportContentItems(exportParams);

      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `theme_articles_${new Date().toISOString().slice(0, 10)}.csv`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Export failed:', error);
      alert('Failed to export articles. Please try again.');
    } finally {
      setExporting(false);
    }
  };

  const formatDate = (dateString) => {
    if (!dateString) return '-';
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
    });
  };

  return (
    <Dialog
      open={open}
      onClose={onClose}
      maxWidth="xl"
      fullWidth
      PaperProps={{ sx: { height: '90vh' } }}
    >
      <DialogTitle>
        <Box display="flex" justifyContent="space-between" alignItems="center">
          <Box display="flex" alignItems="center">
            <ArticleIcon sx={{ mr: 1, color: 'primary.main' }} />
            <Typography variant="h6">Browse Articles</Typography>
            {data && (
              <Chip
                label={`${data.total} items`}
                size="small"
                sx={{ ml: 2 }}
              />
            )}
            <Button
              size="small"
              variant="outlined"
              startIcon={exporting ? <CircularProgress size={14} /> : <DownloadIcon />}
              onClick={handleExport}
              disabled={!data?.total || exporting}
              sx={{ ml: 2 }}
            >
              {exporting ? 'Exporting...' : 'CSV'}
            </Button>
          </Box>
          <IconButton onClick={onClose} size="small">
            <CloseIcon />
          </IconButton>
        </Box>
      </DialogTitle>

      <DialogContent>
        {/* Filters Row */}
        <Box display="flex" gap={2} mb={2} flexWrap="wrap">
          {/* Search */}
          <TextField
            size="small"
            placeholder="Search title, source, or ticker..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            sx={{ minWidth: 280 }}
            InputProps={{
              startAdornment: (
                <InputAdornment position="start">
                  <SearchIcon fontSize="small" />
                </InputAdornment>
              ),
            }}
          />

          {/* Source Type Filter */}
          <FormControl size="small" sx={{ minWidth: 140 }}>
            <InputLabel>Source Type</InputLabel>
            <Select
              value={sourceType}
              label="Source Type"
              onChange={(e) => {
                setSourceType(e.target.value);
                setPage(0);
              }}
            >
              {SOURCE_TYPE_OPTIONS.map((opt) => (
                <MenuItem key={opt.value} value={opt.value}>
                  {opt.label}
                </MenuItem>
              ))}
            </Select>
          </FormControl>

          {/* Sentiment Filter */}
          <FormControl size="small" sx={{ minWidth: 140 }}>
            <InputLabel>Sentiment</InputLabel>
            <Select
              value={sentiment}
              label="Sentiment"
              onChange={(e) => {
                setSentiment(e.target.value);
                setPage(0);
              }}
            >
              {SENTIMENT_OPTIONS.map((opt) => (
                <MenuItem key={opt.value} value={opt.value}>
                  {opt.label}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </Box>

        {/* Table */}
        {isLoading ? (
          <Box display="flex" justifyContent="center" alignItems="center" minHeight={400}>
            <CircularProgress />
          </Box>
        ) : error ? (
          <Box display="flex" justifyContent="center" alignItems="center" minHeight={400}>
            <Typography color="error">Error loading articles: {error.message}</Typography>
          </Box>
        ) : (
          <>
            <TableContainer sx={{ maxHeight: 'calc(90vh - 280px)' }}>
              <Table stickyHeader size="small">
                <TableHead>
                  <TableRow>
                    <TableCell sx={{ minWidth: 250 }}>
                      <TableSortLabel
                        active={orderBy === 'title'}
                        direction={orderBy === 'title' ? order : 'asc'}
                        onClick={() => handleSort('title')}
                      >
                        Title
                      </TableSortLabel>
                    </TableCell>
                    <TableCell sx={{ minWidth: 100 }}>
                      <TableSortLabel
                        active={orderBy === 'source_name'}
                        direction={orderBy === 'source_name' ? order : 'asc'}
                        onClick={() => handleSort('source_name')}
                      >
                        Source
                      </TableSortLabel>
                    </TableCell>
                    <TableCell sx={{ minWidth: 80 }}>Type</TableCell>
                    <TableCell sx={{ minWidth: 100 }}>
                      <TableSortLabel
                        active={orderBy === 'published_at'}
                        direction={orderBy === 'published_at' ? order : 'asc'}
                        onClick={() => handleSort('published_at')}
                      >
                        Published
                      </TableSortLabel>
                    </TableCell>
                    <TableCell sx={{ minWidth: 120 }}>Status</TableCell>
                    <TableCell sx={{ minWidth: 180 }}>Themes</TableCell>
                    <TableCell sx={{ minWidth: 80 }}>Sentiment</TableCell>
                    <TableCell sx={{ minWidth: 150 }}>Tickers</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {data?.items?.map((item) => (
                    <TableRow key={item.id} hover>
                      {/* Title */}
                      <TableCell>
                        <Box sx={{ maxWidth: 300 }}>
                          {item.url ? (
                            <Link
                              href={item.url}
                              target="_blank"
                              rel="noopener noreferrer"
                              sx={{
                                display: 'flex',
                                alignItems: 'flex-start',
                                textDecoration: 'none',
                                color: 'primary.main',
                                '&:hover': { textDecoration: 'underline' },
                              }}
                            >
                              <Typography
                                variant="body2"
                                sx={{
                                  overflow: 'hidden',
                                  textOverflow: 'ellipsis',
                                  display: '-webkit-box',
                                  WebkitLineClamp: 2,
                                  WebkitBoxOrient: 'vertical',
                                  fontWeight: 500,
                                }}
                              >
                                {item.title || (item.content ? item.content.slice(0, 100) + (item.content.length > 100 ? '...' : '') : 'Untitled')}
                              </Typography>
                              <OpenInNewIcon sx={{ fontSize: 12, ml: 0.5, flexShrink: 0, mt: 0.5 }} />
                            </Link>
                          ) : (
                            <Typography
                              variant="body2"
                              sx={{
                                overflow: 'hidden',
                                textOverflow: 'ellipsis',
                                display: '-webkit-box',
                                WebkitLineClamp: 2,
                                WebkitBoxOrient: 'vertical',
                              }}
                            >
                              {item.title || (item.content ? item.content.slice(0, 100) + (item.content.length > 100 ? '...' : '') : 'Untitled')}
                            </Typography>
                          )}
                          {item.author && (
                            <Typography variant="caption" color="text.secondary">
                              by {item.author}
                            </Typography>
                          )}
                        </Box>
                      </TableCell>

                      {/* Source Name */}
                      <TableCell>
                        <Typography variant="body2" noWrap sx={{ maxWidth: 120 }}>
                          {item.source_name || '-'}
                        </Typography>
                      </TableCell>

                      {/* Source Type */}
                      <TableCell>
                        <Chip
                          label={item.source_type}
                          size="small"
                          sx={{
                            backgroundColor: sourceTypeColors[item.source_type] || '#6B7280',
                            color: 'white',
                            fontSize: '10px',
                            height: 20,
                          }}
                        />
                      </TableCell>

                      {/* Published Date */}
                      <TableCell>
                        <Typography variant="body2" sx={{ fontFamily: 'monospace', fontSize: '11px' }}>
                          {formatDate(item.published_at)}
                        </Typography>
                      </TableCell>

                      {/* Processing Status */}
                      <TableCell>
                        {item.processing_status ? (
                          <Chip
                            label={item.processing_status.replaceAll('_', ' ')}
                            size="small"
                            color={processingStatusColors[item.processing_status] || 'default'}
                            sx={{ fontSize: '10px', height: 20, textTransform: 'capitalize' }}
                          />
                        ) : (
                          <Typography variant="body2" color="text.secondary">-</Typography>
                        )}
                      </TableCell>

                      {/* Themes */}
                      <TableCell>
                        <Box display="flex" gap={0.5} flexWrap="wrap">
                          {item.themes.slice(0, 3).map((theme) => (
                            <Tooltip key={theme.id} title={theme.name}>
                              <Chip
                                label={theme.name.length > 15 ? `${theme.name.slice(0, 15)}...` : theme.name}
                                size="small"
                                variant="outlined"
                                color="primary"
                                sx={{ fontSize: '10px', height: 20 }}
                              />
                            </Tooltip>
                          ))}
                          {item.themes.length > 3 && (
                            <Tooltip title={item.themes.slice(3).map(t => t.name).join(', ')}>
                              <Chip
                                label={`+${item.themes.length - 3}`}
                                size="small"
                                sx={{ fontSize: '10px', height: 20 }}
                              />
                            </Tooltip>
                          )}
                        </Box>
                      </TableCell>

                      {/* Sentiment */}
                      <TableCell>
                        {item.primary_sentiment ? (
                          <Chip
                            label={item.primary_sentiment}
                            size="small"
                            color={sentimentColors[item.primary_sentiment] || 'default'}
                            sx={{ fontSize: '10px', height: 20 }}
                          />
                        ) : (
                          <Typography variant="body2" color="text.secondary">-</Typography>
                        )}
                      </TableCell>

                      {/* Tickers */}
                      <TableCell>
                        <Box display="flex" gap={0.25} flexWrap="wrap">
                          {item.tickers.slice(0, 4).map((ticker) => (
                            <Chip
                              key={ticker}
                              label={ticker}
                              size="small"
                              sx={{
                                fontSize: '9px',
                                height: 18,
                                backgroundColor: '#1976d2',
                                color: 'white',
                              }}
                            />
                          ))}
                          {item.tickers.length > 4 && (
                            <Tooltip title={item.tickers.slice(4).join(', ')}>
                              <Chip
                                label={`+${item.tickers.length - 4}`}
                                size="small"
                                sx={{ fontSize: '9px', height: 18 }}
                              />
                            </Tooltip>
                          )}
                        </Box>
                      </TableCell>
                    </TableRow>
                  ))}
                  {data?.items?.length === 0 && (
                    <TableRow>
                      <TableCell colSpan={8} align="center">
                        <Typography color="text.secondary" py={4}>
                          No articles found matching your filters
                        </Typography>
                      </TableCell>
                    </TableRow>
                  )}
                </TableBody>
              </Table>
            </TableContainer>

            {/* Pagination */}
            <TablePagination
              component="div"
              count={data?.total || 0}
              page={page}
              onPageChange={handleChangePage}
              rowsPerPage={rowsPerPage}
              onRowsPerPageChange={handleChangeRowsPerPage}
              rowsPerPageOptions={ROWS_PER_PAGE_OPTIONS}
            />
          </>
        )}
      </DialogContent>
    </Dialog>
  );
}

export default ArticleBrowserModal;
