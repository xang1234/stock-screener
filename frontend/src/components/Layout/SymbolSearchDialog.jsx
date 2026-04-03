import { useEffect, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  Box,
  CircularProgress,
  Dialog,
  DialogContent,
  DialogTitle,
  List,
  ListItemButton,
  ListItemText,
  TextField,
  Typography,
} from '@mui/material';
import { useNavigate } from 'react-router-dom';

import { searchStocks } from '../../api/stocks';

function SymbolSearchDialog({ open, onClose }) {
  const navigate = useNavigate();
  const [query, setQuery] = useState('');
  const [debouncedQuery, setDebouncedQuery] = useState('');
  const [highlightedIndex, setHighlightedIndex] = useState(0);

  useEffect(() => {
    if (!open) {
      setQuery('');
      setDebouncedQuery('');
      setHighlightedIndex(0);
      return;
    }

    const timer = setTimeout(() => {
      setDebouncedQuery(query.trim());
    }, 200);

    return () => clearTimeout(timer);
  }, [open, query]);

  const { data: results = [], isFetching } = useQuery({
    queryKey: ['stockSymbolSearch', debouncedQuery],
    queryFn: () => searchStocks(debouncedQuery, 8),
    enabled: open && debouncedQuery.length > 0,
    staleTime: 60_000,
  });

  useEffect(() => {
    setHighlightedIndex(0);
  }, [results.length, debouncedQuery]);

  const handleSelect = (symbol) => {
    if (!symbol) return;
    navigate(`/stock/${symbol.toUpperCase()}`);
    onClose();
  };

  const handleKeyDown = (event) => {
    if (event.key === 'ArrowDown') {
      event.preventDefault();
      if (results.length > 0) {
        setHighlightedIndex((prev) => (prev + 1) % results.length);
      }
      return;
    }

    if (event.key === 'ArrowUp') {
      event.preventDefault();
      if (results.length > 0) {
        setHighlightedIndex((prev) => (prev - 1 + results.length) % results.length);
      }
      return;
    }

    if (event.key === 'Enter') {
      event.preventDefault();
      if (results[highlightedIndex]) {
        handleSelect(results[highlightedIndex].symbol);
      } else if (query.trim()) {
        handleSelect(query.trim());
      }
      return;
    }

    if (event.key === 'Escape') {
      onClose();
    }
  };

  return (
    <Dialog
      open={open}
      onClose={onClose}
      fullWidth
      maxWidth="sm"
    >
      <DialogTitle>Find Symbol</DialogTitle>
      <DialogContent>
        <TextField
          autoFocus
          fullWidth
          placeholder="Type a symbol or company name"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          onKeyDown={handleKeyDown}
          margin="dense"
          inputProps={{ 'aria-label': 'Search symbols' }}
        />

        <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 1 }}>
          Press `/` anywhere in the app to open search. Use arrow keys to navigate results.
        </Typography>

        <Box sx={{ mt: 2, minHeight: 220 }}>
          {isFetching ? (
            <Box display="flex" justifyContent="center" alignItems="center" minHeight={180}>
              <CircularProgress size={24} />
            </Box>
          ) : results.length > 0 ? (
            <List dense>
              {results.map((result, index) => (
                <ListItemButton
                  key={result.symbol}
                  selected={index === highlightedIndex}
                  onClick={() => handleSelect(result.symbol)}
                  onMouseEnter={() => setHighlightedIndex(index)}
                >
                  <ListItemText
                    primary={`${result.symbol} ${result.name ? `· ${result.name}` : ''}`}
                    secondary={[result.sector, result.industry].filter(Boolean).join(' · ')}
                  />
                </ListItemButton>
              ))}
            </List>
          ) : debouncedQuery ? (
            <Box sx={{ pt: 4 }}>
              <Typography variant="body2" color="text.secondary">
                No active-universe matches found. Press Enter to open `{query.trim().toUpperCase()}` directly.
              </Typography>
            </Box>
          ) : (
            <Box sx={{ pt: 4 }}>
              <Typography variant="body2" color="text.secondary">
                Search the active universe by symbol or company name.
              </Typography>
            </Box>
          )}
        </Box>
      </DialogContent>
    </Dialog>
  );
}

export default SymbolSearchDialog;
