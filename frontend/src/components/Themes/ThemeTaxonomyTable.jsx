/**
 * ThemeTaxonomyTable — L1 grouped view of themes with expandable L2 children.
 *
 * Renders L1 parent themes as bold header rows with expand/collapse.
 * L2 children are loaded on-demand via React Query when expanded.
 */
import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  Box,
  Chip,
  CircularProgress,
  Collapse,
  IconButton,
  LinearProgress,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TablePagination,
  Typography,
} from '@mui/material';
import KeyboardArrowDownIcon from '@mui/icons-material/KeyboardArrowDown';
import KeyboardArrowRightIcon from '@mui/icons-material/KeyboardArrowRight';
import { getL1Rankings, getL1Children } from '../../api/themes';

const CATEGORY_COLORS = {
  technology: 'primary',
  healthcare: 'success',
  energy: 'warning',
  defense: 'error',
  financials: 'info',
  materials: 'default',
  consumer: 'secondary',
  industrials: 'default',
  macro: 'info',
  crypto: 'warning',
  real_estate: 'default',
  other: 'default',
};

const MomentumBar = ({ score }) => {
  if (score == null) return <Box sx={{ color: 'text.secondary', fontSize: '11px', fontFamily: 'monospace' }}>-</Box>;
  const color = score >= 70 ? 'success' : score >= 50 ? 'warning' : 'error';
  return (
    <Box sx={{ display: 'flex', alignItems: 'center', minWidth: 80 }}>
      <Box sx={{ width: '100%', mr: 0.5 }}>
        <LinearProgress variant="determinate" value={Math.min(score, 100)} color={color} sx={{ height: 6, borderRadius: 3 }} />
      </Box>
      <Box sx={{ minWidth: 28, fontSize: '11px', fontWeight: 600, fontFamily: 'monospace' }}>
        {score?.toFixed(0)}
      </Box>
    </Box>
  );
};

const formatReturn = (val) => {
  if (val == null) return '-';
  const pct = (val * 100).toFixed(1);
  const color = val >= 0 ? 'success.main' : 'error.main';
  return <Box component="span" sx={{ color, fontFamily: 'monospace', fontSize: '11px' }}>{val > 0 ? '+' : ''}{pct}%</Box>;
};

/** Expandable L2 children rows for a single L1 theme */
function L1ChildrenRows({ l1Id, open, onThemeClick }) {
  const { data, isLoading } = useQuery({
    queryKey: ['l1Children', l1Id],
    queryFn: () => getL1Children(l1Id),
    enabled: open,
    staleTime: 60000,
  });

  if (!open) return null;

  if (isLoading) {
    return (
      <TableRow>
        <TableCell colSpan={8} sx={{ py: 1, pl: 6 }}>
          <CircularProgress size={16} sx={{ mr: 1 }} />
          <Typography variant="caption" color="text.secondary">Loading children...</Typography>
        </TableCell>
      </TableRow>
    );
  }

  if (!data?.children?.length) {
    return (
      <TableRow>
        <TableCell colSpan={8} sx={{ py: 1, pl: 6 }}>
          <Typography variant="caption" color="text.secondary">No L2 themes assigned</Typography>
        </TableCell>
      </TableRow>
    );
  }

  return data.children.map((child) => (
    <TableRow
      key={child.id}
      hover
      sx={{
        bgcolor: 'action.hover',
        cursor: 'pointer',
        '&:hover': { bgcolor: 'action.selected' },
      }}
      onClick={() => onThemeClick?.({ id: child.id, name: child.display_name })}
    >
      <TableCell sx={{ pl: 6, width: 40 }} />
      <TableCell>
        <Box sx={{ pl: 2 }}>
          <Typography variant="body2" sx={{ fontSize: '12px' }}>
            {child.display_name}
          </Typography>
          {child.l1_assignment_method && (
            <Chip
              label={child.l1_assignment_method}
              size="small"
              variant="outlined"
              sx={{ height: 16, fontSize: '9px', ml: 1 }}
            />
          )}
        </Box>
      </TableCell>
      <TableCell>
        <Chip label={child.lifecycle_state} size="small" sx={{ height: 18, fontSize: '10px' }} />
      </TableCell>
      <TableCell align="center">
        <MomentumBar score={child.momentum_score} />
      </TableCell>
      <TableCell align="right">
        <Typography variant="body2" sx={{ fontFamily: 'monospace', fontSize: '11px' }}>
          {child.mentions_7d}
        </Typography>
      </TableCell>
      <TableCell align="right">
        <Typography variant="body2" sx={{ fontFamily: 'monospace', fontSize: '11px' }}>
          {child.num_constituents}
        </Typography>
      </TableCell>
      <TableCell />
      <TableCell />
    </TableRow>
  ));
}


export default function ThemeTaxonomyTable({
  pipeline = 'technical',
  categoryFilter = null,
  onThemeClick,
}) {
  const [expandedL1, setExpandedL1] = useState({});
  const [page, setPage] = useState(0);
  const pageSize = 50;

  const { data, isLoading } = useQuery({
    queryKey: ['l1Rankings', pipeline, categoryFilter, page],
    queryFn: () => getL1Rankings({ pipeline, category: categoryFilter, limit: pageSize, offset: page * pageSize }),
    staleTime: 60000,
  });

  const toggleExpand = (l1Id) => {
    setExpandedL1((prev) => ({ ...prev, [l1Id]: !prev[l1Id] }));
  };

  if (isLoading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="300px">
        <CircularProgress />
      </Box>
    );
  }

  const rankings = data?.rankings || [];
  const total = data?.total || 0;

  return (
    <Paper elevation={1}>
      <Box sx={{ p: 1.5, borderBottom: 1, borderColor: 'divider', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <Box sx={{ fontSize: '14px', fontWeight: 600 }}>
          L1 Theme Groups
        </Box>
        <Chip label={`${total} groups`} size="small" />
      </Box>

      <TableContainer sx={{ maxHeight: 'calc(100vh - 400px)' }}>
        <Table stickyHeader size="small">
          <TableHead>
            <TableRow>
              <TableCell sx={{ width: 40 }} />
              <TableCell>Theme Group</TableCell>
              <TableCell>Category</TableCell>
              <TableCell align="center">Momentum</TableCell>
              <TableCell align="right">Mentions 7d</TableCell>
              <TableCell align="right">Stocks</TableCell>
              <TableCell align="right">Return 1w</TableCell>
              <TableCell align="right">RS vs SPY</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {rankings.map((l1) => {
              const isExpanded = !!expandedL1[l1.id];
              return (
                <Box component="tbody" key={l1.id}>
                  <TableRow
                    hover
                    sx={{
                      bgcolor: 'grey.50',
                      cursor: 'pointer',
                      '& td': { fontWeight: 600 },
                    }}
                    onClick={() => toggleExpand(l1.id)}
                  >
                    <TableCell sx={{ width: 40 }}>
                      <IconButton size="small" sx={{ p: 0 }}>
                        {isExpanded ? <KeyboardArrowDownIcon /> : <KeyboardArrowRightIcon />}
                      </IconButton>
                    </TableCell>
                    <TableCell>
                      <Box display="flex" alignItems="center" gap={1}>
                        <Typography variant="body2" fontWeight="bold" sx={{ fontSize: '13px' }}>
                          {l1.display_name}
                        </Typography>
                        <Chip
                          label={`${l1.num_l2_children} themes`}
                          size="small"
                          variant="outlined"
                          sx={{ height: 18, fontSize: '10px' }}
                        />
                      </Box>
                    </TableCell>
                    <TableCell>
                      {l1.category && (
                        <Chip
                          label={l1.category}
                          size="small"
                          color={CATEGORY_COLORS[l1.category] || 'default'}
                          sx={{ height: 20, fontSize: '10px', textTransform: 'capitalize' }}
                        />
                      )}
                    </TableCell>
                    <TableCell align="center">
                      <MomentumBar score={l1.momentum_score} />
                    </TableCell>
                    <TableCell align="right">
                      <Typography variant="body2" sx={{ fontFamily: 'monospace', fontSize: '12px', fontWeight: 700 }}>
                        {l1.mentions_7d}
                      </Typography>
                    </TableCell>
                    <TableCell align="right">
                      <Typography variant="body2" sx={{ fontFamily: 'monospace', fontSize: '12px', fontWeight: 700 }}>
                        {l1.num_constituents}
                      </Typography>
                    </TableCell>
                    <TableCell align="right">
                      {formatReturn(l1.basket_return_1w)}
                    </TableCell>
                    <TableCell align="right">
                      <Typography variant="body2" sx={{ fontFamily: 'monospace', fontSize: '11px' }}>
                        {l1.basket_rs_vs_spy != null ? l1.basket_rs_vs_spy.toFixed(0) : '-'}
                      </Typography>
                    </TableCell>
                  </TableRow>
                  <L1ChildrenRows
                    l1Id={l1.id}
                    open={isExpanded}
                    onThemeClick={onThemeClick}
                  />
                </Box>
              );
            })}
          </TableBody>
        </Table>
      </TableContainer>

      {total > pageSize && (
        <TablePagination
          component="div"
          count={total}
          page={page}
          onPageChange={(_, newPage) => setPage(newPage)}
          rowsPerPage={pageSize}
          rowsPerPageOptions={[pageSize]}
        />
      )}
    </Paper>
  );
}
