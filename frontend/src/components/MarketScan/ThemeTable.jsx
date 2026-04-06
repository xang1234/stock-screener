/**
 * Theme Table Component
 *
 * Displays stocks grouped by subgroups with:
 * - RS sparkline (30-day)
 * - Price sparkline (30-day)
 * - Price change bars for 1d, 5d, 2w, 1m, 3m
 *
 * Subgroups are collapsible sections.
 */
import { useState, Fragment } from 'react';
import {
  Box,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Typography,
  IconButton,
  Collapse,
  Paper,
} from '@mui/material';
import ExpandLessIcon from '@mui/icons-material/ExpandLess';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import RSSparkline from '../Scan/RSSparkline';
import PriceSparkline from '../Scan/PriceSparkline';
import PriceChangeBar from './PriceChangeBar';
import AddToWatchlistMenu from '../common/AddToWatchlistMenu';
import { updateSubgroup } from '../../api/userThemes';

const PRICE_PERIODS = [
  { key: '1d', label: '1D' },
  { key: '5d', label: '5D' },
  { key: '2w', label: '2W' },
  { key: '1m', label: '1M' },
  { key: '3m', label: '3M' },
];

function ThemeTable({ themeData, onRefresh }) {
  const [collapsedGroups, setCollapsedGroups] = useState({});

  const toggleGroup = async (subgroupId, currentState) => {
    const newState = !currentState;

    // Optimistic update
    setCollapsedGroups((prev) => ({
      ...prev,
      [subgroupId]: newState,
    }));

    // Persist to backend
    try {
      await updateSubgroup(subgroupId, { is_collapsed: newState });
    } catch (err) {
      console.error('Failed to update collapse state:', err);
      // Revert on error
      setCollapsedGroups((prev) => ({
        ...prev,
        [subgroupId]: currentState,
      }));
    }
  };

  const { subgroups, price_change_bounds } = themeData;

  if (!subgroups || subgroups.length === 0) {
    return (
      <Box textAlign="center" py={4}>
        <Typography color="text.secondary">
          No subgroups yet. Click the settings icon to add subgroups and stocks.
        </Typography>
      </Box>
    );
  }

  return (
    <TableContainer component={Paper} variant="outlined" sx={{ maxHeight: 'calc(100vh - 180px)' }}>
      <Table size="small" stickyHeader>
        <TableHead>
          <TableRow sx={{ '& th': { py: 0.5, fontSize: '0.75rem' } }}>
            <TableCell width={28} sx={{ bgcolor: 'background.paper', px: 0.5 }}></TableCell>
            <TableCell sx={{ bgcolor: 'background.paper', width: 45, maxWidth: 45, px: 0.5 }}>Symbol</TableCell>
            <TableCell sx={{ bgcolor: 'background.paper', width: 150, maxWidth: 150, px: 0.5 }}>Company</TableCell>
            <TableCell align="center" sx={{ bgcolor: 'background.paper', width: 140 }}>
              RS (30d)
            </TableCell>
            <TableCell align="center" sx={{ bgcolor: 'background.paper', width: 90 }}>
              Price (30d)
            </TableCell>
            {PRICE_PERIODS.map((period) => (
              <TableCell
                key={period.key}
                align="center"
                sx={{ bgcolor: 'background.paper', width: 95, px: 0.25 }}
              >
                {period.label}
              </TableCell>
            ))}
          </TableRow>
        </TableHead>
        <TableBody>
          {subgroups.map((subgroup) => {
            // Use local state if available, otherwise use server state
            const isCollapsed =
              collapsedGroups[subgroup.id] !== undefined
                ? collapsedGroups[subgroup.id]
                : subgroup.is_collapsed;

            return (
              <Fragment key={`sg-${subgroup.id}`}>
                {/* Subgroup Header Row */}
                <TableRow
                  sx={{
                    bgcolor: '#1F97F4',
                    color: '#fff',
                    cursor: 'pointer',
                    '&:hover': {
                      bgcolor: '#3AA3F6',
                    },
                  }}
                  onClick={() => toggleGroup(subgroup.id, isCollapsed)}
                >
                  <TableCell sx={{ p: 0, px: 0.5, width: 28 }}>
                    <IconButton size="small" sx={{ p: 0.25, color: 'inherit' }}>
                      {isCollapsed ? <ExpandMoreIcon fontSize="small" /> : <ExpandLessIcon fontSize="small" />}
                    </IconButton>
                  </TableCell>
                  <TableCell colSpan={4 + PRICE_PERIODS.length} sx={{ py: 0.5, color: 'inherit' }}>
                    <Typography variant="subtitle2" fontWeight={600} color="inherit">
                      {subgroup.name}
                      <Typography
                        component="span"
                        variant="caption"
                        sx={{ ml: 1, color: 'rgba(255, 255, 255, 0.8)' }}
                      >
                        ({subgroup.stocks.length} stocks)
                      </Typography>
                    </Typography>
                  </TableCell>
                </TableRow>

                {/* Stock Rows (collapsible) */}
                {!isCollapsed &&
                  subgroup.stocks.map((stock) => (
                    <TableRow key={stock.id} hover sx={{ '& td': { py: 0.25 } }}>
                      <TableCell width={28} sx={{ py: 0.25, px: 0.5 }}>
                        <AddToWatchlistMenu symbols={stock.symbol} size="small" />
                      </TableCell>
                      <TableCell sx={{ py: 0.25, px: 0.5, width: 45, maxWidth: 45 }}>
                        <Typography variant="body2" fontWeight={500} sx={{ fontSize: '0.75rem' }}>
                          {stock.symbol}
                        </Typography>
                      </TableCell>
                      <TableCell sx={{ py: 0.25, px: 0.5, maxWidth: 150, overflow: 'hidden' }}>
                        <Typography
                          variant="caption"
                          sx={{
                            fontSize: '0.6rem',
                            display: 'block',
                            overflow: 'hidden',
                            textOverflow: 'ellipsis',
                            whiteSpace: 'nowrap',
                          }}
                          title={stock.company_name || ''}
                        >
                          {stock.company_name || '-'}
                        </Typography>
                      </TableCell>
                      <TableCell align="center" sx={{ p: '2px' }}>
                        <RSSparkline
                          data={stock.rs_data}
                          trend={stock.rs_trend}
                          width={130}
                          height={24}
                        />
                      </TableCell>
                      <TableCell align="center" sx={{ p: '2px' }}>
                        <PriceSparkline
                          data={stock.price_data}
                          trend={stock.price_trend}
                          change1d={stock.change_1d}
                          width={80}
                          height={22}
                          showChange={false}
                        />
                      </TableCell>
                      {PRICE_PERIODS.map((period) => {
                        const changeKey = `change_${period.key}`;
                        const change = stock[changeKey];
                        const bounds = price_change_bounds[period.key] || { min: 0, max: 0 };
                        return (
                          <TableCell key={period.key} align="center" sx={{ p: '1px 2px' }}>
                            <PriceChangeBar
                              value={change}
                              min={bounds.min}
                              max={bounds.max}
                              width={90}
                              height={16}
                            />
                          </TableCell>
                        );
                      })}
                    </TableRow>
                  ))}
              </Fragment>
            );
          })}
        </TableBody>
      </Table>
    </TableContainer>
  );
}

export default ThemeTable;
